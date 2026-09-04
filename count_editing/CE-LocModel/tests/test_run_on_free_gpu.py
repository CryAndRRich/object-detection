"""Test cho tools/run_on_free_gpu.py — chọn GPU và thử lại khi job chết.

Chạy: python tests/test_run_on_free_gpu.py   (không cần GPU, nvidia-smi giả lập)

Lịch sử để không lặp lại: script này từng có ngưỡng free-memory tối thiểu, và
ngưỡng đó hỏng HAI lần theo hai cách khác nhau:

  1. Hard-code 15000 MiB cho mọi job (2026-08-31): variant (d) và cả 2 job eval
     bị bỏ qua trong 1 giây, dù GPU còn 10.607 MiB thừa sức chạy.
  2. Suy ngưỡng từ tên script + tham số (2026-09-04):
     tools/visualize_predictions.py bị đoán nhầm thành job train (10.580 MiB)
     chỉ vì tên không chứa "eval", rồi chờ GPU vô ích 2 tiếng.

Giờ KHÔNG còn ngưỡng: cứ chọn GPU trống nhất rồi chạy. `test_never_refuses_to_run`
là chốt chặn cho cả hai kiểu hỏng trên.
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "tools", "run_on_free_gpu.py")


def load():
    """Nạp lại module sạch mỗi test (các test monkeypatch query_gpus/subprocess)."""
    spec = importlib.util.spec_from_file_location("rofg", _SRC)
    m = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = ["run_on_free_gpu.py"]
    spec.loader.exec_module(m)
    sys.argv = old
    return m


def run_main(m, argv):
    sys.argv = ["run_on_free_gpu.py"] + argv
    try:
        m.main()
    except SystemExit as e:
        return e.code
    return None


def stub_run(m, rc=0, record=None):
    class R:
        returncode = rc

    def run(cmd, env=None, cwd=None):
        if record is not None:
            record.append(env["CUDA_VISIBLE_DEVICES"])
        return R()
    m.subprocess.run = run


# --- Chọn GPU -------------------------------------------------------------

def test_picks_most_free_among_idle():
    m = load()
    m.query_gpus = lambda: [(0, 5000, 24576, 0), (1, 20000, 24576, 3), (2, 24000, 24576, 99)]
    (idx, free, _, _), had_idle = m.pick_gpu(m.query_gpus())
    assert idx == 1 and had_idle is True, (idx, had_idle)
    print("OK chọn GPU rảnh compute có nhiều free memory nhất (bỏ qua GPU 2 dù trống hơn)")


def test_falls_back_when_all_busy():
    m = load()
    m.query_gpus = lambda: [(0, 3000, 24576, 95), (1, 9000, 24576, 99)]
    (idx, _, _, _), had_idle = m.pick_gpu(m.query_gpus())
    assert idx == 1 and had_idle is False, (idx, had_idle)
    print("OK mọi GPU đều bận -> vẫn chọn cái trống nhất, không từ chối chạy")


def test_never_refuses_to_run():
    """HỒI QUY cho cả hai lần hỏng của cơ chế ngưỡng: dù GPU trống rất ít, job
    vẫn PHẢI được chạy. Nếu thiếu VRAM thật thì torch báo OOM rõ ràng trong vài
    giây, tốt hơn hẳn việc ngồi chờ hàng giờ rồi bỏ cuộc."""
    m = load()
    m.query_gpus = lambda: [(0, 700, 24576, 0), (1, 300, 24576, 100)]
    seen = []
    stub_run(m, 0, seen)
    rc = run_main(m, ["--", "tools/visualize_predictions.py", "--checkpoint", "a.pth"])
    assert rc == 0 and seen == ["0"], (rc, seen)
    print("OK GPU chỉ còn 700 MiB -> VẪN chạy (không có ngưỡng nào chặn)")


def test_forced_gpu_skips_detection():
    m = load()
    def boom():
        raise AssertionError("không được gọi nvidia-smi khi đã ép --gpu")
    m.query_gpus = boom
    seen = []
    stub_run(m, 0, seen)
    rc = run_main(m, ["--gpu", "2", "--", "train_w_args.py"])
    assert rc == 0 and seen == ["2"], (rc, seen)
    print("OK --gpu ép được, bỏ qua auto-detect hoàn toàn")


# --- Thử lại --------------------------------------------------------------

def test_retries_then_gives_up():
    """Job chết phải thử lại đúng --retries lần rồi dừng, không lặp vô hạn:
    một bug code thật sẽ hỏng mọi lần."""
    m = load()
    m.query_gpus = lambda: [(1, 21000, 24576, 0)]
    n = {"run": 0}

    class RBad:
        returncode = 1

    def run(cmd, env=None, cwd=None):
        n["run"] += 1
        return RBad()
    m.subprocess.run = run
    m.time.sleep = lambda s: None

    rc = run_main(m, ["--retries", "2", "--", "train_w_args.py"])
    assert rc == 1 and n["run"] == 3, (rc, n)
    print(f"OK job chết -> chạy {n['run']} lần (1 + 2 retry) rồi dừng")


def test_rereads_nvidia_smi_each_retry():
    """Mỗi lần thử lại phải đọc nvidia-smi MỚI, để có thể chuyển sang GPU khác
    khi GPU cũ vừa bị job khác chiếm."""
    m = load()
    calls = {"smi": 0}

    def q():
        calls["smi"] += 1
        return ([(0, 20000, 24576, 0)] if calls["smi"] == 1
                else [(0, 500, 24576, 99), (1, 22000, 24576, 0)])
    m.query_gpus = q
    seen = []
    n = {"run": 0}

    class R:
        returncode = 0

    class RBad:
        returncode = 1

    def run(cmd, env=None, cwd=None):
        n["run"] += 1
        seen.append(env["CUDA_VISIBLE_DEVICES"])
        return RBad() if n["run"] == 1 else R()
    m.subprocess.run = run
    m.time.sleep = lambda s: None

    rc = run_main(m, ["--retries", "2", "--", "train_w_args.py"])
    assert rc == 0 and seen == ["0", "1"], (rc, seen)
    print(f"OK lần 1 GPU {seen[0]} chết -> đọc lại nvidia-smi -> lần 2 đổi sang GPU {seen[1]}")


def test_success_runs_once():
    m = load()
    m.query_gpus = lambda: [(0, 20000, 24576, 0)]
    n = {"run": 0}

    class R:
        returncode = 0

    def run(cmd, env=None, cwd=None):
        n["run"] += 1
        return R()
    m.subprocess.run = run
    rc = run_main(m, ["--", "train_w_args.py"])
    assert rc == 0 and n["run"] == 1, (rc, n)
    print("OK job chạy ngon -> đúng 1 lần, không thử lại thừa")


def test_no_threshold_flags_left():
    """Cờ --min-free-mib / --wait-for-gpu đã bị xoá hẳn; lệnh cũ dùng chúng
    phải báo lỗi rõ ràng thay vì âm thầm bỏ qua."""
    m = load()
    m.query_gpus = lambda: [(0, 20000, 24576, 0)]
    stub_run(m, 0)
    try:
        rc = run_main(m, ["--min-free-mib", "5000", "--", "train_w_args.py"])
    except SystemExit as e:
        rc = e.code
    assert rc == 2, f"argparse phải từ chối cờ đã xoá, nhận rc={rc}"
    print("OK --min-free-mib đã xoá hẳn, lệnh cũ báo lỗi thay vì im lặng")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)}/{len(fns)} test PASS")

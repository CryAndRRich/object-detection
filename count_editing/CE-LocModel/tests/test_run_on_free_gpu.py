"""Test cho tools/run_on_free_gpu.py — chọn GPU, chờ GPU, suy ngưỡng bộ nhớ.

Chạy: python tests/test_run_on_free_gpu.py   (không cần GPU, nvidia-smi được giả lập)

Hai lỗi thật mà bộ test này chốt lại để không tái diễn:
  1. (2026-08-31) Không chọn được GPU thì script THOÁT NGAY thay vì chờ, nên
     variant (d) và cả 2 job eval bị bỏ qua trong đúng 1 giây khi job trước vừa
     nhả GPU chưa kịp.
  2. Ngưỡng --min-free-mib hard-code 15000 MiB cho MỌI job, không đo từ đâu.
     Thực tế (d) chỉ cần ~3400 MiB, eval ~1200 MiB — thừa hơn 4x, và chính nó
     là nguyên nhân trực tiếp của (1) trong lần chạy đó.
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "tools", "run_on_free_gpu.py")


def load():
    """Nạp lại module sạch mỗi test (các test có monkeypatch query_gpus/subprocess)."""
    spec = importlib.util.spec_from_file_location("rofg", _SRC)
    m = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = ["run_on_free_gpu.py"]
    spec.loader.exec_module(m)
    sys.argv = old_argv
    return m


def run_main(m, argv):
    sys.argv = ["run_on_free_gpu.py"] + argv
    try:
        m.main()
    except SystemExit as e:
        return e.code
    return None


# --- 1. Chờ GPU -----------------------------------------------------------

def test_waits_then_runs():
    m = load()
    calls = {"smi": 0, "run": 0, "dev": None}

    def q():
        calls["smi"] += 1
        if calls["smi"] <= 3:
            return [(0, 2000, 24576, 100), (1, 1000, 24576, 99)]
        return [(0, 2000, 24576, 100), (1, 21000, 24576, 0)]
    m.query_gpus = q

    class R:
        returncode = 0

    def run(cmd, env=None, cwd=None):
        calls["run"] += 1
        calls["dev"] = env["CUDA_VISIBLE_DEVICES"]
        return R()
    m.subprocess.run = run
    m.time.sleep = lambda s: None

    rc = run_main(m, ["--wait-for-gpu", "3600", "--gpu-poll", "1", "--", "train_w_args.py"])
    assert rc == 0, rc
    assert calls["smi"] == 4 and calls["run"] == 1 and calls["dev"] == "1", calls
    print(f"OK chờ qua {calls['smi'] - 1} lần GPU đầy rồi chạy trên GPU {calls['dev']}")


def test_wait_budget_is_finite():
    m = load()
    n = {"smi": 0}

    def q():
        n["smi"] += 1
        if n["smi"] > 500:
            raise RuntimeError("LẶP VÔ HẠN")
        return [(0, 100, 24576, 100)]
    m.query_gpus = q
    t = [0.0]
    m.time.time = lambda: t[0]
    m.time.sleep = lambda s: t.__setitem__(0, t[0] + s)

    rc = run_main(m, ["--wait-for-gpu", "600", "--gpu-poll", "100", "--", "train_w_args.py"])
    assert rc == 1, rc
    assert n["smi"] == 7, n["smi"]  # t = 0,100,...,600
    print(f"OK dừng sau {n['smi']} lần thử, hết đúng ngân sách 600s (không lặp vô hạn)")


def test_default_no_wait():
    """Mặc định --wait-for-gpu 0 giữ nguyên hành vi cũ: thoát ngay."""
    m = load()
    n = {"smi": 0}

    def q():
        n["smi"] += 1
        return [(0, 100, 24576, 100)]
    m.query_gpus = q
    m.time.sleep = lambda s: None

    rc = run_main(m, ["--", "train_w_args.py"])
    assert rc == 1 and n["smi"] == 1, (rc, n)
    print("OK mặc định vẫn thoát ngay sau 1 lần đọc (không hồi quy)")


def test_dead_job_uses_retries_not_wait():
    """Job CHẾT là chuyện khác hẳn thiếu GPU: phải giới hạn theo --retries,
    nếu không một bug code thật sẽ chạy lại vô tận."""
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

    rc = run_main(m, ["--retries", "2", "--wait-for-gpu", "3600", "--", "train_w_args.py"])
    assert rc == 1 and n["run"] == 3, (rc, n)
    print(f"OK job chết -> chạy {n['run']} lần (1 + 2 retry) rồi dừng, không chờ vô hạn")


# --- 2. Chọn GPU ----------------------------------------------------------

def test_memory_is_hard_filter():
    """GPU rảnh compute nhưng thiếu memory KHÔNG bao giờ được chọn — đó đúng là
    thứ gây OOM."""
    m = load()
    gpus = [(0, 1000, 24576, 0), (1, 20000, 24576, 95)]
    picked, had_idle = m.pick_gpu(gpus, 15000)
    assert picked[0] == 1 and had_idle is False, (picked, had_idle)
    print("OK bỏ qua GPU rảnh-nhưng-thiếu-memory, chọn GPU đủ memory dù đang bận")


def test_prefers_idle_among_fitting():
    m = load()
    gpus = [(0, 24000, 24576, 99), (1, 16000, 24576, 5)]
    picked, had_idle = m.pick_gpu(gpus, 15000)
    assert picked[0] == 1 and had_idle is True, picked
    print("OK trong số GPU đủ memory, ưu tiên cái rảnh compute dù ít memory hơn")


# --- 3. Suy ngưỡng bộ nhớ -------------------------------------------------

def test_threshold_scales_with_batch_size():
    m = load()
    vals = [m.infer_min_free_mib(f"train_w_args.py --batch_size {b}".split())
            for b in (1, 2, 4, 8, 16, 32, 64, 128)]
    assert vals == sorted(vals) and len(set(vals)) == len(vals), vals
    print(f"OK ngưỡng tăng đơn điệu theo batch_size: {vals}")


def test_eval_cheaper_than_train():
    """Eval không giữ activation cho backward, không có optimizer state -> phải
    rẻ hơn hẳn train. Ngưỡng cũ 15000 áp chung cho cả hai là sai bản chất."""
    m = load()
    tr = m.infer_min_free_mib("train_w_args.py --batch_size 32".split())
    ev = m.infer_min_free_mib("eval_detection.py --variant x --k 30".split())
    assert ev < tr, (ev, tr)
    assert tr < 15000 and ev < 15000, (tr, ev)
    print(f"OK train({tr}) > eval({ev}), cả hai đều < 15000 cũ")


def test_eval_scales_with_k():
    """Nhánh 1-box chạy K mẫu SONG SONG trong 1 batch nên K mới là thứ quyết
    định bộ nhớ của eval, không phải batch_size."""
    m = load()
    a = m.infer_min_free_mib("eval_detection.py --k 30".split())
    b = m.infer_min_free_mib("eval_detection.py --k 300".split())
    assert b > a, (a, b)
    print(f"OK eval k=30 -> {a} MiB, k=300 -> {b} MiB")


def test_bad_flag_values_dont_crash():
    m = load()
    base = m.infer_min_free_mib("train_w_args.py".split())
    assert m.infer_min_free_mib("train_w_args.py --batch_size abc".split()) == base
    assert m.infer_min_free_mib("train_w_args.py --batch_size".split()) > 0
    assert m.infer_min_free_mib("train_w_args.py --batch_size=32".split()) == \
        m.infer_min_free_mib("train_w_args.py --batch_size 32".split())
    print("OK giá trị flag rác/thiếu -> rơi về mặc định; parse được cả dạng --flag=giá_trị")


def test_real_commands_fit_yesterdays_gpu():
    """Hồi quy trực tiếp cho sự cố 2026-08-31: GPU 1 khi đó còn 10607 MiB free
    và cả 3 job đều bị bỏ qua. Với ngưỡng suy ra, cả 3 đều phải chạy được."""
    m = load()
    free_that_night = 10607
    cmds = {
        "(d) train": "train_w_args.py --variant detect/d_coco_classhead --task coco "
                     "--epochs 60 --batch_size 32 --lr 5e-5 --num_workers 8",
        "eval (c)": "eval_detection.py --checkpoint a.pth --variant detect/c_transformer_multibox "
                    "--split test --nms 0.5 --box_renewal --use_ensemble",
        "eval (d)": "eval_detection.py --checkpoint a.pth --variant detect/d_coco_classhead "
                    "--dataset coco --nms 0.5 --box_renewal --use_ensemble",
    }
    for name, c in cmds.items():
        v = m.infer_min_free_mib(c.split())
        assert v <= free_that_night, f"{name}: {v} > {free_that_night}"
        print(f"OK {name:<10} cần {v:5d} MiB <= {free_that_night} MiB mà GPU 1 đang có tối hôm đó")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)}/{len(fns)} test PASS")

"""Chạy một script Python bất kỳ (train_w_args.py, eval_detection.py, ...) trên
GPU trống nhất của server, tự động chọn qua nvidia-smi.

Port lại nguyên logic từ
object-detection/diffu_grounding_dino/tools/run_train.py — cùng lý do: server
này chạy chung với người/job khác, nên không thể cứ mặc định GPU 0. Khác với
bản gốc (chỉ wrap main.py cố định), script này wrap MỘT SCRIPT PYTHON BẤT KỲ
truyền vào — vì CE-LocModel có nhiều entry point cần chạy trên server
(train_w_args.py, eval_detection.py, test_mul_box.py, tools/*.py).

  1. Query nvidia-smi cho mọi GPU: free memory + utilization.
  2. Chọn GPU TRỐNG NHẤT: ưu tiên GPU không bận tính toán (utilization dưới
     ngưỡng); trong cùng nhóm thì lấy cái nhiều free memory nhất.
  3. Set CUDA_VISIBLE_DEVICES=<gpu đã chọn> rồi chạy script Python truyền vào
     với mọi tham số còn lại.

KHÔNG có ngưỡng free-memory tối thiểu. Đã thử hai lần và hỏng cả hai:

  - Bản 1 hard-code 15000 MiB cho mọi job. 2026-08-31: variant (d) và cả 2 job
    eval bị bỏ qua trong đúng 1 giây, dù GPU 1 còn 10.607 MiB thừa sức chạy.
  - Bản 2 suy ngưỡng từ tên script + tham số. 2026-09-04:
    tools/visualize_predictions.py bị đoán nhầm là job train (10.580 MiB) chỉ
    vì tên nó không chứa "eval", rồi ngồi chờ GPU suốt 2 tiếng vô ích.

Bài học: đoán nhu cầu bộ nhớ từ dòng lệnh luôn sai ở đâu đó, và khi sai thì
hậu quả (job không chạy) tệ hơn hẳn thứ nó định phòng (OOM, mà torch báo lỗi
rõ ràng và chỉ mất vài giây). Cứ chọn GPU trống nhất rồi chạy; nếu thật sự
không đủ VRAM thì traceback OOM nói thẳng, và --retries sẽ thử lại trên GPU
khác (mỗi lần thử đọc nvidia-smi mới).

Cách dùng — script Python cần chạy đứng ngay sau --, mọi tham số của NÓ đứng
sau đó:

    python tools/run_on_free_gpu.py -- train_w_args.py --variant detect/e_cosine_multibox
    python tools/run_on_free_gpu.py -- eval_detection.py --checkpoint ... --variant ...
    python tools/run_on_free_gpu.py -- tools/visualize_predictions.py --checkpoint ...

Muốn ép 1 GPU cụ thể: --gpu <index> (đặt TRƯỚC --).
"""

import argparse
import os
import subprocess
import time
import sys

UTIL_BUSY_THRESHOLD = 50  # phần trăm; GPU trên ngưỡng này coi là đang bận tính toán


def query_gpus():
    """``[(index, free_mib, total_mib, util_percent), ...]`` từ nvidia-smi."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    gpus = []
    for line in out.strip().splitlines():
        idx, used, total, util = (int(x.strip()) for x in line.split(","))
        gpus.append((idx, total - used, total, util))
    return gpus


def pick_gpu(gpus):
    """GPU trống nhất: ``(gpu, had_idle)``.

    Ưu tiên GPU rảnh compute (``utilization < UTIL_BUSY_THRESHOLD``); trong
    nhóm đã chọn thì lấy cái nhiều free memory nhất. Không loại GPU nào vì
    thiếu memory — xem docstring đầu file.
    """
    idle = [g for g in gpus if g[3] < UTIL_BUSY_THRESHOLD]
    pool = idle if idle else gpus
    return max(pool, key=lambda g: g[1]), bool(idle)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gpu", type=int, default=None,
                        help="ép chạy trên GPU index cụ thể, bỏ qua auto-detect")
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="số lần thử lại khi job chết. Server dùng chung nên GPU có thể bị job khác "
        "chiếm mất trong khoảng giữa lúc đọc nvidia-smi và lúc thật sự cấp phát -- lần thử "
        "lại đọc nvidia-smi MỚI nên có thể chọn GPU khác.",
    )
    parser.add_argument("--retry-wait", type=int, default=60,
                        help="số giây chờ trước mỗi lần thử lại")
    parser.add_argument(
        "target", nargs=argparse.REMAINDER,
        help="script Python cần chạy + tham số của nó, đặt sau --, ví dụ: "
        "-- train_w_args.py --variant detect/e_cosine_multibox",
    )
    args = parser.parse_args()

    target = args.target
    if target and target[0] == "--":
        target = target[1:]
    if not target:
        parser.error("thiếu script cần chạy — đặt sau --, ví dụ: -- train_w_args.py --variant a_cnn_1box")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable] + target

    for attempt in range(1, args.retries + 2):
        rc = _pick_and_run(args, cmd, repo_root)
        if rc == 0:
            raise SystemExit(0)
        if attempt > args.retries:
            print(f"HẾT {args.retries} lần thử lại, lần nào cũng thoát != 0 -- dừng. Nếu traceback "
                  f"ở trên là CUDA out of memory thì GPU đang bị chiếm liên tục (đợi rồi chạy lại, "
                  f"hoặc ép --gpu <id>, hoặc giảm --batch_size). Nếu là lỗi khác thì thử lại không "
                  f"giúp được gì -- sửa code.", flush=True)
            raise SystemExit(rc)
        print(f"\n[thử lại {attempt}/{args.retries}] job thoát với mã {rc}. Trên server dùng chung "
              f"nguyên nhân thường gặp là OOM (GPU bị chiếm mất giữa lúc đọc nvidia-smi và lúc cấp "
              f"phát), nhưng mã thoát không phân biệt được OOM với lỗi code -- NẾU cả {args.retries} "
              f"lần đều hỏng thì đọc traceback ở trên, rất có thể là bug thật. "
              f"Đợi {args.retry_wait}s rồi đọc lại trạng thái GPU...\n", flush=True)
        time.sleep(args.retry_wait)


def _pick_and_run(args, cmd, repo_root):
    if args.gpu is not None:
        chosen_index = args.gpu
        print(f"dùng GPU {chosen_index} (ép bằng --gpu)", flush=True)
    else:
        gpus = query_gpus()
        print("trạng thái GPU hiện tại:", flush=True)
        for idx, free, total, util in gpus:
            print(f"  GPU {idx}: free {free:6d} MiB / {total} MiB, utilization {util:3d}%")
        (chosen_index, free, total, util), had_idle = pick_gpu(gpus)
        if had_idle:
            print(f"chọn GPU {chosen_index} (free {free} MiB, utilization {util}%)", flush=True)
        else:
            print(f"chọn GPU {chosen_index} (free {free} MiB, utilization {util}%) -- MỌI GPU đều "
                  f"đang bận compute (>{UTIL_BUSY_THRESHOLD}%), nên job sẽ tranh chấp và chạy chậm "
                  f"hơn, nhưng vẫn chạy.", flush=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(chosen_index)

    print(f"chạy: CUDA_VISIBLE_DEVICES={chosen_index} {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env, cwd=repo_root).returncode


if __name__ == "__main__":
    main()

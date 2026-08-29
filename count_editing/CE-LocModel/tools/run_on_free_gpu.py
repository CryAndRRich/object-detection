"""Chạy một script Python bất kỳ (train_w_args.py, test_mul_box.py, ...) trên
GPU trống nhất của server, tự động chọn qua nvidia-smi.

Port lại nguyên logic từ
object-detection/diffu_grounding_dino/tools/run_train.py — cùng lý do: server
này chạy chung với người/job khác, nên không thể cứ mặc định GPU 0. Khác với
bản gốc (chỉ wrap main.py cố định), script này wrap MỘT SCRIPT PYTHON BẤT KỲ
truyền vào — vì CE-LocModel có 2 entry point cần chạy trên server
(train_w_args.py, test_mul_box.py), không phải 1.

  1. Query nvidia-smi cho mọi GPU: free memory + utilization.
  2. Loại trước các GPU không đủ ``--min-free-mib`` free memory -- điều kiện
     CỨNG, không GPU nào qua được thì thoát lỗi thay vì chọn liều rồi OOM
     giữa chừng.
  3. Trong số các GPU đã đủ memory, ưu tiên GPU KHÔNG bận tính toán
     (utilization dưới ngưỡng); nếu tất cả đều bận tính toán, chọn GPU nhiều
     free memory nhất trong số đó rồi cảnh báo.
  4. Set CUDA_VISIBLE_DEVICES=<gpu đã chọn> rồi exec script Python truyền vào
     với mọi tham số còn lại.

Cách dùng — script Python cần chạy đứng ngay sau --, mọi tham số của NÓ đứng
sau đó:

    python tools/run_on_free_gpu.py -- train_w_args.py --variant resnet34_transformer
    python tools/run_on_free_gpu.py -- test_mul_box.py --checkpoint checkpoints/resnet34_transformer/best_model.pth --variant resnet34_transformer

Muốn ép 1 GPU cụ thể: --gpu <index> (đặt TRƯỚC --).
"""

import argparse
import os
import subprocess
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


def pick_gpu(gpus, min_free_mib):
    """``(None, False)`` if no GPU has enough free memory; otherwise ``(gpu, had_idle)``.

    Memory is filtered FIRST and is a hard requirement -- a GPU below
    ``min_free_mib`` is never chosen no matter how idle its compute is, since
    that is exactly what OOMs. Only among the GPUs that fit is idle compute
    (``utilization < UTIL_BUSY_THRESHOLD``) preferred.
    """
    fits = [g for g in gpus if g[1] >= min_free_mib]
    if not fits:
        return None, False
    idle = [g for g in fits if g[3] < UTIL_BUSY_THRESHOLD]
    pool = idle if idle else fits
    chosen = max(pool, key=lambda g: g[1])
    return chosen, bool(idle)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gpu", type=int, default=None, help="ép chạy trên GPU index cụ thể, bỏ qua auto-detect")
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=15000,
        help="loại GPU có free memory dưới mức này trước khi xét utilization. Đổi giá trị này "
        "theo batch_size/kiến trúc đang chạy — đừng để mặc định rồi hy vọng khớp.",
    )
    parser.add_argument(
        "target", nargs=argparse.REMAINDER,
        help="script Python cần chạy + tham số của nó, đặt sau --, ví dụ: "
        "-- train_w_args.py --variant resnet34_transformer",
    )
    args = parser.parse_args()

    target = args.target
    if target and target[0] == "--":
        target = target[1:]
    if not target:
        parser.error("thiếu script cần chạy — đặt sau --, ví dụ: -- train_w_args.py --variant resnet18_cnn")

    if args.gpu is not None:
        chosen_index = args.gpu
        print(f"dùng GPU {chosen_index} (ép bằng --gpu)")
    else:
        gpus = query_gpus()
        print("trạng thái GPU hiện tại:")
        for idx, free, total, util in gpus:
            print(f"  GPU {idx}: free {free:6d} MiB / {total} MiB, utilization {util:3d}%")
        picked, had_idle = pick_gpu(gpus, args.min_free_mib)
        if picked is None:
            print(f"LỖI: không GPU nào còn >= {args.min_free_mib} MiB free -- không đủ cho batch_size "
                  f"hiện tại. Đợi rồi thử lại, giảm batch_size, hoặc chỉnh --min-free-mib nếu đã biết "
                  f"chắc mức thực tế cần thấp hơn.")
            raise SystemExit(1)
        chosen_index, free, total, util = picked
        if not had_idle:
            print(f"CẢNH BÁO: GPU {chosen_index} đủ free memory ({free} MiB) nhưng utilization {util}% "
                  f"(>{UTIL_BUSY_THRESHOLD}%) -- chọn vì không GPU nào vừa đủ memory vừa rảnh compute, "
                  f"sẽ tranh chấp compute với job khác nên train chậm hơn, nhưng không OOM.")
        else:
            print(f"chọn GPU {chosen_index} (free {free} MiB, utilization {util}%)")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(chosen_index)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable] + target
    print(f"chạy: CUDA_VISIBLE_DEVICES={chosen_index} {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=repo_root)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

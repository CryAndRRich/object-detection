"""Chạy main.py trên GPU trống nhất của server, tự động chọn qua nvidia-smi.

Server hiện tại chạy chung với người/job khác (không như Kaggle được cấp riêng),
nên không thể cứ mặc định GPU 0 -- có thể đang bận, hoặc không đủ free memory.
Script này:

  1. Query nvidia-smi cho mọi GPU: free memory + utilization.
  2. Loại trước các GPU không đủ ``--min-free-mib`` free memory -- đây là điều
     kiện CỨNG, không GPU nào qua được thì thoát lỗi thay vì chọn liều rồi OOM
     giữa chừng (từng gặp đúng kiểu lỗi này: chọn GPU rảnh compute nhưng không
     đủ VRAM cho batch_size đang chạy).
  3. Trong số các GPU đã đủ memory, ưu tiên GPU KHÔNG bận tính toán
     (utilization dưới ngưỡng); nếu tất cả đều bận tính toán, chọn GPU nhiều
     free memory nhất trong số đó rồi cảnh báo (không chặn train, chỉ báo để
     bạn biết đang tranh chấp compute -- sẽ chạy chậm hơn nhưng không OOM).
  4. Set CUDA_VISIBLE_DEVICES=<gpu đã chọn> rồi exec main.py với mọi tham số
     truyền vào script này.

    python tools/run_train.py -c config/cfg_odvg_diffusion.py \
        --datasets config/datasets_coco_minitrain.json \
        --output_dir output/diffu_run1 \
        --pretrain_model_path ../weights/diffu_grounding_dino/groundingdino_swint_ogc.pth \
        --finetune_ignore time_ diffusion

Chỉ chạy single-GPU (server này không có nhiều GPU rảnh cùng lúc để torchrun đa
GPU như trên Kaggle 2xT4 trước đây). Muốn ép 1 GPU cụ thể: --gpu <index>.
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
    """``None`` if no GPU has enough free memory; otherwise ``(gpu, had_idle)``.

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
        help="loại GPU có free memory dưới mức này trước khi xét utilization. Mặc định 8192 "
        "khớp mức đo thật cho batch_size=4 (~8118 MiB đo trên T4) -- ĐỔI GIÁ TRỊ NÀY nếu chạy "
        "batch_size khác, đừng để mặc định rồi hy vọng khớp.",
    )
    args, main_args = parser.parse_known_args()

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
    cmd = [sys.executable, os.path.join(repo_root, "main.py")] + main_args
    print(f"chạy: CUDA_VISIBLE_DEVICES={chosen_index} {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=repo_root)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

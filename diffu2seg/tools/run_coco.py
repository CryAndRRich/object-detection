#!/usr/bin/env python3
r"""Chạy Diffu2Seg trên COCO val2017 với ĐÚNG cấu hình paper.

MỤC ĐÍCH: reproduce phần training-free của Diffuse2Seg trong đúng chế độ dữ liệu
mà paper báo số — vật thưa (trung vị 4 vật/ảnh), vật to (53,7 x 62,2 px), nhiều
class. Không so gì với CE-130; đây là một lần chạy độc lập.

CẤU HÌNH LÀ CỦA PAPER, KHÔNG PHẢI CỦA DỰ ÁN: canvas 1120, grid_r 140, stride 6,
p=1,6, lam=1e-5, tau_att=0,55, t=150 (config/coco_paper.py). Không chỉnh gì cho
hợp dữ liệu — chỉnh rồi thì kết quả nói về cách ta chỉnh, không nói về phương
pháp.

                    HAI THỨ KHÔNG REPRODUCE ĐƯỢC, NÓI TRƯỚC

1. **Không có AP mask trong bảng của paper.** Bảng đó là số của Mask2Former đã
   train trên pseudo-mask (bước 3). Ta bỏ hẳn bước train. Training-free thì
   không có score, không có ranking, nên AP KHÔNG ĐỊNH NGHĨA ĐƯỢC. Cái đo được
   ở đây là `oracle_recall` và `mean_bestIoU` — chất lượng mask/box thô.

2. **SD 1.5 chứ không phải SD2.** Mọi repo `stabilityai/stable-diffusion-2*`
   trả HTTP 401 từ 2026-09-15. Quan trọng hơn: `t=150` là giá trị paper tinh
   chỉnh CHO SD2, thang timestep của SD1.5 không nhất thiết đặt đặc trưng tốt
   nhất ở cùng chỗ. Đây là nguồn sai khác thật với số của paper.

CHI PHÍ: A là (19600, 19600) fp32 = 1,54 GB, cộng SD1.5 fp16. Vừa A30 24 GB,
nhưng CHỈ VÌ compute_g khai triển thành matmul — dạng literal sẽ là 815 TB.
Ước tính ~20-40 s/ảnh, nên mặc định --limit 50 chứ không phải cả 5000 ảnh.

CHẠY (TRÊN SERVER):
    export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
    LOG=/mnt/disk1/aiotlab/haitn/log/d2s_coco_$(date +%Y%m%d_%H%M%S).log
    nohup python tools/run_on_free_gpu.py -- tools/run_coco.py --limit 50 \
        --out /mnt/disk1/aiotlab/haitn/log/d2s_coco.json > "$LOG" 2>&1 &
    echo "PID $! -> $LOG"
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base import Diffu2SegConfig          # noqa: E402
from d2s.pipeline import build_affinity, segment_image  # noqa: E402
from data.coco_val import CocoVal                # noqa: E402
from utils.metrics import fmt_time, quality_one_image, summarise  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=50,
                    help="số ảnh; mặc định 50 vì ~20-40 s/ảnh ở grid_r=140")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--canvas", type=int, default=1120,
                    help="1120 = cấu hình paper (grid_r 140). 512 -> grid_r 64, nhẹ hơn 22x")
    ap.add_argument("--stride", type=int, default=6, help="prompt stride, paper dùng 6")
    ap.add_argument("--p", type=float, default=None, help="ghi đè p (paper: 1.6)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-boxes", action="store_true")
    args = ap.parse_args()

    over = {"canvas": args.canvas, "prompt_stride_cells": args.stride}
    if args.p is not None:
        over["p"] = args.p
    cfg = Diffu2SegConfig(**{**Diffu2SegConfig().__dict__, **over}).validate()

    root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
    ds = CocoVal(os.path.join(root, "coco", "annotations", "instances_val2017.json"),
                 os.path.join(root, "coco", "val2017"), canvas=cfg.canvas)

    end = min(args.start + args.limit, len(ds))
    n = end - args.start

    print("=" * 74)
    print("Diffu2Seg trên COCO val2017 — CẤU HÌNH PAPER")
    print(f"  timestamp   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ảnh         : {n} (từ {args.start}), tập có {len(ds)} ảnh dùng được "
          f"({ds.n_dropped} ảnh bị bỏ vì không có box non-crowd)")
    print(f"  canvas      : {cfg.canvas}  grid_r={cfg.grid_r}  "
          f"1 ô = {cfg.canvas / cfg.grid_r:.0f} px  A = "
          f"{cfg.n_tokens ** 2 * 4 / 1e9:.2f} GB fp32")
    print(f"  p           : {cfg.p}   lam={cfg.lam}  tau_prop={cfg.tau_prop}  "
          f"max_iter={cfg.max_iter}")
    print(f"  prompts     : stride {cfg.prompt_stride_cells} ô  "
          f"quantile={cfg.mask_quantile}  rel_floor={cfg.mask_rel_floor}")
    print(f"  SD          : t={cfg.timesteps[0]}  tau_att={cfg.tau_att}  "
          f"prompt_text={cfg.prompt_text!r}")
    print(f"  model       : {cfg.model_source}")
    print(f"  device      : {args.device}  ({platform.node()})")
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        print(f"  gpu         : {torch.cuda.get_device_name(args.device)}")
    print(f"  command     : {' '.join(sys.argv)}")
    print("=" * 74 + "\n")

    from d2s.attention import StableDiffusion2AttentionAggregator
    t0 = time.time()
    agg = StableDiffusion2AttentionAggregator(
        timestep=cfg.timesteps[0], attention_resolution=cfg.grid_r,
        weight_down_block_0=cfg.w_down_0, weight_down_block_1=cfg.w_down_1,
        weight_up_block_0=cfg.w_up_0, weight_up_block_1=cfg.w_up_1,
        weight_up_block_2=cfg.w_up_2, hugging_face_model_id=cfg.model_source,
        prompt_text=cfg.prompt_text, device=args.device, torch_dtype=torch.float16)
    print(f"model loaded in {time.time() - t0:.1f}s\n")

    best_all, hits, n_gt_total, n_pred_total = [], 0, 0, 0
    n_not_conv, n_iters, f_maxes = 0, [], []
    pad_removed, big_removed, ceilings = 0, 0, []
    per_image = []
    t_start = time.time()

    for i in range(args.start, end):
        s = ds[i]
        A = build_affinity(s["image"], cfg, agg)
        out = segment_image(s["image"], s["valid_h"], cfg, A=A,
                            valid_w=s["valid_w"])

        best, hit, n_gt = quality_one_image(out["boxes"], s["gt_cxcywh"],
                                            size=cfg.canvas)
        best_all.append(best)
        hits += hit
        n_gt_total += n_gt
        n_pred_total += out["n_boxes"]
        n_iters.append(out["n_iter"])
        f_maxes.append(out["f_max"])
        pad_removed += out["filter_info"]["n_in_padding"]
        big_removed += out["filter_info"]["n_too_large"]
        if not out["converged"]:
            n_not_conv += 1

        ceil = ds.resolution_ceiling(i, cfg.grid_r)
        if not np.isnan(ceil):
            ceilings.append(ceil)

        rec = {"file_name": s["file_name"], "image_id": s["image_id"],
               "n_gt": n_gt, "n_boxes": out["n_boxes"], "n_iter": out["n_iter"],
               "converged": out["converged"], "recall": hit / max(n_gt, 1),
               "resolution_ceiling": ceil, "W": s["W"], "H": s["H"]}
        if args.save_boxes:
            rec["boxes"] = out["boxes"].tolist()
        per_image.append(rec)

        done = i - args.start + 1
        el = time.time() - t_start
        print(f"  [{done:4d}/{n} {100 * done / n:5.1f}%] {s['file_name']:20s} "
              f"{s['W']:4d}x{s['H']:<4d} "
              f"n_gt={n_gt:3d} n_pred={out['n_boxes']:4d} "
              f"recall={hit / max(n_gt, 1):.2f} n_iter={out['n_iter']:3d} | "
              f"{el / done:5.1f}s/ảnh | elapsed {fmt_time(el)} | "
              f"ETA {fmt_time(el / done * (n - done))}", flush=True)
        if done == 1 and torch.cuda.is_available() and args.device.startswith("cuda"):
            print(f"         max_memory_allocated = "
                  f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
                  f"| n_prompts={out['n_prompts']}", flush=True)

    res = summarise(best_all, hits, n_gt_total,
                    n_pred_total=n_pred_total, n_images=n)
    elapsed = time.time() - t_start

    print("\n" + "=" * 74)
    print("KẾT QUẢ — COCO val2017, cấu hình paper")
    print(f"  oracle_recall  : {res['oracle_recall']:.4f}   "
          f"({res['n_hit']}/{res['n_gt']} GT ở IoU>=0.5)")
    print(f"  mean_bestIoU   : {res['mean_bestIoU']:.4f}")
    print(f"  median_bestIoU : {res['median_bestIoU']:.4f}")
    print(f"  box/ảnh        : {n_pred_total / max(n, 1):.1f}  "
          f"(GT {n_gt_total / max(n, 1):.1f}/ảnh)")
    print(f"  thời lượng     : {fmt_time(elapsed)} ({elapsed / max(n, 1):.1f}s/ảnh)")
    print(f"  ngoại suy {len(ds)} ảnh: {fmt_time(elapsed / max(n, 1) * len(ds))}")

    print("\nCHẨN ĐOÁN")
    conv = 1.0 - n_not_conv / max(n, 1)
    print(f"  hội tụ         : {100 * conv:.1f} %  "
          f"(n_iter median {np.median(n_iters):.0f}/{cfg.max_iter})")
    print(f"  f_max median   : {np.median(f_maxes):.2e}")
    print(f"  box bị loại    : {pad_removed} rơi vùng pad, {big_removed} quá lớn")
    if conv < 0.90:
        print("     ⚠️ dưới 90 % hội tụ -> số trên mô tả CAP, không mô tả cơ chế")
    if ceilings:
        print(f"  trần độ phân giải: {100 * np.mean(ceilings):.1f} % GT có cạnh "
              f"ngắn >= 1 ô (grid_r={cfg.grid_r})")

    print("\n⚠️ KHÔNG SO ĐƯỢC VỚI BẢNG AP TRONG PAPER: bảng đó là Mask2Former đã")
    print("   train trên pseudo-mask (bước 3), ta bỏ hẳn bước train. Training-free")
    print("   không có score nên AP không định nghĩa được.")
    print("⚠️ Chạy trên SD 1.5, không phải SD2 như paper (SD2 gated). t=150 là giá")
    print("   trị paper tinh chỉnh CHO SD2 — nguồn sai khác thật, không bỏ qua.")
    print("=" * 74)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"tool": "run_coco", "dataset": "coco_val2017",
                       "n_images": n, "start": args.start,
                       "config": cfg.to_dict(), "results": res,
                       "elapsed_sec": elapsed,
                       "diagnostics": {
                           "convergence_rate": conv,
                           "median_n_iter": float(np.median(n_iters)),
                           "median_f_max": float(np.median(f_maxes)),
                           "n_boxes_in_padding": pad_removed,
                           "n_boxes_too_large": big_removed,
                           "mean_resolution_ceiling":
                               float(np.mean(ceilings)) if ceilings else None},
                       "per_image": per_image}, f, indent=2)
        print(f"  -> {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

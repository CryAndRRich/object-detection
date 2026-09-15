#!/usr/bin/env python3
r"""Chạy GĐ1 đầy đủ trên một split CE-130.

CHỈ CHẠY SAU KHI CỬA CHẶN 1 ĐẠT hoặc XÁM. Nếu cửa chặn 1 KHÔNG ĐẠT thì `p` không
có tác dụng đo được và việc chạy 908 ảnh chỉ tốn 2-4 giờ để xác nhận lại điều đã
biết trên 100 ảnh.

TRẦN DO ĐỘ PHÂN GIẢI — in ra cùng kết quả, không phải phần phụ:
`grid_r=64` nghĩa là một ô lưới = 8 px canvas. Vật nào có cạnh ngắn dưới 1 ô thì
**không biểu diễn được**, bất kể `p` làm gì. Trên ảnh rất rộng (1918x384, aspect
4,99) hệ số thu nhỏ là 0,267 nên vật trung vị 39x32 px co còn 1,30 x 1,07 ô —
round-trip IoU chỉ ~0,35 **dù mask hoàn hảo**. Hiếm (3,5 % ảnh val có aspect > 2,0)
nhưng phải in ra, nếu không `oracle_recall` thấp sẽ bị đọc nhầm thành "cơ chế
hỏng" thay vì "độ phân giải chặn".

KHÔNG CÓ AP50 Ở ĐÂY, và đó là TÍNH CHẤT chứ không phải thiếu sót: training-free
nên không có score, không có ranking, nên AP không định nghĩa được. `oracle_recall`
và `mean_bestIoU` là đúng bộ metric cho thứ này (xem utils/metrics.py).

CHẠY (TRÊN SERVER — xem README):
    export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
    LOG=/mnt/disk1/aiotlab/haitn/log/d2s_stage1_val_$(date +%Y%m%d_%H%M%S).log
    nohup python tools/run_stage1.py --split val \
        --out /mnt/disk1/aiotlab/haitn/log/d2s_stage1_val.json > "$LOG" 2>&1 &
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
from data.ce130_coco import CE130Coco            # noqa: E402
from utils.metrics import fmt_time, quality_one_image, summarise  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--p", type=float, default=None, help="override config p")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-boxes", action="store_true",
                    help="also dump per-image boxes (larger json)")
    args = ap.parse_args()

    cfg = Diffu2SegConfig().validate()
    if args.p is not None:
        cfg = Diffu2SegConfig(**{**cfg.__dict__, "p": args.p}).validate()

    root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
    ds = CE130Coco(os.path.join(root, "ce130_coco", f"ce130_agnostic_{args.split}.json"),
                   os.path.join(root, "all_phase2_V2"), cfg.canvas)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    # Header: everything needed to reproduce this run, printed before any work.
    print("=" * 74)
    print("Diffu2Seg GĐ1 — p-Laplacian + một mức + connected components")
    print(f"  timestamp   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  split       : {args.split}   n_images={n}/{len(ds)}")
    print(f"  grid_r      : {cfg.grid_r}  (canvas {cfg.canvas}, 1 ô = "
          f"{cfg.canvas / cfg.grid_r:.0f} px)")
    print(f"  p           : {cfg.p}        lam={cfg.lam}  tau_prop={cfg.tau_prop}  "
          f"max_iter={cfg.max_iter}")
    print(f"  prompts     : stride {cfg.prompt_stride_cells} ô  "
          f"quantile={cfg.mask_quantile}  rel_floor={cfg.mask_rel_floor}")
    print(f"  SD2         : t={cfg.timesteps[0]}  tau_att={cfg.tau_att}  "
          f"prompt_text={cfg.prompt_text!r}")
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
    print(f"SD2 loaded in {time.time() - t0:.1f}s\n")

    best_all, hits, n_gt_total, n_pred_total = [], 0, 0, 0
    n_not_conv, n_iters, f_maxes = 0, [], []
    pad_removed, big_removed, ceilings = 0, 0, []
    per_image = []
    t_start = time.time()

    for i in range(n):
        s = ds[i]
        A = build_affinity(s["image"], cfg, agg)
        out = segment_image(s["image"], s["valid_h"], cfg, A=A)

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

        rec = {"file_name": s["file_name"], "n_gt": n_gt, "n_boxes": out["n_boxes"],
               "n_iter": out["n_iter"], "converged": out["converged"],
               "recall": hit / max(n_gt, 1), "resolution_ceiling": ceil,
               "aspect": s["W"] / s["H"]}
        if args.save_boxes:
            rec["boxes"] = out["boxes"].tolist()
        per_image.append(rec)

        el = time.time() - t_start
        print(f"  [{i + 1:4d}/{n} {100 * (i + 1) / n:5.1f}%] {s['file_name']:28s} "
              f"n_gt={n_gt:3d} n_pred={out['n_boxes']:4d} n_iter={out['n_iter']:3d} | "
              f"oracle_recall={hits / max(n_gt_total, 1):.4f} | "
              f"{el / (i + 1):5.2f}s/ảnh | elapsed {fmt_time(el)} | "
              f"ETA {fmt_time(el / (i + 1) * (n - i - 1))}", flush=True)
        if i == 0 and torch.cuda.is_available() and args.device.startswith("cuda"):
            print(f"         max_memory_allocated = "
                  f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
                  f"| n_prompts={out['n_prompts']}", flush=True)

    res = summarise(best_all, hits, n_gt_total,
                    n_pred_total=n_pred_total, n_images=n)
    elapsed = time.time() - t_start

    print("\n" + "=" * 74)
    print("KẾT QUẢ")
    print(f"  oracle_recall  : {res['oracle_recall']:.4f}   "
          f"({res['n_hit']}/{res['n_gt']} GT)")
    print(f"  mean_bestIoU   : {res['mean_bestIoU']:.4f}")
    print(f"  median_bestIoU : {res['median_bestIoU']:.4f}")
    print(f"  box/ảnh        : {n_pred_total / max(n, 1):.1f}")
    print(f"  thời lượng     : {fmt_time(elapsed)} ({elapsed / max(n, 1):.2f}s/ảnh)")
    if n < len(ds):
        print(f"  ngoại suy {len(ds)} ảnh: {fmt_time(elapsed / max(n, 1) * len(ds))}")

    print("\nCHẨN ĐOÁN")
    conv = 1.0 - n_not_conv / max(n, 1)
    print(f"  hội tụ         : {100 * conv:.1f} %  "
          f"(n_iter median {np.median(n_iters):.0f}/{cfg.max_iter})")
    print(f"  f_max median   : {np.median(f_maxes):.2e}   "
          f"(anchor yếu -> nghiệm có thể tụt về 0)")
    print(f"  box bị loại    : {pad_removed} rơi vùng pad, {big_removed} quá lớn")
    if big_removed > 0.05 * max(n_pred_total, 1):
        print("     ⚠️ nhiều box 'quá lớn' -> nghi logic pad hỏng, không phải ảnh lạ")
    if conv < 0.90:
        print("     ⚠️ dưới 90 % hội tụ -> số trên mô tả CAP, không mô tả cơ chế")

    if ceilings:
        print(f"\nTRẦN DO ĐỘ PHÂN GIẢI (grid_r={cfg.grid_r})")
        print(f"  GT có cạnh ngắn >= 1 ô : {100 * np.mean(ceilings):.1f} % "
              f"(trung bình theo ảnh)")
        print(f"  ảnh có trần < 90 %     : {int((np.array(ceilings) < 0.9).sum())}/"
              f"{len(ceilings)}")
        print("  ⚠️ oracle_recall KHÔNG THỂ vượt trần này. Nếu hai số sát nhau thì")
        print("     nút thắt là ĐỘ PHÂN GIẢI (thử r=96), không phải p.")

    print("\n⚠️ KHÔNG CÓ AP50: training-free nên không có score, AP không định")
    print("   nghĩa được. Đó là tính chất của hướng này, không phải thiếu sót.")
    print(f"⚠️ Đo trên '{args.split}'. Số A/C1/E1/D.1 trong docs đo trên TEST —")
    print("   khác split, KHÔNG so trực tiếp.")
    print("=" * 74)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"tool": "run_stage1", "split": args.split, "n_images": n,
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

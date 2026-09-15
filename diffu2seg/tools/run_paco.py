#!/usr/bin/env python3
r"""Chạy Diffuse2Seg trên PACO-LVIS val với ĐÚNG cấu hình paper, đo AR_1000.

PACO là bộ DUY NHẤT trong 5 bộ của bảng training-free mà dự án lấy được:
SA-1B chỉ tải theo shard 10 GB và tập val 1000 ảnh của họ không công bố;
ADE20K bản có instance cần tài khoản MIT CSAIL được duyệt (bản tải tự do
ADEChallengeData2016 là SEMANTIC, không có instance); EntitySeg bản low-res
11,4 GB chia 3 shard không tách val; UVO là video Kinetics-400 cần xin quyền.

MỐC CỦA PAPER TRÊN PACO (Table 2, AR_1000):
    Diffuse2Seg 13,6  |  DiffSeg 9,8  |  M2N2 9,6  |  UnSAM 9,3  |  CutLER 10,7

                ⚠️ BA ĐIỀU PHẢI ĐỌC TRƯỚC KHI XEM SỐ

1. **GĐ1 có MỘT mức, paper có SÁU.** PACO là part segmentation: 65,6 % mục tiêu
   là bộ phận ("chair:apron"), 34,4 % là vật nguyên. Một cái ghế đồng thời là 1
   mask OBJECT và ~8 mask PART, và 13,6 của paper đạt được NHỜ 6 mức KL cluster
   (GĐ2, chưa viết). Số thấp ở đây KHÔNG phân biệt được "implement sai" với
   "chưa có GĐ2". Đây là bộ mà thiếu multi-granularity bị phạt nặng nhất.

2. **Trần độ phân giải 90,7 %.** Ở canvas 1120 / grid_r 140, chỉ 90,7 % GT có
   cạnh ngắn >= 1 ô (trung vị 4,23 ô, p10 1,09). Mask hoàn hảo cũng không vượt
   được số đó. Tool in trần thực đo trên đúng tập ảnh đã chạy.

3. **SD 1.5 chứ không phải SD2**, và `t=150` là giá trị paper tinh chỉnh CHO
   SD2. Thang timestep của SD1.5 không nhất thiết đặt đặc trưng tốt nhất ở cùng
   chỗ. Nguồn sai khác thật, không bỏ qua khi báo cáo.

AR_1000 KHÁC `oracle_recall`: AR trung bình recall trên 10 ngưỡng IoU 0,5→0,95
và ghép cặp MỘT-MỘT; `oracle_recall` chỉ là số hạng đầu (t=0,5) và cho phép một
box phủ nhiều GT. AR luôn NHỎ HƠN. Tool in cả hai để thấy khoảng cách, nhưng
**chỉ AR_1000 mới so được với 13,6**.

CHẠY (TRÊN SERVER):
    export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
    LOG=/mnt/disk1/aiotlab/haitn/log/d2s_paco_$(date +%Y%m%d_%H%M%S).log
    nohup python tools/run_on_free_gpu.py -- tools/run_paco.py --limit 50 \
        --out /mnt/disk1/aiotlab/haitn/log/d2s_paco.json > "$LOG" 2>&1 &
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

from config.base import Diffu2SegConfig                    # noqa: E402
from d2s.pipeline import build_affinity, segment_image     # noqa: E402
from data.paco_val import PacoVal                          # noqa: E402
from utils.ar_metrics import (IOU_THRESHOLDS, SIZE_BANDS,  # noqa: E402
                              ar_one_image, summarise_ar)
from utils.mask_ops import mask_iou_matrix, masks_to_original  # noqa: E402
from utils.metrics import fmt_time, quality_one_image     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=50,
                    help="số ảnh; mặc định 50 vì ~20-40 s/ảnh ở grid_r=140")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--canvas", type=int, default=1120,
                    help="1120 = cấu hình paper (grid_r 140)")
    ap.add_argument("--stride", type=int, default=6, help="prompt stride, paper dùng 6")
    ap.add_argument("--p", type=float, default=None, help="ghi đè p (paper: 1.6)")
    ap.add_argument("--quantile", type=float, default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    over = {"canvas": args.canvas, "prompt_stride_cells": args.stride}
    if args.p is not None:
        over["p"] = args.p
    if args.quantile is not None:
        over["mask_quantile"] = args.quantile
    cfg = Diffu2SegConfig(**{**Diffu2SegConfig().__dict__, **over}).validate()

    root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
    ds = PacoVal(os.path.join(root, "paco", "paco_lvis_v1_val.json"),
                 os.path.join(root, "paco", "images"), canvas=cfg.canvas)

    end = min(args.start + args.limit, len(ds))
    n = end - args.start

    print("=" * 76)
    print("Diffuse2Seg trên PACO-LVIS val — CẤU HÌNH PAPER, đo AR_1000")
    print(f"  timestamp   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ảnh         : {n} (từ {args.start}) / {len(ds)}")
    print(f"  GT          : {ds.n_poly} polygon (OBJECT) + {ds.n_rle} RLE (PART)")
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
    print("=" * 76)
    print("MỐC paper trên PACO (AR_1000): Diffuse2Seg 13,6 | DiffSeg 9,8 | "
          "M2N2 9,6 | UnSAM 9,3")
    print("⚠️ paper đạt 13,6 với SÁU mức granularity; GĐ1 có MỘT. PACO là bộ")
    print("   part-segmentation nên đây là chỗ thiếu GĐ2 bị phạt nặng nhất.")
    print("=" * 76 + "\n")

    from d2s.attention import StableDiffusion2AttentionAggregator
    t0 = time.time()
    agg = StableDiffusion2AttentionAggregator(
        timestep=cfg.timesteps[0], attention_resolution=cfg.grid_r,
        weight_down_block_0=cfg.w_down_0, weight_down_block_1=cfg.w_down_1,
        weight_up_block_0=cfg.w_up_0, weight_up_block_1=cfg.w_up_1,
        weight_up_block_2=cfg.w_up_2, hugging_face_model_id=cfg.model_source,
        prompt_text=cfg.prompt_text, device=args.device, torch_dtype=torch.float16)
    print(f"model loaded in {time.time() - t0:.1f}s\n")

    hits = np.zeros(len(IOU_THRESHOLDS), dtype=np.int64)
    band_hits = {k: np.zeros(len(IOU_THRESHOLDS), dtype=np.int64) for k in SIZE_BANDS}
    band_n = {k: 0 for k in SIZE_BANDS}
    n_gt_total = n_pred_total = 0
    part_hits = np.zeros(len(IOU_THRESHOLDS), dtype=np.int64)
    obj_hits = np.zeros(len(IOU_THRESHOLDS), dtype=np.int64)
    n_part = n_obj = 0

    orc_hits, orc_gt = 0, 0          # oracle_recall, để thấy khoảng cách với AR
    n_not_conv, n_iters, ceilings, n_over_cap = 0, [], [], 0
    per_image = []
    t_start = time.time()

    for i in range(args.start, end):
        s = ds[i]
        A = build_affinity(s["image"], cfg, agg)
        out = segment_image(s["image"], s["valid_h"], cfg, A=A, valid_w=s["valid_w"])

        pred = masks_to_original(out["masks"], s["valid_w"], s["valid_h"],
                                 s["W"], s["H"])
        if len(pred) > 1000:
            n_over_cap += 1
        iou = mask_iou_matrix(pred, s["gt_masks"])
        h, n_gt, per_band = ar_one_image(iou, s["gt_areas"], n_pred=len(pred))

        hits += h
        n_gt_total += n_gt
        n_pred_total += len(pred)
        for k in SIZE_BANDS:
            band_hits[k] += per_band[k][0]
            band_n[k] += per_band[k][1]

        # Tách OBJECT / PART: cùng ma trận IoU, chỉ đổi tập GT được tính.
        ip = s["is_part"]
        if ip.any():
            hp, ngp, _ = ar_one_image(iou[:, ip], s["gt_areas"][ip])
            part_hits += hp
            n_part += ngp
        if (~ip).any():
            ho, ngo, _ = ar_one_image(iou[:, ~ip], s["gt_areas"][~ip])
            obj_hits += ho
            n_obj += ngo

        _, oh, og = quality_one_image(out["boxes"], s["gt_cxcywh"], size=cfg.canvas)
        orc_hits += oh
        orc_gt += og

        n_iters.append(out["n_iter"])
        if not out["converged"]:
            n_not_conv += 1
        ceil = ds.resolution_ceiling(i, cfg.grid_r)
        if not np.isnan(ceil):
            ceilings.append(ceil)

        per_image.append({
            "file_name": s["file_name"], "image_id": s["image_id"],
            "n_gt": n_gt, "n_part": int(ip.sum()), "n_pred": int(len(pred)),
            "hits_at_50": int(h[0]), "n_iter": out["n_iter"],
            "converged": out["converged"], "resolution_ceiling": ceil,
            "W": s["W"], "H": s["H"]})

        done = i - args.start + 1
        run_ar = (hits / max(n_gt_total, 1)).mean()
        el = time.time() - t_start
        print(f"  [{done:4d}/{n} {100 * done / n:5.1f}%] {s['file_name']:20s} "
              f"{s['W']:4d}x{s['H']:<4d} "
              f"n_gt={n_gt:3d}({int(ip.sum()):3d}p) n_pred={len(pred):4d} "
              f"hit@50={h[0]:3d} | AR_1000={100 * run_ar:5.2f} | "
              f"{el / done:5.1f}s/ảnh | elapsed {fmt_time(el)} | "
              f"ETA {fmt_time(el / done * (n - done))}", flush=True)
        if done == 1 and torch.cuda.is_available() and args.device.startswith("cuda"):
            print(f"         max_memory_allocated = "
                  f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
                  f"| n_prompts={out['n_prompts']}", flush=True)

    res = summarise_ar(hits, n_gt_total, band_hits, band_n)
    elapsed = time.time() - t_start

    print("\n" + "=" * 76)
    print("KẾT QUẢ — PACO-LVIS val, cấu hình paper")
    print(f"  AR_1000        : {100 * res['AR_1000']:6.2f}   "
          f"(paper: Diffuse2Seg 13,6 | M2N2 9,6 | DiffSeg 9,8)")
    print(f"  AR_S / AR_M / AR_L : {100 * res['AR_S']:5.2f} / "
          f"{100 * res['AR_M']:5.2f} / {100 * res['AR_L']:5.2f}")
    print(f"     n_gt theo band  : S {res['n_gt_S']}  M {res['n_gt_M']}  "
          f"L {res['n_gt_L']}")
    print(f"  recall@0.50    : {100 * res['recall_at_50']:6.2f}   "
          f"recall@0.75: {100 * res['recall_at_75']:6.2f}")

    ar_part = float((part_hits / max(n_part, 1)).mean())
    ar_obj = float((obj_hits / max(n_obj, 1)).mean())
    print(f"\n  TÁCH THEO LOẠI (chỗ thiếu GĐ2 lộ ra)")
    print(f"    AR trên OBJECT : {100 * ar_obj:6.2f}  (n={n_obj})")
    print(f"    AR trên PART   : {100 * ar_part:6.2f}  (n={n_part})")
    print(f"    ⚠️ GĐ1 một mức: nếu AR_PART << AR_OBJECT thì nút thắt là")
    print(f"       multi-granularity (GĐ2), không phải cơ chế lan truyền.")

    print(f"\n  box/ảnh        : {n_pred_total / max(n, 1):.1f}   "
          f"(GT {n_gt_total / max(n, 1):.1f}/ảnh)")
    print(f"  thời lượng     : {fmt_time(elapsed)} ({elapsed / max(n, 1):.1f}s/ảnh)")
    print(f"  ngoại suy {len(ds)} ảnh: {fmt_time(elapsed / max(n, 1) * len(ds))}")

    print("\nCHẨN ĐOÁN")
    conv = 1.0 - n_not_conv / max(n, 1)
    print(f"  hội tụ         : {100 * conv:.1f} %  "
          f"(n_iter median {np.median(n_iters):.0f}/{cfg.max_iter})")
    if conv < 0.90:
        print("     ⚠️ dưới 90 % hội tụ -> số trên mô tả CAP, không mô tả cơ chế")
    if n_over_cap:
        print(f"  ⚠️ {n_over_cap} ảnh vượt 1000 proposal -> bị cắt bớt. Cắt theo "
              f"THỨ TỰ (không có score để xếp hạng), nên quy tắc cắt bắt đầu "
              f"ảnh hưởng kết quả.")
    if ceilings:
        print(f"  trần độ phân giải: {100 * np.mean(ceilings):.1f} % GT có cạnh "
              f"ngắn >= 1 ô (grid_r={cfg.grid_r})")
        print(f"     ⚠️ AR_1000 KHÔNG THỂ vượt trần này. Sát nhau -> nút thắt là "
              f"ĐỘ PHÂN GIẢI.")
    print(f"  oracle_recall@0.5 (box): {orc_hits / max(orc_gt, 1):.4f}")
    print(f"     ⚠️ KHÔNG so với 13,6. Đây là ngưỡng IoU đơn 0,5, ghép nhiều-một,")
    print(f"        tính trên BOX — luôn cao hơn AR_1000. Chỉ để tham chiếu nội bộ.")

    print("\n⚠️ Chạy SD 1.5 (SD2 bị gated); t=150 là giá trị paper tinh chỉnh CHO SD2.")
    print("⚠️ GĐ1 = MỘT mức granularity; paper dùng SÁU. Trên PACO (65,6 % mục tiêu")
    print("   là PART) đây là khoảng cách lớn nhất giữa ta và bảng của họ.")
    print("=" * 76)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"tool": "run_paco", "dataset": "paco_lvis_val",
                       "n_images": n, "start": args.start,
                       "config": cfg.to_dict(), "results": res,
                       "ar_object": ar_obj, "n_gt_object": n_obj,
                       "ar_part": ar_part, "n_gt_part": n_part,
                       "oracle_recall_box_at50": orc_hits / max(orc_gt, 1),
                       "elapsed_sec": elapsed,
                       "paper_reference": {"Diffuse2Seg": 13.6, "DiffSeg": 9.8,
                                           "M2N2": 9.6, "UnSAM": 9.3,
                                           "CutLER": 10.7,
                                           "note": "paper dùng 6 mức granularity, "
                                                   "GĐ1 ở đây chỉ có 1"},
                       "diagnostics": {
                           "convergence_rate": conv,
                           "median_n_iter": float(np.median(n_iters)),
                           "n_images_over_1000_proposals": n_over_cap,
                           "mean_resolution_ceiling":
                               float(np.mean(ceilings)) if ceilings else None},
                       "per_image": per_image}, f, indent=2)
        print(f"  -> {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

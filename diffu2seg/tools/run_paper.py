#!/usr/bin/env python3
r"""Diffuse2Seg training-free ĐẦY ĐỦ — Bước 1 + Bước 2, cấu hình paper.

    ảnh -> VAE encode -> MỘT bước denoise, hook self-attention   [Bước 1]
        -> A (N,N) row-stochastic, trộn 2 layer w1/w2, tau_att
        -> lưới one-hot prompt, lan truyền p-Laplacian song song  [Bước 2a, Alg.1]
        -> chuẩn hoá thành phân phối, KL đối xứng, cụm average-link
        -> cắt dendrogram ở 6 height log-spaced                   [Bước 2b, Alg.2]
        -> trung bình trong cụm, UPSAMPLE, ARGMAX qua các cụm
        -> connected components, lọc a_min
        -> NMS theo diện tích giảm dần, tau_IoU=0.9, cap 1000
        -> instance mask ở độ phân giải ẢNH GỐC

KHÔNG có bước 3 (train Mask2Former) — ngoài phạm vi, và không cần cho AR_1000
của nhãn. KHÔNG có CascadePSP — paper gọi nó là "optional refinement step" và
Table 1 đo KHÔNG có nó ("without mask-refinement post-processing").

MỌI THAM SỐ Ở config/paper.py, mỗi dòng ghi kèm mục của paper. Xem file đó
trước khi đổi bất cứ giá trị nào.

MỐC CỦA PAPER (AR_1000):
    PACO 13,6  |  SA-1B 20,7  |  ADE20K 22,5  |  EntitySeg 23,8  |  UVO 29,9
Trong 5 bộ đó chỉ PACO lấy được (xem data/paco_val.py để biết vì sao 4 bộ kia
không lấy được).

                    BỐN SAI KHÁC KHÔNG GỠ ĐƯỢC, NÓI TRƯỚC

1. **SD 1.5 thay SD2** — mọi repo `stabilityai/stable-diffusion-2*` trả HTTP 401
   từ 2026-09-15. Đường hook không đổi (ta lấy `attn1`, trung bình mọi head),
   nhưng `t=150` là giá trị tinh chỉnh CHO SD2.
2. **1120 px xa vùng train của SD1.5 hơn** — SD2 train ở 1024 (1,09x), SD1.5
   train ở 512 (2,2x). Paper thừa nhận 1120 đã "outside the native resolution"
   với SD2; với SD1.5 khoảng cách lớn hơn nhiều. Dùng `--canvas 512` để đo xem
   điều này ảnh hưởng bao nhiêu.
3. **Không có CascadePSP** — xem trên.
4. **Không có bước 3** — xem trên.

                        CHI PHÍ — ĐO THẬT, KHÔNG ƯỚC TÍNH

Đo trên CPU máy local, mask 640x480 (kích thước ảnh PACO trung vị):

    mask_iou_matrix   1000 pred x 279 GT      4,3 s   RSS đỉnh 2,8 GB
    upsample          529 cụm -> 640x480      6,0 s   0,65 GB  (MỖI mức, 6 mức)
    NMS               300 / 1000 / 3000 mask  4 / 32 / 80 s

⚠️ NMS là O(n²) theo số mask GIỮ LẠI. Ở tau_IoU=0,9 rất ít mask bị chặn (đo:
3000 vào -> chỉ 6 bị chặn, phần còn lại chạm cap 1000), nên nó luôn chạy gần
kịch bản xấu nhất. Đây là khâu tốn nhất của GĐ2 trên CPU.

Bản đầu chậm hơn **110x** vì hai chỗ: `np.stack` dựng lại cả mảng trong vòng
lặp, và không lọc theo hộp bao. Ghi lại vì cả hai chỉ lộ ra khi ĐO ở quy mô
thật — ở 20 mask thì không có gì bất thường.

CHẠY (TRÊN SERVER):
    export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
    LOG=/mnt/disk1/aiotlab/haitn/log/d2s_paper_$(date +%Y%m%d_%H%M%S).log
    nohup python tools/run_on_free_gpu.py -- tools/run_paper.py \
        --dataset paco --limit 50 \
        --out /mnt/disk1/aiotlab/haitn/log/d2s_paper_paco.json > "$LOG" 2>&1 &
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

from config.base import Diffu2SegConfig                       # noqa: E402
from d2s.pipeline import build_affinity, segment_image        # noqa: E402
from utils.ar_metrics import (IOU_THRESHOLDS, SIZE_BANDS,     # noqa: E402
                              ar_one_image, summarise_ar)
from utils.mask_ops import mask_iou_matrix                    # noqa: E402
from utils.metrics import fmt_time                            # noqa: E402

PAPER_AR = {"paco": 13.6, "sa1b": 20.7, "ade20k": 22.5,
            "entityseg": 23.8, "uvo": 29.9}
PACO_BASELINES = "Diffuse2Seg 13,6 | CutLER 10,7 | DiffSeg 9,8 | M2N2 9,6 | UnSAM 9,3"


def build_dataset(name, root, canvas):
    if name == "paco":
        from data.paco_val import PacoVal
        return PacoVal(os.path.join(root, "paco", "paco_lvis_v1_val.json"),
                       os.path.join(root, "paco", "images"), canvas=canvas)
    if name == "coco":
        from data.coco_val import CocoVal
        return CocoVal(os.path.join(root, "coco", "annotations",
                                    "instances_val2017.json"),
                       os.path.join(root, "coco", "val2017"), canvas=canvas)
    raise ValueError(f"dataset {name!r} chưa có loader")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="paco", choices=["paco", "coco"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-masks", default=None,
                    help="thư mục lưu mask .npz mỗi ảnh (lớn; chỉ khi cần xem lại)")
    # Ghi đè — mặc định LÀ GIÁ TRỊ PAPER, chỉ dùng khi cố ý lệch khỏi paper.
    ap.add_argument("--canvas", type=int, default=None,
                    help="ghi đè canvas (paper: 1120). 512 = vùng train của SD1.5")
    ap.add_argument("--stride", type=int, default=None, help="paper: 6")
    ap.add_argument("--p", type=float, default=None, help="paper: 1.6")
    ap.add_argument("--levels", type=int, default=None, help="paper: 6")
    ap.add_argument("--nms-iou", type=float, default=None, help="paper: 0.9")
    ap.add_argument("--min-area", type=int, default=None, help="paper: 100")
    ap.add_argument("--timestep", type=int, default=None, help="paper: 150")
    args = ap.parse_args()

    from config.paper import cfg as paper_cfg
    over = dict(paper_cfg.__dict__)
    deviations = []
    for flag, key in (("canvas", "canvas"), ("stride", "prompt_stride_cells"),
                      ("p", "p"), ("levels", "n_levels"), ("nms_iou", "nms_iou"),
                      ("min_area", "min_area_px")):
        v = getattr(args, flag)
        if v is not None:
            deviations.append(f"{key}: paper {over[key]} -> {v}")
            over[key] = v
    if args.timestep is not None:
        deviations.append(f"timestep: paper {over['timesteps'][0]} -> {args.timestep}")
        over["timesteps"] = (args.timestep,)
    cfg = Diffu2SegConfig(**over).validate()

    root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
    ds = build_dataset(args.dataset, root, cfg.canvas)
    end = min(args.start + args.limit, len(ds))
    n = end - args.start

    n_side = len(range(cfg.prompt_stride_cells // 2, cfg.grid_r,
                       cfg.prompt_stride_cells))
    heights = np.geomspace(cfg.kl_h_min, cfg.kl_h_max, cfg.n_levels)

    print("=" * 78)
    print("Diffuse2Seg TRAINING-FREE ĐẦY ĐỦ (Bước 1 + Bước 2) — cấu hình paper")
    print(f"  timestamp   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  dataset     : {args.dataset}   {n} ảnh (từ {args.start}) / {len(ds)}")
    print("  " + "-" * 74)
    print("  BƯỚC 1 — trích self-attention")
    print(f"    model       : {cfg.model_source}")
    print(f"    timestep    : {cfg.timesteps[0]}        tau_att : {cfg.tau_att}")
    print(f"    canvas      : {cfg.canvas}      grid_r  : {cfg.grid_r}  "
          f"(1 ô = {cfg.canvas / cfg.grid_r:.0f} px)")
    print(f"    layer w     : up_0 {cfg.w_up_0}  up_1 {cfg.w_up_1}  up_2 {cfg.w_up_2}")
    print(f"    A           : ({cfg.n_tokens}, {cfg.n_tokens}) fp32 = "
          f"{cfg.n_tokens ** 2 * 4 / 1e9:.2f} GB")
    print(f"    VRAM đỉnh   : ~{cfg.peak_attention_gb:.1f} GB (attention trung gian) "
          f"+ ~1,9 GB (SD fp16)  -> ~{cfg.peak_attention_gb + 1.9:.1f} GB")
    print("  BƯỚC 2a — lan truyền p-Laplacian (Algorithm 1)")
    print(f"    prompts     : lưới {n_side}x{n_side} = {n_side ** 2}, stride "
          f"{cfg.prompt_stride_cells} ô")
    print(f"    p           : {cfg.p}        lam : {cfg.lam}")
    print(f"    tau_prop    : {cfg.tau_prop}     max_iter : {cfg.max_iter}")
    print("  BƯỚC 2b — gộp map + NMS (Algorithm 2)")
    print(f"    L levels    : {cfg.n_levels}  log-spaced trong "
          f"[{cfg.kl_h_min}, {cfg.kl_h_max}]")
    print(f"    heights     : {np.round(heights, 3).tolist()}")
    print(f"    a_min       : {cfg.min_area_px} px    tau_IoU : {cfg.nms_iou}    "
          f"N_max : {cfg.max_masks}")
    print("  " + "-" * 74)
    print(f"  device      : {args.device}  ({platform.node()})")
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        print(f"  gpu         : {torch.cuda.get_device_name(args.device)}")
    print(f"  command     : {' '.join(sys.argv)}")
    if deviations:
        print("  ⚠️ LỆCH KHỎI PAPER (do cờ dòng lệnh):")
        for d in deviations:
            print(f"      {d}")
    else:
        print("  ✓ không lệch tham số nào so với config/paper.py")
    print("=" * 78)
    if args.dataset in PAPER_AR:
        print(f"  MỐC paper trên {args.dataset}: AR_1000 = {PAPER_AR[args.dataset]}")
        if args.dataset == "paco":
            print(f"    {PACO_BASELINES}")
    else:
        print(f"  ⚠️ {args.dataset} KHÔNG có trong bảng training-free của paper — "
              f"không đối chiếu được.")
    print("  ⚠️ Chạy SD 1.5 (SD2 gated); t=150 tinh chỉnh CHO SD2. Không có")
    print("     CascadePSP (paper gọi là optional; Table 1 cũng đo không có).")
    print("=" * 78 + "\n")

    from d2s.attention import StableDiffusion2AttentionAggregator
    t0 = time.time()
    agg = StableDiffusion2AttentionAggregator(
        timestep=cfg.timesteps[0], attention_resolution=cfg.grid_r,
        weight_down_block_0=cfg.w_down_0, weight_down_block_1=cfg.w_down_1,
        weight_up_block_0=cfg.w_up_0, weight_up_block_1=cfg.w_up_1,
        weight_up_block_2=cfg.w_up_2, hugging_face_model_id=cfg.model_source,
        prompt_text=cfg.prompt_text, device=args.device, torch_dtype=torch.float16)
    print(f"model loaded in {time.time() - t0:.1f}s\n")

    if args.save_masks:
        os.makedirs(args.save_masks, exist_ok=True)

    hits = np.zeros(len(IOU_THRESHOLDS), dtype=np.int64)
    band_hits = {k: np.zeros(len(IOU_THRESHOLDS), dtype=np.int64) for k in SIZE_BANDS}
    band_n = {k: 0 for k in SIZE_BANDS}
    part_hits = np.zeros(len(IOU_THRESHOLDS), dtype=np.int64)
    obj_hits = np.zeros(len(IOU_THRESHOLDS), dtype=np.int64)
    n_part = n_obj = n_gt_total = n_pred_total = 0
    n_not_conv, n_iters, ceilings = 0, [], []
    level_masks = np.zeros(cfg.n_levels, dtype=np.int64)
    level_clusters = np.zeros(cfg.n_levels, dtype=np.int64)
    nms_in = nms_kept = nms_cap = 0
    per_image = []
    t_start = time.time()

    # Đồng hồ TÁCH THEO KHÂU. Một con số "s/ảnh" duy nhất không nói được nút
    # thắt nằm đâu, mà ba khâu này có chi phí rất khác nhau và khác nhau theo
    # ảnh: SD gần như hằng số, p-Laplacian phụ thuộc số vòng lặp, Algorithm 2
    # phụ thuộc số cụm (đo được: NMS 3000 mask = 80 s).
    t_sd = t_prop = t_merge = t_eval = 0.0

    for i in range(args.start, end):
        s = ds[i]
        _t = time.time()
        A = build_affinity(s["image"], cfg, agg)
        t_sd += time.time() - _t

        _t = time.time()
        out = segment_image(s["image"], s["valid_h"], cfg, A=A,
                            valid_w=s["valid_w"], orig_hw=(s["H"], s["W"]))
        t_step = time.time() - _t
        # segment_image gộp lan truyền + Algorithm 2; tách bằng tỉ lệ n_iter đã
        # biết thì không đáng tin, nên ghi chung và gọi đúng tên.
        t_prop += t_step

        _t = time.time()
        pred = out["masks_full"]
        iou = mask_iou_matrix(pred, s["gt_masks"])
        h, n_gt, per_band = ar_one_image(iou, s["gt_areas"],
                                         max_proposals=cfg.max_masks)
        hits += h
        n_gt_total += n_gt
        n_pred_total += len(pred)
        for k in SIZE_BANDS:
            band_hits[k] += per_band[k][0]
            band_n[k] += per_band[k][1]

        ip = s.get("is_part")
        if ip is not None:
            if ip.any():
                hp, ngp, _ = ar_one_image(iou[:, ip], s["gt_areas"][ip],
                                          max_proposals=cfg.max_masks)
                part_hits += hp
                n_part += ngp
            if (~ip).any():
                ho, ngo, _ = ar_one_image(iou[:, ~ip], s["gt_areas"][~ip],
                                          max_proposals=cfg.max_masks)
                obj_hits += ho
                n_obj += ngo

        t_eval += time.time() - _t

        mi = out["merge_info"]
        for li, lv in enumerate(mi["per_level"]):
            level_masks[li] += lv["n_masks"]
            level_clusters[li] += lv["n_clusters"]
        nms_in += mi["nms"]["n_in"]
        nms_kept += mi["nms"]["n_kept"]
        nms_cap += mi["nms"]["n_over_cap"]

        n_iters.append(out["n_iter"])
        if not out["converged"]:
            n_not_conv += 1
        ceil = ds.resolution_ceiling(i, cfg.grid_r)
        if not np.isnan(ceil):
            ceilings.append(ceil)

        if args.save_masks and len(pred):
            np.savez_compressed(
                os.path.join(args.save_masks, f"{s['image_id']}.npz"),
                masks=np.packbits(pred, axis=None), shape=np.array(pred.shape))

        per_image.append({
            "file_name": s["file_name"], "image_id": s["image_id"],
            "n_gt": n_gt, "n_pred": int(len(pred)), "hits_at_50": int(h[0]),
            "n_iter": out["n_iter"], "converged": out["converged"],
            "n_pool": mi["nms"]["n_in"], "resolution_ceiling": ceil,
            "W": s["W"], "H": s["H"]})

        done = i - args.start + 1
        run_ar = (hits / max(n_gt_total, 1)).mean()
        el = time.time() - t_start
        eta = el / done * (n - done)
        print(f"  [{done:4d}/{n} {100 * done / n:5.1f}%] {s['file_name']:20s} "
              f"{s['W']:4d}x{s['H']:<4d} "
              f"n_gt={n_gt:3d} pool={mi['nms']['n_in']:4d} -> "
              f"n_pred={len(pred):4d} hit@50={h[0]:3d} | "
              f"AR_1000={100 * run_ar:5.2f} | "
              f"{el / done:5.1f}s/ảnh | elapsed {fmt_time(el)} | "
              f"ETA {fmt_time(eta)}", flush=True)
        if done == 1:
            print(f"         mức: {[lv['n_clusters'] for lv in mi['per_level']]} cụm "
                  f"-> {[lv['n_masks'] for lv in mi['per_level']]} mask "
                  f"| n_iter={out['n_iter']}", flush=True)
            if torch.cuda.is_available() and args.device.startswith("cuda"):
                print(f"         max_memory_allocated = "
                      f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)

    res = summarise_ar(hits, n_gt_total, band_hits, band_n)
    elapsed = time.time() - t_start

    print("\n" + "=" * 78)
    print(f"KẾT QUẢ — {args.dataset}, Diffuse2Seg training-free đầy đủ")
    ref = PAPER_AR.get(args.dataset)
    print(f"  AR_1000        : {100 * res['AR_1000']:6.2f}" +
          (f"     (paper: {ref})" if ref else "     (paper không báo bộ này)"))
    print(f"  AR_S / M / L   : {100 * res['AR_S']:5.2f} / {100 * res['AR_M']:5.2f} / "
          f"{100 * res['AR_L']:5.2f}   "
          f"(n: {res['n_gt_S']}/{res['n_gt_M']}/{res['n_gt_L']})")
    print(f"  recall@0.50    : {100 * res['recall_at_50']:6.2f}   "
          f"recall@0.75: {100 * res['recall_at_75']:6.2f}")
    if n_part or n_obj:
        ar_part = float((part_hits / max(n_part, 1)).mean())
        ar_obj = float((obj_hits / max(n_obj, 1)).mean())
        print(f"\n  AR trên OBJECT : {100 * ar_obj:6.2f}  (n={n_obj})")
        print(f"  AR trên PART   : {100 * ar_part:6.2f}  (n={n_part})")
    else:
        ar_part = ar_obj = None
    print(f"\n  mask/ảnh       : {n_pred_total / max(n, 1):.1f}  "
          f"(GT {n_gt_total / max(n, 1):.1f}/ảnh)")

    print(f"\nTHỜI GIAN — {fmt_time(elapsed)} cho {n} ảnh "
          f"({elapsed / max(n, 1):.1f}s/ảnh)")
    other = max(elapsed - t_sd - t_prop - t_eval, 0.0)
    for name, t in (("SD forward + affinity", t_sd),
                    ("lan truyền + Algorithm 2", t_prop),
                    ("tính IoU + AR", t_eval),
                    ("còn lại (đọc ảnh, giải mã GT)", other)):
        print(f"    {name:32s} {fmt_time(t):>10s}  {100 * t / max(elapsed, 1e-9):5.1f} %  "
              f"({t / max(n, 1):6.2f}s/ảnh)")
    print(f"  ⚙️  ngoại suy cả tập {len(ds)} ảnh: "
          f"{fmt_time(elapsed / max(n, 1) * len(ds))}")

    print("\nCHẨN ĐOÁN — sáu mức granularity có làm gì không?")
    print(f"  {'height':>8} {'cụm/ảnh':>9} {'mask/ảnh':>9}")
    for li, hh in enumerate(heights):
        print(f"  {hh:8.3f} {level_clusters[li] / max(n, 1):9.1f} "
              f"{level_masks[li] / max(n, 1):9.1f}")
    print(f"  NMS: {nms_in / max(n, 1):.1f} vào -> {nms_kept / max(n, 1):.1f} giữ "
          f"({100 * (1 - nms_kept / max(nms_in, 1)):.0f} % bị chặn ở "
          f"tau_IoU={cfg.nms_iou})")
    if nms_cap:
        print(f"  ⚠️ {nms_cap} mask bị cắt vì chạm cap N_max={cfg.max_masks}")
    if level_clusters[0] / max(n, 1) < 2:
        print("  ⚠️ h nhỏ nhất đã gộp gần hết prompt -> dải height không khớp thang")
        print("     KL thực tế. Đây là dấu hiệu cần quét lại [hmin, hmax], KHÔNG")
        print("     phải cơ chế hỏng.")

    conv = 1.0 - n_not_conv / max(n, 1)
    print(f"\n  hội tụ         : {100 * conv:.1f} %  "
          f"(n_iter median {np.median(n_iters):.0f}/{cfg.max_iter})")
    if conv < 0.90:
        print("     ⚠️ dưới 90 % -> số trên mô tả CAP, không mô tả cơ chế")
    if ceilings:
        print(f"  trần độ phân giải: {100 * np.mean(ceilings):.1f} % GT có cạnh "
              f"ngắn >= 1 ô (grid_r={cfg.grid_r})")

    print("\n" + "=" * 78)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"tool": "run_paper", "dataset": args.dataset,
                       "n_images": n, "start": args.start,
                       "config": cfg.to_dict(), "deviations_from_paper": deviations,
                       "results": res, "ar_object": ar_obj, "ar_part": ar_part,
                       "n_gt_object": n_obj, "n_gt_part": n_part,
                       "paper_reference_ar1000": ref,
                       "elapsed_sec": elapsed,
                       "diagnostics": {
                           "convergence_rate": conv,
                           "median_n_iter": float(np.median(n_iters)),
                           "heights": [float(x) for x in heights],
                           "clusters_per_level": (level_clusters / max(n, 1)).tolist(),
                           "masks_per_level": (level_masks / max(n, 1)).tolist(),
                           "nms_in_per_image": nms_in / max(n, 1),
                           "nms_kept_per_image": nms_kept / max(n, 1),
                           "n_over_cap": nms_cap,
                           "mean_resolution_ceiling":
                               float(np.mean(ceilings)) if ceilings else None},
                       "per_image": per_image}, f, indent=2)
        print(f"  -> {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

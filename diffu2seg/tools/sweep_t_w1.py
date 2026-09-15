#!/usr/bin/env python3
r"""Quét `timestep` và `w1` (trọng số trộn 2 layer decoder) trên PACO, đo AR_1000.

VÌ SAO HAI THAM SỐ NÀY, VÀ VÌ SAO ĐÁNG QUÉT

Cả hai đều được paper chọn để tối ưu **mAP**, không phải AR — và ta CHỈ đo AR:

  t = 150   §5: "recall degrades across timesteps [...] earlier time steps focus
            more on fine texture, which favors our objective of segmenting
            objects at varying granularities". Họ chọn 150 vì nó "maximizes mAP
            while keeping strong mAR". => hướng quét đúng là t NHỎ HƠN.
  w1 = 0.85 §A.1: "w1 = 0.85 and w2 = 0.15, which lies on the PRECISION
            plateau". Không phải giá trị tối ưu recall.

⚠️ `w1 = 0.85/0.15` KHÔNG phải "giá trị của SD2" — đó là số Diffuse2Seg tự đo
trên SA-1B holdout. M2N2 dùng 0.5/0.5. Nhưng phép đo đó làm trên SD2, còn ta
chạy SD 1.5, nên quét lại là hợp lý.

                        CHI PHÍ — VÌ SAO QUÉT ĐƯỢC

Đo thật trên A30 (50 ảnh, r=140): 20,0 s/ảnh, trong đó SD chỉ 2,3 s (11,5 %),
còn lan truyền + Algorithm 2 chiếm 16,6 s (82,9 %). Nên:

  * Mỗi giá trị `t` PHẢI chạy lại SD (t đổi thì attention đổi).
  * Mỗi giá trị `w1` thì KHÔNG: tool trích từng layer RIÊNG một lần
    (`extract_per_layer`), rồi trộn ngoài. Quét thêm một `w1` chỉ tốn phần
    p-Laplacian, không tốn SD.

Với `--limit 20`, 4 giá trị t và 4 giá trị w1 = 16 cấu hình:
    16 x 20 x ~18 s ~ 1 giờ 40 phút.
Giảm `--limit` nếu cần nhanh hơn; 20 ảnh đã đủ để xếp hạng, KHÔNG đủ để báo số.

                    ⚠️ ĐỌC KẾT QUẢ CHO ĐÚNG

`--limit 20` cho sai số lấy mẫu LỚN. Trên 50 ảnh AR_1000 = 10,36; trên 20 ảnh
con số đó có thể lệch vài điểm chỉ vì ảnh nào rơi vào mẫu. Tool này để CHỌN
cấu hình, không để BÁO số. Chạy lại cấu hình thắng bằng `run_paper.py` với
`--limit` lớn rồi mới trích số.

Và: chọn cấu hình tốt nhất trên cùng tập ảnh dùng để đánh giá là **overfit tập
đó**. Đúng ra phải quét trên một tập, xác nhận trên tập khác — `--start` cho
phép làm điều đó (quét trên ảnh 0..19, xác nhận trên 100..149).

CHẠY (TRÊN SERVER):
    export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
    LOG=/mnt/disk1/aiotlab/haitn/log/d2s_sweep_$(date +%Y%m%d_%H%M%S).log
    nohup python tools/run_on_free_gpu.py -- tools/sweep_t_w1.py \
        --limit 20 --timesteps 50 100 150 300 --w1 1.0 0.85 0.5 0.15 \
        --out /mnt/disk1/aiotlab/haitn/output/d2s_sweep.json > "$LOG" 2>&1 &
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
from d2s.affinity import to_affinity                       # noqa: E402
from d2s.pipeline import segment_image                     # noqa: E402
from utils.ar_metrics import (IOU_THRESHOLDS, SIZE_BANDS,  # noqa: E402
                              ar_one_image, summarise_ar)
from utils.mask_ops import mask_iou_matrix                 # noqa: E402
from utils.metrics import fmt_time                         # noqa: E402

UP0 = '.up_blocks.3.attentions.0.transformer_blocks.0.attn1'
UP1 = '.up_blocks.3.attentions.1.transformer_blocks.0.attn1'


def build_dataset(name, root, canvas):
    if name == "paco":
        from data.paco_val import PacoVal
        return PacoVal(os.path.join(root, "paco", "paco_lvis_v1_val.json"),
                       os.path.join(root, "paco", "images"), canvas=canvas)
    from data.coco_val import CocoVal
    return CocoVal(os.path.join(root, "coco", "annotations",
                                "instances_val2017.json"),
                   os.path.join(root, "coco", "val2017"), canvas=canvas)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="paco", choices=["paco", "coco"])
    ap.add_argument("--limit", type=int, default=20,
                    help="ảnh mỗi cấu hình. 20 đủ để XẾP HẠNG, không đủ để BÁO số")
    ap.add_argument("--start", type=int, default=0,
                    help="dùng --start khác để xác nhận trên tập ảnh khác")
    ap.add_argument("--timesteps", type=int, nargs="+", default=[50, 100, 150, 300],
                    help="paper: 150. §5 gợi ý t nhỏ hơn cho recall cao hơn")
    ap.add_argument("--w1", type=float, nargs="+", default=[1.0, 0.85, 0.5, 0.15],
                    help="trọng số up_blocks.3.attentions.0; w2 = 1 - w1. paper: 0.85")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from config.paper import cfg as paper_cfg
    cfg0 = paper_cfg
    root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
    ds = build_dataset(args.dataset, root, cfg0.canvas)
    end = min(args.start + args.limit, len(ds))
    n = end - args.start
    n_cfg = len(args.timesteps) * len(args.w1)

    print("=" * 78)
    print("QUÉT timestep x w1 — Diffuse2Seg trên " + args.dataset)
    print(f"  timestamp   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ảnh         : {n} (từ {args.start}) / {len(ds)}")
    print(f"  timesteps   : {args.timesteps}      (paper: 150)")
    print(f"  w1          : {args.w1}   (paper: 0.85; w2 = 1 - w1)")
    print(f"  -> {n_cfg} cấu hình x {n} ảnh")
    print(f"  cố định     : canvas {cfg0.canvas} (r={cfg0.grid_r}), p={cfg0.p}, "
          f"tau_att={cfg0.tau_att}, stride={cfg0.prompt_stride_cells},")
    print(f"                L={cfg0.n_levels} trong [{cfg0.kl_h_min}, {cfg0.kl_h_max}], "
          f"tau_IoU={cfg0.nms_iou}, a_min={cfg0.min_area_px}")
    print(f"  device      : {args.device}  ({platform.node()})")
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        print(f"  gpu         : {torch.cuda.get_device_name(args.device)}")
    print(f"  command     : {' '.join(sys.argv)}")
    print("=" * 78)
    print("  ⚠️ Quét để CHỌN cấu hình, KHÔNG để báo số — 20 ảnh có sai số lấy mẫu")
    print("     lớn. Chạy lại cấu hình thắng bằng run_paper.py --limit lớn.")
    print("  ⚠️ Chọn trên cùng tập dùng để đánh giá là OVERFIT tập đó. Xác nhận")
    print("     lại bằng --start khác.")
    print("=" * 78 + "\n")

    from d2s.attention import StableDiffusion2AttentionAggregator
    t0 = time.time()
    agg = StableDiffusion2AttentionAggregator(
        timestep=cfg0.timesteps[0], attention_resolution=cfg0.grid_r,
        weight_down_block_0=0.0, weight_down_block_1=0.0,
        weight_up_block_0=1.0, weight_up_block_1=1.0, weight_up_block_2=0.0,
        hugging_face_model_id=cfg0.model_source, prompt_text=cfg0.prompt_text,
        device=args.device, torch_dtype=torch.float16)
    print(f"model loaded in {fmt_time(time.time() - t0)}\n")

    # acc[(t, w1)] -> {hits, n_gt, band..., n_pred}
    acc = {(t, w): {"hits": np.zeros(len(IOU_THRESHOLDS), dtype=np.int64),
                    "n_gt": 0, "n_pred": 0,
                    "band_hits": {k: np.zeros(len(IOU_THRESHOLDS), dtype=np.int64)
                                  for k in SIZE_BANDS},
                    "band_n": {k: 0 for k in SIZE_BANDS},
                    "part_hits": np.zeros(len(IOU_THRESHOLDS), dtype=np.int64),
                    "obj_hits": np.zeros(len(IOU_THRESHOLDS), dtype=np.int64),
                    "n_part": 0, "n_obj": 0}
           for t in args.timesteps for w in args.w1}

    t_start = time.time()
    done_cfg = 0
    total_units = n * len(args.timesteps)          # SD chạy 1 lần / (ảnh, t)

    for img_i, i in enumerate(range(args.start, end)):
        s = ds[i]
        for t in args.timesteps:
            # MỘT lần SD cho mỗi (ảnh, t): trích riêng hai layer, trộn sau.
            per_layer = agg.extract_per_layer(s["image"], t, paths=[UP0, UP1])
            a0, a1 = per_layer[UP0], per_layer[UP1]

            for w in args.w1:
                # TRỘN TRƯỚC, CHUẨN HOÁ SAU — đúng thứ tự của đường chạy thật.
                # extract_per_layer trả tensor THÔ chính vì lý do này: chuẩn
                # hoá từng layer rồi mới trộn cho kết quả KHÁC (đo: 7,8e-3).
                mixed = a0 * w + a1 * (1.0 - w)
                h, wd = mixed.shape[0], mixed.shape[1]
                denom = mixed.reshape(h, wd, -1).sum(dim=2)[:, :, None, None]
                mixed = mixed / denom.clamp_min(torch.finfo(mixed.dtype).tiny)
                A = to_affinity(mixed, tau_att=cfg0.tau_att, dtype=torch.float32)

                cfg = Diffu2SegConfig(**{**cfg0.__dict__, "timesteps": (t,)}).validate()
                out = segment_image(s["image"], s["valid_h"], cfg, A=A,
                                    valid_w=s["valid_w"], orig_hw=(s["H"], s["W"]))
                iou = mask_iou_matrix(out["masks_full"], s["gt_masks"])
                hh, n_gt, per_band = ar_one_image(iou, s["gt_areas"],
                                                  max_proposals=cfg.max_masks)
                a = acc[(t, w)]
                a["hits"] += hh
                a["n_gt"] += n_gt
                a["n_pred"] += len(out["masks_full"])
                for k in SIZE_BANDS:
                    a["band_hits"][k] += per_band[k][0]
                    a["band_n"][k] += per_band[k][1]
                ip = s.get("is_part")
                if ip is not None:
                    if ip.any():
                        hp, ngp, _ = ar_one_image(iou[:, ip], s["gt_areas"][ip],
                                                  max_proposals=cfg.max_masks)
                        a["part_hits"] += hp
                        a["n_part"] += ngp
                    if (~ip).any():
                        ho, ngo, _ = ar_one_image(iou[:, ~ip], s["gt_areas"][~ip],
                                                  max_proposals=cfg.max_masks)
                        a["obj_hits"] += ho
                        a["n_obj"] += ngo
                del A, out
            del a0, a1, per_layer
            done_cfg += 1

        el = time.time() - t_start
        eta = el / done_cfg * (total_units - done_cfg)
        best = max(acc.items(), key=lambda kv: (kv[1]["hits"] / max(kv[1]["n_gt"], 1)).mean())
        best_ar = (best[1]["hits"] / max(best[1]["n_gt"], 1)).mean()
        print(f"  [{img_i + 1:3d}/{n} {100 * (img_i + 1) / n:5.1f}%] "
              f"{s['file_name']:20s} n_gt={len(s['gt_areas']):3d} | "
              f"dẫn đầu t={best[0][0]} w1={best[0][1]} AR={100 * best_ar:5.2f} | "
              f"elapsed {fmt_time(el)} | ETA {fmt_time(eta)}", flush=True)
        if img_i == 0 and torch.cuda.is_available() and args.device.startswith("cuda"):
            print(f"         max_memory_allocated = "
                  f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)

    # ---------------- bảng kết quả ----------------
    rows = []
    for (t, w), a in acc.items():
        res = summarise_ar(a["hits"], a["n_gt"], a["band_hits"], a["band_n"])
        rows.append({
            "timestep": t, "w1": w, "w2": round(1.0 - w, 4),
            "AR_1000": res["AR_1000"], "AR_S": res["AR_S"], "AR_M": res["AR_M"],
            "AR_L": res["AR_L"], "recall_at_50": res["recall_at_50"],
            "ar_part": float((a["part_hits"] / max(a["n_part"], 1)).mean()),
            "ar_object": float((a["obj_hits"] / max(a["n_obj"], 1)).mean()),
            "n_pred_per_image": a["n_pred"] / max(n, 1), "n_gt": a["n_gt"]})
    rows.sort(key=lambda r: -r["AR_1000"])

    elapsed = time.time() - t_start
    print("\n" + "=" * 78)
    print(f"BẢNG QUÉT — {n} ảnh, {n_cfg} cấu hình, {fmt_time(elapsed)}")
    print(f"  {'t':>5} {'w1':>5} {'w2':>5} | {'AR_1000':>8} {'AR_S':>6} {'AR_M':>6} "
          f"{'AR_L':>6} | {'AR_obj':>7} {'AR_part':>7} | {'mask/ảnh':>9}")
    print("  " + "-" * 74)
    for r in rows:
        star = "  <-- tốt nhất" if r is rows[0] else ""
        paper = "  (paper)" if (r["timestep"] == 150 and abs(r["w1"] - 0.85) < 1e-9) else ""
        print(f"  {r['timestep']:5d} {r['w1']:5.2f} {r['w2']:5.2f} | "
              f"{100 * r['AR_1000']:8.2f} {100 * r['AR_S']:6.2f} "
              f"{100 * r['AR_M']:6.2f} {100 * r['AR_L']:6.2f} | "
              f"{100 * r['ar_object']:7.2f} {100 * r['ar_part']:7.2f} | "
              f"{r['n_pred_per_image']:9.1f}{star}{paper}")

    best = rows[0]
    base = next((r for r in rows if r["timestep"] == 150 and abs(r["w1"] - 0.85) < 1e-9),
                None)
    print("\n" + "=" * 78)
    print(f"TỐT NHẤT: t={best['timestep']}  w1={best['w1']}/w2={best['w2']}  "
          f"AR_1000={100 * best['AR_1000']:.2f}")
    if base:
        d = 100 * (best["AR_1000"] - base["AR_1000"])
        print(f"  cấu hình paper (t=150, w1=0.85): {100 * base['AR_1000']:.2f}  "
              f"-> chênh {d:+.2f} p.p.")
        if abs(d) < 1.0:
            print("  ⚠️ Chênh dưới 1 p.p. trên 20 ảnh NẰM TRONG NHIỄU LẤY MẪU.")
            print("     Chưa đủ để kết luận cấu hình nào hơn — cần --limit lớn hơn.")
    # Lệnh sẵn sàng copy, ĐIỀN SẴN cấu hình thắng và tập ảnh rời nhau.
    OUT = "/mnt/disk1/aiotlab/haitn/output"
    w1_flag = f" --w1 {best['w1']}" if abs(best["w1"] - 0.85) > 1e-9 else ""
    next_start = args.start + n          # tập rời hẳn với tập vừa quét
    print("\n  BA BƯỚC TIẾP THEO (copy thẳng):\n")
    print(f"  # 1. XÁC NHẬN trên tập ảnh KHÁC (tránh overfit tập vừa quét)")
    print(f"  python tools/sweep_t_w1.py --start {next_start} --limit {n} \\")
    print(f"      --timesteps {best['timestep']} 150 --w1 {best['w1']} 0.85 \\")
    print(f"      --out {OUT}/d2s_sweep_confirm.json")
    print(f"  # ~{fmt_time(elapsed / max(n_cfg, 1) * 4)} (4 cấu hình)\n")
    print(f"  # 2. CHẠY ĐỦ cấu hình thắng -> sinh per_image cho bước 3")
    print(f"  python tools/run_paper.py --dataset {args.dataset} --limit {n} \\")
    print(f"      --timestep {best['timestep']}{w1_flag} \\")
    print(f"      --out {OUT}/d2s_paper_best.json")
    print(f"  # ~{fmt_time(elapsed / max(n_cfg, 1))}\n")
    print(f"  # 3. VẼ 20 ảnh tốt nhất của cấu hình đó")
    print(f"  python tools/visualize_best.py \\")
    print(f"      --from-json {OUT}/d2s_paper_best.json \\")
    print(f"      --limit 20 --pick best --out-dir {OUT}/d2s_viz_best")
    print(f"  # ⚠️ 'best' là MẪU CHỌN LỌC. Chạy thêm --pick spread để thấy")
    print(f"  #    phân bố thật, và --pick worst để tìm chỗ hỏng.")
    print("=" * 78)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"tool": "sweep_t_w1", "dataset": args.dataset,
                       "n_images": n, "start": args.start,
                       "timesteps": args.timesteps, "w1_values": args.w1,
                       "fixed_config": cfg0.to_dict(),
                       "elapsed_sec": elapsed,
                       "rows_sorted_by_ar": rows,
                       "best": best, "paper_baseline": base}, f, indent=2)
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

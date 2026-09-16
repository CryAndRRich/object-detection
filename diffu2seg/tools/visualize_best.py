#!/usr/bin/env python3
r"""Vẽ mask GĐ2 lên N ảnh, chọn theo kết quả của một lần chạy trước.

NHÌN ẢNH TRƯỚC KHI TIN BẤT KỲ CON SỐ NÀO — bài học §5 của docs/02-du-lieu-ce130.md:
visualize bắt được 2 lỗi lớn mà toàn bộ test và 3 vòng rà soát code bỏ sót.

CHỌN ẢNH NÀO: đọc `per_image` trong JSON của `run_paper.py`, xếp theo
`recall@50 = hits_at_50 / n_gt`, rồi lấy:

    --pick best    N ảnh recall cao nhất   -> cơ chế chạy ĐÚNG trông thế nào
    --pick worst   N ảnh thấp nhất         -> hỏng ở đâu
    --pick spread  đều khắp dải            <- MẶC ĐỊNH, và là thứ nên xem trước

⚠️ `--pick best` một mình LÀ CHỌN LỌC THIÊN VỊ. 30 ảnh tốt nhất trong 50 trông
sẽ luôn thuyết phục, kể cả khi trung vị recall chỉ 0,27 (đo được ở lần chạy
2026-09-15). Nên mặc định là `spread`, và mỗi ảnh in kèm recall thật của nó
ngay trên tiêu đề để không ai đọc nhầm một mẫu chọn lọc thành kết quả chung.

MÀU: xanh lá = GT | cam = pred khớp GT ở IoU>=0.5 | đỏ = pred không khớp
     (chỉ vẽ tối đa `--max-draw` mask pred, xếp theo diện tích giảm dần —
      trung bình có 509 mask/ảnh, vẽ hết thì không nhìn được gì)

CHẠY (TRÊN SERVER):
    export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
    python tools/run_on_free_gpu.py -- tools/visualize_best.py \
        --from-json /mnt/disk1/aiotlab/haitn/output/d2s_paper_paco.json \
        --limit 30 --pick spread \
        --out-dir /mnt/disk1/aiotlab/haitn/output/d2s_viz
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base import Diffu2SegConfig                  # noqa: E402
from d2s.pipeline import build_affinity, segment_image   # noqa: E402
from utils.mask_ops import mask_iou_matrix               # noqa: E402
from utils.metrics import fmt_time                       # noqa: E402


def build_dataset(name, root, canvas):
    if name == "paco":
        from data.paco_val import PacoVal
        return PacoVal(os.path.join(root, "paco", "paco_lvis_v1_val.json"),
                       os.path.join(root, "paco", "images"), canvas=canvas)
    from data.coco_val import CocoVal
    return CocoVal(os.path.join(root, "coco", "annotations",
                                "instances_val2017.json"),
                   os.path.join(root, "coco", "val2017"), canvas=canvas)


def pick_indices(per_image, n, mode):
    """-> list các (chỉ số trong dataset, recall) theo tiêu chí `mode`."""
    scored = []
    for k, p in enumerate(per_image):
        rec = p["hits_at_50"] / max(p["n_gt"], 1)
        scored.append((k, rec, p))
    scored.sort(key=lambda x: -x[1])

    if mode == "best":
        chosen = scored[:n]
    elif mode == "worst":
        chosen = scored[-n:]
    else:                                   # spread
        if n >= len(scored):
            chosen = scored
        else:
            idx = np.linspace(0, len(scored) - 1, n).round().astype(int)
            chosen = [scored[i] for i in idx]
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-json", required=True,
                    help="JSON của run_paper.py — quyết định ảnh nào được vẽ")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--pick", default="spread", choices=["spread", "best", "worst"])
    ap.add_argument("--max-draw", type=int, default=60,
                    help="số mask pred vẽ tối đa mỗi ảnh (theo diện tích giảm dần)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    with open(args.from_json) as f:
        blob = json.load(f)
    saved_cfg = blob["config"]
    per_image = blob["per_image"]
    dataset = blob.get("dataset", "paco")

    # Dựng lại ĐÚNG cấu hình đã chạy, không dùng mặc định -- nếu không thì ảnh
    # vẽ ra không phải ảnh của con số trong JSON.
    fields = {k: v for k, v in saved_cfg.items()
              if k in Diffu2SegConfig.__dataclass_fields__}
    for k in ("timesteps", "timestep_weights", "kl_thresholds"):
        if k in fields and isinstance(fields[k], list):
            fields[k] = tuple(fields[k])
    cfg = Diffu2SegConfig(**fields).validate()

    root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
    ds = build_dataset(dataset, root, cfg.canvas)
    start = blob.get("start", 0)

    chosen = pick_indices(per_image, args.limit, args.pick)
    os.makedirs(args.out_dir, exist_ok=True)

    recs = np.array([r for _, r, _ in chosen])
    all_rec = np.array([p["hits_at_50"] / max(p["n_gt"], 1) for p in per_image])
    print("=" * 74)
    print(f"VISUALIZE — {len(chosen)} ảnh, chọn kiểu '{args.pick}'")
    print(f"  nguồn       : {args.from_json}")
    print(f"  cấu hình    : t={cfg.timesteps[0]} w1={cfg.w_up_0} canvas={cfg.canvas} "
          f"r={cfg.grid_r} p={cfg.p}")
    print(f"  AR_1000 lần chạy đó : {100 * blob['results']['AR_1000']:.2f}")
    print(f"  recall@50 CẢ {len(per_image)} ảnh  : trung vị {np.median(all_rec):.2f}  "
          f"[{all_rec.min():.2f}, {all_rec.max():.2f}]")
    print(f"  recall@50 {len(chosen)} ảnh được chọn: trung vị {np.median(recs):.2f}  "
          f"[{recs.min():.2f}, {recs.max():.2f}]")
    if args.pick == "best":
        print("  ⚠️ 'best' LÀ MẪU CHỌN LỌC — đừng đọc nó như kết quả chung.")
    print("=" * 74 + "\n")

    from d2s.attention import StableDiffusion2AttentionAggregator
    agg = StableDiffusion2AttentionAggregator(
        timestep=cfg.timesteps[0], attention_resolution=cfg.grid_r,
        weight_down_block_0=cfg.w_down_0, weight_down_block_1=cfg.w_down_1,
        weight_up_block_0=cfg.w_up_0, weight_up_block_1=cfg.w_up_1,
        weight_up_block_2=cfg.w_up_2, hugging_face_model_id=cfg.model_source,
        prompt_text=cfg.prompt_text, device=args.device, torch_dtype=torch.float16)

    t_start = time.time()
    for j, (k, rec_saved, rec_info) in enumerate(chosen):
        i = start + k
        s = ds[i]
        A = build_affinity(s["image"], cfg, agg)
        out = segment_image(s["image"], s["valid_h"], cfg, A=A,
                            valid_w=s["valid_w"], orig_hw=(s["H"], s["W"]))
        pred = out["masks_full"]

        iou = mask_iou_matrix(pred, s["gt_masks"])
        hit_pred = (iou.max(axis=1) >= 0.5) if iou.size else np.zeros(len(pred), bool)
        hit_gt = (iou.max(axis=0) >= 0.5) if iou.size else np.zeros(len(s["gt_masks"]), bool)
        rec = hit_gt.mean() if len(hit_gt) else 0.0

        areas = pred.reshape(len(pred), -1).sum(1) if len(pred) else np.zeros(0)
        order = np.argsort(-areas)[:args.max_draw]

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        raw = s["image"][:int(s["valid_h"] * cfg.canvas),
                         :int(s["valid_w"] * cfg.canvas)]

        axes[0].imshow(raw)
        axes[0].set_title(f"ảnh gốc {s['W']}x{s['H']}")

        # giữa: mask pred, mỗi instance một màu
        axes[1].imshow(s["image"])
        if len(order):
            rng = np.random.default_rng(0)
            overlay = np.zeros((s["H"], s["W"], 4))
            for oi in order:
                c = rng.random(3)
                overlay[pred[oi]] = [c[0], c[1], c[2], 0.55]
            axes[1].imshow(overlay, extent=[0, cfg.canvas * s["valid_w"],
                                            cfg.canvas * s["valid_h"], 0])
        axes[1].set_title(f"{len(pred)} mask (vẽ {len(order)} lớn nhất) | "
                          f"{out['merge_info']['nms']['n_in']} trước NMS")

        # phải: box từ mask, so GT
        axes[2].imshow(s["image"])
        sx = cfg.canvas * s["valid_w"] / s["W"]
        sy = cfg.canvas * s["valid_h"] / s["H"]
        for gi, gm in enumerate(s["gt_masks"]):
            ys, xs = np.where(gm)
            if not len(ys):
                continue
            axes[2].add_patch(Rectangle(
                (xs.min() * sx, ys.min() * sy),
                (xs.max() - xs.min() + 1) * sx, (ys.max() - ys.min() + 1) * sy,
                fill=False, edgecolor="lime" if hit_gt[gi] else "darkgreen",
                lw=2.0 if hit_gt[gi] else 1.0,
                linestyle="-" if hit_gt[gi] else ":"))
        for oi in order:
            ys, xs = np.where(pred[oi])
            if not len(ys):
                continue
            axes[2].add_patch(Rectangle(
                (xs.min() * sx, ys.min() * sy),
                (xs.max() - xs.min() + 1) * sx, (ys.max() - ys.min() + 1) * sy,
                fill=False, edgecolor="orange" if hit_pred[oi] else "red", lw=1.0))
        n_hit = int(hit_gt.sum())
        axes[2].set_title(f"GT {len(s['gt_masks'])} (lá đậm = tìm thấy {n_hit}) | "
                          f"recall@50 = {rec:.2f}")

        for ax in (axes[1], axes[2]):
            if s["valid_h"] < 0.999:
                ax.axhline(s["valid_h"] * cfg.canvas, color="yellow", ls="--", lw=1.2)
            if s["valid_w"] < 0.999:
                ax.axvline(s["valid_w"] * cfg.canvas, color="yellow", ls="--", lw=1.2)
            ax.set_xlim(0, cfg.canvas)
            ax.set_ylim(cfg.canvas, 0)
        for ax in axes:
            ax.axis("off")

        n_part = int(s["is_part"].sum()) if "is_part" in s else -1
        fig.suptitle(
            f"{s['file_name']}  |  t={cfg.timesteps[0]} w1={cfg.w_up_0} r={cfg.grid_r}"
            f"  |  n_gt={len(s['gt_masks'])}"
            + (f" ({n_part} PART)" if n_part >= 0 else "")
            + f"  recall@50={rec:.2f}", fontsize=13)
        fig.tight_layout()

        name = os.path.splitext(os.path.basename(s["file_name"]))[0]
        path = os.path.join(args.out_dir, f"{rec:.2f}_{name}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        # ⚠️ Giải phóng GIỮA các ảnh. A là (h,w,h,w) fp32 = 1,54 GB ở r=140 và
        # current_merged_tensor là một bản nữa. Không xoá thì cả hai sống qua
        # vòng lặp sau, đúng lúc SD đang dựng tensor 6,15 GB cho ảnh kế tiếp.
        # empty_cache() trả phần đã giải phóng về driver — cần trên GPU dùng
        # chung, nơi phần trống thật sự có thể chỉ còn ~18 GB.
        del A, out, pred
        agg.current_merged_tensor = None
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        el = time.time() - t_start
        print(f"  [{j + 1:3d}/{len(chosen)}] {os.path.basename(path):40s} "
              f"n_gt={len(s['gt_masks']):3d} n_pred={len(pred):4d} recall={rec:.2f} | "
              f"elapsed {fmt_time(el)} | "
              f"ETA {fmt_time(el / (j + 1) * (len(chosen) - j - 1))}", flush=True)

    print(f"\n{len(chosen)} ảnh -> {args.out_dir}")
    print("  (tên file bắt đầu bằng recall@50, nên `ls` đã là bảng xếp hạng)")
    print("\n⚠️ NHÌN KỸ: mask có bám vật hay tràn nền? hai vật cùng loại cạnh nhau")
    print("   có bị gộp? các mức granularity có ra mask lồng nhau như mong đợi?")
    print("   vùng dưới/phải vạch vàng là PAD — không được sinh mask ở đó.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

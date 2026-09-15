#!/usr/bin/env python3
r"""Vẽ mask + box của GĐ1 lên ảnh thật, để KIỂM BẰNG MẮT.

VÌ SAO CÓ FILE NÀY: bài học §5 của docs/02-du-lieu-ce130.md — visualize bắt được
**2 lỗi lớn mà toàn bộ test và 3 vòng rà soát code bỏ sót**. Bộ test ở đây chạy
trên affinity giả lập; nó chứng minh phần toán đúng, không chứng minh mask rơi
đúng vật.

NHÌN ẢNH TRƯỚC KHI TIN BẤT KỲ CON SỐ NÀO. Cụ thể cần nhìn:
  - mask có bám vật hay tràn ra nền?
  - hai vật cùng class cạnh nhau có bị gộp làm một không? (CE-130 trung vị 21
    vật CÙNG class mỗi ảnh — đây là chỗ dễ hỏng nhất)
  - box có lệch nửa ô lưới không? (off-by-one = 8 px = ~20 % vật trung vị)
  - vùng pad (dải xám dưới đáy) có sinh mask rác không?

MÀU: xanh lá = GT | cam = box dự đoán hit (IoU>=0.5) | đỏ = box trượt |
     vàng = ranh giới valid_h (dưới nó là pad)

CHẠY (TRÊN SERVER):
    export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
    python tools/visualize_masks.py --split val --limit 8 \
        --out-dir /mnt/disk1/aiotlab/haitn/log/d2s_viz
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base import Diffu2SegConfig          # noqa: E402
from d2s.pipeline import build_affinity, segment_image  # noqa: E402
from data.ce130_coco import CE130Coco            # noqa: E402
from utils.box_ops_np import box_iou, cxcywh_to_xyxy  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ce130",
                    choices=["ce130", "coco", "paco"],
                    help="paco = PACO-LVIS val, bộ DUY NHẤT của bảng training-free "
                         "lấy được; coco = COCO val2017 (không có trong bảng nào)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--canvas", type=int, default=None,
                    help="ghi đè canvas; mặc định 512 cho ce130, 1120 cho coco")
    ap.add_argument("--stride", type=int, default=None,
                    help="ghi đè prompt stride; mặc định 3 cho ce130, 6 cho coco")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--p", type=float, default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    paper_cfg = args.dataset in ("coco", "paco")
    over = {}
    over["canvas"] = args.canvas if args.canvas else (1120 if paper_cfg else 512)
    over["prompt_stride_cells"] = args.stride if args.stride else (6 if paper_cfg else 3)
    if args.p is not None:
        over["p"] = args.p
    cfg = Diffu2SegConfig(**{**Diffu2SegConfig().__dict__, **over}).validate()

    root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
    if args.dataset == "paco":
        from data.paco_val import PacoVal
        ds = PacoVal(os.path.join(root, "paco", "paco_lvis_v1_val.json"),
                     os.path.join(root, "paco", "images"), canvas=cfg.canvas)
    elif args.dataset == "coco":
        from data.coco_val import CocoVal
        ds = CocoVal(os.path.join(root, "coco", "annotations",
                                  "instances_val2017.json"),
                     os.path.join(root, "coco", "val2017"), canvas=cfg.canvas)
    else:
        ds = CE130Coco(
            os.path.join(root, "ce130_coco", f"ce130_agnostic_{args.split}.json"),
            os.path.join(root, "all_phase2_V2"), cfg.canvas)
    os.makedirs(args.out_dir, exist_ok=True)

    from d2s.attention import StableDiffusion2AttentionAggregator
    agg = StableDiffusion2AttentionAggregator(
        timestep=cfg.timesteps[0], attention_resolution=cfg.grid_r,
        weight_down_block_0=cfg.w_down_0, weight_down_block_1=cfg.w_down_1,
        weight_up_block_0=cfg.w_up_0, weight_up_block_1=cfg.w_up_1,
        weight_up_block_2=cfg.w_up_2, hugging_face_model_id=cfg.model_source,
        prompt_text=cfg.prompt_text, device=args.device, torch_dtype=torch.float16)

    end = min(args.start + args.limit, len(ds))
    for i in range(args.start, end):
        s = ds[i]
        A = build_affinity(s["image"], cfg, agg)
        out = segment_image(s["image"], s["valid_h"], cfg, A=A,
                            valid_w=s.get("valid_w", 1.0))

        gt, pred = s["gt_cxcywh"], out["boxes"]
        if len(pred) and len(gt):
            iou = box_iou(cxcywh_to_xyxy(pred) * cfg.canvas,
                          cxcywh_to_xyxy(gt) * cfg.canvas)[0]
            hit = iou.max(axis=1) >= 0.5
        else:
            hit = np.zeros(len(pred), dtype=bool)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # left: the union of all masks, to see what propagation actually covered
        axes[0].imshow(s["image"])
        if len(out["masks"]):
            union = out["masks"].any(axis=0).astype(float)
            axes[0].imshow(np.kron(union, np.ones((cfg.canvas // cfg.grid_r,) * 2)),
                           alpha=0.45, cmap="viridis", vmin=0, vmax=1)
        axes[0].set_title(f"hợp của {out['n_masks']} mask  "
                          f"(prompt: {out['n_prompts']}, n_iter {out['n_iter']})")

        # right: boxes against GT
        axes[1].imshow(s["image"])
        for b in cxcywh_to_xyxy(gt) * cfg.canvas:
            axes[1].add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                        fill=False, edgecolor="lime", lw=1.4))
        for k, b in enumerate(cxcywh_to_xyxy(pred) * cfg.canvas if len(pred) else []):
            axes[1].add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                        fill=False, lw=1.4,
                                        edgecolor="orange" if hit[k] else "red"))
        recall = int(hit.sum()) / max(len(gt), 1)
        axes[1].set_title(f"GT {len(gt)} (lá) | pred {len(pred)} "
                          f"(cam=hit {int(hit.sum())}, đỏ=trượt) | recall {recall:.2f}")

        for ax in axes:
            y = s["valid_h"] * cfg.canvas
            ax.axhline(y, color="yellow", ls="--", lw=1.2)
            # Ảnh dọc (COCO 427x640) pad ở PHẢI chứ không phải ở dưới -- vạch
            # ngang một mình sẽ không cho thấy vùng pad nằm đâu.
            vw = s.get("valid_w", 1.0)
            if vw < 0.999:
                ax.axvline(vw * cfg.canvas, color="yellow", ls="--", lw=1.2)
            ax.set_xlim(0, cfg.canvas)
            ax.set_ylim(cfg.canvas, 0)
            ax.axis("off")

        fig.suptitle(f"{s['file_name']}  {s['W']}x{s['H']}  "
                     f"p={cfg.p}  r={cfg.grid_r}  "
                     f"(dưới vạch vàng là PAD)", fontsize=11)
        fig.tight_layout()

        name = s["file_name"].replace("/", "_").replace(".jpg", "")
        path = os.path.join(args.out_dir, f"{name}_p{cfg.p}.png")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{i + 1}/{end}] {path}  "
              f"n_gt={len(gt)} n_pred={len(pred)} recall={recall:.2f}")

    print(f"\n{end - args.start} ảnh -> {args.out_dir}")
    print("⚠️ NHÌN ẢNH TRƯỚC KHI TIN SỐ. Chú ý: mask tràn nền? hai vật cùng class")
    print("   bị gộp? box lệch nửa ô? vùng dưới vạch vàng có sinh mask rác?")
    return 0


if __name__ == "__main__":
    sys.exit(main())

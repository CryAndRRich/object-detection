"""Vẽ box dự đoán lên ảnh để NHÌN model đang làm gì, thay vì chỉ đọc P/R.

Vì sao cần: mọi số hiện có (§17) đều là tổng hợp trên 779 ảnh. Chúng nói được
"tốt hơn 1,89×" nhưng KHÔNG nói được model sai kiểu gì — box lệch chỗ, hay đúng
chỗ mà sai kích thước, hay tụ hết vào giữa ảnh? Ba khả năng đó dẫn tới ba hướng
sửa khác nhau, và chỉ nhìn ảnh mới phân biệt được.

Mỗi ảnh xuất ra 3 panel cạnh nhau:
    GT | dự đoán (top-k theo score) | chồng lên nhau
Kèm số liệu của riêng ảnh đó (P/R/số box khớp) in trên tiêu đề.

Lưu ý về SỐ BOX VẼ: model sinh 300 box (x4 bước nếu ensemble) — vẽ hết thì đen
kịt, không nhìn được gì. Mặc định chỉ vẽ `--topk 30` box điểm cao nhất. Đây là
lựa chọn để NHÌN, không phải để đo: P/R in kèm vẫn tính trên TOÀN BỘ box như
eval_detection.py, nên đừng nhầm hai con số.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # server không có màn hình
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_detection_dataset import CE130DetectionDataset  # noqa: E402
from data.coco_detection_dataset import CocoDetectionDataset  # noqa: E402
from models.diffusion_module import ObjectPlacementPolicy  # noqa: E402
from test_mul_box import sample_boxes, sample_boxes_multibox  # noqa: E402
from utils.matcher import _cxcywh_to_xyxy, box_iou_normalized  # noqa: E402
from eval_detection import nms_class_agnostic, greedy_match  # noqa: E402


def to_pixels(boxes_xyxy, W, H):
    """[-1,1] xyxy -> pixel của canvas đã resize+pad (đúng khung ảnh đang vẽ).

    Sau khi `_cxcywh_to_xyxy` được sửa (2026-09-04), CẢ BỐN số đều nằm trong
    cùng hệ [-1,1], nên phép đổi là như nhau cho mọi thành phần. Chính việc
    trộn hai hệ trước đây (tâm [-1,1], extent [0,1]) là thứ làm box GT vẽ ra chỉ
    rộng bằng nửa vật thể.
    """
    b = (boxes_xyxy + 1.0) / 2.0
    return torch.stack([b[:, 0] * W, b[:, 1] * H, b[:, 2] * W, b[:, 3] * H], dim=-1)


def draw(ax, img, boxes_px, color, title, lw=1.2, alpha=0.85):
    ax.imshow(img)
    for x1, y1, x2, y2 in boxes_px.tolist():
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=color, linewidth=lw, alpha=alpha))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--dataset", choices=["ce130", "coco"], default="ce130")
    p.add_argument("--split", default="test")
    p.add_argument("--all_phase2_dir", default="../../data/all_phase2_V2")
    p.add_argument("--coco_root", default="../../data")
    p.add_argument("--n_images", type=int, default=8)
    p.add_argument("--topk", type=int, default=30,
                   help="số box điểm cao nhất đem VẼ (không ảnh hưởng P/R in kèm)")
    p.add_argument("--k", type=int, default=30, help="nhánh 1-box: số mẫu mỗi ảnh")
    p.add_argument("--nms", type=float, default=0.5)
    p.add_argument("--use_ensemble", action="store_true")
    p.add_argument("--box_renewal", action="store_true")
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--out_dir", default=None,
                   help="mặc định: <thư mục checkpoint>/viz/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", type=int, default=0, help="bỏ qua N ảnh đầu")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.dirname(args.checkpoint)
    out_dir = args.out_dir or os.path.join(ckpt_dir, "viz")
    os.makedirs(out_dir, exist_ok=True)

    model_cfg = yaml.safe_load(open(os.path.join(ckpt_dir, "model_config_final.yaml")))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if isinstance(ckpt, dict) and ckpt.get("args", {}).get("num_steps"):
        model_cfg.setdefault("diffusion", {})["num_timesteps"] = ckpt["args"]["num_steps"]
    model = ObjectPlacementPolicy(model_cfg).to(device)
    model.load_state_dict(state)
    model.eval()

    N = model.num_proposals
    multi = N > 1
    steps = model_cfg.get("diffusion", {}).get("sampling_steps", 4)

    if args.dataset == "coco":
        ds = CocoDetectionDataset(
            os.path.join(args.coco_root, "coco/annotations/instances_val2017.json"),
            os.path.join(args.coco_root, "coco/val2017"), num_proposals=N)
    else:
        ds = CE130DetectionDataset(args.all_phase2_dir, split=args.split, num_proposals=N)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    torch.manual_seed(args.seed)
    print(f"\n{args.variant} | {args.dataset}/{args.split} | N={N} | steps={steps} "
          f"| vẽ top-{args.topk} box\n")
    print(f"{'#':>4} {'ảnh':<24}{'GT':>5}{'pred':>7}{'khớp':>6}{'P':>9}{'R':>9}")
    print("-" * 66)

    done = 0
    for i, batch in enumerate(loader):
        if i < args.start:
            continue
        if done >= args.n_images:
            break
        rgb = batch["pixel_values"].to(device)
        cond = model.encode_condition(rgb, None, batch["text"])

        scores = None
        if multi:
            r = sample_boxes_multibox(model, cond, device, steps,
                                      box_renewal=args.box_renewal,
                                      use_ensemble=args.use_ensemble)
            pred, scores = r if isinstance(r, tuple) else (r, None)
        else:
            pred = sample_boxes(model, cond.expand(args.k, -1), device, None, args.k)

        # NMS giống hệt eval_detection.py để con số P/R in ra so được với báo cáo
        if args.nms is not None:
            pc = pred.cpu()
            if scores is not None:
                order = torch.argsort(scores.cpu(), descending=True)
                pc, pred, scores = pc[order], pred[order.to(pred.device)], scores[order.to(scores.device)]
            keep = nms_class_agnostic(_cxcywh_to_xyxy(pc), args.nms)
            pred = pred[keep.to(pred.device)]
            if scores is not None:
                scores = scores[keep.to(scores.device)]

        rec = ds.samples[i]
        sc = batch["scale"].item()
        if args.dataset == "coco":
            gt = torch.stack([ds._normalize_bbox(b[:4], sc) for b in rec["boxes"]])
        else:
            gt = torch.stack([ds._normalize_bbox(ds._xyxy_to_cxcywh(b), sc)
                              for b in rec["boxes_xyxy"]])

        pred_xyxy = _cxcywh_to_xyxy(pred.cpu())
        gt_xyxy = _cxcywh_to_xyxy(gt)
        n_match = greedy_match(pred_xyxy, gt_xyxy, args.iou_threshold)
        P = n_match / max(pred.shape[0], 1)
        R = n_match / max(gt.shape[0], 1)

        # top-k CHỈ để vẽ: score cao nhất, hoặc k box đầu nếu không có score head
        show = pred_xyxy[:args.topk] if scores is not None else pred_xyxy[:args.topk]

        img = batch["pixel_values"][0].permute(1, 2, 0).numpy()
        H, W = img.shape[:2]
        gt_px, pred_px = to_pixels(gt_xyxy, W, H), to_pixels(show, W, H)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
        draw(axes[0], img, gt_px, "lime", f"GT ({gt.shape[0]} box)", lw=1.6)
        draw(axes[1], img, pred_px, "red",
             f"dự đoán: top-{min(args.topk, show.shape[0])}/{pred.shape[0]} box")
        axes[2].imshow(img)
        for x1, y1, x2, y2 in gt_px.tolist():
            axes[2].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                                edgecolor="lime", linewidth=1.8))
        for x1, y1, x2, y2 in pred_px.tolist():
            axes[2].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                                edgecolor="red", linewidth=1.0, alpha=0.75))
        axes[2].set_title(f"chồng | khớp {n_match}/{gt.shape[0]} @IoU{args.iou_threshold}",
                          fontsize=9)
        axes[2].axis("off")

        cls = batch["text"][0] if batch["text"][0] else "(không text)"
        fig.suptitle(f"{args.variant}  |  {cls}  |  P={P*100:.2f}%  R={R*100:.2f}%  "
                     f"(trên TOÀN BỘ {pred.shape[0]} box, không chỉ top-{args.topk})",
                     fontsize=10)
        fig.tight_layout()
        iid = rec.get("img_id", i)
        fp = os.path.join(out_dir, f"{done:02d}_{iid}.png")
        fig.savefig(fp, dpi=110, bbox_inches="tight")
        plt.close(fig)

        print(f"{done:>4} {str(iid)[:23]:<24}{gt.shape[0]:>5}{pred.shape[0]:>7}"
              f"{n_match:>6}{P*100:>8.2f}%{R*100:>8.2f}%")
        done += 1

    print(f"\nĐã lưu {done} ảnh vào {out_dir}/")
    print("Lấy về máy: scp -r <user>@<host>:" + os.path.abspath(out_dir) + " .")


if __name__ == "__main__":
    main()

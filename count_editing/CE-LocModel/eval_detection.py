"""
Eval cho bài DETECTION có điều kiện theo class trên CE-130.

Tách hẳn khỏi `test_mul_box.py` (vốn gắn chặt với bài add: C-NLL cần {B_j} lấy
từ lookup all_phase2_V2, IoU@K so với đúng 1 box bị xoá). Ở đây target là cả
tập `all_bboxes`, nên metric là Precision/Recall trên tập box.

Cách sinh box:
  - num_proposals == 1 (variant 1/2): chạy K chain độc lập (mặc định K=30) ->
    K box ứng viên. Đây đúng cách người dùng mô tả "1 ảnh lúc infer chạy vài
    chục lần để cho ra nhiều box".
  - num_proposals > 1 (variant 3): 1 chain DDIM ra thẳng N box.

Metric: Precision/Recall @ IoU threshold, greedy 1-1 matching theo IoU giảm dần.
Dùng hình học ĐÚNG (`utils.matcher.box_iou_normalized`), không dùng
`calculate_iou` của repo gốc — xem ghi chú COORDINATE SPACE trong
utils/matcher.py (CE-Loc mã hoá w/h thành 2f-1 nên box nhỏ hơn nửa ảnh có
norm_w âm; coi đó là chiều rộng thật sẽ ra box lộn ngược).

CHƯA có AP: xếp hạng box theo độ tin cậy cần score head, thứ CE-Loc chưa có
(đã ghi trong CLAUDE.md là việc còn thiếu). P/R không cần score nên làm được
ngay; AP để giai đoạn sau.
"""
import argparse
import datetime
import json
import os
import time
import warnings

# Xem ghi chú trong train_w_args.py: cuDNN tự fallback, chỉ ồn log.
warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.ce130_detection_dataset import CE130DetectionDataset
from models.diffusion_module import ObjectPlacementPolicy
from utils.matcher import _cxcywh_to_xyxy, box_iou_normalized
from test_mul_box import sample_boxes, sample_boxes_multibox
from train import load_config


def _fmt(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="đường dẫn best_model.pth")
    p.add_argument("--variant", required=True, help="vd detect/a_cnn_1box")
    p.add_argument("--all_phase2_dir", default="../../data/all_phase2_V2")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--k", type=int, default=30,
                   help="số lần sampling cho variant 1 box/lần (bỏ qua với multi-box)")
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--inference_steps", type=int, default=None,
                   help="số bước reverse; mặc định = num_timesteps (1-box) "
                        "hoặc diffusion.sampling_steps (multi-box)")
    p.add_argument("--max_boxes", type=int, default=None,
                   help="bỏ ảnh có nhiều hơn ngần này box GT")
    p.add_argument("--limit", type=int, default=None,
                   help="chỉ eval N ảnh đầu (để chạy thử nhanh)")
    return p.parse_args()


def greedy_match(pred_xyxy, gt_xyxy, iou_threshold):
    """Ghép 1-1 tham lam theo IoU giảm dần. Trả về số cặp khớp.
    pred_xyxy: [P,4], gt_xyxy: [G,4] (đã là xyxy hình học đúng)."""
    if pred_xyxy.shape[0] == 0 or gt_xyxy.shape[0] == 0:
        return 0
    iou = box_iou_normalized(pred_xyxy, gt_xyxy)  # [P,G]
    cand = (iou >= iou_threshold).nonzero(as_tuple=False)
    if cand.numel() == 0:
        return 0
    scores = iou[cand[:, 0], cand[:, 1]]
    order = torch.argsort(scores, descending=True)
    used_p, used_g, n = set(), set(), 0
    for idx in order.tolist():
        pi, gi = int(cand[idx, 0]), int(cand[idx, 1])
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        n += 1
    return n


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = os.path.dirname(args.checkpoint)
    model_cfg = yaml.safe_load(open(os.path.join(ckpt_dir, "model_config_final.yaml")))

    if not os.path.exists(args.checkpoint):
        print(f"Không thấy checkpoint: {args.checkpoint}")
        return
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "args" in ckpt and ckpt["args"].get("num_steps"):
        model_cfg.setdefault("diffusion", {})["num_timesteps"] = ckpt["args"]["num_steps"]
    trained_task = ckpt.get("args", {}).get("task")
    if trained_task and trained_task != "detect":
        print(f"CẢNH BÁO: checkpoint này train với --task {trained_task!r}, "
              f"không phải 'detect'. Số đo sẽ không có ý nghĩa.")

    model = ObjectPlacementPolicy(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    multi_box = model.num_proposals > 1
    sampling_steps = model_cfg.get("diffusion", {}).get("sampling_steps", 4)
    if multi_box and args.inference_steps is not None:
        sampling_steps = args.inference_steps

    ds = CE130DetectionDataset(args.all_phase2_dir, split=args.split,
                               num_proposals=model.num_proposals,
                               max_boxes=args.max_boxes)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    n_pred_total = n_gt_total = n_matched_total = 0
    per_image = []
    t0 = time.monotonic()
    n_done = 0

    for i, batch in tqdm(enumerate(loader), total=len(loader), desc=f"eval[{args.variant}]"):
        if args.limit is not None and i >= args.limit:
            break
        rgb = batch["pixel_values"].to(device)
        text = batch["text"]

        vis = model.vision_encoder(rgb, None)   # in_channels=3 -> không density
        txt = model.text_encoder(text)
        cond = torch.cat([vis, txt], dim=-1)

        if multi_box:
            pred = sample_boxes_multibox(model, cond, device, sampling_steps)  # [N,4]
            gt = batch["boxes"][0][batch["box_mask"][0]]
        else:
            # K chain độc lập -> K box ứng viên cho cùng một ảnh.
            pred = sample_boxes(model, cond.expand(args.k, -1), device,
                                args.inference_steps, args.k)  # [K,4]
            # với 1-box, dataset không trả "boxes"; dựng lại tập GT đầy đủ
            rec = ds.samples[i]
            gt = torch.stack([
                ds._normalize_bbox(ds._xyxy_to_cxcywh(b), batch["scale"].item())
                for b in rec["boxes_xyxy"]
            ])

        n_match = greedy_match(_cxcywh_to_xyxy(pred.cpu()),
                               _cxcywh_to_xyxy(gt.cpu()), args.iou_threshold)
        n_pred_total += pred.shape[0]
        n_gt_total += gt.shape[0]
        n_matched_total += n_match
        per_image.append({
            "img_id": batch["img_id"][0],
            "n_pred": int(pred.shape[0]),
            "n_gt": int(gt.shape[0]),
            "n_matched": int(n_match),
        })
        n_done += 1

    total_time = time.monotonic() - t0
    precision = n_matched_total / n_pred_total if n_pred_total else 0.0
    recall = n_matched_total / n_gt_total if n_gt_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    results = {
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "task": "detect",
        "split": args.split,
        "n_images": n_done,
        "iou_threshold": args.iou_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred_total": n_pred_total,
        "n_gt_total": n_gt_total,
        "n_matched_total": n_matched_total,
        "num_proposals": model.num_proposals,
        "k_samples": None if multi_box else args.k,
        "sampling_steps": sampling_steps if multi_box else args.inference_steps,
        "eval_time_seconds": total_time,
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Eval xong variant={args.variant}. Tổng thời gian: {_fmt(total_time)}")

    out = os.path.join(ckpt_dir, "eval_detection.json")
    with open(out, "w") as f:
        json.dump({"summary": results, "per_image": per_image}, f, indent=2, ensure_ascii=False)
    print(f"Đã ghi kết quả vào {out}")


if __name__ == "__main__":
    main()

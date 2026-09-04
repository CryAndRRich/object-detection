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

from data.ce130_detection_dataset import CE130DetectionDataset
from data.coco_detection_dataset import CocoDetectionDataset
from models.diffusion_module import ObjectPlacementPolicy
from utils.matcher import _cxcywh_to_xyxy, box_iou_normalized
from test_mul_box import sample_boxes, sample_boxes_multibox
from train import load_config


# COCO đánh giá AP trung bình trên IoU 0.50:0.05:0.95
AP_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def _fmt(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="đường dẫn best_model.pth")
    p.add_argument("--variant", required=True, help="vd detect/a_cnn_1box")
    p.add_argument("--all_phase2_dir", default="../../data/all_phase2_V2")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--dataset", default="ce130", choices=["ce130", "coco"],
                   help="ce130 = CE-130 detection (variant c). coco = COCO-minitrain (variant d).")
    p.add_argument("--coco_root", default="../../data")
    p.add_argument("--k", type=int, default=30,
                   help="số lần sampling cho variant 1 box/lần (bỏ qua với multi-box)")
    p.add_argument("--iou_threshold", type=float, default=0.5)
    p.add_argument("--box_renewal", action="store_true",
                   help="bật box_renewal của DiffusionDet (cần score head)")
    p.add_argument("--renewal_threshold", type=float, default=0.5)
    p.add_argument("--use_ensemble", action="store_true",
                   help="bật use_ensemble của DiffusionDet: gom box mọi bước DDIM (cần score head)")
    p.add_argument("--nms", type=float, default=None,
                   help="ngưỡng IoU cho NMS class-agnostic trên box sinh ra (vd 0.5). "
                        "Mặc định TẮT để giữ nguyên cách chấm của lần chạy đầu. "
                        "Bật lên sẽ bỏ box trùng lặp -> precision tăng, recall gần "
                        "như không đổi. DiffusionDet luôn bật NMS ở bước inference.")
    p.add_argument("--inference_steps", type=int, default=None,
                   help="số bước reverse; mặc định = num_timesteps (1-box) "
                        "hoặc diffusion.sampling_steps (multi-box)")
    p.add_argument("--max_boxes", type=int, default=None,
                   help="bỏ ảnh có nhiều hơn ngần này box GT")
    p.add_argument("--log_every", type=int, default=50,
                   help="in tiến trình mỗi N ảnh (thay cho thanh tqdm cũ)")
    p.add_argument("--limit", type=int, default=None,
                   help="chỉ eval N ảnh đầu (để chạy thử nhanh)")
    return p.parse_args()


def nms_class_agnostic(boxes_xyxy, iou_threshold):
    """NMS class-agnostic: bỏ box trùng lặp, giữ lại tập box rời nhau.

    DiffusionDet dùng `batched_nms(boxes, scores, labels, 0.5)` — nó CẦN score để
    biết trong nhóm box chồng nhau thì giữ cái nào. CE-Loc chưa có score head, nên
    ở đây dùng thứ tự đầu ra của model làm thứ tự ưu tiên (box nào ra trước thì
    được giữ). Kém hơn NMS có score, nhưng vẫn loại được phần lớn box trùng —
    và KHÔNG cần score head, khác với điều tôi từng nói.

    Trả về chỉ số các box được giữ, theo thứ tự gốc.
    """
    n = boxes_xyxy.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.long)
    keep = []
    suppressed = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        if suppressed[i]:
            continue
        keep.append(i)
        if i + 1 < n:
            rest = torch.arange(i + 1, n)
            alive = rest[~suppressed[rest]]
            if alive.numel():
                ious = box_iou_normalized(boxes_xyxy[i:i + 1], boxes_xyxy[alive])[0]
                suppressed[alive[ious > iou_threshold]] = True
    return torch.as_tensor(keep, dtype=torch.long)


def average_precision(all_dets, n_gt_total, iou_threshold):
    """AP kiểu COCO cho MỘT ngưỡng IoU, class-agnostic.

    all_dets: list các (score, is_true_positive) đã gom qua TOÀN BỘ ảnh.
    Xếp hạng theo score giảm dần, tích luỹ TP/FP, rồi tính diện tích dưới đường
    precision-recall bằng nội suy 101 điểm (đúng cách pycocotools làm).

    CẦN score head — không có score thì không xếp hạng được, và AP vô nghĩa.
    """
    if not all_dets or n_gt_total == 0:
        return 0.0
    all_dets = sorted(all_dets, key=lambda x: -x[0])
    tp = np.array([d[1] for d in all_dets], dtype=np.float64)
    fp = 1.0 - tp
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recalls = tp_cum / n_gt_total
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # precision đơn điệu giảm (envelope), như pycocotools
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    rec_pts = np.linspace(0, 1, 101)
    idx = np.searchsorted(recalls, rec_pts, side="left")
    q = np.where(idx < len(precisions), precisions[np.minimum(idx, len(precisions) - 1)], 0.0)
    return float(q.mean())


def match_with_scores(pred_xyxy, scores, gt_xyxy, iou_threshold):
    """Ghép greedy theo SCORE giảm dần (không phải theo IoU) — đúng cách COCO
    đánh giá: detection điểm cao được ưu tiên chọn GT trước.

    Trả về list (score, is_tp) cho từng box dự đoán."""
    P, G = pred_xyxy.shape[0], gt_xyxy.shape[0]
    if P == 0:
        return []
    if G == 0:
        return [(float(s), 0) for s in scores]
    iou = box_iou_normalized(pred_xyxy, gt_xyxy)
    order = torch.argsort(scores, descending=True)
    used_gt = set()
    out = []
    for pi in order.tolist():
        best_iou, best_gi = 0.0, -1
        for gi in range(G):
            if gi in used_gt:
                continue
            v = float(iou[pi, gi])
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_gi >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_gi)
            out.append((float(scores[pi]), 1))
        else:
            out.append((float(scores[pi]), 0))
    return out


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

    if args.dataset == "coco":
        # (d) eval trên COCO val2017 — đúng bộ DiffusionDet dùng để báo AP.
        ds = CocoDetectionDataset(
            os.path.join(args.coco_root, "coco/annotations/instances_val2017.json"),
            os.path.join(args.coco_root, "coco/val2017"),
            num_proposals=model.num_proposals, max_boxes=args.max_boxes)
    else:
        ds = CE130DetectionDataset(args.all_phase2_dir, split=args.split,
                                   num_proposals=model.num_proposals,
                                   max_boxes=args.max_boxes)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    n_pred_total = n_gt_total = n_matched_total = 0
    n_before_nms_total = n_after_nms = 0
    ap_dets = {t: [] for t in AP_THRESHOLDS}
    per_image = []
    t0 = time.monotonic()
    n_done = 0

    n_total = len(loader) if args.limit is None else min(args.limit, len(loader))
    # In mỗi --log_every ảnh thay vì dùng tqdm. Lý do: tqdm vẽ lại thanh tiến
    # trình bằng ký tự \r, thứ chỉ có nghĩa trên terminal -- ghi vào file log
    # (nohup) thì \r nằm nguyên trong file, làm mọi lần vẽ chồng lên cùng MỘT
    # dòng vật lý. Đọc log kiểu đó thấy số nhảy lộn xộn (166/779 rồi 97/779) và
    # rất dễ tưởng là nhiều tiến trình chạy song song. flush=True vì nohup buffer
    # stdout theo block.
    for i, batch in enumerate(loader):
        if args.limit is not None and i >= args.limit:
            break
        rgb = batch["pixel_values"].to(device)
        text = batch["text"]

        # Dùng encode_condition thay vì gọi tay: variant (d) TẮT nhánh text
        # (text_encoder is None) nên `model.text_encoder(text)` sẽ nổ.
        cond = model.encode_condition(rgb, None, text)  # in_channels=3 -> không density

        scores = None
        if multi_box:
            r = sample_boxes_multibox(model, cond, device, sampling_steps,
                                      box_renewal=args.box_renewal,
                                      renewal_threshold=args.renewal_threshold,
                                      use_ensemble=args.use_ensemble)
            pred, scores = r if isinstance(r, tuple) else (r, None)
        else:
            # K chain độc lập -> K box ứng viên cho cùng một ảnh.
            pred = sample_boxes(model, cond.expand(args.k, -1), device,
                                args.inference_steps, args.k)  # [K,4]

        # NMS class-agnostic (nếu bật): bỏ box trùng lặp trước khi chấm điểm.
        # Không có nó thì 300 box của variant (c) tính hết vào mẫu số precision,
        # kể cả khi nhiều box mô tả cùng một vật.
        n_before_nms = pred.shape[0]
        if args.nms is not None:
            pc = pred.cpu()
            if scores is not None:
                # Có score thì xếp theo score giảm dần TRƯỚC khi NMS — đúng
                # `batched_nms(boxes, scores, ...)` của DiffusionDet: box điểm
                # cao được giữ, box trùng điểm thấp bị loại.
                order = torch.argsort(scores.cpu(), descending=True)
                pc, pred, scores = pc[order], pred[order.to(pred.device)], scores[order.to(scores.device)]
            keep = nms_class_agnostic(_cxcywh_to_xyxy(pc), args.nms)
            pred = pred[keep.to(pred.device)]
            if scores is not None:
                scores = scores[keep.to(scores.device)]
        n_after_nms += pred.shape[0]
        n_before_nms_total += n_before_nms

        # GT luôn dựng từ danh sách box GỐC, chung cho cả 3 variant.
        # KHÔNG dùng batch["boxes"][box_mask] cho nhánh multi-box: tensor đó đã bị
        # crop về num_proposals, nên ảnh có >N box sẽ mất bớt GT -> recall bị thổi
        # lên và 3 variant chấm trên 2 tập GT khác nhau (đo được: 37.218 so với
        # 37.812, lệch 594 box = 1,57%).
        if args.dataset == "coco":
            rec = ds.samples[i]
            gt = torch.stack([
                ds._normalize_bbox(b[:4], batch["scale"].item()) for b in rec["boxes"]
            ])
        else:
            rec = ds.samples[i]
            gt = torch.stack([
                ds._normalize_bbox(ds._xyxy_to_cxcywh(b), batch["scale"].item())
                for b in rec["boxes_xyxy"]
            ])

        pred_xyxy, gt_xyxy = _cxcywh_to_xyxy(pred.cpu()), _cxcywh_to_xyxy(gt.cpu())
        n_match = greedy_match(pred_xyxy, gt_xyxy, args.iou_threshold)
        if scores is not None:
            for thr in AP_THRESHOLDS:
                ap_dets[thr].extend(match_with_scores(pred_xyxy, scores.cpu(), gt_xyxy, thr))
        n_pred_total += pred.shape[0]
        n_gt_total += gt.shape[0]
        n_matched_total += n_match
        per_image.append({
            # img_id: CE-130 là str, COCO là int -> DataLoader bọc int thành
            # tensor, mà tensor thì json.dump không serialize được.
            "img_id": (batch["img_id"][0].item()
                       if torch.is_tensor(batch["img_id"][0]) else batch["img_id"][0]),
            "n_pred": int(pred.shape[0]),
            "n_gt": int(gt.shape[0]),
            "n_matched": int(n_match),
        })
        n_done += 1

        if n_done % args.log_every == 0 or n_done == n_total:
            el = time.monotonic() - t0
            rate = n_done / el
            eta = (n_total - n_done) / rate if rate > 0 else 0
            print(f"[eval {args.variant}] {n_done}/{n_total} ảnh"
                  f" | P={n_matched_total / max(n_pred_total, 1) * 100:.4f}%"
                  f" R={n_matched_total / max(n_gt_total, 1) * 100:.2f}%"
                  f" | {rate:.2f} ảnh/s | elapsed {_fmt(el)} | ETA {_fmt(eta)}",
                  flush=True)

    total_time = time.monotonic() - t0
    precision = n_matched_total / n_pred_total if n_pred_total else 0.0
    recall = n_matched_total / n_gt_total if n_gt_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    results = {
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "task": "detect",
        "split": args.split,
        "dataset": args.dataset,
        "n_images": n_done,
        "iou_threshold": args.iou_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred_total": n_pred_total,
        "n_gt_total": n_gt_total,
        "n_matched_total": n_matched_total,
        "nms_iou": args.nms,
        "box_renewal": args.box_renewal,
        "use_ensemble": args.use_ensemble,
        "n_box_before_nms": n_before_nms_total,
        "n_box_after_nms": n_after_nms,
        "num_proposals": model.num_proposals,
        "k_samples": None if multi_box else args.k,
        "sampling_steps": sampling_steps if multi_box else args.inference_steps,
        "eval_time_seconds": total_time,
    }
    if ap_dets[0.5]:
        aps = {t: average_precision(ap_dets[t], n_gt_total, t) for t in AP_THRESHOLDS}
        results["AP"] = float(np.mean(list(aps.values())))     # AP@[.50:.95], kiểu COCO
        results["AP50"] = aps[0.5]
        results["AP75"] = aps[0.75]
        results["AP_per_threshold"] = {f"{t:.2f}": v for t, v in aps.items()}
    else:
        results["AP"] = None
        results["AP_note"] = ("model không có score head (noise_net.num_classes=0) nên "
                              "không xếp hạng được box -> không tính được AP")

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Eval xong variant={args.variant}. Tổng thời gian: {_fmt(total_time)}")

    out = os.path.join(ckpt_dir, "eval_detection.json")
    with open(out, "w") as f:
        json.dump({"summary": results, "per_image": per_image}, f, indent=2, ensure_ascii=False)
    print(f"Đã ghi kết quả vào {out}")


if __name__ == "__main__":
    main()

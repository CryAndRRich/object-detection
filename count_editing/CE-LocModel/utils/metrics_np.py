"""Chấm điểm detection — THUẦN NUMPY, một chỗ duy nhất cho `eval.py` và các tool.

VÌ SAO TÁCH RA: trước 2026-09-25 các hàm này nằm rải ở `eval.py` và
`tools/measure_box_quality.py`, và `eval.py` trộn torch với numpy trong cùng một luồng
(`torch.argsort` trên mảng numpy, `cls[keep]` khi `cls` là None, `evaluate` nhận dict
trong khi nó duyệt tuple). Không test nào chạy trọn luồng đó nên lần eval đầu tiên của
EXPERIMENT A vòng 2 vỡ ngay dòng đầu. Mọi thứ sau model giờ là numpy, và có test chạy
trọn luồng (`tests/test_experiment_a.py`).

GIAO THỨC GIỐNG VÒNG 1 (để so được với E1 AP50 6,36):
  top-k theo score -> NMS không phân lớp -> ghép tham lam theo score giảm dần -> AP.
  `oracle_recall` và `score_AUC` cùng định nghĩa với cột tương ứng của bảng vòng 1:
  tính trên TẤT CẢ N box, không qua top-k/NMS.

Mọi box ở đây là **xyxy**, cùng một thang (chuẩn hoá [0,1] hay pixel đều được — IoU
bất biến theo thang).
"""

import numpy as np

from utils.box_ops_np import box_iou, cxcywh_to_xyxy

__all__ = ["nms_class_agnostic", "ap_from_pr", "evaluate", "roc_auc", "quality",
           "summarise", "COCO_THR"]

COCO_THR = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def nms_class_agnostic(boxes_xyxy, scores, iou_thr=0.5):
    """-> chỉ số giữ lại (numpy int), theo score giảm dần."""
    boxes_xyxy, scores = np.asarray(boxes_xyxy), np.asarray(scores)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        iou = box_iou(boxes_xyxy[i:i + 1], boxes_xyxy[order[1:]])[0][0]
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)


def ap_from_pr(rec, prec):
    """AP kiểu COCO: làm precision giảm đơn điệu rồi tích phân theo recall."""
    m_rec = np.concatenate([[0.0], rec, [1.0]])
    m_pre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(m_pre) - 2, -1, -1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    idx = np.where(m_rec[1:] != m_rec[:-1])[0]
    return float(np.sum((m_rec[idx + 1] - m_rec[idx]) * m_pre[idx + 1]))


def evaluate(predictions, iou_thr=0.5):
    """predictions: list[(boxes_xyxy [K,4], scores [K], gt_xyxy [M,4])] -> dict.

    Mỗi GT chỉ được ghép MỘT lần; box được xét theo score giảm dần trên toàn tập.
    """
    records, total_gt = [], 0
    for boxes, scores, gt in predictions:
        boxes, scores, gt = np.asarray(boxes), np.asarray(scores), np.asarray(gt)
        total_gt += len(gt)
        if len(boxes) == 0:
            continue
        order = np.argsort(-scores, kind="stable")
        used = np.zeros(len(gt), dtype=bool)
        for i in order:
            if len(gt) == 0:
                records.append((scores[i], 0))
                continue
            iou = box_iou(boxes[i:i + 1], gt)[0][0]
            j = int(np.argmax(iou))
            if iou[j] >= iou_thr and not used[j]:
                used[j] = True
                records.append((scores[i], 1))
            else:
                records.append((scores[i], 0))

    if not records or total_gt == 0:
        return {"AP": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pred": 0,
                "n_gt": total_gt}

    records.sort(key=lambda x: -x[0])
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([1 - r[1] for r in records])
    rec = tp / total_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    p, r = float(prec[-1]), float(rec[-1])
    return {"AP": ap_from_pr(rec, prec), "precision": p, "recall": r,
            "f1": 2 * p * r / max(p + r, 1e-9), "n_pred": len(records), "n_gt": total_gt}


def roc_auc(labels, scores):
    """AUC bằng đẳng thức tổng hạng — không cần sklearn, đúng cả khi có điểm bằng nhau.

    Trả nan khi thiếu một lớp (AUC không xác định); nơi gọi lấy trung bình theo ảnh và
    bỏ qua các ảnh đó, thay vì âm thầm chấm 0,5.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def quality(pred_cxcywh, scores, gt_cxcywh, size=512, iou_thr=0.5, pred_cls=None,
            gt_cls=None):
    """Chỉ số cho MỘT ảnh -> (best_iou mỗi GT, số GT được phủ, n_gt, auc).

    `best_iou` lấy trên TẤT CẢ dự đoán, không dùng score và không ghép tham lam: câu hỏi
    là "vật này có được box nào phủ không". Đây là định nghĩa của `oracle_recall` trong
    bảng vòng 1.
    """
    if len(gt_cxcywh) == 0:
        return np.zeros(0), 0, 0, float("nan")
    g = cxcywh_to_xyxy(np.asarray(gt_cxcywh)) * size
    if len(pred_cxcywh) == 0:
        return np.zeros(len(g)), 0, len(g), float("nan")
    p = cxcywh_to_xyxy(np.asarray(pred_cxcywh)) * size

    m = box_iou(p, g)[0]                                   # [P, G]
    if pred_cls is not None and gt_cls is not None:
        m = np.where(np.asarray(pred_cls)[:, None] == np.asarray(gt_cls)[None, :],
                     m, 0.0)
    best = m.max(axis=0)
    hit = int((best >= iou_thr).sum())
    auc = roc_auc((m.max(axis=1) >= iou_thr).astype(int), np.asarray(scores))
    return best, hit, len(g), auc


def summarise(best_all, hits, n_gt, aucs, recall_scored=None):
    b = np.concatenate(best_all) if best_all else np.zeros(0)
    a = np.array([x for x in aucs if not np.isnan(x)])
    out = {
        "oracle_recall": hits / max(n_gt, 1),
        "mean_bestIoU": float(b.mean()) if len(b) else 0.0,
        "median_bestIoU": float(np.median(b)) if len(b) else 0.0,
        "score_AUC": float(a.mean()) if len(a) else float("nan"),
        "score_AUC_n_images": int(len(a)),
        "n_gt": int(n_gt),
    }
    if recall_scored is not None:
        out["recall_scored"] = recall_scored
        out["score_head_cost"] = out["oracle_recall"] - recall_scored
    return out

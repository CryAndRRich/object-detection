"""Box quality tách riêng khỏi score-head quality — THUẦN NUMPY, không cần detectron2.

Dùng bởi ``tools/measure_box_quality_ce130.py`` (EXPERIMENT D). Cùng công thức 3 chỉ số
với ``count_editing/CE-LocModel/tools/measure_box_quality.py`` (bản gốc của ý tưởng này,
viết cho CE-LocModel A/B/C) — viết lại độc lập, KHÔNG import trực tiếp, vì hai stack
không dùng chung dependency (detectron2/COCO-json ở đây so với numpy-dict/CLIP-cache
bên kia).

Vì sao KHÔNG dùng AP làm kết luận chính (đặc tả:
``docs/thiet-ke-experiment-d-diffusiondet-ce130.md`` §4): A/B đã đo được
``score_AUC = 0,512`` (~ngẫu nhiên), tức AP của chúng phản ánh CẢ ranking lẫn box —
không tách được đâu là lỗi hình học đâu là lỗi xếp hạng. So "AP của D" với "AP của
A/B/C" là so hai đại lượng trộn theo tỉ lệ khác nhau.

    oracle_recall   tỉ lệ GT có ít nhất 1 box IoU>=ngưỡng (bỏ qua score) — TRẦN mà một
                    score head hoàn hảo có thể đạt.
    mean_bestIoU    IoU tốt nhất mỗi GT, trung bình — chất lượng hình học thuần.
    score_AUC       score có xếp hạng đúng box khớp lên trên box không khớp không.
                    0,5 = ngẫu nhiên (tham chiếu TUYỆT ĐỐI).
"""

import numpy as np

__all__ = ["roc_auc", "box_iou_xyxy", "quality_one_image", "summarise"]


def roc_auc(labels, scores):
    """AUC qua rank-sum identity — exact cả khi có tie. NaN khi thiếu 1 class."""
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


def box_iou_xyxy(a, b):
    """IoU ma trận [Na, Nb], input xyxy tuyệt đối (pixel)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def quality_one_image(pred_xyxy, scores, gt_xyxy, iou_thr=0.5):
    """Metric cho MỘT ảnh. Trả (best_iou mỗi GT, hit, n_gt, auc, scored_hit).

    ``best_iou``/``hit`` bỏ qua score hoàn toàn (oracle). ``scored_hit`` đếm theo đúng
    cách một greedy matcher dùng score thật sẽ đếm (mỗi GT chỉ được gán 1 lần, ưu tiên
    prediction điểm cao hơn trước) — hiệu ``hit - scored_hit`` là cái giá của score head.
    """
    if len(gt_xyxy) == 0:
        return np.zeros(0), 0, 0, float("nan"), 0
    if len(pred_xyxy) == 0:
        return np.zeros(len(gt_xyxy)), 0, len(gt_xyxy), float("nan"), 0

    m = box_iou_xyxy(pred_xyxy, gt_xyxy)          # [P, G]
    best = m.max(axis=0)
    hit = int((best >= iou_thr).sum())
    auc = roc_auc((m.max(axis=1) >= iou_thr).astype(int), scores)

    used = np.zeros(len(gt_xyxy), dtype=bool)
    scored_hit = 0
    for pi in np.argsort(-np.asarray(scores)):
        row = np.where(used, -1.0, m[pi])
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            used[j] = True
            scored_hit += 1
    return best, hit, len(gt_xyxy), auc, scored_hit


def summarise(best_all, hits, n_gt, aucs, scored_hits):
    """Gộp kết quả nhiều ảnh. An toàn khi ``n_gt == 0`` (không chia cho 0)."""
    b = np.concatenate(best_all) if best_all else np.zeros(0)
    a = np.array([x for x in aucs if not np.isnan(x)])
    oracle_recall = hits / max(n_gt, 1)
    recall_scored = scored_hits / max(n_gt, 1)
    return {
        "oracle_recall": oracle_recall,
        "recall_scored": recall_scored,
        "score_head_cost": oracle_recall - recall_scored,
        "mean_bestIoU": float(b.mean()) if len(b) else 0.0,
        "median_bestIoU": float(np.median(b)) if len(b) else 0.0,
        "score_AUC": float(a.mean()) if len(a) else float("nan"),
        "score_AUC_n_images": int(len(a)),
        "n_gt": int(n_gt),
    }

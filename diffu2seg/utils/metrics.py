"""Box-quality metrics — pure numpy.

COPIED (not imported) from count_editing/CE-LocModel/tools/measure_box_quality.py,
keeping the definitions bit-for-bit so numbers stay comparable with A/B/C1/E1/D.1.

DELIBERATELY ABSENT: score_AUC.
Training-free Diffuse2Seg produces no score for a box -- there is no score head,
no objectness, nothing learned. Emitting some proxy (component area, attention
mass) under the name `score_AUC` would invite a comparison against the 0.4965 -
0.4988 of A/B/C1, which those numbers do not support. If a proxy is ever needed
it must be added under a different key whose name says PROXY out loud.
`test_metrics.py::test_no_score_auc_key` locks this.

TWO TRAPS, both recorded in docs/old/ROUND_1_ARCHIVE.md (phần 03) and both reproduced here on purpose:

1. RAW ACCUMULATION, divided ONCE.
   Accumulate (hits, n_gt) across images and divide at the end. Averaging the
   per-image ratios answers a different question and gives wildly different
   numbers: an image with 1/1 hit and one with 1/100 give 2/101 = 0.0198 raw,
   but 0.505 as a mean of ratios.

2. AN IMAGE WITH NO PREDICTIONS CONTRIBUTES ITS n_gt, NOT ZERO.
   Returning (0, 0) silently drops those objects from the denominator and
   inflates oracle_recall. Diffuse2Seg CAN legitimately return zero boxes for an
   image (every prompt landed in padding, or every component was filtered), so
   this path is live here, not theoretical.
"""

import numpy as np

from .box_ops_np import box_iou, cxcywh_to_xyxy

__all__ = ["quality_one_image", "summarise", "fmt_time"]


def quality_one_image(pred_cxcywh, gt_cxcywh, size=512, iou_thr=0.5):
    """Metrics for ONE image. Returns (best_iou per GT, n_hit, n_gt).

    `best_iou` is taken over ALL predictions, with no score involved and no
    greedy assignment: the question is "was this object covered at all". A
    greedy matcher answers a different one (it can hand a GT's best box to
    another GT), which is why eval-style AP lives elsewhere.

    No `scores` argument -- see the module docstring.
    """
    g_raw = np.asarray(gt_cxcywh, dtype=np.float64).reshape(-1, 4)
    if len(g_raw) == 0:
        return np.zeros(0), 0, 0

    p_raw = np.asarray(pred_cxcywh, dtype=np.float64).reshape(-1, 4)
    if len(p_raw) == 0:
        # TRAP 2: n_gt, not 0. These objects were missed; they must stay in the
        # denominator.
        return np.zeros(len(g_raw)), 0, len(g_raw)

    g = cxcywh_to_xyxy(g_raw) * size
    p = cxcywh_to_xyxy(p_raw) * size

    m = box_iou(p, g)[0]                       # [P, G]
    best = m.max(axis=0)                       # per GT
    hit = int((best >= iou_thr).sum())
    return best, hit, len(g)


def summarise(best_all, hits, n_gt, n_pred_total=0, n_images=0, extra=None):
    """Aggregate. `hits` and `n_gt` MUST already be raw sums -- see TRAP 1."""
    b = np.concatenate(best_all) if best_all else np.zeros(0)
    out = {
        "oracle_recall": hits / max(n_gt, 1),
        "mean_bestIoU": float(b.mean()) if len(b) else 0.0,
        "median_bestIoU": float(np.median(b)) if len(b) else 0.0,
        "n_gt": int(n_gt),
        "n_hit": int(hits),
        "n_pred_total": int(n_pred_total),
        "n_images": int(n_images),
    }
    if extra:
        out.update(extra)
    return out


def fmt_time(seconds):
    """3661 -> '1h01m01s'. Dùng cho cả elapsed lẫn ETA.

    COPIED (không import chéo) từ count_editing/CE-LocModel/train.py:96, giữ
    nguyên định dạng để log của hai sub-project đọc giống nhau.
    """
    seconds = int(max(seconds, 0))
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")

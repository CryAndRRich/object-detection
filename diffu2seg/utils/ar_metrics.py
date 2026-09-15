"""AR_1000 — the metric Diffuse2Seg actually reports.

WHY A SEPARATE FILE FROM metrics.py: `oracle_recall` there is a project metric,
fixed at IoU 0.5, with no proposal cap, and comparable to A/B/C1/E1/D.1. AR_1000
is the PAPER's metric. Mixing them in one module would invite reading a number
computed one way against a table built the other way.

                        THE DEFINITION, FROM THE PAPER

  "we mainly consider mean average recall (AR_1000/mAR) [...]. It averages recall
   over IoU matching thresholds from 0.5 to 0.95. We always consider up to 1000
   predicted masks per image for metric calculation. In addition, we report
   average recall filtered by object size, denoted as AR_S, AR_M, and AR_L,
   following the COCO definition."

So, precisely:

  AR = mean over t in {0.50, 0.55, ..., 0.95} of recall(t)
  recall(t) = (number of GT matched at IoU >= t) / (total GT)
  at most 1000 predictions per image enter the computation

`oracle_recall` is AR's single term at t=0.50 -- it is ALWAYS the largest, so
AR is always lower. Reading one against the other overstates by roughly 2x on
typical data. That is the whole reason this file exists.

                       ONE-TO-ONE MATCHING, AND WHY

Recall here is COCO-style: each prediction can claim AT MOST ONE ground truth,
greedily, best IoU first. This differs from `quality_one_image`'s "was this
object covered at all", which lets one excellent prediction satisfy several
overlapping GT. On PACO that difference is not academic: a chair and its eight
parts overlap heavily, and a single chair-shaped mask would otherwise be
credited with finding all nine.

                        SIZE BANDS (COCO DEFINITION)

  S: area <  32^2 = 1024 px^2
  M: 32^2 <= area < 96^2 = 9216 px^2
  L: area >= 96^2

Area is measured in ORIGINAL IMAGE pixels, never canvas pixels -- the canvas
rescales every image differently, so a canvas-based band would put the same
object in different bands depending on the image it came from.
"""

import numpy as np

__all__ = ["IOU_THRESHOLDS", "ar_one_image", "summarise_ar", "SIZE_BANDS"]

# 0.50, 0.55, ..., 0.95 -- ten thresholds, the COCO set.
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)

SIZE_BANDS = {"S": (0.0, 32.0 ** 2), "M": (32.0 ** 2, 96.0 ** 2),
              "L": (96.0 ** 2, float("inf"))}

MAX_PROPOSALS = 1000


def _greedy_match(iou):
    """Greedy one-to-one assignment, best IoU first.

    iou: (P, G). Returns `best` (G,), the IoU each GT was matched at, or 0.0
    when it was never claimed.

    Greedy on the globally-sorted pair list, not per-GT argmax: per-GT argmax
    can hand the same prediction to two GT, which one-to-one forbids.
    """
    P, G = iou.shape
    best = np.zeros(G, dtype=np.float64)
    if P == 0 or G == 0:
        return best

    order = np.argsort(iou, axis=None)[::-1]
    used_p = np.zeros(P, dtype=bool)
    used_g = np.zeros(G, dtype=bool)
    n_done = 0
    limit = min(P, G)
    for flat in order:
        v = iou.flat[flat]
        if v <= 0.0:
            break
        p, g = divmod(int(flat), G)
        if used_p[p] or used_g[g]:
            continue
        used_p[p] = used_g[g] = True
        best[g] = v
        n_done += 1
        if n_done == limit:
            break
    return best


def ar_one_image(iou, gt_areas, max_proposals=MAX_PROPOSALS, n_pred=None):
    """Per-image contribution to AR. Returns (hits_per_threshold, n_gt, per_band).

    Args:
        iou:       (P, G) IoU matrix, predictions x ground truth. Already capped
                   to `max_proposals` rows by the caller, or capped here.
        gt_areas:  (G,) GT area in ORIGINAL image pixels, for the size bands.
        n_pred:    predictions BEFORE the cap, for reporting.

    Returns:
        hits:      (10,) int, GT matched at each IoU threshold.
        n_gt:      int.
        per_band:  {"S": (hits(10,), n_gt), "M": ..., "L": ...}

    NOTE the cap is applied by TRUNCATION, taking the first `max_proposals`
    rows. Diffuse2Seg produces no score, so there is no ranking to take the
    "top" 1000 by -- and inventing one (area, attention mass) would silently
    make the metric depend on that invented proxy. The paper's 1000 is a
    generous ceiling; if a run ever exceeds it, the log says so, because at that
    point the truncation rule starts to matter and must be stated.
    """
    iou = np.asarray(iou, dtype=np.float64)
    if iou.ndim != 2:
        iou = iou.reshape(-1, 0) if iou.size == 0 else iou
    G = iou.shape[1]
    gt_areas = np.asarray(gt_areas, dtype=np.float64).reshape(-1)
    assert len(gt_areas) == G, f"{len(gt_areas)} areas for {G} GT"

    if iou.shape[0] > max_proposals:
        iou = iou[:max_proposals]

    best = _greedy_match(iou)
    hits = np.array([(best >= t).sum() for t in IOU_THRESHOLDS], dtype=np.int64)

    per_band = {}
    for name, (lo, hi) in SIZE_BANDS.items():
        sel = (gt_areas >= lo) & (gt_areas < hi)
        b = best[sel]
        per_band[name] = (
            np.array([(b >= t).sum() for t in IOU_THRESHOLDS], dtype=np.int64),
            int(sel.sum()),
        )
    return hits, G, per_band


def summarise_ar(hits, n_gt, per_band_hits=None, per_band_n=None, extra=None):
    """Aggregate raw sums into AR_1000 and the per-size AR.

    `hits` is a (10,) sum over images, `n_gt` a scalar sum -- RAW, divided once,
    for the same reason metrics.py states: averaging per-image ratios answers a
    different question.
    """
    hits = np.asarray(hits, dtype=np.float64)
    recalls = hits / max(n_gt, 1)
    out = {
        "AR_1000": float(recalls.mean()),
        "recall_at_50": float(recalls[0]),
        "recall_at_75": float(recalls[5]),
        "recall_per_threshold": {f"{t:.2f}": float(r)
                                 for t, r in zip(IOU_THRESHOLDS, recalls)},
        "n_gt": int(n_gt),
    }
    if per_band_hits is not None:
        for name in SIZE_BANDS:
            h = np.asarray(per_band_hits[name], dtype=np.float64)
            n = per_band_n[name]
            out[f"AR_{name}"] = float((h / max(n, 1)).mean())
            out[f"n_gt_{name}"] = int(n)
    if extra:
        out.update(extra)
    return out

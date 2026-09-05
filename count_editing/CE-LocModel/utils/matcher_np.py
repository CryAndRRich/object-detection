"""Prediction <-> ground-truth matcher — pure numpy, NO torch dependency.

SOURCE OF TRUTH. The torch port must be mechanical, with `allclose` tests.

TWO OPTIONS, HUNGARIAN BY DEFAULT:

  Hungarian 1-to-1 (default) — `scipy.optimize.linear_sum_assignment`.

  SimOTA dynamic-k (opt-in flag) — follows `diffusiondet/loss.py`, read for
  reference, NOT imported. Measured to DEGENERATE into Hungarian on CE-130:
    - k = clamp(sum of that GT's top-10 IoU, min=1), i.e. proportional to IoU
    - CE-130 objects occupy just 0.41 % of image area -> IoU collapses fast
    - already at t=100 (98.6 % signal left) IoU is only ~0.15 -> k = 1.0
    - round 1 measured: IoU 0.10 -> k=1.0, i.e. 45/300 slots vs Hungarian's 48
  So it adds complex code for no gain while the model is still weak, and it is
  also LESS LABEL-STABLE (55 % of pairs preserved) — which contradicts making
  "label stability" the number-one warning metric.

TWO ROUND-1 CENTER-PRIOR BUGS (fixed here):
  1. Used `is_in_boxes OR is_in_centers`; correct is **AND** (loss.py:403).
  2. Assigned positive labels directly to in-region slots; correct is to **add a
     +100 penalty to the cost matrix** (loss.py:364) and let the matcher decide.
  Measured: the center prior does NOT stabilise labels (55 % -> 51 %) because
  under AND only 2.1 % of pairs are eligible. OFF by default.

COST WEIGHTS = LOSS WEIGHTS (the DETR/DiffusionDet principle): 5.0 L1 + 2.0 GIoU
+ 2.0 class. If they diverge, the matcher picks pairs the loss disagrees with.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .box_ops_np import cxcywh_to_xyxy, generalized_box_iou, box_iou, pairwise_l1

__all__ = ["build_cost", "hungarian_match", "simota_match", "match"]

COST_L1, COST_GIOU, COST_CLASS = 5.0, 2.0, 2.0
ALPHA, GAMMA = 0.25, 2.0


def _focal_cost(scores):
    """Focal-style classification cost, same as `loss.py:305-308`.

    neg_cost - pos_cost per prediction; the matcher then prefers slots the model
    is ALREADY confident about, creating a positive feedback loop that stabilises
    the assignment.
    """
    p = np.clip(np.asarray(scores, dtype=np.float64).reshape(-1), 1e-8, 1 - 1e-8)
    pos = -ALPHA * ((1 - p) ** GAMMA) * np.log(p)
    neg = (1 - ALPHA) * (p ** GAMMA) * (-np.log(1 - p))
    return (neg - pos)[:, None]


def build_cost(pred_boxes, gt_boxes, scores=None, need_iou=True):
    """Cost matrix [N, M]. Boxes in canonical cxcywh [0,1].

    Returns (cost, iou) — `iou` is reused for dynamic-k, avoiding a second pass.
    `need_iou=False` skips it entirely for Hungarian, which never reads it (the
    torch port does the same; keeping the signatures identical keeps the
    allclose comparison tests meaningful).
    """
    p = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    g = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    p_xyxy, g_xyxy = cxcywh_to_xyxy(p), cxcywh_to_xyxy(g)

    # An untrained model can emit inverted boxes -> sort the corners before GIoU,
    # otherwise GIoU returns nonsense (round 1 saw 5e5)
    p_xyxy = np.stack([
        np.minimum(p_xyxy[:, 0], p_xyxy[:, 2]), np.minimum(p_xyxy[:, 1], p_xyxy[:, 3]),
        np.maximum(p_xyxy[:, 0], p_xyxy[:, 2]), np.maximum(p_xyxy[:, 1], p_xyxy[:, 3]),
    ], axis=1)

    cost = COST_L1 * pairwise_l1(p, g) + COST_GIOU * (1.0 - generalized_box_iou(p_xyxy, g_xyxy))
    if scores is not None:
        cost = cost + COST_CLASS * _focal_cost(scores)
    return cost, (box_iou(p_xyxy, g_xyxy)[0] if need_iou else None)


def _center_prior_mask(pred_boxes, gt_boxes, radius_ratio=2.5):
    """`is_in_boxes` AND `is_in_centers` (loss.py:403).

    `center_radius` is PROPORTIONAL to the GT's sqrt(w*h), not DiffusionDet's
    constant 2.5 (which is measured in FPN strides — we have no FPN, and CE-130
    objects cover only 0.41 % of the image).
    """
    p = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    g = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    cx, cy = p[:, 0][:, None], p[:, 1][:, None]

    gx, gy, gw, gh = g[:, 0][None], g[:, 1][None], g[:, 2][None], g[:, 3][None]
    in_boxes = (
        (cx > gx - gw / 2) & (cx < gx + gw / 2) & (cy > gy - gh / 2) & (cy < gy + gh / 2)
    )
    r = radius_ratio * np.sqrt(gw * gh)
    in_centers = (cx > gx - r) & (cx < gx + r) & (cy > gy - r) & (cy < gy + r)
    return in_boxes & in_centers


def hungarian_match(pred_boxes, gt_boxes, scores=None):
    """1-to-1 assignment. Returns (pred_idx, gt_idx) — two arrays of equal length."""
    g = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    if g.shape[0] == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    cost, _ = build_cost(pred_boxes, g, scores, need_iou=False)
    return linear_sum_assignment(cost)


def simota_match(pred_boxes, gt_boxes, scores=None, use_center_prior=False,
                 radius_ratio=2.5, top_k=10):
    """SimOTA dynamic-k. One GT takes k proposals; one proposal matches AT MOST 1 GT.

    "1-to-k" means one GT receives k proposals, NOT the reverse — seen from the
    proposal side it is still 1-to-1 (the `anchor_matching_gt > 1` dedup block).
    """
    g = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    n = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4).shape[0]
    if g.shape[0] == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    cost, iou = build_cost(pred_boxes, g, scores)
    if use_center_prior:
        cost = cost + 100.0 * (~_center_prior_mask(pred_boxes, g, radius_ratio))

    m = g.shape[0]
    matching = np.zeros((n, m), dtype=bool)

    # k = clamp(sum of that GT's top-k IoU, min=1)
    k_cap = min(top_k, n)
    topk_iou = np.sort(iou, axis=0)[-k_cap:]
    dynamic_k = np.clip(topk_iou.sum(0).astype(int), 1, None)

    for j in range(m):
        idx = np.argsort(cost[:, j])[: dynamic_k[j]]
        matching[idx, j] = True

    def _dedup(mat):
        """A proposal keeps only its lowest-cost GT (loss.py:424-427)."""
        multi = mat.sum(1) > 1
        if multi.any():
            best = np.argmin(cost[multi], axis=1)
            mat[multi] = False
            mat[np.where(multi)[0], best] = True
        return mat

    matching = _dedup(matching)

    # Rescue loop for unmatched GT (loss.py:428-438). The dedup above can strip a
    # GT of all its proposals; DiffusionDet penalises already-used proposals by
    # +1e5, reassigns the cheapest remaining proposal to each empty GT, and
    # repeats until every GT has one.
    cost = cost.copy()
    for _ in range(100):
        empty = matching.sum(0) == 0
        if not empty.any():
            break
        cost[matching.sum(1) > 0] += 1e5
        for j in np.where(empty)[0]:
            matching[np.argmin(cost[:, j]), j] = True
        matching = _dedup(matching)
    assert not (matching.sum(0) == 0).any(), "a GT is still unmatched"

    pred_idx, gt_idx = np.where(matching)
    return pred_idx, gt_idx


def match(pred_boxes, gt_boxes, scores=None, method="hungarian", **kw):
    """Shared entry point. `method` is 'hungarian' (default) or 'simota'."""
    if method == "hungarian":
        return hungarian_match(pred_boxes, gt_boxes, scores)
    if method == "simota":
        return simota_match(pred_boxes, gt_boxes, scores, **kw)
    raise ValueError(f"invalid method: {method!r}")

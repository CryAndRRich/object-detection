"""TORCH port of `matcher_np.py` — MECHANICAL. Runs under `no_grad`.

HUNGARIAN BY DEFAULT. SimOTA was measured to degenerate to k=1 on CE-130 (objects
cover just 0.41 % of the area -> IoU collapses fast, already ~0.15 at t=100), so
it adds complex code for no gain while the model is still weak, and it is also
less label-stable (55 % of pairs preserved) — contradicting warning metric #1.

Center prior (SimOTA only): `is_in_boxes` AND `is_in_centers`, adding +100 to the
COST (not assigning labels directly — round 1 got both of these wrong). OFF by
default.
"""

import torch
from scipy.optimize import linear_sum_assignment

from .box_ops import cxcywh_to_xyxy, box_iou, generalized_box_iou, sanitize_boxes

__all__ = ["build_cost", "hungarian_match", "simota_match", "match"]

COST_L1, COST_GIOU, COST_CLASS = 5.0, 2.0, 2.0
ALPHA, GAMMA = 0.25, 2.0


def _focal_cost(scores):
    """neg_cost - pos_cost, same as loss.py:305-308."""
    p = scores.reshape(-1).sigmoid().clamp(1e-8, 1 - 1e-8)
    pos = -ALPHA * ((1 - p) ** GAMMA) * p.log()
    neg = (1 - ALPHA) * (p ** GAMMA) * (-(1 - p).log())
    return (neg - pos)[:, None]


@torch.no_grad()
def build_cost(pred_boxes, gt_boxes, scores=None):
    """Cost [N,M] + iou [N,M]. Boxes in canonical cxcywh [0,1]. `scores` are LOGITS."""
    p_xyxy = sanitize_boxes(cxcywh_to_xyxy(pred_boxes))
    g_xyxy = cxcywh_to_xyxy(gt_boxes)

    l1 = torch.cdist(pred_boxes, gt_boxes, p=1)
    giou = generalized_box_iou(p_xyxy, g_xyxy)
    cost = COST_L1 * l1 + COST_GIOU * (1.0 - giou)
    if scores is not None:
        cost = cost + COST_CLASS * _focal_cost(scores)
    return cost, box_iou(p_xyxy, g_xyxy)[0]


def _center_prior_mask(pred_boxes, gt_boxes, radius_ratio=2.5):
    """AND, with `center_radius` PROPORTIONAL to the GT's sqrt(w*h) (not
    DiffusionDet's constant 2.5 — that is in FPN strides, which we do not have)."""
    cx, cy = pred_boxes[:, 0:1], pred_boxes[:, 1:2]
    gx, gy = gt_boxes[:, 0][None], gt_boxes[:, 1][None]
    gw, gh = gt_boxes[:, 2][None], gt_boxes[:, 3][None]

    in_boxes = ((cx > gx - gw / 2) & (cx < gx + gw / 2) &
                (cy > gy - gh / 2) & (cy < gy + gh / 2))
    r = radius_ratio * (gw * gh).sqrt()
    in_centers = (cx > gx - r) & (cx < gx + r) & (cy > gy - r) & (cy < gy + r)
    return in_boxes & in_centers


@torch.no_grad()
def hungarian_match(pred_boxes, gt_boxes, scores=None):
    """1-to-1 assignment -> (pred_idx, gt_idx) on the same device."""
    dev = pred_boxes.device
    if gt_boxes.shape[0] == 0:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z
    cost, _ = build_cost(pred_boxes, gt_boxes, scores)
    r, c = linear_sum_assignment(cost.detach().float().cpu().numpy())
    return (torch.as_tensor(r, dtype=torch.long, device=dev),
            torch.as_tensor(c, dtype=torch.long, device=dev))


@torch.no_grad()
def simota_match(pred_boxes, gt_boxes, scores=None, use_center_prior=False,
                 radius_ratio=2.5, top_k=10):
    """One GT takes k proposals; one proposal matches AT MOST 1 GT."""
    dev = pred_boxes.device
    n, m = pred_boxes.shape[0], gt_boxes.shape[0]
    if m == 0:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return z, z

    cost, iou = build_cost(pred_boxes, gt_boxes, scores)
    if use_center_prior:
        cost = cost + 100.0 * (~_center_prior_mask(pred_boxes, gt_boxes, radius_ratio))
    cost = cost.clone()

    matching = torch.zeros(n, m, dtype=torch.bool, device=dev)
    k_cap = min(top_k, n)
    dynamic_k = iou.topk(k_cap, dim=0).values.sum(0).int().clamp(min=1)
    for j in range(m):
        matching[cost[:, j].topk(int(dynamic_k[j]), largest=False).indices, j] = True

    def _dedup(mat):
        multi = mat.sum(1) > 1
        if multi.any():
            best = cost[multi].argmin(dim=1)
            mat[multi] = False
            mat[multi.nonzero(as_tuple=True)[0], best] = True
        return mat

    matching = _dedup(matching)

    # Rescue loop for unmatched GT (loss.py:428-438): the dedup above can strip a
    # GT of all its proposals -> penalise used proposals by +1e5 and reassign.
    for _ in range(100):
        empty = matching.sum(0) == 0
        if not empty.any():
            break
        cost[matching.sum(1) > 0] += 1e5
        for j in empty.nonzero(as_tuple=True)[0]:
            matching[cost[:, j].argmin(), j] = True
        matching = _dedup(matching)
    assert not (matching.sum(0) == 0).any(), "a GT is still unmatched"

    pi, gi = matching.nonzero(as_tuple=True)
    return pi, gi


def match(pred_boxes, gt_boxes, scores=None, method="hungarian", **kw):
    if method == "hungarian":
        return hungarian_match(pred_boxes, gt_boxes, scores)
    if method == "simota":
        return simota_match(pred_boxes, gt_boxes, scores, **kw)
    raise ValueError(f"invalid method: {method!r}")

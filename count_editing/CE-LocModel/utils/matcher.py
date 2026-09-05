"""Bản TORCH của `matcher_np.py` — port CƠ HỌC. Chạy dưới `no_grad`.

MẶC ĐỊNH HUNGARIAN. SimOTA đo được thoái hoá về k=1 trên CE-130 (vật chỉ chiếm
0,41 % diện tích -> IoU sụp rất nhanh, ngay t=100 đã chỉ ~0,15), nên nó thêm code
phức tạp mà không mang lại gì ở giai đoạn model chưa tốt, lại kém ổn định nhãn
hơn (55 % cặp giữ nguyên) — mâu thuẫn với chỉ số cảnh báo số 1.

Center prior (chỉ SimOTA): `is_in_boxes` AND `is_in_centers`, cộng +100 vào COST
(không gán nhãn trực tiếp — vòng 1 sai cả hai chỗ). Mặc định TẮT.
"""

import torch
from scipy.optimize import linear_sum_assignment

from .box_ops import cxcywh_to_xyxy, box_iou, generalized_box_iou, sanitize_boxes

__all__ = ["build_cost", "hungarian_match", "simota_match", "match"]

COST_L1, COST_GIOU, COST_CLASS = 5.0, 2.0, 2.0
ALPHA, GAMMA = 0.25, 2.0


def _focal_cost(scores):
    """neg_cost - pos_cost, giống loss.py:305-308."""
    p = scores.reshape(-1).sigmoid().clamp(1e-8, 1 - 1e-8)
    pos = -ALPHA * ((1 - p) ** GAMMA) * p.log()
    neg = (1 - ALPHA) * (p ** GAMMA) * (-(1 - p).log())
    return (neg - pos)[:, None]


@torch.no_grad()
def build_cost(pred_boxes, gt_boxes, scores=None):
    """Cost [N,M] + iou [N,M]. Box ở hệ chuẩn cxcywh [0,1]. `scores` là LOGIT."""
    p_xyxy = sanitize_boxes(cxcywh_to_xyxy(pred_boxes))
    g_xyxy = cxcywh_to_xyxy(gt_boxes)

    l1 = torch.cdist(pred_boxes, gt_boxes, p=1)
    giou = generalized_box_iou(p_xyxy, g_xyxy)
    cost = COST_L1 * l1 + COST_GIOU * (1.0 - giou)
    if scores is not None:
        cost = cost + COST_CLASS * _focal_cost(scores)
    return cost, box_iou(p_xyxy, g_xyxy)[0]


def _center_prior_mask(pred_boxes, gt_boxes, radius_ratio=2.5):
    """AND, và `center_radius` TỈ LỆ theo sqrt(w*h) của GT (không phải hằng 2,5
    của DiffusionDet — vốn tính theo stride FPN mà ta không có)."""
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
    """Gán 1-1 -> (pred_idx, gt_idx) trên cùng device."""
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
    """Một GT nhận k proposal; một proposal khớp TỐI ĐA 1 GT."""
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

    def _khu_trung(mat):
        multi = mat.sum(1) > 1
        if multi.any():
            best = cost[multi].argmin(dim=1)
            mat[multi] = False
            mat[multi.nonzero(as_tuple=True)[0], best] = True
        return mat

    matching = _khu_trung(matching)

    # Vòng cứu GT chưa được khớp (loss.py:428-438): khử trùng có thể làm một GT
    # mất hết proposal -> phạt +1e5 vào proposal đã dùng rồi gán lại.
    for _ in range(100):
        trong = matching.sum(0) == 0
        if not trong.any():
            break
        cost[matching.sum(1) > 0] += 1e5
        for j in trong.nonzero(as_tuple=True)[0]:
            matching[cost[:, j].argmin(), j] = True
        matching = _khu_trung(matching)
    assert not (matching.sum(0) == 0).any(), "còn GT không được khớp"

    pi, gi = matching.nonzero(as_tuple=True)
    return pi, gi


def match(pred_boxes, gt_boxes, scores=None, method="hungarian", **kw):
    if method == "hungarian":
        return hungarian_match(pred_boxes, gt_boxes, scores)
    if method == "simota":
        return simota_match(pred_boxes, gt_boxes, scores, **kw)
    raise ValueError(f"method không hợp lệ: {method!r}")

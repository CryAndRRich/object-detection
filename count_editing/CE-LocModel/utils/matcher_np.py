"""Matcher gán prediction <-> ground truth — numpy thuần, KHÔNG phụ thuộc torch.

NGUỒN CHÂN LÝ. Bản torch phải port cơ học kèm test `allclose`.

HAI LỰA CHỌN, MẶC ĐỊNH HUNGARIAN:

  Hungarian 1-1 (mặc định) — `scipy.optimize.linear_sum_assignment`.

  SimOTA dynamic-k (cờ bật) — theo `diffusiondet/loss.py`, đọc đối chiếu, KHÔNG
  import. Đo được nó THOÁI HOÁ về Hungarian trên CE-130:
    - k = clamp(sum(top-10 IoU của GT đó), min=1), tỉ lệ với IoU
    - vật CE-130 chỉ chiếm 0,41 % diện tích ảnh -> IoU sụp rất nhanh
    - ngay t=100 (còn 98,6 % tín hiệu) IoU đã chỉ ~0,15 -> k = 1,0
    - vòng 1 đo: IoU 0,10 -> k=1,0, tức 45/300 slot so với 48 của Hungarian
  Nên nó thêm code phức tạp mà không mang lại gì ở giai đoạn model chưa tốt, và
  còn KÉM ỔN ĐỊNH NHÃN hơn (55 % cặp giữ nguyên) — mâu thuẫn với việc đặt "ổn
  định nhãn" làm chỉ số cảnh báo số 1.

HAI LỖI VÒNG 1 VỀ CENTER PRIOR (đã sửa ở đây):
  1. Dùng `is_in_boxes OR is_in_centers`; đúng là **AND** (loss.py:403).
  2. Gán nhãn dương trực tiếp cho slot trong vùng; đúng là **cộng phạt +100 vào
     cost matrix** (loss.py:364) rồi để matcher quyết định.
  Đo được center prior KHÔNG ổn định hoá nhãn (55 % -> 51 %) vì với AND chỉ 2,1 %
  cặp là hợp lệ. Mặc định TẮT.

TRỌNG SỐ COST = trọng số LOSS (nguyên tắc DETR/DiffusionDet): 5,0 L1 + 2,0 GIoU
+ 2,0 class. Lệch nhau thì matcher chọn cặp mà loss không đồng ý.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .box_ops_np import cxcywh_to_xyxy, generalized_box_iou, box_iou, pairwise_l1

__all__ = ["build_cost", "hungarian_match", "simota_match", "match"]

COST_L1, COST_GIOU, COST_CLASS = 5.0, 2.0, 2.0
ALPHA, GAMMA = 0.25, 2.0


def _focal_cost(scores):
    """Cost phân loại kiểu focal, giống `loss.py:305-308`.

    neg_cost - pos_cost cho mỗi prediction; matcher ưu tiên slot model ĐÃ tự tin
    là vật -> tạo vòng phản hồi dương ổn định hoá việc gán.
    """
    p = np.clip(np.asarray(scores, dtype=np.float64).reshape(-1), 1e-8, 1 - 1e-8)
    pos = -ALPHA * ((1 - p) ** GAMMA) * np.log(p)
    neg = (1 - ALPHA) * (p ** GAMMA) * (-np.log(1 - p))
    return (neg - pos)[:, None]


def build_cost(pred_boxes, gt_boxes, scores=None):
    """Cost matrix [N, M]. Box ở hệ chuẩn cxcywh [0,1].

    Trả về (cost, iou) — `iou` dùng lại cho dynamic-k, khỏi tính hai lần.
    """
    p = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    g = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    p_xyxy, g_xyxy = cxcywh_to_xyxy(p), cxcywh_to_xyxy(g)

    # box dự đoán có thể lộn ngược khi model chưa train -> kẹp lại trước GIoU,
    # nếu không GIoU cho giá trị vô nghĩa (vòng 1: 5e5)
    p_xyxy = np.stack([
        np.minimum(p_xyxy[:, 0], p_xyxy[:, 2]), np.minimum(p_xyxy[:, 1], p_xyxy[:, 3]),
        np.maximum(p_xyxy[:, 0], p_xyxy[:, 2]), np.maximum(p_xyxy[:, 1], p_xyxy[:, 3]),
    ], axis=1)

    cost = COST_L1 * pairwise_l1(p, g) + COST_GIOU * (1.0 - generalized_box_iou(p_xyxy, g_xyxy))
    if scores is not None:
        cost = cost + COST_CLASS * _focal_cost(scores)
    return cost, box_iou(p_xyxy, g_xyxy)[0]


def _center_prior_mask(pred_boxes, gt_boxes, radius_ratio=2.5):
    """`is_in_boxes` AND `is_in_centers` (loss.py:403).

    `center_radius` TỈ LỆ theo sqrt(w*h) của GT, không phải hằng 2,5 của
    DiffusionDet (vốn tính theo stride FPN — ta không có FPN, và vật CE-130 chỉ
    chiếm 0,41 % diện tích ảnh).
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
    """Gán 1-1. Trả về (pred_idx, gt_idx) — hai mảng cùng độ dài."""
    g = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    if g.shape[0] == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    cost, _ = build_cost(pred_boxes, g, scores)
    return linear_sum_assignment(cost)


def simota_match(pred_boxes, gt_boxes, scores=None, use_center_prior=False,
                 radius_ratio=2.5, top_k=10):
    """SimOTA dynamic-k. Một GT nhận k proposal; một proposal khớp TỐI ĐA 1 GT.

    "1-to-k" nghĩa là một GT nhận k proposal, KHÔNG phải ngược lại — nhìn từ phía
    proposal thì vẫn là 1-1 (khối khử trùng `anchor_matching_gt > 1`).
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

    # k = clamp(sum(top-k IoU của GT đó), min=1)
    k_cap = min(top_k, n)
    topk_iou = np.sort(iou, axis=0)[-k_cap:]
    dynamic_k = np.clip(topk_iou.sum(0).astype(int), 1, None)

    for j in range(m):
        idx = np.argsort(cost[:, j])[: dynamic_k[j]]
        matching[idx, j] = True

    def _khu_trung(mat):
        """Một proposal chỉ giữ GT có cost thấp nhất (loss.py:424-427)."""
        multi = mat.sum(1) > 1
        if multi.any():
            best = np.argmin(cost[multi], axis=1)
            mat[multi] = False
            mat[np.where(multi)[0], best] = True
        return mat

    matching = _khu_trung(matching)

    # Vòng cứu GT chưa được khớp (loss.py:428-438). Khử trùng ở trên có thể làm
    # một GT mất hết proposal; DiffusionDet phạt +1e5 vào proposal đã dùng rồi
    # gán lại proposal rẻ nhất cho GT trống, lặp tới khi mọi GT đều có.
    cost = cost.copy()
    for _ in range(100):
        trong = matching.sum(0) == 0
        if not trong.any():
            break
        cost[matching.sum(1) > 0] += 1e5
        for j in np.where(trong)[0]:
            matching[np.argmin(cost[:, j]), j] = True
        matching = _khu_trung(matching)
    assert not (matching.sum(0) == 0).any(), "còn GT không được khớp"

    pred_idx, gt_idx = np.where(matching)
    return pred_idx, gt_idx


def match(pred_boxes, gt_boxes, scores=None, method="hungarian", **kw):
    """Cửa vào chung. `method` là 'hungarian' (mặc định) hoặc 'simota'."""
    if method == "hungarian":
        return hungarian_match(pred_boxes, gt_boxes, scores)
    if method == "simota":
        return simota_match(pred_boxes, gt_boxes, scores, **kw)
    raise ValueError(f"method không hợp lệ: {method!r}")

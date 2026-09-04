"""
Hungarian matching between N predicted boxes and M ground-truth boxes, for the
multi-box variant (c) of CE-Loc (sinh N box cùng lúc, theo cơ chế DiffusionDet).

Cross-checked against `object-detection/diffusiondet/diffusiondet/loss.py`
(`HungarianMatcherDynamicK`) — NOT ported/imported from there. DiffusionDet's own
matcher is a SimOTA-style dynamic-K assignment gated on a classification score
(`pred_logits`) and an anchor-center-region mask (`get_in_boxes_info`), because it
is matching across many object CATEGORIES at once. CE-Loc's multi-box variant has
no class head — every box in one image is the same category (the sample's single
"class" text) — so a plain 1-to-1 Hungarian assignment on box cost alone is the
right-sized tool, not a simplification that drops something CE-Loc needs.

Cost = cost_bbox * L1(cx,cy,w,h) + cost_giou * (1 - GIoU), same weighting formula
as DiffusionDet's `loss_boxes` (see loss.py:159-201): L1 on normalized cxcywh +
GIoU on xyxy. Padding predictions (beyond num_gt) are left unmatched by
`linear_sum_assignment` itself (it only returns min(N, M) pairs) — the caller
uses `matched_pred_idx` to know which of the N predictions actually have a target.

COORDINATE SPACE — read before touching the geometry helpers. CE-Loc normalizes
BOTH centers and sizes with the same affine map (data/dataset.py::_normalize_bbox,
inherited verbatim from the original repo):

    norm_cx = (cx / target_w) * 2 - 1        norm_w = (w / target_w) * 2 - 1

For the CENTER that is the usual [0,size] -> [-1,1] mapping, but for the SIZE it
means a box occupying a fraction f of the image has norm_w = 2f - 1, so every box
smaller than half the image has a NEGATIVE norm_w (measured: 100% of real boxes in
samples/train). Feeding that straight into a cxcywh->xyxy conversion yields x2 < x1,
i.e. inverted boxes, and IoU/GIoU then returns nonsense (measured: GIoU of a box
with itself = 5e5 instead of 1.0). So every geometric helper here first converts
size back to a true width/height via (norm + 1) / 2 before building corners.
(The original repo's own `calculate_iou` in test_mul_box.py does NOT do this — it
treats norm_w as a raw width. That flaw is left untouched there so the reported
IoU@10/IoU@30 numbers stay comparable with the paper's, but it must not leak into
the matcher/loss, where inverted boxes would corrupt training rather than a metric.)
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _cxcywh_to_xyxy(boxes):
    """CE-Loc-normalized [-1,1] cxcywh -> xyxy corners in the same [-1,1] frame.

    `w`/`h` arrive as (2 * fraction_of_image - 1) — see the COORDINATE SPACE note
    in the module docstring — so they are mapped back to a true extent with
    (w + 1) / 2 before halving. Without this the corners come out inverted for
    any box smaller than half the image, which is essentially all of them.
    """
    cx, cy, w, h = boxes.unbind(-1)
    # clamp(min=0): a predicted norm_w below -1 would mean a negative true
    # width, which makes x2 < x1 and sends GIoU to garbage (the union area
    # goes negative). DiffusionDet asserts against degenerate boxes instead
    # (util/box_ops.py:51-52); clamping is the non-fatal equivalent, since
    # here the offending boxes come from an untrained network, not from data.
    # `true_w` ở đây là PHÂN SỐ của chiều rộng ảnh (hệ [0,1]), trong khi `cx`
    # nằm trong hệ [-1,1] — tức MỘT ĐƠN VỊ của cx chỉ bằng NỬA ảnh, còn một đơn
    # vị của true_w bằng CẢ ảnh. Trộn hai thang này khi dựng góc làm box hẹp đi
    # đúng 2 lần so với khoảng cách giữa các tâm.
    #
    # LỖI NÀY ĐÃ TỪNG TỒN TẠI (phát hiện 2026-09-04 qua visualize: box GT vẽ ra
    # không bao nổi vật thể, chỉ rộng bằng nửa). Hệ quả: IoU đúng khi hai box
    # CÙNG TÂM, nhưng sai — luôn thấp hơn thật — ngay khi tâm lệch nhau, vì độ
    # lệch tâm bị đo ở thang gấp đôi extent. Đo được: hai box lệch nhau chút ít
    # cho IoU 0,309 thay vì 0,553 thật.
    #
    # Sửa: nhân extent lên 2 để đưa về CÙNG hệ [-1,1] với tâm. Kiểm bằng
    # round-trip pixel (tests/test_matcher_coords.py).
    true_w = ((w + 1.0)).clamp(min=0.0)   # (w+1)/2 phân số ảnh, x2 để vào hệ [-1,1]
    true_h = ((h + 1.0)).clamp(min=0.0)
    return torch.stack([cx - true_w / 2, cy - true_h / 2,
                        cx + true_w / 2, cy + true_h / 2], dim=-1)


def box_iou_normalized(boxes1_xyxy, boxes2_xyxy):
    """Pairwise plain IoU, boxes1: [N,4], boxes2: [M,4] -> [N,M]. Inputs must
    already be xyxy from `_cxcywh_to_xyxy` (so their extents are true widths,
    not CE-Loc's (2f-1)-encoded sizes)."""
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def generalized_box_iou(boxes1_xyxy, boxes2_xyxy):
    """Pairwise GIoU, boxes1: [N,4], boxes2: [M,4] -> [N,M]. Standard formula
    (Rezatofighi et al., CVPR 2019 Eq. 1). Inputs must come from
    `_cxcywh_to_xyxy`, which already clamps away negative extents, so the
    degenerate case DiffusionDet asserts on (util/box_ops.py:51-52) cannot
    reach the area arithmetic here.
    """
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])

    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    lt_c = torch.min(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb_c = torch.max(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]

    return iou - (area_c - union) / area_c.clamp(min=1e-6)


@torch.no_grad()
def hungarian_match(pred_boxes_cxcywh, gt_boxes_cxcywh, cost_bbox=1.0, cost_giou=1.0):
    """
    pred_boxes_cxcywh: [N, 4] tensor, normalized [-1, 1] cxcywh (one image's predictions)
    gt_boxes_cxcywh:   [M, 4] tensor, same normalization, M <= N (padding already
                       stripped by the caller — this function assumes every row of
                       gt_boxes_cxcywh is a real target)

    Returns (pred_idx, gt_idx): 1-D LongTensors of length min(N, M) — pred_idx[i]
    is matched to gt_idx[i]. Predictions not in pred_idx are unmatched (padding
    slots or, when M < N, simply extra proposals with no target this step).
    """
    N = pred_boxes_cxcywh.shape[0]
    M = gt_boxes_cxcywh.shape[0]
    if M == 0:
        return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))

    pred_xyxy = _cxcywh_to_xyxy(pred_boxes_cxcywh)
    gt_xyxy = _cxcywh_to_xyxy(gt_boxes_cxcywh)

    l1 = torch.cdist(pred_boxes_cxcywh, gt_boxes_cxcywh, p=1)  # [N, M]
    giou = generalized_box_iou(pred_xyxy, gt_xyxy)  # [N, M], higher = more overlap

    cost = cost_bbox * l1 + cost_giou * (1.0 - giou)  # [N, M], lower = better match
    cost_np = cost.cpu().numpy()
    pred_idx, gt_idx = linear_sum_assignment(cost_np)
    return torch.as_tensor(pred_idx, dtype=torch.long), torch.as_tensor(gt_idx, dtype=torch.long)

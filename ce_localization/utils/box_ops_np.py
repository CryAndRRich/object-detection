"""Hình học box bằng numpy (không torch): đổi xyxy <-> cxcywh, IoU, lọc box suy biến. Dùng cho
chấm điểm (`utils/metrics_np.py`, `engine/evaluate.py`) và quét dữ liệu (`data/dataset.py`)."""

import numpy as np

__all__ = ["xyxy_to_cxcywh", "cxcywh_to_xyxy", "box_iou", "filter_degenerate"]


# --------------------------------------------------------------------------
# Steps (1) and (6): change format, NOT scale
# --------------------------------------------------------------------------

def xyxy_to_cxcywh(boxes):
    """[..., 4] (x1,y1,x2,y2) -> (cx,cy,w,h). Units unchanged."""
    b = np.asarray(boxes, dtype=np.float64)
    x1, y1, x2, y2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], axis=-1)


def cxcywh_to_xyxy(boxes):
    """[..., 4] (cx,cy,w,h) -> (x1,y1,x2,y2). Units unchanged."""
    b = np.asarray(boxes, dtype=np.float64)
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=-1)


# --------------------------------------------------------------------------
# Steps (2)+(3): original-image pixels -> canonical [0,1]
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Steps (4) and (5): canonical <-> diffusion space
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# IoU / GIoU — take xyxy, shared by matcher, loss and tests alike
# --------------------------------------------------------------------------

def _area_xyxy(b):
    return np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)


def box_iou(boxes1, boxes2):
    """Pairwise IoU. boxes1 [N,4], boxes2 [M,4] xyxy -> (iou [N,M], union [N,M])."""
    b1 = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)
    b2 = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)
    a1, a2 = _area_xyxy(b1)[:, None], _area_xyxy(b2)[None, :]

    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    union = a1 + a2 - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou, union


# --------------------------------------------------------------------------
# Augmentation + data hygiene
# --------------------------------------------------------------------------


def filter_degenerate(boxes_xyxy, min_size=0.0):
    """Drop boxes with w or h <= min_size. Returns (clean_boxes, keep_mask).

    Measured 14 of 37,110 CE-130 train boxes are degenerate. Must be filtered
    BEFORE the matcher AND before computing GIoU.
    """
    b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    keep = (b[:, 2] - b[:, 0] > min_size) & (b[:, 3] - b[:, 1] > min_size)
    return b[keep], keep

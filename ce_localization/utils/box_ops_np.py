"""Hình học box bằng numpy (xyxy): IoU, IoA, lọc box suy biến."""

import numpy as np

__all__ = ["box_iou", "ioa", "filter_degenerate"]


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


def ioa(boxes, objects):
    """[K,4], [M,4] xyxy -> [K,M] giao / diện tích box (box diện tích 0 -> 0) — `on_object` của eval."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    o = np.asarray(objects, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(b[:, None, :2], o[None, :, :2])
    rb = np.minimum(b[:, None, 2:], o[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(area > 0, inter / area, 0.0)


def filter_degenerate(boxes_xyxy, min_size=0.0):
    """Bỏ box có w hoặc h <= min_size -> (box còn lại, mask giữ)."""
    b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    keep = (b[:, 2] - b[:, 0] > min_size) & (b[:, 3] - b[:, 1] > min_size)
    return b[keep], keep

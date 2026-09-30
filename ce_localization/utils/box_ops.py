"""Hình học box bằng torch: đổi xyxy <-> cxcywh, IoU, GIoU (loss / matcher, `engine/criterion.py`)."""

import torch

__all__ = ["xyxy_to_cxcywh", "cxcywh_to_xyxy", "box_iou", "generalized_box_iou"]


def xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _area(b):
    return (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)


def box_iou(b1, b2):
    """[N,4] x [M,4] xyxy -> (iou [N,M], union [N,M])."""
    a1, a2 = _area(b1)[:, None], _area(b2)[None, :]
    lt = torch.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = torch.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = a1 + a2 - inter
    return torch.where(union > 0, inter / union.clamp(min=1e-12), torch.zeros_like(union)), union


def generalized_box_iou(b1, b2):
    """GIoU [-1,1]. Has a gradient even for DISJOINT boxes — essential for tiny objects."""
    iou, union = box_iou(b1, b2)
    lt = torch.minimum(b1[:, None, :2], b2[None, :, :2])
    rb = torch.maximum(b1[:, None, 2:], b2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    enc = wh[..., 0] * wh[..., 1]
    return iou - (enc - union) / enc.clamp(min=1e-12)

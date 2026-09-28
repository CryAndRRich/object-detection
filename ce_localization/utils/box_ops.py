"""TORCH port of `box_ops_np.py` — MECHANICAL, formula for formula.

Every function here must match the numpy version to within 1e-6; the comparison
tests live in `tests/test_torch_vs_numpy.py`. This is the gate that catches the
exact class of bug that killed round 1 (dividing twice, forgetting the clamp,
forgetting snr_scale) WITHOUT needing to train.

THE ONE CANONICAL SYSTEM: cxcywh [0,1]. `snr_scale` lives only inside diffusion.
"""

import torch

__all__ = [
    "xyxy_to_cxcywh", "cxcywh_to_xyxy", "encode_diffusion", "decode_diffusion",
    "box_iou", "generalized_box_iou", "sanitize_boxes",
]


def xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def encode_diffusion(boxes_norm, snr_scale=2.0):
    """cxcywh [0,1] -> x_start. Small objects give NEGATIVE w — normal, as in DiffusionDet."""
    return (boxes_norm * 2.0 - 1.0) * snr_scale


def decode_diffusion(x, snr_scale=2.0):
    """x_start -> cxcywh [0,1]. Clamp BEFORE dividing.

    ROUND-1 BUG: dividing by 2 once more -> boxes at half size, IoU 0.25.
    """
    return (x.clamp(-snr_scale, snr_scale) / snr_scale + 1.0) / 2.0


def sanitize_boxes(b_xyxy):
    """Sort corners of inverted boxes (an untrained model can emit w<0) before GIoU.

    Without this step GIoU returns nonsense — round 1 measured 5e5.
    """
    x1 = torch.minimum(b_xyxy[..., 0], b_xyxy[..., 2])
    y1 = torch.minimum(b_xyxy[..., 1], b_xyxy[..., 3])
    x2 = torch.maximum(b_xyxy[..., 0], b_xyxy[..., 2])
    y2 = torch.maximum(b_xyxy[..., 1], b_xyxy[..., 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)


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

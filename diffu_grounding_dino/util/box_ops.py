"""Bounding-box utilities.

Boxes come in two conventions:
  * ``cxcywh``: (center_x, center_y, width, height), normalized to [0, 1].
  * ``xyxy``:   (x_min, y_min, x_max, y_max).

Written from the definitions in the DETR / Generalized-IoU papers
(Rezatofighi et al., CVPR 2019) rather than copied from any reference repo.
"""

import torch
from torch import Tensor


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0], dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    """Area of ``xyxy`` boxes. Shape (N, 4) -> (N,)."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor):
    """Pairwise IoU between two sets of ``xyxy`` boxes.

    Returns ``(iou, union)`` both of shape (N, M) so that callers computing GIoU
    can reuse the union without recomputing it.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / union, union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise generalized IoU, ``GIoU = IoU - |C \\ (A u B)| / |C|``.

    ``C`` is the smallest enclosing box. Result is in [-1, 1], shape (N, M).
    Both inputs must be ``xyxy`` with ``x1 >= x0`` and ``y1 >= y0``.
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1 is not a valid xyxy box"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2 is not a valid xyxy box"

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    enclosing = wh[..., 0] * wh[..., 1]

    return iou - (enclosing - union) / enclosing.clamp(min=1e-7)


def box_xyxy_to_xywh(boxes: Tensor) -> Tensor:
    """``xyxy`` -> COCO-style ``(x_min, y_min, w, h)``."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([x0, y0, x1 - x0, y1 - y0], dim=-1)

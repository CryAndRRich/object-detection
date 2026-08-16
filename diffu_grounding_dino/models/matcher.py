"""Bipartite matching between predictions and ground truth.

DETR-style set prediction needs a one-to-one assignment before the loss can be
computed. The cost combines a focal-loss-shaped classification term with L1 and
GIoU box terms, and the assignment is solved exactly with the Hungarian algorithm.

The open-vocabulary twist: a "class" is a span of text tokens, not an index. The
classification cost of matching query ``q`` to target of category ``c`` is
therefore the *average* focal cost of ``q`` over the tokens of ``c``, which is
what ``label_map`` (row-normalized) encodes.
"""

from typing import List, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

MATCHER_TYPES = ("HungarianMatcher", "SimpleMinsumMatcher")


def _focal_cost_terms(out_prob: Tensor, alpha: float, gamma: float) -> Tuple[Tensor, Tensor]:
    """Per-token positive and negative focal costs. Shapes follow ``out_prob``."""
    neg = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
    pos = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
    return pos, neg


class HungarianMatcher(nn.Module):
    """Optimal one-to-one assignment by ``scipy.optimize.linear_sum_assignment``."""

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha: float = 0.25):
        super().__init__()
        assert cost_class or cost_bbox or cost_giou, "at least one cost must be non-zero"
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs: dict, targets: List[dict], label_map: Tensor) -> List[Tuple[Tensor, Tensor]]:
        """
        Args:
            outputs: ``pred_logits`` (bs, nq, max_text_len), ``pred_boxes`` (bs, nq, 4).
            targets: per image, ``labels`` (n,) as indices into this image's category
                list and ``boxes`` (n, 4) cxcywh normalized.
            label_map: (num_categories, max_text_len) 0/1 token mask per category.

        Returns:
            Per image, ``(pred_indices, target_indices)``.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # bs*nq, max_text_len
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # bs*nq, 4

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        if tgt_ids.numel() == 0:
            cost_class = torch.zeros_like(cost_bbox)
        else:
            pos_cost, neg_cost = _focal_cost_terms(out_prob, self.focal_alpha, 2.0)
            token_map = label_map.to(out_prob.device).index_select(0, tgt_ids.to(out_prob.device))
            # Average over the category's tokens: a two-token name must not cost
            # twice as much as a one-token name.
            token_map = token_map / token_map.sum(-1, keepdim=True).clamp(min=1e-6)
            # Single matmul in place of a per-target python loop.
            cost_class = (pos_cost - neg_cost) @ token_map.T

        cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost = cost.view(bs, num_queries, -1).cpu()
        # A padded text slot contributes -inf logits; NaN/inf must not reach the
        # solver, which would otherwise raise or return a garbage assignment.
        cost = torch.nan_to_num(cost, nan=0.0, posinf=0.0, neginf=0.0)

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        for i, chunk in enumerate(cost.split(sizes, -1)):
            if sizes[i] == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)))
                continue
            row, col = linear_sum_assignment(chunk[i])
            indices.append((torch.as_tensor(row, dtype=torch.int64), torch.as_tensor(col, dtype=torch.int64)))
        return indices


class SimpleMinsumMatcher(nn.Module):
    """Greedy alternative: every target takes its cheapest query, collisions allowed.

    Cheaper than Hungarian and occasionally more stable early in training, but it
    can assign one query to several targets. Kept for ablation only.
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha: float = 0.25):
        super().__init__()
        assert cost_class or cost_bbox or cost_giou, "at least one cost must be non-zero"
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs: dict, targets: List[dict], label_map: Tensor = None) -> List[Tuple[Tensor, Tensor]]:
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        if tgt_ids.numel() == 0:
            cost_class = torch.zeros_like(cost_bbox)
        elif label_map is not None:
            pos_cost, neg_cost = _focal_cost_terms(out_prob, self.focal_alpha, 2.0)
            token_map = label_map.to(out_prob.device).index_select(0, tgt_ids.to(out_prob.device))
            token_map = token_map / token_map.sum(-1, keepdim=True).clamp(min=1e-6)
            cost_class = (pos_cost - neg_cost) @ token_map.T
        else:
            pos_cost, neg_cost = _focal_cost_terms(out_prob, self.focal_alpha, 2.0)
            cost_class = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]

        cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost = torch.nan_to_num(cost.view(bs, num_queries, -1), nan=0.0, posinf=0.0, neginf=0.0)

        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        for i, (chunk, size) in enumerate(zip(cost.split(sizes, -1), sizes)):
            if size == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)))
                continue
            idx_i = chunk[i].min(0)[1]
            indices.append((idx_i.cpu().to(torch.int64), torch.arange(size, dtype=torch.int64)))
        return indices


def build_matcher(args) -> nn.Module:
    matcher_type = args.matcher_type
    assert matcher_type in MATCHER_TYPES, f"unknown matcher_type {matcher_type!r}"
    cls = HungarianMatcher if matcher_type == "HungarianMatcher" else SimpleMinsumMatcher
    return cls(
        cost_class=args.set_cost_class,
        cost_bbox=args.set_cost_bbox,
        cost_giou=args.set_cost_giou,
        focal_alpha=args.focal_alpha,
    )


__all__ = ["HungarianMatcher", "SimpleMinsumMatcher", "build_matcher"]

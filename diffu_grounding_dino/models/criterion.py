"""Set-prediction loss, with optional timestep reweighting.

Two things make this different from a plain DETR criterion.

**Text-token classification.** There is no class index. Each ground-truth box is
supervised against the *token positions* of its category in the prompt, so the
target is a 0/1 map over the 256 text slots, and the loss is a token-wise binary
focal loss masked to the real tokens.

**Timestep weighting (the diffusion part).** When a sample's reference points were
noised at timestep ``t``, its supervision is worth less the noisier it was: a
query starting from near-pure noise cannot be expected to land on the box. Each
image therefore carries a weight ``w(t)`` from the diffusion schedule
(``RefPointDiffusion.loss_weight``), applied to the box and classification losses
of the main and auxiliary decoder outputs.

The ``interm_outputs`` branch is deliberately left unweighted: those predictions
come from the encoder's proposal head, which never saw a timestep, so scaling them
by ``w(t)`` would add noise to a signal that has nothing to do with diffusion.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import get_world_size, is_dist_avail_and_initialized
from util.vl_utils import create_positive_map


class SetCriterion(nn.Module):
    """Hungarian-matched box + token-classification loss.

    Args:
        matcher: assignment module.
        weight_dict: loss name -> coefficient. Also drives which losses are summed
            in the training loop.
        focal_alpha / focal_gamma: focal loss parameters.
        losses: subset of ``("labels", "boxes", "cardinality")``.
        use_timestep_weighting: honour the ``t_weight`` argument of ``forward``.
            When ``False`` the argument is ignored, which is what keeps the
            non-diffusion baseline numerically identical.
    """

    def __init__(
        self,
        matcher,
        weight_dict: Dict[str, float],
        focal_alpha: float,
        focal_gamma: float,
        losses: List[str],
        max_text_len: int = 256,
        use_timestep_weighting: bool = False,
    ):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.max_text_len = max_text_len
        self.use_timestep_weighting = use_timestep_weighting

    # ------------------------------------------------------------------ #
    # individual losses
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, t_weight=None):
        """Absolute error in the predicted object count. Logging only, no gradient."""
        pred_logits = outputs["pred_logits"]
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=pred_logits.device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        return {"cardinality_error": F.l1_loss(card_pred.float(), tgt_lengths.float())}

    def loss_boxes(self, outputs, targets, indices, num_boxes, t_weight=None):
        """L1 + GIoU on the matched boxes, in normalized cxcywh."""
        assert "pred_boxes" in outputs
        batch_idx, src_idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][batch_idx, src_idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        if src_boxes.numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero, "loss_xy": zero.detach(), "loss_hw": zero.detach()}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")  # (N, 4)
        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )  # (N,)

        weight = self._per_query_weight(t_weight, batch_idx)
        if weight is not None:
            loss_bbox = loss_bbox * weight[:, None]
            loss_giou = loss_giou * weight

        losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        with torch.no_grad():
            losses["loss_xy"] = loss_bbox[..., :2].sum() / num_boxes
            losses["loss_hw"] = loss_bbox[..., 2:].sum() / num_boxes
        return losses

    def token_sigmoid_binary_focal_loss(self, outputs, targets, indices, num_boxes, t_weight=None):
        """Binary focal loss over text-token slots.

        Padding slots carry ``-inf`` logits from ``ContrastiveEmbed``; they are
        removed by ``masked_select`` *before* any arithmetic, so no ``inf`` ever
        enters the loss or its gradient.
        """
        pred_logits = outputs["pred_logits"]
        new_targets = outputs["one_hot"].to(pred_logits.device)
        text_mask = outputs["text_mask"]

        assert new_targets.dim() == 3 and pred_logits.dim() == 3, "expected (bs, nq, num_tokens)"
        bs, num_queries, _ = pred_logits.shape

        weight_full = None
        apply_weight = t_weight is not None and self.use_timestep_weighting
        per_image_weight = (
            t_weight.to(pred_logits.device)[:, None, None].expand_as(pred_logits) if apply_weight else None
        )

        if text_mask is not None:
            mask = text_mask[:, None, :].expand(bs, num_queries, text_mask.shape[1])
            if apply_weight:
                weight_full = torch.masked_select(per_image_weight, mask)
            pred_logits = torch.masked_select(pred_logits, mask)
            new_targets = torch.masked_select(new_targets, mask)
        elif apply_weight:
            weight_full = per_image_weight

        new_targets = new_targets.float()
        prob = pred_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, new_targets, reduction="none")
        p_t = prob * new_targets + (1 - prob) * (1 - new_targets)
        loss = ce_loss * ((1 - p_t) ** self.focal_gamma)

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * new_targets + (1 - self.focal_alpha) * (1 - new_targets)
            loss = alpha_t * loss

        if weight_full is not None:
            loss = loss * weight_full

        # Normalized by this rank's positive count, not the global one. That is what
        # upstream does; changing it would shift the classification/box balance and
        # break comparability with the published baseline numbers.
        num_pos = max(sum(len(idx[0]) for idx in indices), 1)
        return {"loss_ce": loss.sum() / num_pos}

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _per_query_weight(self, t_weight: Optional[Tensor], batch_idx: Optional[Tensor]) -> Optional[Tensor]:
        """Expand a per-image ``w(t)`` to the matched queries, or return ``None``."""
        if t_weight is None or not self.use_timestep_weighting:
            return None
        if batch_idx is None:
            return t_weight
        return t_weight.to(batch_idx.device).index_select(0, batch_idx)

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, t_weight=None):
        loss_map = {
            "labels": self.token_sigmoid_binary_focal_loss,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"unknown loss {loss!r}"
        return loss_map[loss](outputs, targets, indices, num_boxes, t_weight=t_weight)

    def _build_label_maps(self, token, cat_list, caption) -> List[Tensor]:
        """Per image, a (num_categories, max_text_len) token mask."""
        label_maps = []
        for j, categories in enumerate(cat_list):
            label_maps.append(
                create_positive_map(
                    token[j], list(range(len(categories))), categories, caption[j], max_text_len=self.max_text_len
                )
            )
        return label_maps

    def _match(self, outputs, targets, label_map_list):
        """Match one image at a time: each image has its own category vocabulary."""
        indices = []
        for j in range(len(targets)):
            single = {
                "pred_logits": outputs["pred_logits"][j].unsqueeze(0),
                "pred_boxes": outputs["pred_boxes"][j].unsqueeze(0),
            }
            indices.extend(self.matcher(single, [targets[j]], label_map_list[j]))
        return indices

    def _build_one_hot(self, shape, indices, targets, label_map_list, device) -> Tensor:
        """Scatter each matched query's category token mask into a (bs, nq, L) target."""
        one_hot = torch.zeros(shape, dtype=torch.float, device=device)
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            labels = targets[i]["labels"].cpu()[tgt_idx.cpu()]
            one_hot[i, src_idx.to(device)] = label_map_list[i][labels].to(device=device, dtype=one_hot.dtype)
        return one_hot

    # ------------------------------------------------------------------ #
    def forward(self, outputs, targets, cat_list, caption, t_weight=None, return_indices=False):
        """
        Args:
            outputs: model output dict; must carry ``token`` and ``text_mask``.
            targets: per image, ``labels`` and ``boxes``.
            cat_list: per image, the category names in prompt order.
            caption: per image, the prompt string.
            t_weight: (bs,) diffusion loss weights, or ``None``.

        Returns:
            Dict of scalar losses (plus the indices list if ``return_indices``).
        """
        device = outputs["pred_logits"].device
        if t_weight is not None and not self.use_timestep_weighting:
            t_weight = None

        label_map_list = self._build_label_maps(outputs["token"], cat_list, caption)

        indices = self._match(outputs, targets, label_map_list)
        outputs["one_hot"] = self._build_one_hot(
            outputs["pred_logits"].shape, indices, targets, label_map_list, device
        )
        indices_list = [] if return_indices else None
        indices0_copy = indices

        # Normalizer shared across all decoder layers, averaged over ranks so that
        # every rank scales its gradients identically.
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, t_weight=t_weight))

        # Auxiliary decoder layers: re-matched, because a different layer's boxes
        # can prefer a different assignment.
        if "aux_outputs" in outputs:
            for idx, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_indices = self._match(aux_outputs, targets, label_map_list)
                aux_outputs["one_hot"] = self._build_one_hot(
                    outputs["pred_logits"].shape, aux_indices, targets, label_map_list, device
                )
                aux_outputs["text_mask"] = outputs["text_mask"]
                if return_indices:
                    indices_list.append(aux_indices)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_boxes, t_weight=t_weight)
                    losses.update({f"{k}_{idx}": v for k, v in l_dict.items()})

        # Encoder proposal branch: no timestep, hence no t_weight.
        if "interm_outputs" in outputs:
            interm_outputs = outputs["interm_outputs"]
            interm_indices = self._match(interm_outputs, targets, label_map_list)
            interm_outputs["one_hot"] = self._build_one_hot(
                outputs["pred_logits"].shape, interm_indices, targets, label_map_list, device
            )
            interm_outputs["text_mask"] = outputs["text_mask"]
            if return_indices:
                indices_list.append(interm_indices)
            for loss in self.losses:
                l_dict = self.get_loss(loss, interm_outputs, targets, interm_indices, num_boxes, t_weight=None)
                losses.update({f"{k}_interm": v for k, v in l_dict.items()})

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list
        return losses


__all__ = ["SetCriterion"]

"""Turn model outputs into COCO-style detections.

The model scores each query against *text tokens*, so producing per-category scores
means projecting those token scores through the category->token map of the eval
prompt. Then the usual DETR trick: take the top-k over the flattened
(query, category) grid, so one query may contribute detections for more than one
category.
"""

from typing import List, Tuple

import torch
from torch import Tensor, nn
from torchvision.ops.boxes import nms

from util.box_ops import box_cxcywh_to_xyxy
from util.vl_utils import build_caption, create_positive_map

# COCO ships 80 classes numbered inside a 91-slot id space; contiguous index -> id.
# Kept for the converters, which need the inverse direction.
COCO_CONTIGUOUS_TO_ID = {
    0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10,
    10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21,
    20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34,
    30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44,
    40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55,
    50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65,
    60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79,
    70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90,
}


class PostProcess(nn.Module):
    """Scores + boxes in absolute pixels, ready for ``CocoEvaluator``.

    Args:
        num_select: how many (query, category) pairs to keep per image.
        nms_iou_threshold: optional class-agnostic NMS; ``<= 0`` disables it.
            Off by default -- set prediction is supposed to make NMS unnecessary,
            and it stays that way with diffusion since the box count is fixed.
    """

    def __init__(self, num_select: int = 100, text_encoder_type: str = "bert-base-uncased", nms_iou_threshold: float = -1, args=None):
        super().__init__()
        from .text import get_tokenizer

        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold
        self.tokenizer = get_tokenizer(text_encoder_type)
        # Must match the model's width: pred_logits is max_text_len wide and the
        # positive map is multiplied against it.
        self.max_text_len = int(getattr(args, "max_text_len", 256))

        cat_list, category_ids = self._eval_categories(args)
        caption = build_caption(cat_list)
        tokenized = self.tokenizer(caption, padding="longest", return_tensors="pt")
        pos_map = create_positive_map(
            tokenized, list(range(len(cat_list))), cat_list, caption, max_text_len=self.max_text_len
        )

        # ``labels`` in the results is read by the evaluator as a dataset category
        # id, so row i of the positive map must be category id i. COCO numbers 80
        # classes inside a 91-slot space, and converted VOC/CrowdHuman json may
        # start at 1 -- re-index whenever the ids are not already 0..N-1.
        if list(category_ids) != list(range(len(cat_list))):
            remapped = torch.zeros((max(category_ids) + 1, pos_map.shape[1]))
            for row, category_id in enumerate(category_ids):
                remapped[category_id] = pos_map[row]
            pos_map = remapped

        # Normalize each row so a multi-token category is not favoured by having
        # more tokens to sum over.
        row_sums = pos_map.sum(-1, keepdim=True)
        pos_map = torch.where(row_sums > 0, pos_map / row_sums.clamp(min=1e-6), pos_map)
        self.register_buffer("positive_map", pos_map, persistent=False)
        self.cat_list = cat_list
        self.category_ids = list(category_ids)

    @staticmethod
    def _eval_categories(args) -> Tuple[List[str], List[int]]:
        """``(names, dataset_category_ids)`` for the evaluation prompt.

        Reading them from the val json keeps names and ids consistent by
        construction; ``label_list`` is the fallback for datasets evaluated without
        pycocotools, where ids are assumed contiguous from 0.
        """
        if getattr(args, "use_coco_eval", False):
            from pycocotools.coco import COCO

            coco = COCO(args.coco_val_path)
            cat_ids = sorted(coco.getCatIds())
            cats = coco.loadCats(cat_ids)
            return [c["name"] for c in cats], [int(c["id"]) for c in cats]

        names = list(args.label_list)
        assert names, "either use_coco_eval must be True or label_list must be non-empty"
        return names, list(range(len(names)))

    @torch.no_grad()
    def forward(self, outputs, target_sizes: Tensor, not_to_xyxy: bool = False, test: bool = False):
        """
        Args:
            outputs: ``pred_logits`` (bs, nq, num_tokens), ``pred_boxes`` (bs, nq, 4).
            target_sizes: (bs, 2) original ``(h, w)`` per image.

        Returns:
            Per image, a dict of ``scores``, ``labels``, ``boxes`` (absolute xyxy).
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        prob_to_label = prob_to_token @ pos_maps.T  # bs, nq, num_categories

        num_select = min(self.num_select, prob_to_label[0].numel())
        topk_values, topk_indexes = torch.topk(prob_to_label.view(prob_to_label.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, prob_to_label.shape[2], rounding_mode="trunc")
        labels = topk_indexes % prob_to_label.shape[2]

        boxes = out_bbox if not_to_xyxy else box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
        if self.nms_iou_threshold > 0:
            kept = [nms(r["boxes"], r["scores"], iou_threshold=self.nms_iou_threshold) for r in results]
            results = [{k: v[i] for k, v in r.items()} for r, i in zip(results, kept)]
        return results


__all__ = ["COCO_CONTIGUOUS_TO_ID", "PostProcess"]

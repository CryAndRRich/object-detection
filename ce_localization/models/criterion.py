"""Loss giống DiffusionDet: 5·L1 + 2·GIoU + 2·Focal, ở MỌI tầng, CỘNG (không chia).

  - L1 + GIoU: chỉ trên cặp đã ghép (box không ghép không có đích toạ độ).
  - Focal: trên cả N slot (slot không ghép có đích rõ ràng: score = 0).
  - Chia cho số cặp đã ghép (clamp >= 1) — như `diffusiondet/loss.py:119,192`.
  - Matcher chạy LẠI ở mỗi tầng (`loss.py:253-255`); loss các tầng CỘNG, mỗi tầng phụ cùng
    trọng số với tầng cuối (`detector.py:148-151`). Chia trung bình = train nhánh box ở lr
    thấp hơn 6 lần. `stats["loss_mean"]` giữ con số chia trung bình chỉ để đọc log.

KHÔNG có epsilon loss: matcher hoán vị nên không epsilon nào thuộc về một cặp cố định.
Head score focal (alpha 0,25) không phân biệt sẽ hội tụ về HẰNG SỐ ⇒ lúc suy luận dùng
top-k, không dùng ngưỡng tuyệt đối.
"""

import torch
import torch.nn.functional as F

from ce_localization.utils.box_ops import (box_iou, cxcywh_to_xyxy,
                                           generalized_box_iou, sanitize_boxes)
from ce_localization.utils.matcher import match

__all__ = ["SetCriterion", "sigmoid_focal_loss"]

W_L1, W_GIOU, W_CLASS = 5.0, 2.0, 2.0
ALPHA, GAMMA = 0.25, 2.0


class SetCriterion:
    def __init__(self, matcher_method="simota", **matcher_kw):
        self.method = matcher_method
        self.matcher_kw = matcher_kw

    @classmethod
    def from_config(cls, cfg):
        m = cfg["matcher"]
        if m["method"] == "simota":
            return cls("simota", use_center_prior=m["use_center_prior"],
                       radius_ratio=m["center_radius"], top_k=m.get("top_k", 10))
        return cls(m["method"])

    def __call__(self, layers, targets):
        """
        layers  : list[(boxes [B,N,4], logits [B,N])], cũ trước
        targets : list[B] tensor [M_i, 4] cxcywh trong [0,1]
        -> (loss, stats, indices của tầng CUỐI)
        """
        if not isinstance(layers, list) or not layers:
            raise TypeError("SetCriterion cần một list [(boxes, logits)] khác rỗng")

        total, per_layer, indices = 0.0, [], None
        for boxes, logits in layers:
            loss, st, indices = self._forward_one(boxes, logits, targets)
            total = total + loss                      # CỘNG, không chia
            per_layer.append(st)

        n = len(per_layer)
        stats = {k: sum(st[k] for st in per_layer) / n for k in per_layer[0]}
        stats["n_layers"] = n
        stats["loss"] = float(total)                  # con số thật đi vào backward
        stats["loss_mean"] = float(total) / n
        for k in ("loss", "iou_matched", "n_matched"):
            stats[f"{k}_per_layer"] = [st[k] for st in per_layer]
        for k in ("loss", "iou_matched", "n_matched", "loss_l1", "loss_giou", "loss_ce"):
            stats[f"{k}_final"] = per_layer[-1][k]
        return total, stats, indices

    def _forward_one(self, pred_boxes, pred_logits, targets):
        """Một tầng. pred_boxes [B,N,4] cxcywh [0,1], pred_logits [B,N]."""
        dev = pred_boxes.device
        l1_all, giou_all, iou_all = [], [], []
        tgt_score = torch.zeros_like(pred_logits)
        indices, n_matched = [], 0

        for i, gt in enumerate(targets):
            if gt.numel() == 0:
                indices.append((torch.zeros(0, dtype=torch.long, device=dev),) * 2)
                continue
            pi, gi = match(pred_boxes[i].detach(), gt, pred_logits[i].detach(),
                           method=self.method, **self.matcher_kw)
            indices.append((pi, gi))
            if len(pi) == 0:
                continue
            n_matched += len(pi)
            tgt_score[i, pi] = 1.0

            p, g = pred_boxes[i][pi], gt[gi]
            l1_all.append(F.l1_loss(p, g, reduction="none").sum(-1))
            p_xyxy = sanitize_boxes(cxcywh_to_xyxy(p))
            g_xyxy = cxcywh_to_xyxy(g)
            giou_all.append(1.0 - torch.diagonal(generalized_box_iou(p_xyxy, g_xyxy)))
            # IoU THẬT (>= 0) để báo cáo, không phải GIoU (âm khi rời nhau).
            iou_all.append(torch.diagonal(box_iou(p_xyxy, g_xyxy)[0]).detach())

        den = max(n_matched, 1)
        zero = pred_boxes.sum() * 0.0
        loss_l1 = torch.cat(l1_all).sum() / den if l1_all else zero
        loss_giou = torch.cat(giou_all).sum() / den if giou_all else zero
        loss_ce = sigmoid_focal_loss(pred_logits, tgt_score).sum() / den

        total = W_L1 * loss_l1 + W_GIOU * loss_giou + W_CLASS * loss_ce
        stats = {
            "loss": float(total),
            "loss_l1": float(loss_l1),
            "loss_giou": float(loss_giou),
            "loss_ce": float(loss_ce),
            "n_matched": n_matched,
            "iou_matched": float(torch.cat(iou_all).mean()) if iou_all else 0.0,
        }
        return total, stats, indices


def sigmoid_focal_loss(logits, targets, alpha=ALPHA, gamma=GAMMA):
    """Sigmoid focal như DiffusionDet (`use_focal=True`). 1 chiều + sigmoid tương đương
    toán học 2 chiều + softmax."""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss

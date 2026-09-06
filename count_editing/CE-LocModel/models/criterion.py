"""Loss IDENTICAL TO DIFFUSIONDET: 5.0 L1 + 2.0 GIoU + 2.0 Focal.

Weights read from `diffusiondet/config.py:34-36,43-44`.

  - L1 + GIoU: ONLY on matched pairs (an unmatched box has no coordinate target).
  - Focal: on ALL N slots (an unmatched slot has a well-defined target: score = 0).
  - Normalised by / num_boxes = the number of matched pairs, clamp(min=1) for
    images with 0 GT.

NO epsilon loss (meaningless under set matching: the matcher permutes, so no
epsilon belongs to prediction p and corresponds to GT g at once -> you would be
training on the noise of padding boxes). NO deep supervision (single-forward
architecture, no intermediate stage to supervise).

WHY BOTH L1 AND GIoU: L1 penalises ABSOLUTE error, so with CE-130 objects (median
0.41 % of image area) it essentially ignores small objects; GIoU penalises
RELATIVE overlap and has a gradient even when two boxes are disjoint.

SCORE-HEAD TRAP: focal with alpha=0.25 and a non-discriminating head converges to
a CONSTANT. Round 1 landed at 0.263 (16 % positive) < the 0.5 threshold -> every
box filtered out -> the argmax fallback kept exactly 1 box ("the viz only draws
one box", once misdiagnosed as a tooling bug). So use TOP-K at inference, never
an absolute threshold.
"""

import torch
import torch.nn.functional as F

from utils.box_ops import box_iou, cxcywh_to_xyxy, generalized_box_iou, sanitize_boxes
from utils.matcher import match

__all__ = ["SetCriterion"]

W_L1, W_GIOU, W_CLASS = 5.0, 2.0, 2.0
ALPHA, GAMMA = 0.25, 2.0


class SetCriterion:
    def __init__(self, matcher_method="hungarian", **matcher_kw):
        self.method = matcher_method
        self.matcher_kw = matcher_kw

    def __call__(self, pred_boxes, pred_logits, targets, labels=None):
        """
        pred_boxes  : [B, N, 4] cxcywh [0,1]
        pred_logits : [B, N]   (A/B, 1-D score) or [B, N, C] (A.2, C-way class head)
        targets     : list[B] tensor [M_i, 4] cxcywh [0,1]
        labels      : list[B] tensor [M_i] long, REQUIRED when pred_logits is 3-D.
                      Class index of each GT box; ignored in the 1-D case.
        -> (total loss, dict of components, list of matched index pairs)

        The multi-class path changes only WHICH target the focal loss is pushed
        towards. Everything else -- matcher, L1, GIoU, the /num_matched
        normalisation -- is shared, so A.1 and A.2 differ by the class pathway and
        nothing else.
        """
        dev = pred_boxes.device
        multiclass = pred_logits.dim() == 3
        if multiclass and labels is None:
            raise ValueError("pred_logits is [B,N,C] but no `labels` were given")

        l1_all, giou_all, iou_all = [], [], []
        # Zeros everywhere = "background": with focal, background is not a class,
        # it is simply every logit being low (DiffusionDet's 80-not-81 convention).
        tgt_score = torch.zeros_like(pred_logits)
        indices, n_matched = [], 0

        for i, gt in enumerate(targets):
            if gt.numel() == 0:
                indices.append((torch.zeros(0, dtype=torch.long, device=dev),) * 2)
                continue

            # The matcher's class cost wants ONE score per proposal. With a C-way
            # head the meaningful score is the box's confidence in the class it is
            # being considered for -- but the matcher is class-agnostic here, so it
            # gets the max over classes, i.e. "how object-like is this proposal".
            # Using the raw [N,C] tensor instead would silently broadcast and
            # produce a cost matrix of the wrong shape.
            m_scores = pred_logits[i].detach()
            if multiclass:
                m_scores = m_scores.max(dim=-1).values

            pi, gi = match(pred_boxes[i].detach(), gt, m_scores,
                           method=self.method, **self.matcher_kw)
            indices.append((pi, gi))
            if len(pi) == 0:
                continue
            n_matched += len(pi)
            if multiclass:
                tgt_score[i, pi, labels[i][gi].to(dev)] = 1.0
            else:
                tgt_score[i, pi] = 1.0

            p, g = pred_boxes[i][pi], gt[gi]
            l1_all.append(F.l1_loss(p, g, reduction="none").sum(-1))

            p_xyxy = sanitize_boxes(cxcywh_to_xyxy(p))
            g_xyxy = cxcywh_to_xyxy(g)
            giou = torch.diagonal(generalized_box_iou(p_xyxy, g_xyxy))
            giou_all.append(1.0 - giou)
            # REAL IoU (>= 0) for reporting, NOT GIoU (which goes negative when
            # boxes are disjoint). A wrong tracking metric is worse than none —
            # this is what tells you whether coordinates are improving, separate
            # from the loss.
            iou_all.append(torch.diagonal(box_iou(p_xyxy, g_xyxy)[0]).detach())

        den = max(n_matched, 1)
        loss_l1 = torch.cat(l1_all).sum() / den if l1_all else pred_boxes.sum() * 0.0
        loss_giou = torch.cat(giou_all).sum() / den if giou_all else pred_boxes.sum() * 0.0

        # SUM over every element then / num_matched, exactly as DiffusionDet does.
        # With a C-way head this sums B*N*C terms rather than B*N, but that is what
        # DiffusionDet itself does with its 80 classes -- the extra terms are all
        # well-classified negatives, which focal already drives towards zero. Any
        # other normalisation would make A.1 and A.2 use different loss scales and
        # break the comparison.
        #
        # EXPECT A HUGE loss_ce AT INITIALISATION and do not "fix" it: with C=80 and
        # every logit at 0, this term is exactly 80x the 1-D value (measured: 520 vs
        # 6.5). Focal collapses it as soon as the negatives are pushed down --
        # measured 520 -> 3.16 at p=0.1 and 0.02 at p=0.02 -- so it falls within the
        # first epoch. What it does mean is that A.2's early gradients are large, so
        # gradient clipping matters more there, and that loss_ce is NOT comparable
        # between A.1 and A.2 in absolute terms. Compare AP and IoU.
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
    """Sigmoid focal loss — same as DiffusionDet (`use_focal=True`).

    1 dim + sigmoid is MATHEMATICALLY EQUIVALENT to 2 dims + softmax (softmax
    depends only on the DIFFERENCE of the two logits -> one redundant degree of
    freedom). DiffusionDet with focal likewise uses 80 dims, NOT 81 —
    "background" simply means every logit is low.
    """
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss

"""Loss + matcher của DiffusionDet, port TỪNG DÒNG từ `refs/repos/DiffusionDet/diffusiondet/loss.py`
(`SetCriterionDynamicK` :228-268, `loss_labels` :91-157, `loss_boxes` :159-206,
`HungarianMatcherDynamicK` :296-451), cho MỘT lớp + focal.

Giữ nguyên hai hành vi của bản gốc (có test):
  - chuẩn hoá mỗi loss bằng SỐ QUERY ĐÃ GHÉP ở GPU này (`num_boxes` all-reduce ở :241-245 bị ghi
    đè trong `loss_labels` :119 và `loss_boxes` :192);
  - trong vòng cứu GT chưa ghép, bước khử trùng dùng lại `anchor_matching_gt` CŨ (:436-441).
Khác bản gốc MỘT chỗ, bắt buộc: vòng cứu (`while`, :429-441) có TRẦN `MAX_RESCUE` lượt. Khi số GT
> số query (CE-130 có ảnh 501 box, N = 200: 30 ảnh train) bản gốc có thể lặp VÔ HẠN: mọi query đã
ghép, mỗi lượt đều +1e5 như nhau nên argmin theo cột không đổi, còn bước khử trùng dùng
`anchor_matching_gt` cũ cứ gỡ lại đúng GT đó (test tái hiện được). Hết trần thì GT còn lại KHÔNG
được ghép — không có đích toạ độ, như ảnh nhiều GT hơn query ở Hungarian. Khi bản gốc tự dừng
(mọi ca có số GT <= số query mà test thử) thì kết quả trùng bản gốc.
KHÔNG dùng lại `ce_localization/models/criterion.py` / `utils/matcher.py`: khác bản gốc
(top_k 10, bán kính theo sqrt(wh), không +10000, L1 trên cxcywh).
"""

import torch
import torch.nn.functional as F

from ce_localization.models.criterion import sigmoid_focal_loss
from ce_localization.utils.box_ops import box_iou, cxcywh_to_xyxy, generalized_box_iou, xyxy_to_cxcywh

__all__ = ["build_targets", "get_in_boxes_info", "dynamic_k_matching", "match", "AlphaCriterion"]


def build_targets(gt_list, whwh):
    """list [M_i,4] xyxy tuyệt đối + whwh [B,4] -> list dict như `prepare_targets` của bản gốc."""
    out = []
    for gt, wh in zip(gt_list, whwh):
        gt = gt.float()
        out.append({
            "labels": torch.zeros(len(gt), dtype=torch.long, device=gt.device),
            "boxes": xyxy_to_cxcywh(gt / wh),            # chuẩn hoá cxcywh
            "boxes_xyxy": gt,                            # tuyệt đối xyxy
            "image_size_xyxy": wh,
            "image_size_xyxy_tgt": wh[None].repeat(len(gt), 1),
        })
    return out


def get_in_boxes_info(boxes_cxcywh, gts_cxcywh, center_radius=2.5):
    """(:376-405) box và GT đều cxcywh TUYỆT ĐỐI. -> (fg_mask [Q], in_box_and_center [Q,G])."""
    gx = cxcywh_to_xyxy(gts_cxcywh)
    cx, cy = boxes_cxcywh[:, 0:1], boxes_cxcywh[:, 1:2]
    in_boxes = (cx > gx[:, 0][None]) & (cx < gx[:, 2][None]) & (cy > gx[:, 1][None]) & (cy < gx[:, 3][None])
    gw, gh = (gx[:, 2] - gx[:, 0])[None], (gx[:, 3] - gx[:, 1])[None]
    in_centers = ((cx > gts_cxcywh[:, 0][None] - center_radius * gw)
                  & (cx < gts_cxcywh[:, 0][None] + center_radius * gw)
                  & (cy > gts_cxcywh[:, 1][None] - center_radius * gh)
                  & (cy < gts_cxcywh[:, 1][None] + center_radius * gh))
    fg = in_boxes.any(1) | in_centers.any(1)
    return fg, in_boxes & in_centers


MAX_RESCUE = 100


def dynamic_k_matching(cost, ious, num_gt, ota_k=5, max_rescue=MAX_RESCUE):
    """(:407-451) -> (selected_query [Q] bool, gt_indices [#selected]). `cost` bị sửa tại chỗ."""
    matching = torch.zeros_like(cost)
    topk_ious, _ = torch.topk(ious, min(ota_k, ious.shape[0]), dim=0)
    dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
    for g in range(num_gt):
        k = min(int(dynamic_ks[g]), cost.shape[0])
        _, pos = torch.topk(cost[:, g], k=k, largest=False)
        matching[:, g][pos] = 1.0
    anchor_matching_gt = matching.sum(1)
    if (anchor_matching_gt > 1).sum() > 0:
        _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
        matching[anchor_matching_gt > 1] *= 0
        matching[anchor_matching_gt > 1, cost_argmin] = 1
    for _ in range(max_rescue):
        if not (matching.sum(0) == 0).any():
            break
        matched_q = matching.sum(1) > 0
        cost[matched_q] += 100000.0
        unmatched = torch.nonzero(matching.sum(0) == 0, as_tuple=False).squeeze(1)
        for g in unmatched:
            matching[:, g][torch.argmin(cost[:, g])] = 1.0
        if (matching.sum(1) > 1).sum() > 0:
            # bản gốc dùng `anchor_matching_gt` CŨ ở đây (:436-441) — giữ nguyên
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching[anchor_matching_gt > 1] *= 0
            matching[anchor_matching_gt > 1, cost_argmin] = 1
    selected = matching.sum(1) > 0
    gt_indices = matching[selected].max(1)[1]
    return selected, gt_indices


@torch.no_grad()
def match(pred_logits, pred_boxes, targets, alpha=0.25, gamma=2.0, w_cls=2.0, w_l1=5.0,
          w_giou=2.0, ota_k=5, center_radius=2.5):
    """SimOTA (:296-374). pred_logits [B,Q,1], pred_boxes [B,Q,4] xyxy tuyệt đối."""
    prob = pred_logits.sigmoid()
    indices = []
    for b, tgt in enumerate(targets):
        if len(tgt["labels"]) == 0:
            indices.append((torch.zeros(prob.shape[1], dtype=torch.bool, device=prob.device),
                            torch.zeros(0, dtype=torch.long, device=prob.device)))
            continue
        boxes, p = pred_boxes[b], prob[b]
        gt_abs = tgt["boxes_xyxy"]
        fg, in_bc = get_in_boxes_info(xyxy_to_cxcywh(boxes), xyxy_to_cxcywh(gt_abs), center_radius)
        ious = box_iou(boxes, gt_abs)[0]
        neg = (1 - alpha) * (p ** gamma) * (-(1 - p + 1e-8).log())
        pos = alpha * ((1 - p) ** gamma) * (-(p + 1e-8).log())
        cost_class = pos[:, tgt["labels"]] - neg[:, tgt["labels"]]
        cost_bbox = torch.cdist(boxes / tgt["image_size_xyxy"], gt_abs / tgt["image_size_xyxy_tgt"], p=1)
        cost_giou = -generalized_box_iou(boxes, gt_abs)
        cost = w_l1 * cost_bbox + w_cls * cost_class + w_giou * cost_giou + 100.0 * (~in_bc)
        cost[~fg] = cost[~fg] + 10000.0
        indices.append(dynamic_k_matching(cost, ious, gt_abs.shape[0], ota_k))
    return indices


class AlphaCriterion:
    """Loss ở MỌI stage (deep supervision, bắt buộc vì box detach giữa các stage), matcher chạy
    lại ở từng stage, trọng số 2·focal + 5·L1 + 2·GIoU, CỘNG các stage."""

    def __init__(self, cfg_loss, cfg_matcher):
        self.alpha, self.gamma = cfg_loss["alpha"], cfg_loss["gamma"]
        self.w_cls, self.w_l1, self.w_giou = (cfg_loss["class_weight"], cfg_loss["l1_weight"],
                                              cfg_loss["giou_weight"])
        self.ota_k, self.radius = cfg_matcher["ota_k"], cfg_matcher["center_radius"]

    def loss_one(self, logits, boxes, targets):
        """Một stage. logits [B,Q,1], boxes [B,Q,4] xyxy tuyệt đối."""
        idx = match(logits, boxes, targets, self.alpha, self.gamma, self.w_cls, self.w_l1,
                    self.w_giou, self.ota_k, self.radius)
        tgt_cls = torch.zeros_like(logits)
        src_b, src_n, tgt_n, tgt_x = [], [], [], []
        for b, ((sel, gi), tgt) in enumerate(zip(idx, targets)):
            if len(gi) == 0:
                continue
            tgt_cls[b, sel, 0] = 1.0
            src_b.append(boxes[b][sel])
            src_n.append(boxes[b][sel] / tgt["image_size_xyxy"])
            tgt_n.append(tgt["boxes"][gi])
            tgt_x.append(tgt["boxes_xyxy"][gi])
        n_matched = sum(len(x) for x in src_b)
        loss_ce = sigmoid_focal_loss(logits.flatten(0, 1), tgt_cls.flatten(0, 1),
                                     self.alpha, self.gamma).sum() / max(n_matched, 1)
        if n_matched:
            sb, sn, tn, tx = (torch.cat(x) for x in (src_b, src_n, tgt_n, tgt_x))
            loss_l1 = F.l1_loss(sn, cxcywh_to_xyxy(tn), reduction="none").sum() / n_matched
            loss_giou = (1 - torch.diag(generalized_box_iou(sb, tx))).sum() / n_matched
            with torch.no_grad():
                iou = torch.diag(box_iou(sb, tx)[0]).mean()
        else:
            loss_l1 = loss_giou = boxes.sum() * 0.0
            iou = torch.zeros((), device=boxes.device)
        total = self.w_cls * loss_ce + self.w_l1 * loss_l1 + self.w_giou * loss_giou
        return total, {"loss_ce": loss_ce.detach(), "loss_bbox": loss_l1.detach(),
                       "loss_giou": loss_giou.detach(), "n_matched": n_matched,
                       "iou_matched": iou}

    def __call__(self, all_logits, all_boxes, targets):
        """all_logits [S,B,Q,1], all_boxes [S,B,Q,4] -> (loss, stats)."""
        total, per = 0.0, []
        for s in range(all_logits.shape[0]):
            loss, st = self.loss_one(all_logits[s], all_boxes[s], targets)
            total = total + loss
            per.append({"loss": loss.detach(), **st})
        stats = {"loss": total.detach(),
                 "loss_per_stage": [p["loss"] for p in per],
                 **{f"{k}_final": per[-1][k] for k in ("loss_ce", "loss_bbox", "loss_giou",
                                                       "n_matched", "iou_matched")}}
        return total, stats

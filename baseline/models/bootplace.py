"""BootPlace (CVPR 2025; tham khảo `refs/BOOTPLACE/models/bootplace.py`). Tên tham số như bản gốc ⇒ nạp thẳng `BOOTPLACE_Cityscapes`.

Ảnh nền -> DETR -> Q vùng đặt vật (cxcywh chuẩn hoá theo vùng ảnh thật) + logit lớp; nhánh box nhân đặc trưng với MỘT vô hướng tính từ
box các vật đang có (`cMLP.loc_proj` trên Q × 4 số, ô thừa = 0). Patch vật cần đặt chỉ vào điểm khớp vật–vùng (`pred_clip`).
Loss Hungarian: CE lớp (no-object trọng số `eos_coef`), L1 + GIoU box (chia `num_boxes` TOÀN batch — truyền vào khi cộng dồn / nhiều
GPU), `clip`; aux loss mọi tầng decoder (trừ clip).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from baseline.models.detr import HungarianMatcher, box_cxcywh_to_xyxy, build_backbone, build_transformer, generalized_box_iou

__all__ = ["NUM_CLASSES", "BOOTPLACE", "SetCriterion", "build_bootplace", "build_criterion"]

NUM_CLASSES = 8                  # lớp Cityscapes của checkpoint công bố; CE-130 không phân lớp ⇒ mọi đích nhãn 0


class cMLP(nn.Module):  # noqa: N801  (tên gốc cho khoá checkpoint)
    def __init__(self, d, out, num_layers, num_queries):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(num_layers - 1)] + [nn.Linear(d, out)])
        self.loc_proj = nn.Sequential(nn.Linear(num_queries * 4, num_queries), nn.ReLU(inplace=True), nn.Linear(num_queries, 1))

    def forward(self, x, loc):
        x = x * self.loc_proj(loc.flatten(1)).view(-1, 1, 1)          # [L,B,Q,D] × [B,1,1]
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class BOOTPLACE(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries):
        super().__init__()
        self.num_queries, self.backbone, self.transformer = num_queries, backbone, transformer
        d = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, d)
        self.input_proj = nn.Conv2d(backbone.num_channels, d, kernel_size=1)
        self.class_embed = nn.Linear(d, num_classes + 1)
        self.bbox_embed = cMLP(d, 4, 3, num_queries)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, x, mask, patches, scene_boxes):
        """x [B,3,H,W] đã chuẩn hoá, mask [B,H,W] (True = đệm), patches [B,4,P·64,64], scene_boxes list [N_i <= Q, 4]."""
        sb = x.new_zeros(len(scene_boxes), self.num_queries, 4)
        for i, b in enumerate(scene_boxes):
            if len(b) > self.num_queries:
                raise ValueError(f"{len(b)} box cảnh > num_queries {self.num_queries} — cắt bớt phía dữ liệu")
            sb[i, :len(b)] = b
        f, m, pos = self.backbone(x, mask)
        hs, pf = self.transformer(self.input_proj(f), m, patches, self.query_embed.weight, pos)
        # cosine (query, patch) × nhiệt độ học được. Bản gốc chuẩn hoá patch theo trục SỐ PATCH (ghi chú của nó tưởng [B,D,P]) — sửa
        # thành theo trục đặc trưng (cosine thật); finetune học lại `logit_scale`
        clip = self.logit_scale.exp() * torch.bmm(F.normalize(hs[-1], dim=-1), F.normalize(pf, dim=-1).transpose(1, 2))   # [B,Q,P]
        cls, box = self.class_embed(hs), self.bbox_embed(hs, sb).sigmoid()
        return {"pred_logits": cls[-1], "pred_boxes": box[-1], "pred_clip": clip,
                "aux_outputs": [{"pred_logits": c, "pred_boxes": b} for c, b in zip(cls[:-1], box[:-1])]}


class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, eos_coef):
        super().__init__()
        self.matcher, self.weight_dict = matcher, weight_dict
        w = torch.ones(NUM_CLASSES + 1)
        w[-1] = eos_coef
        self.register_buffer("empty_weight", w)

    def _losses(self, out, targets, idx, num_boxes):
        src = (torch.cat([torch.full_like(s, i) for i, (s, _) in enumerate(idx)]), torch.cat([s for s, _ in idx]))
        logits = out["pred_logits"]
        tgt_cls = torch.full(logits.shape[:2], NUM_CLASSES, dtype=torch.int64, device=logits.device)
        tgt_cls[src] = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, idx)])
        sb = out["pred_boxes"][src]
        tb = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, idx)])
        giou = torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(sb), box_cxcywh_to_xyxy(tb)))
        return {"loss_ce": F.cross_entropy(logits.transpose(1, 2), tgt_cls, self.empty_weight),
                "loss_bbox": F.l1_loss(sb, tb, reduction="sum") / num_boxes, "loss_giou": (1 - giou).sum() / num_boxes}

    def forward(self, outputs, targets, num_boxes=None):
        """Tính ở fp32. -> (loss box — L1 + GIoU, TỔNG chia `num_boxes` của cả batch; loss TB theo ảnh — CE + clip; thành phần tầng cuối
        chưa nhân trọng số). Cộng dồn micro-batch: loss box cộng thẳng, loss TB nhân tỉ lệ ảnh của micro."""
        num_boxes = num_boxes or max(float(sum(len(t["labels"]) for t in targets)), 1.0)
        w = self.weight_dict
        main = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        parts = self._losses(main, targets, self.matcher(main, targets), num_boxes)
        clip = outputs["pred_clip"]
        parts["loss_clip"] = -(torch.log(1 - clip.softmax(-2) + 1e-6).mean() + torch.log(1 - clip.transpose(1, 2).softmax(-1) + 1e-6).mean()) / 2
        box = w["loss_bbox"] * parts["loss_bbox"] + w["loss_giou"] * parts["loss_giou"]
        mean = w["loss_ce"] * parts["loss_ce"] + w["loss_clip"] * parts["loss_clip"]
        for aux in outputs["aux_outputs"]:
            p = self._losses(aux, targets, self.matcher(aux, targets), num_boxes)
            box = box + w["loss_bbox"] * p["loss_bbox"] + w["loss_giou"] * p["loss_giou"]
            mean = mean + w["loss_ce"] * p["loss_ce"]
        return box, mean, parts


def build_bootplace(cfg):
    """cfg (khoá như argparse bản gốc): backbone, dilation, train_backbone, hidden_dim, dropout, nheads, dim_feedforward, enc_layers,
    dec_layers, num_queries."""
    return BOOTPLACE(build_backbone(cfg), build_transformer(cfg), NUM_CLASSES, cfg["num_queries"])


def build_criterion(loss):
    w = {"loss_ce": loss["ce_loss_coef"], "loss_bbox": loss["bbox_loss_coef"], "loss_giou": loss["giou_loss_coef"],
         "loss_clip": loss["clip_loss_coef"]}
    return SetCriterion(HungarianMatcher(loss["set_cost_class"], loss["set_cost_bbox"], loss["set_cost_giou"]), w, loss["eos_coef"])

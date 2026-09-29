"""Head 6 stage của ALPHA — docs/EXPERIMENT_ALPHA.md mục 2.4.

Stage k (trọng số riêng), nhận box b_k [B,N,4] xyxy tuyệt đối và token q_{k-1}:
  a. RoIAlign(P2..P5, b_k)          -> [B*N, 256, 7, 7]
  b. làm phẳng -> Linear(12544, 256) -> r_k           (1 RoI = 1 token, GIỮ bố cục trong box)
  c. q = LayerNorm(q_{k-1} + r_k)                       (stage 1: q = r_1)
  d-f. nn.TransformerDecoderLayer(norm_first) của DP: self-attn giữa N token -> cross-attn tới
       memory -> FFN. KHÔNG mask causal, KHÔNG pos_emb theo chỉ số slot (proposal là TẬP).
  g. tower cls / reg + head như DiffusionDet (`refs/repos/DiffusionDet/diffusiondet/head.py`
     :205-231, :288-326) -> logit [B,N,1], box = apply_deltas(delta, b_k)
Giữa hai stage: box `.detach()` (head.py:168), token q giữ gradient (như `obj_features`).
KHÔNG có DynamicConv, KHÔNG có FiLM thời gian: t chỉ vào qua token trong memory.
"""

import math

import torch
import torch.nn as nn

from ce_localization.alpha.roi import MultiLevelRoIAlign

__all__ = ["apply_deltas", "AlphaStage", "AlphaHead", "BBOX_WEIGHTS", "SCALE_CLAMP"]

BBOX_WEIGHTS = (2.0, 2.0, 1.0, 1.0)
SCALE_CLAMP = math.log(100000.0 / 16)


def apply_deltas(deltas, boxes, weights=BBOX_WEIGHTS, scale_clamp=SCALE_CLAMP):
    """Y hệt `RCNNHead.apply_deltas`: boxes [K,4] xyxy tuyệt đối, deltas [K,4] -> [K,4] xyxy."""
    boxes = boxes.to(deltas.dtype)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    wx, wy, ww, wh = weights
    dx, dy = deltas[:, 0] / wx, deltas[:, 1] / wy
    dw = (deltas[:, 2] / ww).clamp(max=scale_clamp)
    dh = (deltas[:, 3] / wh).clamp(max=scale_clamp)
    px = dx * widths + ctr_x
    py = dy * heights + ctr_y
    pw = torch.exp(dw) * widths
    ph = torch.exp(dh) * heights
    return torch.stack([px - 0.5 * pw, py - 0.5 * ph, px + 0.5 * pw, py + 0.5 * ph], dim=1)


def _tower(d, n):
    layers = []
    for _ in range(n):
        layers += [nn.Linear(d, d, bias=False), nn.LayerNorm(d), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class AlphaStage(nn.Module):
    def __init__(self, d_model=256, n_head=4, dim_feedforward=1024, dropout=0.3,
                 roi_channels=256, roi_size=7, num_cls=1, num_reg=3, first=False):
        super().__init__()
        self.roi_proj = nn.Linear(roi_channels * roi_size * roi_size, d_model)
        # stage 1: q = r_1, không LayerNorm -> không tạo (DDP / tham số thừa)
        self.norm_in = None if first else nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.cls_tower = _tower(d_model, num_cls)
        self.reg_tower = _tower(d_model, num_reg)
        self.class_logits = nn.Linear(d_model, 1)
        self.bboxes_delta = nn.Linear(d_model, 4)

    def forward(self, feats, boxes, q_prev, memory, mem_mask, pooler):
        """feats list P2..P5 ; boxes [B,N,4] (đã detach) ; q_prev [B,N,d] | None.
        -> (logits [B,N,1], pred_boxes [B,N,4], q [B,N,d])."""
        B, N = boxes.shape[:2]
        roi = pooler(feats, boxes)                                         # [B*N,C,7,7]
        r = self.roi_proj(roi.flatten(1)).view(B, N, -1)
        q = r if self.norm_in is None else self.norm_in(q_prev + r)
        q = self.decoder(q, memory, memory_key_padding_mask=mem_mask)
        logits = self.class_logits(self.cls_tower(q))
        delta = self.bboxes_delta(self.reg_tower(q))
        pred = apply_deltas(delta.reshape(-1, 4), boxes.reshape(-1, 4)).view(B, N, 4)
        return logits, pred, q


class AlphaHead(nn.Module):
    def __init__(self, n_stage=6, d_model=256, n_head=4, dim_feedforward=1024, dropout=0.3,
                 prior_prob=0.01):
        super().__init__()
        self.pooler = MultiLevelRoIAlign(output_size=7, sampling_ratio=2)
        self.stages = nn.ModuleList([
            AlphaStage(d_model, n_head, dim_feedforward, dropout, first=(i == 0))
            for i in range(n_stage)])
        self._init(prior_prob)

    def _init(self, prior_prob):
        """- tầng decoder: `_init_weights` của DP (Linear/MHA normal 0.02, bias 0, LN 1/0);
        - roi_proj, tower, head: xavier_uniform như DiffusionDet (`head.py:112-121`);
        - bias focal −log((1−p)/p) gán TƯỜNG MINH cho `class_logits.bias` — KHÔNG gán theo shape
          như bản gốc (với 1 lớp, cách đó trúng mọi tham số có chiều cuối 1 hoặc 2)."""
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for st in self.stages:
            for m in st.decoder.modules():
                if isinstance(m, nn.MultiheadAttention):
                    nn.init.normal_(m.in_proj_weight, mean=0.0, std=0.02)
                    nn.init.zeros_(m.in_proj_bias)
                elif isinstance(m, nn.Linear):          # gồm cả out_proj của MHA
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            for m in [st.roi_proj, st.class_logits, st.bboxes_delta, *st.cls_tower, *st.reg_tower]:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(st.class_logits.bias, bias_value)

    def forward(self, feats, init_boxes, memory, mem_mask=None):
        """-> (logits [S,B,N,1], boxes [S,B,N,4]) cho MỌI stage (loss ở cả 6 stage)."""
        boxes, q = init_boxes, None
        all_logits, all_boxes = [], []
        for st in self.stages:
            logits, pred, q = st(feats, boxes, q, memory, mem_mask, self.pooler)
            all_logits.append(logits)
            all_boxes.append(pred)
            boxes = pred.detach()
        return torch.stack(all_logits), torch.stack(all_boxes)

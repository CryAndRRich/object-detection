"""Head refine 6 stage kiểu DiffusionDet cho bài ADD (GAMMA2, docs/EXPERIMENT_GAMMA.md mục 13), chép theo
`refs/repos/DiffusionDet/diffusiondet/head.py` (`DynamicHead` :66-162, `RCNNHead` :165-274, `DynamicConv` :303-353).

Mỗi stage (weight riêng), MỘT box mỗi mẫu:
    roi   = RoIAlignV2(P2..P5, b_k) 7×7                       (`MultiLevelRoIAlign` = ROIPooler ROIAlignV2 của detectron2)
    pro   = mean(roi) ở stage 1 | obj_features của stage trước
    pro   = LN(pro + CrossAttn(pro -> memory [t ; text ; vis]))   ← THAY self-attn giữa các box (1 box: vô nghĩa)
    pro   = LN(pro + DynamicConv(pro, roi))                         (y DiffusionDet)
    obj   = LN(pro + FFN(pro))                                      (relu, 2048)
    fc    = obj · (1 + scale) + shift,  (scale, shift) = block_time_mlp(time_mlp(t))   (y DiffusionDet)
    delta = Linear(reg tower 3 × (Linear không bias -> LN -> ReLU)) ; b_{k+1} = apply_deltas(delta, b_k), detach
`geo_hidden` (GAMMA3, `model.geo`): đầu mỗi stage `pro = pro · (1 + γ) + β`, (γ, β) = MLP_stage(geo_features(b_k, vật đang có))
(models/geo.py) — TRƯỚC cross-attn; lớp cuối MLP khởi tạo 0 ⇒ lúc đầu = GAMMA2. `geo=None` khi gọi = tắt (eval `_nogeo`).
Không nhánh class (một lần sinh = một box). Post-norm, dropout 0, d 256, 8 head, FFN 2048, dynamic 64 × 2 như cấu hình mặc
định của DiffusionDet (`diffusiondet/config.py`). Khởi tạo: xavier_uniform mọi tham số ≥ 2 chiều (`_reset_parameters`).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ce_localization.models.geo import GEO_DIM, geo_features
from ce_localization.models.head import apply_deltas
from ce_localization.models.roi import MultiLevelRoIAlign

__all__ = ["SinusoidalPositionEmbeddings", "DynamicConv", "DynamicStage", "DynamicRefineHead"]


class SinusoidalPositionEmbeddings(nn.Module):
    """`SinusoidalPositionEmbeddings` của DiffusionDet (head.py:28-40)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half = self.dim // 2
        e = math.log(10000) / (half - 1)
        e = torch.exp(torch.arange(half, device=time.device) * -e)
        e = time.float()[:, None] * e[None, :]
        return torch.cat((e.sin(), e.cos()), dim=-1)


class DynamicConv(nn.Module):
    """`DynamicConv` của DiffusionDet (head.py:303-353): 2 ma trận sinh từ `pro_features` biến đổi 49 ô của RoI."""

    def __init__(self, hidden_dim=256, dim_dynamic=64, num_dynamic=2, pooler_resolution=7):
        super().__init__()
        self.hidden_dim, self.dim_dynamic, self.num_dynamic = hidden_dim, dim_dynamic, num_dynamic
        self.num_params = hidden_dim * dim_dynamic
        self.dynamic_layer = nn.Linear(hidden_dim, num_dynamic * self.num_params)
        self.norm1 = nn.LayerNorm(dim_dynamic)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU(inplace=True)
        self.out_layer = nn.Linear(hidden_dim * pooler_resolution ** 2, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, pro_features, roi_features):
        """pro_features [1, N, d], roi_features [49, N, d] -> [N, d]."""
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)
        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)
        features = self.activation(self.norm1(torch.bmm(features, param1)))
        features = self.activation(self.norm2(torch.bmm(features, param2)))
        features = self.out_layer(features.flatten(1))
        return self.activation(self.norm3(features))


class DynamicStage(nn.Module):
    """`RCNNHead` của DiffusionDet, MỘT box mỗi mẫu, self-attn -> cross-attn tới memory, bỏ nhánh class."""

    def __init__(self, d_model=256, dim_feedforward=2048, nhead=8, dropout=0.0, dim_dynamic=64, num_dynamic=2,
                 num_reg=3, pooler_resolution=7):
        super().__init__()
        self.d_model = d_model
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.inst_interact = DynamicConv(d_model, dim_dynamic, num_dynamic, pooler_resolution)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout1, self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 2))
        reg = []
        for _ in range(num_reg):
            reg += [nn.Linear(d_model, d_model, False), nn.LayerNorm(d_model), nn.ReLU(inplace=True)]
        self.reg_module = nn.ModuleList(reg)
        self.bboxes_delta = nn.Linear(d_model, 4)

    def forward(self, features, boxes, pro_features, pooler, time_emb, memory, need_weights=False, film=None):
        """boxes [B,K,4] xyxy pixel (detach), pro_features [B·K, d] | None, time_emb [B·K, 4d], memory [B·K, M, d],
        film (γ, β) [B·K, d] | None. -> (box [B,K,4], obj_features [B·K, d], attention [B·K, M] | None)."""
        B, K = boxes.shape[:2]
        N = B * K
        roi = pooler(features, boxes)                                       # [N, d, 7, 7]
        if pro_features is None:
            pro_features = roi.view(N, self.d_model, -1).mean(-1)
        if film is not None:                                                # geo: FiLM trước cross-attn
            pro_features = pro_features * (1 + film[0]) + film[1]
        roi = roi.view(N, self.d_model, -1).permute(2, 0, 1)                # [49, N, d]
        a, w = self.cross_attn(pro_features[:, None], memory, memory, need_weights=need_weights)
        pro = self.norm1(pro_features + self.dropout1(a[:, 0]))
        pro = self.norm2(pro + self.dropout2(self.inst_interact(pro[None], roi)))
        obj = self.norm3(pro + self.dropout3(self.linear2(self.dropout(F.relu(self.linear1(pro))))))
        scale, shift = self.block_time_mlp(time_emb).chunk(2, dim=1)
        reg = obj * (scale + 1) + shift
        for layer in self.reg_module:
            reg = layer(reg)
        pred = apply_deltas(self.bboxes_delta(reg), boxes.reshape(-1, 4)).view(B, K, 4)
        return pred, obj, (None if w is None else w[:, 0])


class DynamicRefineHead(nn.Module):
    """`DynamicHead` của DiffusionDet: time_mlp + 6 `DynamicStage`, box detach giữa các stage, trả box MỌI stage."""

    def __init__(self, n_stage=6, d_model=256, dim_feedforward=2048, nhead=8, dropout=0.0, dim_dynamic=64, num_dynamic=2,
                 num_reg=3, geo_hidden=None):
        super().__init__()
        self.pooler = MultiLevelRoIAlign(output_size=7, sampling_ratio=2)
        self.stages = nn.ModuleList([DynamicStage(d_model, dim_feedforward, nhead, dropout, dim_dynamic, num_dynamic, num_reg)
                                     for _ in range(n_stage)])
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(d_model), nn.Linear(d_model, d_model * 4), nn.GELU(),
                                      nn.Linear(d_model * 4, d_model * 4))
        self.geo_films = (nn.ModuleList([nn.Sequential(nn.Linear(GEO_DIM, geo_hidden), nn.SiLU(),
                                                       nn.Linear(geo_hidden, 2 * d_model)) for _ in range(n_stage)])
                          if geo_hidden else None)
        for p in self.parameters():                                         # `_reset_parameters` của DiffusionDet
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.geo_films is not None:                                      # SAU xavier: bắt đầu đúng bằng không geo
            for f in self.geo_films:
                nn.init.zeros_(f[-1].weight)
                nn.init.zeros_(f[-1].bias)

    def forward(self, features, init_boxes, t, memory, need_weights=False, geo=None):
        """features P2..P5, init_boxes [B,K,4], t [B·K], memory [B·K, M, d], geo (vật [B·K, M, 4], mask [B·K, M]) | None
        -> (box mọi stage [S, B·K, 4], attention [S, B·K, M] | None)."""
        if geo is not None and self.geo_films is None:
            raise ValueError("head không có nhánh geo (geo_hidden None) mà vẫn truyền vật")
        time = self.time_mlp(t)
        boxes, pro, outs, attn = init_boxes, None, [], []
        for si, st in enumerate(self.stages):
            film = None
            if geo is not None:
                film = self.geo_films[si](geo_features(boxes.reshape(-1, 4), *geo)).chunk(2, dim=-1)
            pred, pro, w = st(features, boxes, pro, self.pooler, time, memory, need_weights, film)
            outs.append(pred.flatten(0, 1))
            attn.append(w)
            boxes = pred.detach()
        return torch.stack(outs), (torch.stack(attn) if need_weights else None)

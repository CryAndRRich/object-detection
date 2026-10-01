"""BoxRefiner — bộ sinh MỘT box cho bài ADD (`model.arch: box_refiner`, GAMMA1 — docs/EXPERIMENT_GAMMA.md mục 11).
Giữ phần điều kiện của CE-Loc (SpatialSoftmax trên C5 + text + t), thay U-Net 1D + FiLM bằng 6 tầng RoI + cross-attention
kiểu ALPHA; mỗi tầng ra một box, loss ở MỌI tầng (deep supervision như DiffusionDet).

    ảnh [B,3|4,T,T] -> R-50 + FPN (`models/backbone.py`, MỘT lượt) -> P2..P5 (RoIAlign) + C5 (layer4, 2048 kênh)
    memory [t ; text ; vis] = `MemoryEncoder(kind="spatial_softmax")` của ALPHA1 đặt trên C5:
        vis = SpatialSoftmax có mask vùng thật trên C5 -> Linear(4096, d)            (như GAMMA0)
    1 mẫu = 1 box nhiễu x_t -> b_1 ; tầng k = 1..6 (weight riêng), MỘT query, KHÔNG self-attention:
        r_k = Linear(RoIAlign(P2..P5, b_k))   ;   q_k = r_1 | LN(q_{k-1} + r_k)
        q_k = q_k + CrossAttn(LN(q_k), memory)   ;   q_k = q_k + FFN(LN(q_k))         (pre-norm như DP)
        b_{k+1} = apply_deltas(reg(q_k), b_k), detach trước tầng sau (như DiffusionDet)
    Các mẫu (cùng ảnh hay khác ảnh) không thấy nhau: K mẫu / ảnh = K hàng độc lập.

Khuếch tán như GAMMA0: lịch β tuyến tính 1e-4..0,02, T = 1000; box cxcywh / (nw, nh) · 2 − 1 (= hàm của DiffusionDet
với snr = 1). Khác GAMMA0: dự đoán x0 (box), không ε. Loss = Σ_k [5·L1(b / whwh) + 2·(1 − GIoU)] với lỗ mới nhất
(trọng số loss box của DiffusionDet, mỗi tầng như nhau).
Box nhiễu vào tầng 1: kẹp [−1, 1] rồi w, h >= MIN_WH (tỉ lệ vùng thật). Lý do: snr 1 ⇒ ~16 % box nhiễu ở t lớn có
w hoặc h <= 0 sau kẹp; apply_deltas nhân delta với w = 0 nên box kẹt suy biến qua cả 6 tầng.
Suy luận: DDIM (eta 1 như DiffusionDet) S bước từ nhiễu thuần; mỗi bước x̂0 = box tầng cuối (kẹp [−1, 1]), ε suy lại
từ x̂0; đầu ra = box tầng cuối của bước cuối, KHÔNG kẹp (như DiffusionDet; box suy biến / tràn ảnh do bộ chấm báo).
"""

import torch
import torch.nn as nn

from ce_localization.engine.diffusion import diffusion_from_boxes
from ce_localization.models.backbone import ResNet50FPN
from ce_localization.models.box_policy import boxes_to_unit
from ce_localization.models.head import _tower, apply_deltas
from ce_localization.models.memory import MemoryEncoder
from ce_localization.models.roi import MultiLevelRoIAlign
from ce_localization.utils.box_ops import cxcywh_to_xyxy
from ce_localization.utils.diffusion_math import ddim_time_pairs, linear_alphas_cumprod, predict_noise_from_start

__all__ = ["BoxRefiner", "RefineStage", "RefineHead", "noisy_to_boxes", "paired_giou", "MIN_WH", "MEM_TOKENS"]

C5_CHANNELS = 2048
C5_STRIDE = 32
MIN_WH = 0.02                    # w, h tối thiểu của box nhiễu, tỉ lệ (nw, nh) — ~10 px trên canvas 512
MEM_TOKENS = ("t", "text", "vis")


def noisy_to_boxes(x, whwh):
    """x [...,4] không gian khuếch tán (snr 1) -> xyxy pixel: kẹp [−1, 1], w, h >= MIN_WH."""
    b = (x.clamp(-1, 1) + 1) / 2
    b = torch.cat([b[..., :2], b[..., 2:].clamp(min=MIN_WH)], dim=-1)
    return cxcywh_to_xyxy(b) * whwh


def paired_giou(a, b):
    """GIoU từng cặp [N,4] xyxy -> [N] (box_ops.generalized_box_iou là ma trận N×M)."""
    area = lambda x: (x[:, 2] - x[:, 0]).clamp(min=0) * (x[:, 3] - x[:, 1]).clamp(min=0)  # noqa: E731
    inter = (torch.minimum(a[:, 2:], b[:, 2:]) - torch.maximum(a[:, :2], b[:, :2])).clamp(min=0).prod(-1)
    union = area(a) + area(b) - inter
    iou = inter / union.clamp(min=1e-12)
    enc = (torch.maximum(a[:, 2:], b[:, 2:]) - torch.minimum(a[:, :2], b[:, :2])).clamp(min=0).prod(-1)
    return iou - (enc - union) / enc.clamp(min=1e-12)


class RefineStage(nn.Module):
    """Một tầng: RoI -> token -> cross-attn tới memory -> FFN -> delta box. Phép toán của `nn.TransformerDecoderLayer`
    (norm_first, gelu) BỎ khối self-attention."""

    def __init__(self, d_model=256, n_head=4, dim_feedforward=1024, dropout=0.3, roi_channels=256, roi_size=7,
                 num_reg=3, first=False):
        super().__init__()
        self.roi_proj = nn.Linear(roi_channels * roi_size * roi_size, d_model)
        self.norm_in = None if first else nn.LayerNorm(d_model)
        self.norm_ca = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.drop_ca = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, d_model))
        self.drop_ff = nn.Dropout(dropout)
        self.reg_tower = _tower(d_model, num_reg)
        self.bboxes_delta = nn.Linear(d_model, 4)

    def forward(self, feats, boxes, q_prev, memory, pooler, need_weights=False):
        """boxes [B,K,4] xyxy (đã detach), memory [B·K, M, d], q_prev [B·K, d] | None.
        -> (box [B,K,4], q [B·K, d], attention [B·K, M] (TB các head) | None)."""
        B, K = boxes.shape[:2]
        r = self.roi_proj(pooler(feats, boxes).flatten(1))
        q = r if self.norm_in is None else self.norm_in(q_prev + r)
        h = self.norm_ca(q)[:, None]
        a, w = self.cross_attn(h, memory, memory, need_weights=need_weights)
        q = q + self.drop_ca(a[:, 0])
        q = q + self.drop_ff(self.ffn(self.norm_ff(q)))
        delta = self.bboxes_delta(self.reg_tower(q))
        pred = apply_deltas(delta, boxes.reshape(-1, 4)).view(B, K, 4)
        return pred, q, (None if w is None else w[:, 0])


class RefineHead(nn.Module):
    def __init__(self, n_stage=6, d_model=256, n_head=4, dim_feedforward=1024, dropout=0.3):
        super().__init__()
        self.pooler = MultiLevelRoIAlign(output_size=7, sampling_ratio=2)
        self.stages = nn.ModuleList([RefineStage(d_model, n_head, dim_feedforward, dropout, first=(i == 0))
                                     for i in range(n_stage)])
        self._init()

    def _init(self):
        """Như `DecoderHead._init` của ALPHA: khối attention / FFN theo DP (normal 0,02, bias 0, LN 1/0);
        roi_proj, tower, delta xavier như DiffusionDet. Không có nhánh score."""
        for st in self.stages:
            for m in [st.cross_attn, st.ffn, st.norm_ca, st.norm_ff, *([st.norm_in] if st.norm_in else [])]:
                for mm in m.modules():
                    if isinstance(mm, nn.MultiheadAttention):
                        nn.init.normal_(mm.in_proj_weight, mean=0.0, std=0.02)
                        nn.init.zeros_(mm.in_proj_bias)
                    elif isinstance(mm, nn.Linear):          # gồm cả out_proj của MHA
                        nn.init.normal_(mm.weight, mean=0.0, std=0.02)
                        nn.init.zeros_(mm.bias)
                    elif isinstance(mm, nn.LayerNorm):
                        nn.init.ones_(mm.weight)
                        nn.init.zeros_(mm.bias)
            for m in [st.roi_proj, st.bboxes_delta, *st.reg_tower]:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

    def forward(self, feats, init_boxes, memory, need_weights=False):
        """-> (box mọi tầng [S,B,K,4], attention [S, B·K, M] | None)."""
        boxes, q, outs, attn = init_boxes, None, [], []
        for st in self.stages:
            pred, q, w = st(feats, boxes, q, memory, self.pooler, need_weights)
            outs.append(pred)
            attn.append(w)
            boxes = pred.detach()
        return torch.stack(outs), (torch.stack(attn) if need_weights else None)


class BoxRefiner(nn.Module):
    def __init__(self, in_channels=3, pretrained_backbone=True, d_model=256, n_stage=6, n_head=4, dim_feedforward=1024,
                 dropout=0.3, text_dim=512, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, l1_weight=5.0,
                 giou_weight=2.0):
        super().__init__()
        self.backbone = ResNet50FPN(d_model, pretrained=pretrained_backbone, in_channels=in_channels)
        self.memory = MemoryEncoder("spatial_softmax", d_model, text_dim, feat_channels=C5_CHANNELS,
                                    feat_stride=C5_STRIDE)
        self.head = RefineHead(n_stage, d_model, n_head, dim_feedforward, dropout)
        self.num_timesteps = num_timesteps
        self.l1_weight, self.giou_weight = l1_weight, giou_weight
        self.register_buffer("alphas_cumprod", linear_alphas_cumprod(num_timesteps, beta_start, beta_end),
                             persistent=False)
        self.track_attn = False                 # bật khi eval: cộng dồn attention lên [t ; text ; vis] mỗi tầng
        self._attn_sum, self._attn_n = None, 0

    def encode_image(self, images, valid_hw):
        """-> (P2..P5, token vis [B,1,d]). Không phụ thuộc t: MỘT lần mỗi ảnh."""
        f, c5 = self.backbone.forward_with_c5(images)
        vis, _ = self.memory.image_tokens(c5, valid_hw)
        return [f["p2"], f["p3"], f["p4"], f["p5"]], vis

    def refine(self, feats, vis, text_raw, t, boxes, need_weights=False):
        """boxes [B,K,4] xyxy, t [B·K] -> (box mọi tầng [S, B·K, 4], attention [S, B·K, 3] | None)."""
        K = boxes.shape[1]
        mem, _ = self.memory(t, text_raw.repeat_interleave(K, 0), vis.repeat_interleave(K, 0))
        out, attn = self.head(feats, boxes, mem, need_weights)
        return out.flatten(1, 2), attn

    def forward(self, images, text_raw, valid_hw, target, whwh, k=1, generator=None):
        """target [B,4] xyxy pixel (lỗ mới nhất), whwh [B,4]; k bộ (t, ε) độc lập mỗi ảnh.
        -> (loss vô hướng, {"loss", "loss_per_stage" [S]})."""
        feats, vis = self.encode_image(images, valid_hw)
        B, dev = images.shape[0], images.device
        wk = whwh.repeat_interleave(k, 0)
        gt = target.repeat_interleave(k, 0)
        x0 = diffusion_from_boxes(gt, wk, 1.0)
        t = torch.randint(0, self.num_timesteps, (B * k,), device=dev, generator=generator)
        noise = torch.randn(x0.shape, device=dev, generator=generator)
        ab = self.alphas_cumprod[t].unsqueeze(-1)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        preds, _ = self.refine(feats, vis, text_raw, t, noisy_to_boxes(xt, wk).view(B, k, 4))
        per = torch.stack([self.l1_weight * ((p - gt) / wk).abs().sum(-1).mean()
                           + self.giou_weight * (1 - paired_giou(p, gt)).mean() for p in preds])
        loss = per.sum()
        return loss, {"loss": loss.detach(), "loss_per_stage": per.detach()}

    @torch.no_grad()
    def sample(self, images, text_raw, valid_hw, n_samples, generator=None, steps=1, eta=1.0, return_stages=False):
        """n_samples mẫu ĐỘC LẬP mỗi ảnh, DDIM `steps` bước. -> [B, n_samples, 4] trong [−1, 1] (cùng quy ước
        `box_policy.unit_to_boxes`); `return_stages` -> (box, box mọi tầng ở MỖI bước: list [S,B,K,4] xyxy pixel)."""
        feats, vis = self.encode_image(images, valid_hw)
        B, K, dev = images.shape[0], n_samples, images.device
        whwh = torch.stack([valid_hw[:, 1], valid_hw[:, 0], valid_hw[:, 1], valid_hw[:, 0]], 1).float()
        wk = whwh.repeat_interleave(K, 0)
        ac = self.alphas_cumprod
        x = torch.randn((B * K, 4), device=dev, generator=generator)
        stages = []
        for t, t_next in ddim_time_pairs(self.num_timesteps, steps):
            tb = torch.full((B * K,), t, device=dev, dtype=torch.long)
            preds, attn = self.refine(feats, vis, text_raw, tb, noisy_to_boxes(x, wk).view(B, K, 4), self.track_attn)
            if attn is not None:
                s = attn.float().sum(1)                                          # [S, 3]
                self._attn_sum = s if self._attn_sum is None else self._attn_sum + s
                self._attn_n += attn.shape[1]
            if return_stages:
                stages.append(preds.view(-1, B, K, 4).cpu())
            if t_next < 0:                       # đầu ra = box tầng cuối, không kẹp (như DiffusionDet)
                x = boxes_to_unit(preds[-1], wk)
                break
            x0 = diffusion_from_boxes(preds[-1], wk, 1.0)    # kẹp [−1, 1] chỉ để suy ε (như DiffusionDet)
            eps = predict_noise_from_start(x, t, x0, ac)
            a, a_next = ac[t], ac[t_next]
            sigma = eta * ((1 - a / a_next) * (1 - a_next) / (1 - a)).sqrt()
            c = (1 - a_next - sigma ** 2).sqrt()
            x = x0 * a_next.sqrt() + c * eps + sigma * torch.randn(x.shape, device=dev, generator=generator)
        out = x.view(B, K, 4)
        return (out, stages) if return_stages else out

    def pop_attn(self):
        """-> {tầng: {t, text, vis}} attention TB từ lần pop trước (None nếu chưa ghi), rồi xoá."""
        if self._attn_sum is None:
            return None
        m = (self._attn_sum / max(self._attn_n, 1)).tolist()
        self._attn_sum, self._attn_n = None, 0
        return [dict(zip(MEM_TOKENS, row)) for row in m]

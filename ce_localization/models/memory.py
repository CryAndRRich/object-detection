"""Memory mà proposal cross-attend tới — kiểu Diffusion Policy
(`refs/repos/diffusion_policy/diffusion_policy/model/diffusion/transformer_for_diffusion.py`
:44-80 dựng, :300-313 forward):

    tokens = [time_emb ; Linear(obs) ...] + cond_pos_emb  ->  encoder MLP từng token
    (n_cond_layers = 0: Linear(d,4d) -> Mish -> Linear(4d,d); p_drop_emb = 0 như config hybrid)

Ba biến thể (docs/EXPERIMENT_ALPHA.md mục 3), chọn bằng `model.memory`:
    none            [t ; text]                                   ALPHA0
    spatial_softmax [t ; text ; 1 token SpatialSoftmax(P5)]      ALPHA1
    grid            [t ; text ; mỗi ô P5 thật là 1 token + PE2D] ALPHA2
                    `grid_size=G`: cắt ĐÚNG vùng ảnh thật trên P5 rồi adaptive_avg_pool về G×G
                    -> luôn G² token, không cần mask (canvas 1024: P5 32×32 -> 16×16)

Chỉ tạo module mà biến thể dùng: DDP chạy `find_unused_parameters=False`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MEMORY_KINDS", "SinusoidalTimeEmbedding", "valid_cells_mask", "masked_spatial_softmax", "sine_pos_2d",
           "MemoryEncoder"]

MEMORY_KINDS = ("none", "spatial_softmax", "grid")


class SinusoidalTimeEmbedding(nn.Module):
    """Time embedding DDPM chuẩn = `SinusoidalPosEmb` của Diffusion Policy."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = math.log(10000) / (half - 1)
        f = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -f)
        a = t.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)


def valid_cells_mask(valid_hw, H, W, stride):
    """[B,2] (nh, nw) pixel -> [B,H,W] bool, True = ô thuộc vùng ảnh thật.

    Một hàng/cột là thật nếu mép trên/trái của nó nằm trong vùng thật: số hàng thật =
    ceil(nh / stride). Ô cuối có thể chỉ phủ một phần vùng thật — vẫn tính là thật.
    """
    rows = torch.arange(H, device=valid_hw.device)[None, :] * stride < valid_hw[:, :1]   # [B,H]
    cols = torch.arange(W, device=valid_hw.device)[None, :] * stride < valid_hw[:, 1:2]  # [B,W]
    return rows[:, :, None] & cols[:, None, :]


def masked_spatial_softmax(feat, valid, valid_hw, stride):
    """SpatialSoftmax như CE-Loc gốc (không temperature, không conv 1x1) nhưng CHE ô đệm.

    feat [B,C,H,W], valid [B,H,W] bool, valid_hw [B,2] (nh, nw) pixel.
    -> [B, C, 2] toạ độ kỳ vọng (x, y) — ghi TƯỜNG MINH x trước y (bản gốc dùng meshgrid 'ij'
       nên toạ độ đầu là DỌC). Toạ độ = tâm ô / kích thước vùng thật, cùng quy ước với box
       chuẩn hoá theo `images_whwh`.
    """
    B, C, H, W = feat.shape
    logits = feat.flatten(2).float()                                         # [B,C,HW]
    logits = logits.masked_fill(~valid.flatten(1)[:, None, :], float("-inf"))
    att = torch.softmax(logits, dim=-1)
    dev = feat.device
    cx = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5) * stride    # tâm ô, pixel
    cy = (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) * stride
    gx = (cx[None, None, :] / valid_hw[:, 1, None, None].float()).expand(B, H, W).flatten(1)
    gy = (cy[None, :, None] / valid_hw[:, 0, None, None].float()).expand(B, H, W).flatten(1)
    ex = (att * gx[:, None, :]).sum(-1)
    ey = (att * gy[:, None, :]).sum(-1)
    return torch.stack([ex, ey], dim=-1).to(feat.dtype)


def sine_pos_2d(valid, num_pos_feats=128, temperature=10000.0):
    """PE sin/cos 2D kiểu DETR (`PositionEmbeddingSine`, normalize=True, scale 2π), chuẩn hoá
    theo VÙNG THẬT: cumsum chỉ trên ô thật, chia giá trị cuối. valid [B,H,W] -> [B,2F,H,W]."""
    v = valid.float()
    y = v.cumsum(1)
    x = v.cumsum(2)
    eps, scale = 1e-6, 2 * math.pi
    y = y / (y[:, -1:, :].amax(dim=2, keepdim=True).clamp(min=1) + eps) * scale
    x = x / (x[:, :, -1:].amax(dim=1, keepdim=True).clamp(min=1) + eps) * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=valid.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    px = x[..., None] / dim_t
    py = y[..., None] / dim_t
    px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), dim=4).flatten(3)
    py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), dim=4).flatten(3)
    return torch.cat((py, px), dim=3).permute(0, 3, 1, 2)


class MemoryEncoder(nn.Module):
    """Dựng memory [B, M, d] + key_padding_mask [B, M] (True = CHE) hoặc None."""

    def __init__(self, kind="none", d_model=256, text_dim=512, feat_channels=256, feat_stride=32,
                 grid_size=None):
        super().__init__()
        if kind not in MEMORY_KINDS:
            raise ValueError(f"model.memory={kind!r}, phải là một trong {MEMORY_KINDS}")
        self.kind, self.d_model, self.stride = kind, d_model, feat_stride
        self.grid_size = grid_size
        self.time_emb = SinusoidalTimeEmbedding(d_model)          # thô, như DP: không MLP riêng
        self.text_proj = nn.Linear(text_dim, d_model)
        n_fixed = 3 if kind == "spatial_softmax" else 2            # [t ; text (; ss)]
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, n_fixed, d_model))
        if kind == "spatial_softmax":
            self.ss_proj = nn.Linear(2 * feat_channels, d_model)
        if kind == "grid":
            self.grid_proj = nn.Linear(feat_channels, d_model)
        self.encoder = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.Mish(),
                                     nn.Linear(4 * d_model, d_model))
        self._init_dp()

    def _init_dp(self):
        """`TransformerForDiffusion._init_weights`: Linear normal(0, 0.02), bias 0; cond_pos_emb
        normal(0, 0.02)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cond_pos_emb, mean=0.0, std=0.02)

    def image_tokens(self, p5, valid_hw):
        """Phần ảnh của memory — tính MỘT lần mỗi ảnh (không phụ thuộc t).
        -> (tokens [B,K,d] hoặc None, mask [B,K] hoặc None)."""
        if self.kind == "none":
            return None, None
        B, C, H, W = p5.shape
        valid = valid_cells_mask(valid_hw, H, W, self.stride)
        if self.kind == "spatial_softmax":
            xy = masked_spatial_softmax(p5, valid, valid_hw, self.stride)          # [B,C,2]
            tok = self.ss_proj(xy.flatten(1))[:, None] + self.cond_pos_emb[:, 2:3]
            return tok, None
        if self.grid_size:
            G = self.grid_size
            rows, cols = valid.any(2).sum(1), valid.any(1).sum(1)                  # số hàng / cột thật
            p5 = torch.cat([F.adaptive_avg_pool2d(p5[b:b + 1, :, :int(rows[b]), :int(cols[b])], G)
                            for b in range(B)])                                     # [B,C,G,G]
            valid = torch.ones(B, G, G, dtype=torch.bool, device=p5.device)
        pos = sine_pos_2d(valid, self.d_model // 2)                                # [B,d,H,W]
        tok = self.grid_proj(p5.flatten(2).transpose(1, 2)) + pos.flatten(2).transpose(1, 2)
        return tok, (None if self.grid_size else ~valid.flatten(1))

    def grid_geometry(self, valid_hw, H, W):
        """Tâm (pixel canvas) + mask thật của từng token lưới -> (cx [B,K], cy [B,K], valid [B,K]).
        Dùng cho chẩn đoán attention; khớp cách `image_tokens` dựng lưới (có / không pool)."""
        dev = valid_hw.device
        if self.grid_size:
            G = self.grid_size
            f = (torch.arange(G, device=dev, dtype=torch.float32) + 0.5) / G
            nh, nw = valid_hw[:, 0].float(), valid_hw[:, 1].float()
            rows = torch.ceil(nh / self.stride) * self.stride                      # vùng pool phủ
            cols = torch.ceil(nw / self.stride) * self.stride
            cy = (f[None, :, None] * rows[:, None, None]).expand(-1, G, G)
            cx = (f[None, None, :] * cols[:, None, None]).expand(-1, G, G)
            return cx.flatten(1), cy.flatten(1), torch.ones(len(nh), G * G, dtype=torch.bool, device=dev)
        c_y = (torch.arange(H, device=dev) + 0.5) * self.stride
        c_x = (torch.arange(W, device=dev) + 0.5) * self.stride
        cy, cx = torch.meshgrid(c_y, c_x, indexing="ij")
        B = valid_hw.shape[0]
        valid = valid_cells_mask(valid_hw, H, W, self.stride).flatten(1)
        return cx.flatten()[None].expand(B, -1), cy.flatten()[None].expand(B, -1), valid

    def forward(self, t, text_raw, img_tok=None, img_mask=None):
        """t [B] long, text_raw [B,512] -> (memory [B,M,d], key_padding_mask [B,M] | None)."""
        fixed = torch.stack([self.time_emb(t), self.text_proj(text_raw)], dim=1)
        fixed = fixed + self.cond_pos_emb[:, :2]
        toks = fixed if img_tok is None else torch.cat([fixed, img_tok], dim=1)
        mem = self.encoder(toks)
        if img_mask is None:
            return mem, None
        pad = torch.zeros(img_mask.shape[0], 2, dtype=torch.bool, device=img_mask.device)
        return mem, torch.cat([pad, img_mask], dim=1)

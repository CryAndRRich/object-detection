"""`ClipBoxRefiner` — BoxRefiner của bản CE-Loc cập nhật tác giả gửi (`refs/CE-Loc-update/models/box_refiner.py`, read-only; viết lại
ĐÚNG công thức, giữ tên module / tham số của tác giả). Nằm TRONG mạng khử nhiễu, gọi ở mọi bước, trước U-Net 1D.

    một lần mỗi ảnh (`encode_context`): rgb canvas -> bicubic về độ phân giải ViT -> CLIP ViT FROZEN -> hidden_states[-2] bỏ CLS
        -> LN + Linear (+ Linear(density pool về lưới g×g)) = token ảnh / map ảnh; token chữ CLIP text (tính sẵn) -> LN + Linear
    mỗi lần gọi, x_t [R,4]: 9 token box = RoIAlign(map ảnh, x_t) 3×3 + BoxFourierEmb(x_t) + type emb ; RoPE 2D
        `num_layers` RefinerBlock (AdaLN-Zero theo t) trên [text | image | sample], mask nhóm `attention_flow` -> head -> refined = x_t + Δ

Khác bản tác giả ở CÁCH tính, không ở toán (có test so bản sao nguyên văn): luồng ngữ cảnh [text | image] không nhìn token box ⇒ tính
MỘT lần mỗi (ảnh, t) rồi R box attend K / V đã cache (`ctx_index` chọn ảnh); CLIP ViT không là module con (không vào state_dict /
optimizer / DDP; `BoxPolicy.clip_sha` kiểm vân tay khi nạp). Phần train được tắt autocast (fp32).
"""

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from ce_localization.models.unet1d import SinusoidalPosEmb
from ce_localization.utils.box_coder import to_linear_unit

__all__ = ["GROUPS", "CLIP_MEAN", "CLIP_STD", "DEFAULT_ATTENTION_FLOW", "REFINER_DEFAULTS", "apply_rope_2d", "rope_cos_sin",
           "rope_rotate", "BoxFourierEmb", "RefinerBlock", "ClipBoxRefiner", "load_clip", "weights_sha256", "refiner_config",
           "text_model_name"]

GROUPS = ('text', 'image', 'sample')
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Which groups each query group may attend to (query -> keys)
DEFAULT_ATTENTION_FLOW = {
    'sample': ['sample', 'image', 'text'],
    'image': ['image', 'text'],
    'text': ['text', 'image'],
}

# Mặc định trong CODE của tác giả; `text_model_name` (thêm): CLIP text cho token chữ, None = `clip_model_name`; `aux_giou_weight` (bản
# log-scale): loss phụ = aux_loss_weight · [MSE(refined, x0) + aux_giou_weight · (1 − GIoU)].
REFINER_DEFAULTS = {
    'clip_model_name': 'openai/clip-vit-base-patch16', 'text_model_name': None, 'vision_layer': -2, 'd_model': 256,
    'num_layers': 4, 'num_heads': 8, 'mlp_ratio': 4.0, 'dropout': 0.0, 'rope_theta': 100.0, 'roi_size': 3, 'num_stages': 1,
    'use_density': True, 'unet_input': 'concat', 'aux_loss_weight': 0.0, 'aux_giou_weight': 0.0,
    'attention_flow': DEFAULT_ATTENTION_FLOW,
}


def refiner_config(refiner):
    """Khối config `model.refiner` -> dict đủ khoá (mặc định của tác giả) nếu bật, None nếu không."""
    return {**REFINER_DEFAULTS, **refiner} if refiner and refiner.get('enabled') else None


def text_model_name(rcfg):
    """Tên CLIP text sinh token chữ của refiner."""
    return rcfg.get('text_model_name') or rcfg['clip_model_name']


def weights_sha256(module):
    """sha256 (32 byte) của mọi tensor trong state_dict (tên + byte float32, theo thứ tự), bỏ buffer `position_ids` (có / không
    tuỳ phiên bản transformers). Vân tay để kiểm CLIP frozen nạp lại từ HF đúng bản đã train."""
    h = hashlib.sha256()
    for name, t in module.state_dict().items():
        if name.endswith('position_ids'):
            continue
        h.update(name.encode())
        h.update(t.detach().to('cpu', torch.float32).contiguous().numpy().tobytes())
    return h.digest()


def load_clip(vision_name, text_name):
    """-> (CLIPVisionModel `vision_name` FROZEN ở eval mode, hidden size của CLIP text `text_name`). Test thay hàm này (không tải gì)."""
    from transformers import CLIPTextConfig, CLIPVisionModel
    vision = CLIPVisionModel.from_pretrained(vision_name)
    vision.eval().requires_grad_(False)
    return vision, CLIPTextConfig.from_pretrained(text_name).hidden_size


def rope_cos_sin(pos, dh, theta, dtype):
    """cos / sin [B, 1, N, Dh/2] của `apply_rope_2d` — tách ra để q, k và mọi block cùng vị trí dùng chung (cùng phép tính)."""
    quarter = dh // 4
    freqs = theta ** (-torch.arange(quarter, device=pos.device, dtype=torch.float32) / quarter)
    angles = (pos.float()[..., None] * freqs).flatten(-2)  # [B, N, Dh/2]
    return angles.cos()[:, None].to(dtype), angles.sin()[:, None].to(dtype)


def rope_rotate(x, cos_sin):
    cos, sin = cos_sin
    dh = x.shape[-1]
    x1, x2 = x[..., :dh // 2], x[..., dh // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def apply_rope_2d(x, pos, theta):
    """
    Axial 2D RoPE with continuous positions.
    x: [B, heads, N, Dh], pos: [B, N, 2] as (x, y) in patch units.
    A quarter of the rotation pairs use x, the other quarter use y.
    """
    return rope_rotate(x, rope_cos_sin(pos, x.shape[-1], theta, x.dtype))


class BoxFourierEmb(nn.Module):
    """Embeds a [B, 4] box with multi-frequency sin/cos features."""
    def __init__(self, out_dim, num_freqs=8):
        super().__init__()
        self.register_buffer('freqs', (2.0 ** torch.arange(num_freqs)) * math.pi, persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(4 * 2 * num_freqs + 4, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, box):
        x = box[..., None] * self.freqs                      # [B, 4, F]
        feats = torch.cat([x.sin(), x.cos()], dim=-1).flatten(1)
        return self.proj(torch.cat([feats, box], dim=-1))


class RefinerBlock(nn.Module):
    """Pre-norm transformer block with RoPE attention and AdaLN-Zero timestep modulation."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, rope_theta=100.0):
        super().__init__()
        assert dim % num_heads == 0 and (dim // num_heads) % 4 == 0, \
            "head_dim must be divisible by 4 for 2D RoPE"
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim),
        )
        # AdaLN-Zero: shift/scale/gate for attn and mlp, zero-init so the block starts as identity
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    # ---------- `forward` của tác giả tách thành các mảnh cho đường tách đôi (cùng phép tính; test so với bản nguyên văn)
    def modulation(self, t_emb):
        """-> (shift1, scale1, gate1, shift2, scale2, gate2), mỗi cái [B, 1, D]."""
        return self.ada(t_emb)[:, None].chunk(6, dim=-1)

    def qkv_rope(self, x, cos_sin, shift1, scale1):
        """-> q, k, v [B, heads, N, Dh] (q, k đã quay RoPE); cos_sin = `rope_cos_sin` của vị trí token (tính một lần, dùng chung)."""
        B, N, D = x.shape
        h = self.norm1(x) * (1 + scale1) + shift1
        q, k, v = self.qkv(h).view(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        return rope_rotate(q, cos_sin), rope_rotate(k, cos_sin), v

    def attend(self, q, k, v, attn_mask):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                              dropout_p=self.dropout if self.training else 0.0)

    def finish(self, x, h, gate1, shift2, scale2, gate2):
        """Phần sau attention: x [B, N, D], h [B, heads, N, Dh]."""
        B, N, D = x.shape
        x = x + gate1 * self.proj(h.transpose(1, 2).reshape(B, N, D))
        h = self.norm2(x) * (1 + scale2) + shift2
        return x + gate2 * self.mlp(h)


class ClipBoxRefiner(nn.Module):
    """
    Refines a (noisy) box sample by letting its visual tokens attend to image/text context.

    1. Frozen CLIP ViT -> image patch tokens (once per image); CLIP text tokens precomputed per class name.
    2. The sample box is RoIAligned on the image token map -> roi_size^2 sample tokens.
    3. Transformer over [text | image | sample] with 2D RoPE, group-level attention masks
       and AdaLN-Zero timestep conditioning.
    4. MLP maps refined sample tokens to a box delta: refined = sample + delta (shape [B, 4]).
    Optionally repeated for `num_stages` (re-RoIAlign around the updated box each stage).
    `vision_model` = CLIPVisionModel frozen (của `load_clip`), `text_hidden` = cỡ token chữ CLIP.
    """
    def __init__(self, cfg, vision_model, text_hidden):
        super().__init__()
        cfg = {**REFINER_DEFAULTS, **(cfg or {})}
        self.box_coder = None                     # `BoxPolicy` đặt khi `model.box_coder` là log (thuộc tính thường, không vào state_dict)
        dim = cfg['d_model']
        num_heads = cfg['num_heads']
        num_layers = cfg['num_layers']
        mlp_ratio = cfg['mlp_ratio']
        dropout = cfg['dropout']
        rope_theta = cfg['rope_theta']
        self.vision_layer = cfg['vision_layer']
        self.roi_size = cfg['roi_size']
        self.num_stages = cfg['num_stages']
        self.use_density = cfg['use_density']

        # 1. Frozen CLIP ViT — KHÔNG đăng ký module con (ngoài state_dict / parameters / DDP); `_apply` kéo theo thiết bị
        vision_model.eval().requires_grad_(False)
        object.__setattr__(self, 'vision_backbone', vision_model)
        vcfg = vision_model.config
        self.clip_res = vcfg.image_size
        self.grid = vcfg.image_size // vcfg.patch_size
        self.register_buffer('clip_mean', torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('clip_std', torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

        # 2. Token projections / embeddings
        self.image_proj = nn.Sequential(nn.LayerNorm(vcfg.hidden_size), nn.Linear(vcfg.hidden_size, dim))
        self.text_proj = nn.Sequential(nn.LayerNorm(text_hidden), nn.Linear(text_hidden, dim))
        if self.use_density:
            self.density_proj = nn.Linear(1, dim)
        self.type_emb = nn.Parameter(torch.randn(len(GROUPS), dim) * 0.02)
        self.box_emb = BoxFourierEmb(dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim),
        )

        # 3. Transformer stages + 4. box heads
        n_tok = self.roi_size ** 2
        self.stages = nn.ModuleList([
            nn.ModuleList([RefinerBlock(dim, num_heads, mlp_ratio, dropout, rope_theta)
                           for _ in range(num_layers)])
            for _ in range(self.num_stages)
        ])
        self.heads = nn.ModuleList()
        for _ in range(self.num_stages):
            head = nn.Sequential(
                nn.LayerNorm(dim * n_tok), nn.Linear(dim * n_tok, dim), nn.GELU(), nn.Linear(dim, 4),
            )
            # Zero-init: refiner starts as identity (refined == sample)
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.heads.append(head)

        self.set_attention_flow(cfg['attention_flow'])

    def _apply(self, fn, *args, **kwargs):
        """`.to(dev)` / `.double()` / channels_last: kéo theo CLIP (không phải module con)."""
        super()._apply(fn, *args, **kwargs)
        self.vision_backbone._apply(fn, *args, **kwargs)
        return self

    def set_attention_flow(self, flow):
        """flow: {query_group: [key_groups]} (not saved in state_dict)."""
        allow = torch.zeros(len(GROUPS), len(GROUPS), dtype=torch.bool)
        for q, keys in flow.items():
            if q not in GROUPS or any(k not in GROUPS for k in keys):
                raise ValueError(f"attention_flow groups must be in {GROUPS}, got {q}: {keys}")
            for k in keys:
                allow[GROUPS.index(q), GROUPS.index(k)] = True
        for i, g in enumerate(GROUPS):
            if not allow[i].any():
                raise ValueError(f"attention_flow: group '{g}' must attend to at least one group")
        device = self.type_emb.device
        # token ảnh / chữ không nhìn token box ⇒ luồng ngữ cảnh không phụ thuộc box ⇒ tính một lần mỗi (ảnh, t) (đường duy nhất)
        s = GROUPS.index('sample')
        if allow[GROUPS.index('text'), s] or allow[GROUPS.index('image'), s]:
            raise ValueError("attention_flow: token ảnh / chữ nhìn token box chưa hỗ trợ (đường tách đôi cần luồng ngữ cảnh "
                             "không phụ thuộc box)")
        self.register_buffer('group_allow', allow.to(device), persistent=False)

    @torch.no_grad()
    def clip_features(self, rgb):
        """rgb [B,3,T,T] in [0, 1] -> token CLIP ViT frozen [B, g*g, C] (hidden_states[vision_layer], bỏ CLS). Theo autocast ngoài (fp16
        khi AMP). Ảnh train không augment ⇒ tính sẵn được (`data/clip_cache.py`)."""
        dt = self.type_emb.dtype
        pixels = F.interpolate(rgb.to(dt).contiguous(), size=(self.clip_res, self.clip_res), mode='bicubic',
                               align_corners=False, antialias=True)
        pixels = (pixels.clamp(0, 1) - self.clip_mean) / self.clip_std
        vis_out = self.vision_backbone(pixel_values=pixels, output_hidden_states=True)
        return vis_out.hidden_states[self.vision_layer][:, 1:]  # drop CLS -> [B, g*g, C]

    def encode_context(self, rgb, density, text_tokens, text_mask, img_feats=None):
        """Timestep-independent context, compute once per image. rgb [B,3,T,T] / density [B,1,T,T] in [0, 1];
        text_tokens [B, L, text_hidden] (`last_hidden_state` CLIP text, đệm phải), text_mask [B, L] bool (True = token thật);
        img_feats: `clip_features(rgb)` tính sẵn (cache train) | None = tính."""
        dt = self.type_emb.dtype                                              # fp32 (train / AMP); float64 trong test
        if img_feats is None:
            img_feats = self.clip_features(rgb)
        B, g = rgb.shape[0], self.grid
        with torch.autocast(device_type=rgb.device.type, enabled=False):     # phần train được: fp32
            img_tok = self.image_proj(img_feats.to(dt))
            if self.use_density:
                dens = F.adaptive_avg_pool2d(density.to(dt), g).flatten(1)[..., None]  # [B, g*g, 1]
                img_tok = img_tok + self.density_proj(dens)
            img_map = img_tok.transpose(1, 2).reshape(B, -1, g, g)               # for RoIAlign
            return {
                'image_tokens': img_tok,
                'image_map': img_map,
                'text_tokens': self.text_proj(text_tokens.to(dt)),
                'text_mask': text_mask.bool(),
            }

    @staticmethod
    def index_context(ctx, index):
        """Chọn hàng ngữ cảnh theo `index` [R] (nhân bản cho từng mẫu — `expand_context` của tác giả, tổng quát)."""
        return {k: v.index_select(0, index) for k, v in ctx.items()}

    def _sample_tokens(self, box, img_map, roi_index=None):
        """RoIAlign the box on the image token map. box: [R, 4] (cx, cy, w, h) mã hoá theo `self.box_coder` (None / linear: [-1, 1];
        log — DELTA5: giải mã về linear trước, như `BoxCoder.decode(box, g)` của bản log-scale); roi_index [R]: map ảnh của từng box
        (None = box i ↔ ảnh i, như tác giả)."""
        box = to_linear_unit(box, self.box_coder)
        B, g, k = box.shape[0], self.grid, self.roi_size
        # [-1, 1] -> patch-grid units; clamp because noisy samples can be anywhere
        cx = ((box[:, 0] + 1) / 2 * g).clamp(0, g)
        cy = ((box[:, 1] + 1) / 2 * g).clamp(0, g)
        w = ((box[:, 2] + 1) / 2 * g).clamp(0.5, g)
        h = ((box[:, 3] + 1) / 2 * g).clamp(0.5, g)
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

        idx = torch.arange(B, device=box.device) if roi_index is None else roi_index
        rois = torch.stack([idx.to(box.dtype), x1, y1, x2, y2], dim=1)
        feats = roi_align(img_map.contiguous(), rois, output_size=k, spatial_scale=1.0,
                          sampling_ratio=2, aligned=True)                 # [B, D, k, k]
        tokens = feats.flatten(2).transpose(1, 2)                        # [B, k*k, D] (y-major)

        # Continuous positions of each RoI cell center, same order as tokens
        steps = (torch.arange(k, device=box.device, dtype=box.dtype) + 0.5) / k
        xs = x1[:, None] + steps * (x2 - x1)[:, None]
        ys = y1[:, None] + steps * (y2 - y1)[:, None]
        pos = torch.stack([xs[:, None, :].expand(B, k, k), ys[:, :, None].expand(B, k, k)], dim=-1)
        return tokens, pos.reshape(B, k * k, 2)

    def _attn_mask(self, text_mask, n_img, n_samp):
        B, L = text_mask.shape
        device = text_mask.device
        gid = torch.cat([
            torch.full((L,), 0, device=device),
            torch.full((n_img,), 1, device=device),
            torch.full((n_samp,), 2, device=device),
        ])
        flow = self.group_allow[gid][:, gid]                               # [N, N]
        key_valid = torch.cat([text_mask, text_mask.new_ones(B, n_img + n_samp)], dim=1)
        return (flow[None] & key_valid[:, None, :])[:, None]               # [B, 1, N, N]

    def _ctx_tokens(self, ctx, dtype):
        """-> (token ngữ cảnh [C, L + n_img, D], vị trí [C, L + n_img, 2], L, n_img)."""
        C, g = ctx['image_tokens'].shape[0], self.grid
        device = ctx['image_tokens'].device
        text_tok = ctx['text_tokens'] + self.type_emb[0]
        img_tok = ctx['image_tokens'] + self.type_emb[1]
        L, n_img = text_tok.shape[1], img_tok.shape[1]
        # Image patch centers in grid units; text has no location, so place it at the canvas center
        coords = torch.arange(g, device=device, dtype=dtype) + 0.5
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        img_pos = torch.stack([xx, yy], dim=-1).reshape(1, n_img, 2).expand(C, -1, -1)
        text_pos = torch.full((C, L, 2), g / 2, device=device, dtype=dtype)
        return torch.cat([text_tok, img_tok], dim=1), torch.cat([text_pos, img_pos], dim=1), L, n_img

    def forward(self, sample, timestep, ctx, ctx_index=None):
        """sample [R, 4] noisy box, timestep [R], ctx của `encode_context` (C hàng), ctx_index [R] long: hàng ngữ cảnh của từng mẫu
        (None = mẫu i ↔ ngữ cảnh i). Mọi mẫu cùng hàng ngữ cảnh PHẢI cùng t (vòng mock / DDPM: đúng; train k nhiễu / ảnh: nhân
        ngữ cảnh bằng `index_context` rồi gọi với ctx_index None). Returns [R, 4]. Tắt autocast (fp32)."""
        with torch.autocast(device_type=sample.device.type, enabled=False):
            return self._forward_split(sample.to(self.type_emb.dtype), timestep, ctx, ctx_index)

    def _forward_split(self, sample, timestep, ctx, ctx_index):
        R = sample.shape[0]
        C = ctx['image_tokens'].shape[0]
        idx = torch.arange(R, device=sample.device) if ctx_index is None else ctx_index
        # t của từng hàng ngữ cảnh (mọi mẫu cùng hàng cùng t — xem docstring)
        t_ctx = timestep.new_zeros(C).scatter(0, idx, timestep)
        t_emb_c = self.time_emb(t_ctx)

        x_ctx0, pos_c, L, n_img = self._ctx_tokens(ctx, sample.dtype)
        n_samp = self.roi_size ** 2
        full = self._attn_mask(ctx['text_mask'], n_img, n_samp)               # [C, 1, N, N], thứ tự khoá [text | image | sample]
        Nc = L + n_img
        mask_c = full[:, :, :Nc, :Nc]
        mask_s = full[:, :, Nc:, :].index_select(0, idx)                      # [R, 1, n_samp, N]

        blk0 = self.stages[0][0]
        rope = lambda pos: rope_cos_sin(pos, blk0.qkv.out_features // 3 // blk0.num_heads, blk0.rope_theta, sample.dtype)  # noqa: E731
        rope_c = rope(pos_c)                                                  # vị trí như nhau ở mọi block / stage
        box = sample
        for blocks, head in zip(self.stages, self.heads):
            # Detach the box used to build tokens (DETR-style iterative refinement)
            box_in = box.detach()
            samp_tok, samp_pos = self._sample_tokens(box_in, ctx['image_map'], idx)
            rope_s = rope(samp_pos)
            xs = samp_tok + self.box_emb(box_in)[:, None] + self.type_emb[2]
            xc = x_ctx0                                                       # tác giả dựng lại chuỗi từ token ban đầu mỗi stage
            last = len(blocks) - 1
            for li, blk in enumerate(blocks):
                mc = blk.modulation(t_emb_c)                                  # ngữ cảnh: [C, 1, D] × 6
                ms = [m.index_select(0, idx) for m in mc]                     # mẫu: cùng t với hàng ngữ cảnh của nó
                qc, kc, vc = blk.qkv_rope(xc, rope_c, mc[0], mc[1])
                qs, ks, vs = blk.qkv_rope(xs, rope_s, ms[0], ms[1])
                k_all = torch.cat([kc.index_select(0, idx), ks], dim=2)
                v_all = torch.cat([vc.index_select(0, idx), vs], dim=2)
                xs_next = blk.finish(xs, blk.attend(qs, k_all, v_all, mask_s), ms[2], ms[3], ms[4], ms[5])
                if li < last:                                                 # block cuối: head chỉ đọc token box
                    xc = blk.finish(xc, blk.attend(qc, kc, vc, mask_c), mc[2], mc[3], mc[4], mc[5])
                xs = xs_next

            # Keep only the refined sample tokens and map back to box space
            box = box + head(xs.flatten(1))
        return box

    @torch.no_grad()
    def gate_stats(self, ts=(0, 250, 500, 750, 999)):
        """Chẩn đoán (log train): TB |gate| attention / MLP của AdaLN mỗi block (TB trên các t trong `ts`) và ‖weight‖ lớp cuối head.
        Cả hai = 0 lúc khởi tạo; còn ≈ 0 sau train = refiner không làm gì."""
        dev = self.type_emb.device
        t_emb = self.time_emb(torch.tensor(ts, device=dev))
        out = {}
        for si, blocks in enumerate(self.stages):
            for bi, blk in enumerate(blocks):
                m = blk.modulation(t_emb)
                out[f"s{si}b{bi}"] = [float(m[2].abs().mean()), float(m[5].abs().mean())]
            out[f"head{si}"] = float(self.heads[si][-1].weight.norm())
        return out

"""U-Net 1D có điều kiện của Diffusion Policy, đúng bản CE-Loc gốc (`refs/repos/Count-Editing/CE-LocModel/models/noise_pred_net.py`):
`down_dims (64, 128, 256)`, kernel 3, GroupNorm 8, embedding t 256-d, FiLM chỉ cộng bias, horizon 1 (một box). cond = `[emb(t) ;
global_cond]` vào mọi khối residual. `output_dim` (bản cập nhật của tác giả): vào [x_t ; refined] 8 kênh, ra ε 4 kênh.

Box attention (GAMMA4): box nhiễu cross-attend tới box CÁC VẬT ĐANG CÓ ở 6 vị trí (sau cặp ResBlock mỗi tầng: down × 3, mid, up × 2).
Token vật mã hoá GƯƠNG (`encode_objects`): box vật đi qua CHÍNH các ResBlock đó với t = 0, lấy feature ở 6 vị trí. `ObjCrossAttn`:
`h + out_proj(MHA(LN(h) -> LN(token)))`, out_proj khởi tạo 0, tính fp32. Lấy mẫu (`ObjKV`): K / V của token vật tính MỘT lần mỗi ảnh
cho mọi bước và mọi mẫu (cùng phép toán, không nhân token theo số mẫu).
"""

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SinusoidalPosEmb", "ConditionalResidualBlock1D", "ObjCrossAttn", "ObjKV", "ConditionalUnet1D"]


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x):
        out = self.conv(x)
        if x.shape[-1] == 1 and out.shape[-1] == 2:          # horizon = 1: cắt về 1 như bản gốc
            return out[..., :1]
        return out


class _Unsqueeze(nn.Module):
    """Thay `Rearrange('batch t -> batch t 1')` của bản gốc (không tham số)."""

    def forward(self, x):
        return x.unsqueeze(-1)


class ConditionalResidualBlock1D(nn.Module):
    """Nhánh `cond_predict_scale=False` của bản gốc: FiLM chỉ cộng bias."""

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels), _Unsqueeze())
        self.residual_conv = (nn.Conv1d(in_channels, out_channels, 1)
                              if in_channels != out_channels else nn.Identity())

    def forward(self, x, cond):
        out = self.blocks[0](x) + self.cond_encoder(cond)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ObjCrossAttn(nn.Module):
    """Một vị trí: h [N, d, 1] (box nhiễu) attend token vật [N, M, d] (mask [N, M], True = có vật). Ảnh 0 vật: cộng 0."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.ln_q, self.ln_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, h, tok, mask):
        with torch.autocast(device_type=h.device.type, enabled=False):
            q = self.ln_q(h[..., 0].float())[:, None]
            kv = self.ln_kv(tok.float())
            none = ~mask.any(1)
            ok = mask | (none[:, None] & (torch.arange(mask.shape[1], device=mask.device) == 0))   # tránh softmax toàn −inf
            a = self.attn(q, kv, kv, key_padding_mask=~ok, need_weights=False)[0][:, 0]
            a = a * (~none).float()[:, None]
        return h + a.to(h.dtype)[..., None]

    def kv(self, tok):
        """Token vật [N, M, d] -> (K, V) [N, heads, M, dh] fp32 — phần chiếu K / V của `attn` (không đổi qua các bước lấy mẫu)."""
        with torch.autocast(device_type=tok.device.type, enabled=False):
            N, M, d = tok.shape
            H = self.attn.num_heads
            w, b = self.attn.in_proj_weight, self.attn.in_proj_bias
            kv = F.linear(self.ln_kv(tok.float()), w[d:], b[d:]).view(N, M, 2, H, d // H).permute(2, 0, 3, 1, 4)
            return kv[0], kv[1]

    def forward_kv(self, h, kv, mask, n):
        """= `forward` với token mỗi ảnh nhân `n` lần (hàng i·n .. i·n + n − 1 thuộc ảnh i), dùng K / V của `kv`: h [N·n, d, 1]."""
        with torch.autocast(device_type=h.device.type, enabled=False):
            k, v = kv
            N, H, _, dh = k.shape
            d = H * dh
            q = F.linear(self.ln_q(h[..., 0].float()), self.attn.in_proj_weight[:d], self.attn.in_proj_bias[:d])
            q = q.view(N, n, H, dh).transpose(1, 2)                                              # [N, H, n, dh]
            none = ~mask.any(1)
            ok = mask | (none[:, None] & (torch.arange(mask.shape[1], device=mask.device) == 0))
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=ok[:, None, None, :])          # [N, H, n, dh]
            a = self.attn.out_proj(a.transpose(1, 2).reshape(N, n, d)) * (~none).float()[:, None, None]
        return h + a.reshape(N * n, d).to(h.dtype)[..., None]


class ObjKV(NamedTuple):
    """Token vật đã chiếu K / V (`ConditionalUnet1D.object_kv`) cho N ảnh, mỗi ảnh `n` hàng box liền nhau."""
    kv: list
    mask: torch.Tensor
    n: int


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=256, down_dims=(64, 128, 256), kernel_size=3, n_groups=8,
                 obj_heads=4, obj_bucket=512, output_dim=None):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        cond_dim = dsed + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        def block(i, o):
            return ConditionalResidualBlock1D(i, o, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups)

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([block(mid_dim, mid_dim), block(mid_dim, mid_dim)])
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                block(dim_in, dim_out), block(dim_out, dim_out),
                Downsample1d(dim_out) if not is_last else nn.Identity()]))
        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)              # như bản gốc: không bao giờ True
            self.up_modules.append(nn.ModuleList([
                block(dim_out * 2, dim_in), block(dim_in, dim_in),
                Upsample1d(dim_in) if not is_last else nn.Identity()]))
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim if output_dim is None else output_dim, 1),
        )
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed))
        self.obj_bucket = obj_bucket                         # luồng vật: số hàng làm tròn lên bội số này (xem `encode_objects`)
        site_dims = list(down_dims) + [mid_dim] + [d_in for d_in, _ in reversed(in_out[1:])]
        self.obj_xattn = nn.ModuleList([ObjCrossAttn(d, obj_heads) for d in site_dims])

    def _trunk(self, x, g, site):
        """Thân U-Net (down / mid / up, chưa final conv); `site(i, x) -> x` gọi ở 6 vị trí (sau cặp ResBlock mỗi tầng)."""
        h, i = [], 0
        for resnet, resnet2, downsample in self.down_modules:
            x = site(i, resnet2(resnet(x, g), g))
            i += 1
            h.append(x)
            x = downsample(x)
        for mid in self.mid_modules:
            x = mid(x, g)
        x = site(i, x)
        i += 1
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = upsample(site(i, resnet2(resnet(x, g), g)))
            i += 1
        return x

    def encode_objects(self, objs, mask, global_cond, index=None):
        """Mã hoá GƯƠNG box vật: objs [N, M, C] (cùng chuẩn hoá với đầu vào U-Net), mask [N, M], global_cond [N, G] -> list 6 token
        [N, M, d_ℓ] (ô đệm = 0). Chỉ tính hàng có vật; số hàng làm tròn lên bội `obj_bucket` bằng hàng đệm (Conv1d dài 1 + GroupNorm
        tính độc lập từng hàng ⇒ hàng thật không đổi) để `cudnn.benchmark` chỉ dò thuật toán cho vài cỡ (Kaggle T4: 1,56 -> 0,33 s / iter).
        `index` = `mask.nonzero(as_tuple=True)` tính sẵn (nonzero trên GPU ép CPU chờ GPU)."""
        N, M = mask.shape
        bi, mi = mask.nonzero(as_tuple=True) if index is None else index
        if bi.numel() == 0:                                  # cả batch không có vật: token 0 (XAttn tự cộng 0)
            return [objs.new_zeros(N, M, x.ln_q.normalized_shape[0]) for x in self.obj_xattn]
        t0 = self.diffusion_step_encoder(torch.zeros(N, dtype=torch.long, device=objs.device))
        g = torch.cat([t0, global_cond], dim=-1)[bi]
        x = objs[bi, mi]
        R = x.shape[0]
        pad = -R % self.obj_bucket if self.obj_bucket and self.obj_bucket > 1 else 0
        if pad:
            x = torch.cat([x, x.new_zeros(pad, x.shape[1])])
            g = torch.cat([g, g[:1].expand(pad, -1)])
        feats = []

        def collect(_, x_):
            feats.append(x_[:R, :, 0])
            return x_
        self._trunk(x[..., None], g, collect)
        return [f.new_zeros(N, M, f.shape[-1]).index_put((bi, mi), f) for f in feats]

    def object_kv(self, obj_tokens, obj_mask, n):
        """Token vật của N ảnh (`encode_objects`) -> `ObjKV` cho N·n hàng (n mẫu / ảnh): K / V chiếu một lần."""
        return ObjKV([x.kv(t) for x, t in zip(self.obj_xattn, obj_tokens)], obj_mask, n)

    def forward(self, sample, timestep, global_cond, obj_tokens=None, obj_mask=None):
        """sample [B, 1, C] (horizon 1), timestep [B] long, global_cond [B, G] -> [B, 1, out]. obj_tokens (list 6 của `encode_objects`)
        + obj_mask [B, M]: cross-attend box vật; `ObjKV` (lấy mẫu, obj_mask bỏ qua); None = tắt (eval `_noobj`)."""
        x = sample.permute(0, 2, 1)                          # b h t -> b t h
        timesteps = timestep.expand(sample.shape[0])
        g = torch.cat([self.diffusion_step_encoder(timesteps), global_cond], dim=-1)

        def site(i, x_):
            if obj_tokens is None:
                return x_
            if isinstance(obj_tokens, ObjKV):
                return self.obj_xattn[i].forward_kv(x_, obj_tokens.kv[i], obj_tokens.mask, obj_tokens.n)
            return self.obj_xattn[i](x_, obj_tokens[i], obj_mask)
        x = self._trunk(x, g, site)
        return self.final_conv(x).permute(0, 2, 1)

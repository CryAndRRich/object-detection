"""U-Net 1D có điều kiện của Diffusion Policy, đúng bản CE-Loc gốc dùng
(`refs/repos/Count-Editing/CE-LocModel/models/{components,noise_pred_net}.py`; viết lại từ bản
`celoc_paper/celoc_model.py` đã nạp strict được checkpoint của bài, xoá 2026-10-01).

Cấu hình CE-Loc gốc: `down_dims (64, 128, 256)`, kernel 3, GroupNorm 8, embedding bước khuếch tán 256-d,
FiLM CHỈ cộng bias (`cond_predict_scale=False`). Horizon = 1 (một box 4 số): `Upsample1d` cắt đầu ra dài 2
về 1 như bản gốc. cond = `[emb(t) ; global_cond]` vào MỌI khối residual.
"""

import math

import torch
import torch.nn as nn

__all__ = ["SinusoidalPosEmb", "ConditionalResidualBlock1D", "ConditionalUnet1D"]


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


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=256,
                 down_dims=(64, 128, 256), kernel_size=3, n_groups=8):
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
            nn.Conv1d(start_dim, input_dim, 1),
        )
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed))

    def forward(self, sample, timestep, global_cond):
        """sample [B, 1, 4] (horizon 1), timestep [B] long, global_cond [B, G] -> [B, 1, 4]."""
        x = sample.permute(0, 2, 1)                          # b h t -> b t h
        timesteps = timestep.expand(sample.shape[0])
        g = torch.cat([self.diffusion_step_encoder(timesteps), global_cond], dim=-1)
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet2(resnet(x, g), g)
            h.append(x)
            x = downsample(x)
        for mid in self.mid_modules:
            x = mid(x, g)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = upsample(resnet2(resnet(x, g), g))
        return self.final_conv(x).permute(0, 2, 1)

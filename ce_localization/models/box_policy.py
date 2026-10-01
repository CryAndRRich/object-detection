"""BoxPolicy — bộ sinh MỘT box kiểu CE-Loc gốc / Diffusion Policy cho bài ADD (docs/EXPERIMENT_GAMMA.md).

    ảnh [B,3|4,T,T] -> R-50 + FPN của ALPHA (`models/backbone.py`) -> P5 [B,256,T/32,T/32]
        -> SpatialSoftmax có mask vùng thật (`models/memory.masked_spatial_softmax`, như token của ALPHA1)
        -> 256 điểm (x, y) -> Linear(512, vis_dim)                                   ┐
    text: CLIP pooler 512 (bảng `models/text.TextTable`) -> Linear(512, text_dim) -> Mish ┤ cond [B, vis+text]
    box nhiễu v_t [B,1,4] + t -> U-Net 1D (`models/unet1d.py`), FiLM cộng bias theo cond -> ε̂

Khác CE-Loc gốc ĐÚNG ở phần ảnh (người dùng chốt 2026-10-01): ResNet18 (BN train, `to_tensor`, SpatialSoftmax
512 kênh C5) -> R-50 + FPN của ALPHA (FrozenBN, chuẩn hoá ImageNet, SpatialSoftmax 256 kênh P5 có mask vùng
thật, density kênh 4 khởi tạo 0). Text, U-Net 1D, lịch β tuyến tính 1e-4..0,02, ε-MSE giữ như bài.

Box trong không gian khuếch tán: cxcywh chia (nw, nh) của VÙNG ẢNH THẬT (cùng quy ước `whwh` của ALPHA) rồi
`·2 − 1` -> [−1, 1], cả w, h (CE-Loc gốc chia canvas 512 — khác chỗ phần đệm, không đổi bản chất).
Train: mỗi ảnh rút `noise_per_image` bộ (t, ε) độc lập — backbone chỉ tính MỘT lần mỗi ảnh, U-Net rất rẻ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ce_localization.models.backbone import ResNet50FPN
from ce_localization.models.memory import masked_spatial_softmax, valid_cells_mask
from ce_localization.models.unet1d import ConditionalUnet1D
from ce_localization.utils.diffusion_math import linear_alphas_cumprod

__all__ = ["BoxPolicy", "boxes_to_unit", "unit_to_boxes", "ddpm_sample"]

P5_STRIDE = 32


def boxes_to_unit(boxes_xyxy, whwh):
    """xyxy pixel canvas [...,4] + whwh [...,4] (nw, nh, nw, nh) -> cxcywh/(nw,nh) ·2 − 1."""
    b = boxes_xyxy / whwh
    c = torch.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2,
                     b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], dim=-1)
    return c * 2 - 1


def unit_to_boxes(u, whwh):
    """Ngược `boxes_to_unit`: [...,4] trong [−1,1] -> xyxy pixel canvas. KHÔNG kẹp (w, h âm giữ nguyên để
    đo tỉ lệ box suy biến; IoU tự kẹp w, h về 0)."""
    c = (u + 1) / 2
    xyxy = torch.stack([c[..., 0] - c[..., 2] / 2, c[..., 1] - c[..., 3] / 2,
                        c[..., 0] + c[..., 2] / 2, c[..., 1] + c[..., 3] / 2], dim=-1)
    return xyxy * whwh


@torch.no_grad()
def ddpm_sample(eps_fn, n_rows, alphas_cumprod, generator=None, device="cpu"):
    """DDPM tổ tiên đúng công thức (Ho et al. 2020), T bước, phương sai β̃_t, không kẹp — bản `sample_ddpm`
    đúng của CE-Loc gốc viết lại (bài dùng vòng "mock" `x -= eps/100`, không phải DDPM).
    eps_fn(x [R,4], t [R] long) -> ε̂ [R,4]. -> x0 [R,4]."""
    ab = alphas_cumprod.to(device)
    ab_prev = torch.cat([ab.new_ones(1), ab[:-1]])
    alphas = ab / ab_prev
    betas = 1 - alphas
    x = torch.randn((n_rows, 4), device=device, generator=generator)
    for t in reversed(range(ab.shape[0])):
        tb = torch.full((n_rows,), t, device=device, dtype=torch.long)
        eps = eps_fn(x, tb)
        mean = (x - betas[t] / torch.sqrt(1 - ab[t]) * eps) / torch.sqrt(alphas[t])
        if t > 0:
            var = betas[t] * (1 - ab_prev[t]) / (1 - ab[t])
            x = mean + torch.sqrt(var) * torch.randn(x.shape, device=device, generator=generator)
        else:
            x = mean
    return x


class BoxPolicy(nn.Module):
    def __init__(self, in_channels=3, pretrained_backbone=True, fpn_dim=256, vis_dim=128, text_in=512,
                 text_dim=128, step_embed_dim=256, down_dims=(64, 128, 256), kernel_size=3, n_groups=8,
                 num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.backbone = ResNet50FPN(fpn_dim, pretrained=pretrained_backbone, in_channels=in_channels)
        self.vis_proj = nn.Linear(2 * fpn_dim, vis_dim)
        self.text_proj = nn.Sequential(nn.Linear(text_in, text_dim), nn.Mish())
        self.noise_net = ConditionalUnet1D(4, vis_dim + text_dim, step_embed_dim, tuple(down_dims),
                                           kernel_size, n_groups)
        self.num_timesteps = num_timesteps
        self.register_buffer("alphas_cumprod", linear_alphas_cumprod(num_timesteps, beta_start, beta_end),
                             persistent=False)

    def condition(self, images, text_raw, valid_hw):
        """-> cond [B, vis_dim + text_dim]. Phần ảnh không phụ thuộc t: tính MỘT lần mỗi ảnh."""
        p5 = self.backbone.forward_p5(images)
        valid = valid_cells_mask(valid_hw, p5.shape[2], p5.shape[3], P5_STRIDE)
        xy = masked_spatial_softmax(p5, valid, valid_hw, P5_STRIDE)                # [B,256,2] (x, y)
        return torch.cat([self.vis_proj(xy.flatten(1)), self.text_proj(text_raw)], dim=-1)

    def forward(self, images, text_raw, valid_hw, x0, k=1, generator=None):
        """ε-MSE như `ObjectPlacementPolicy.compute_loss` của bài, k bộ (t, ε) mỗi ảnh.
        x0 [B,4] trong [−1,1]. -> loss vô hướng (trung bình trên B·k·4)."""
        cond = self.condition(images, text_raw, valid_hw).repeat_interleave(k, dim=0)
        x0 = x0.repeat_interleave(k, dim=0)
        dev = x0.device
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=dev, generator=generator)
        noise = torch.randn(x0.shape, device=dev, generator=generator)
        ab = self.alphas_cumprod[t].unsqueeze(-1)
        noisy = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        pred = self.noise_net(noisy.unsqueeze(1), t, cond).squeeze(1)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, images, text_raw, valid_hw, n_samples, generator=None):
        """n_samples mẫu ĐỘC LẬP mỗi ảnh (ảnh mã hoá một lần, các mẫu khử nhiễu song song).
        -> [B, n_samples, 4] trong [−1,1]."""
        cond = self.condition(images, text_raw, valid_hw).repeat_interleave(n_samples, dim=0)
        x = ddpm_sample(lambda x, t: self.noise_net(x.unsqueeze(1), t, cond).squeeze(1), cond.shape[0],
                        self.alphas_cumprod, generator=generator, device=cond.device)
        return x.view(images.shape[0], n_samples, 4)

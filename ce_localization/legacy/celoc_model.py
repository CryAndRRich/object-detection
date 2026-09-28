"""CE-Loc GỐC (bài add) viết lại đúng từng phép tính của refs/repos/Count-Editing/CE-LocModel:
`models/{text_encoder,components,noise_pred_net,diffusion_module}.py`. Tên module/param trùng
bản gốc để nạp strict `weights/celoc/best_model.pth`. Thêm: bỏ density (vision 3 kênh), sampler
DDPM đúng công thức cạnh sampler "mock" gốc, và IoU.

Cấu hình suy ra từ checkpoint gốc (config yaml không có trong repo):
  U-Net down_dims (64,128,256), kernel 3, GroupNorm 8, FiLM chỉ cộng bias (cond_predict_scale
  False); vis 128 + text 128; T = 1000, beta tuyến tính 1e-4..0.02; ε-prediction, MSE.

Hai lỗi của repo gốc, GIỮ để so được với số của bài (và có bản đúng bên cạnh):
  1. `test_mul_box.py` lấy mẫu bằng "mock update": 100 bước t = 99..0, `x -= eps/100` — không
     phải DDPM (model train với t trong [0, 1000)). -> `sample_mock` (gốc) vs `sample_ddpm`.
  2. `calculate_iou` tính trên box đã chuẩn hoá [-1,1] kể cả w, h (w = 46 px -> -0,82) -> số vô
     nghĩa. -> `iou_original_formula` (gốc) vs `iou_pixels` (đúng, sau khi giải chuẩn hoá).
`text_encoder.py:47` đảo `set_grad_enabled`, nhưng CLIP frozen + input là token nên không có
đồ thị grad nào -> dùng `no_grad`, output y hệt.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ce_localization.legacy.celoc_vision import TARGET, SpatialVisualEncoder

__all__ = ["CLIPTextEncoder", "ConditionalUnet1D", "ObjectPlacementPolicy", "load_policy",
           "sample_mock", "sample_ddpm", "iou_pixels", "iou_original_formula", "denormalize"]

CLIP_NAME = "openai/clip-vit-base-patch32"


# ----------------------------------------------------------------------------- text

class CLIPTextEncoder(nn.Module):
    """CLIP text frozen -> pooler_output -> Linear -> Mish."""

    def __init__(self, model_name=CLIP_NAME, output_dim=128, pretrained=True):
        super().__init__()
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
        if pretrained:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.backbone = CLIPTextModel.from_pretrained(model_name)
        else:                                    # test: đúng kiến trúc, không tải gì
            self.tokenizer = None
            self.backbone = CLIPTextModel(CLIPTextConfig())
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.hidden_size = self.backbone.config.hidden_size
        self.projection = nn.Linear(self.hidden_size, output_dim)
        self.activation = nn.Mish()

    def forward(self, text_list):
        dev = self.projection.weight.device
        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            pooled = self.backbone(**inputs).pooler_output
        return self.activation(self.projection(pooled))


# ----------------------------------------------------------------------------- U-Net 1D

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
    """Thay `Rearrange('batch t -> batch t 1')` (không tham số, giữ chỉ số Sequential)."""

    def forward(self, x):
        return x.unsqueeze(-1)


class ConditionalResidualBlock1D(nn.Module):
    """Nhánh cond_predict_scale=False của bản gốc (checkpoint: cond_encoder ra C, không 2C)."""

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


# ----------------------------------------------------------------------------- policy

class ObjectPlacementPolicy(nn.Module):
    """vision (ResNet18 + SpatialSoftmax) + text (CLIP frozen) -> global_cond 256 -> U-Net ε."""

    def __init__(self, use_density=True, pretrained_vision=True, pretrained_text=True,
                 num_timesteps=1000, vis_dim=128, text_dim=128, down_dims=(64, 128, 256),
                 kernel_size=3, n_groups=8, clip_name=CLIP_NAME):
        super().__init__()
        self.use_density = use_density
        self.vision_encoder = SpatialVisualEncoder(vis_dim, in_channels=4 if use_density else 3,
                                                   pretrained=pretrained_vision)
        self.text_encoder = CLIPTextEncoder(clip_name, text_dim, pretrained=pretrained_text)
        self.noise_net = ConditionalUnet1D(4, global_cond_dim=vis_dim + text_dim, down_dims=down_dims,
                                           kernel_size=kernel_size, n_groups=n_groups)
        self.num_timesteps = num_timesteps
        betas = torch.linspace(0.0001, 0.02, num_timesteps)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

    def condition(self, rgb, density, text):
        vis = self.vision_encoder(rgb, density if self.use_density else None)[0]
        return torch.cat([vis, self.text_encoder(text)], dim=-1)

    def compute_loss(self, rgb, density, text, gt_bbox, generator=None):
        """Như `ObjectPlacementPolicy.compute_loss` gốc: t đều mỗi ảnh, MSE trên ε."""
        cond = self.condition(rgb, density, text)
        B = rgb.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=rgb.device, generator=generator)
        noise = torch.randn(gt_bbox.shape, device=gt_bbox.device, generator=generator)
        ab = self.alphas_cumprod[t].unsqueeze(-1)
        noisy = torch.sqrt(ab) * gt_bbox + torch.sqrt(1 - ab) * noise
        pred = self.noise_net(noisy.unsqueeze(1), t, cond).squeeze(1)
        return F.mse_loss(pred, noise)

    def forward(self, rgb, density, text, gt_bbox, generator=None):
        """= compute_loss. DDP chỉ đồng bộ gradient khi bước tính đi qua `model(...)`."""
        return self.compute_loss(rgb, density, text, gt_bbox, generator=generator)


def load_policy(ckpt_path, device="cpu", pretrained_text=True):
    """Nạp checkpoint gốc hoặc checkpoint của `legacy/train.py`. Density suy từ conv1.
    Chỉ tha `position_ids` của CLIP (buffer, có/không tuỳ phiên bản transformers)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    use_density = int(sd["vision_encoder.backbone.0.weight"].shape[1]) == 4
    model = ObjectPlacementPolicy(use_density=use_density, pretrained_vision=False,
                                  pretrained_text=pretrained_text)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing + unexpected if not k.endswith("position_ids")]
    if bad:
        raise RuntimeError(f"checkpoint lệch kiến trúc: {bad[:10]} ... ({len(bad)} key)")
    return model.to(device).eval(), ck


# ----------------------------------------------------------------------------- lấy mẫu

@torch.no_grad()
def sample_mock(model, cond, n_samples, steps=100, generator=None):
    """Sampler của `test_mul_box.py` gốc: t = steps-1..0, x -= eps / steps. cond [B, G] ->
    [B, n_samples, 4]. KHÔNG phải DDPM — giữ để so với số của bài."""
    B = cond.shape[0]
    c = cond.repeat_interleave(n_samples, dim=0)
    x = torch.randn((B * n_samples, 4), device=cond.device, generator=generator)
    for t in reversed(range(steps)):
        tb = torch.full((x.shape[0],), t, device=cond.device, dtype=torch.long)
        x = x - (1.0 / steps) * model.noise_net(x.unsqueeze(1), tb, c).squeeze(1)
    return x.reshape(B, n_samples, 4)


@torch.no_grad()
def sample_ddpm(model, cond, n_samples, generator=None):
    """DDPM tổ tiên đúng công thức (Ho et al. 2020), T bước, phương sai beta~_t, không clip."""
    B = cond.shape[0]
    c = cond.repeat_interleave(n_samples, dim=0)
    ab = model.alphas_cumprod
    ab_prev = torch.cat([ab.new_ones(1), ab[:-1]])
    alphas = ab / ab_prev
    betas = 1 - alphas
    x = torch.randn((B * n_samples, 4), device=cond.device, generator=generator)
    for t in reversed(range(model.num_timesteps)):
        tb = torch.full((x.shape[0],), t, device=cond.device, dtype=torch.long)
        eps = model.noise_net(x.unsqueeze(1), tb, c).squeeze(1)
        mean = (x - betas[t] / torch.sqrt(1 - ab[t]) * eps) / torch.sqrt(alphas[t])
        if t > 0:
            var = betas[t] * (1 - ab_prev[t]) / (1 - ab[t])
            x = mean + torch.sqrt(var) * torch.randn(x.shape, device=x.device, generator=generator)
        else:
            x = mean
    return x.reshape(B, n_samples, 4)


# ----------------------------------------------------------------------------- IoU

def denormalize(u, target=TARGET):
    """[-1,1] (cx, cy, w, h — cả w, h cũng chuẩn hoá như dataset gốc) -> pixel canvas."""
    return (np.asarray(u, dtype=np.float64) + 1) / 2 * target


def iou_pixels(pred, gt):
    """IoU ĐÚNG: pred [..., 4], gt [..., 4] chuẩn hoá [-1,1] -> giải chuẩn hoá, w/h >= 0."""
    p, g = denormalize(pred), denormalize(gt)
    pw, ph = np.clip(p[..., 2], 0, None), np.clip(p[..., 3], 0, None)
    gw, gh = np.clip(g[..., 2], 0, None), np.clip(g[..., 3], 0, None)
    ix = np.clip(np.minimum(p[..., 0] + pw / 2, g[..., 0] + gw / 2)
                 - np.maximum(p[..., 0] - pw / 2, g[..., 0] - gw / 2), 0, None)
    iy = np.clip(np.minimum(p[..., 1] + ph / 2, g[..., 1] + gh / 2)
                 - np.maximum(p[..., 1] - ph / 2, g[..., 1] - gh / 2), 0, None)
    inter = ix * iy
    return inter / (pw * ph + gw * gh - inter + 1e-9)


def iou_original_formula(pred, gt):
    """`test_mul_box.calculate_iou` gốc, vector hoá, TRÊN GIÁ TRỊ CHUẨN HOÁ (lỗi của bản gốc)."""
    p, g = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    b1x1, b1y1 = p[..., 0] - p[..., 2] / 2, p[..., 1] - p[..., 3] / 2
    b1x2, b1y2 = p[..., 0] + p[..., 2] / 2, p[..., 1] + p[..., 3] / 2
    b2x1, b2y1 = g[..., 0] - g[..., 2] / 2, g[..., 1] - g[..., 3] / 2
    b2x2, b2y2 = g[..., 0] + g[..., 2] / 2, g[..., 1] + g[..., 3] / 2
    inter = (np.maximum(0, np.minimum(b1x2, b2x2) - np.maximum(b1x1, b2x1))
             * np.maximum(0, np.minimum(b1y2, b2y2) - np.maximum(b1y1, b2y1)))
    union = (b1x2 - b1x1) * (b1y2 - b1y1) + (b2x2 - b2x1) * (b2y2 - b2y1) - inter
    return inter / (union + 1e-6)

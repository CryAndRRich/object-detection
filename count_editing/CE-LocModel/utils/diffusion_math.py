"""Bản TORCH của `diffusion_np.py` — port CƠ HỌC.

Bốn lỗi số học vòng 1 (mỗi cái có test âm bản ở bản numpy):
  1. không clamp pred_x_start (1/sqrt(ab) tới 20.291x ở t=999)
  2. không tính LẠI pred_noise từ x_start ĐÃ clamp
  3. x_T nhân snr_scale (phải là N(0,I) std 1,0)
  4. loss trên epsilon dưới set-matching (vô nghĩa: matcher hoán vị)
"""

import math

import torch

from .box_ops import encode_diffusion

__all__ = [
    "cosine_alphas_cumprod", "q_sample", "predict_noise_from_start",
    "ddim_time_pairs", "make_placeholders", "prepare_diffusion_concat",
]


def cosine_alphas_cumprod(num_timesteps=1000, s=0.008, dtype=torch.float64):
    """Giống hệt `cosine_beta_schedule` của DiffusionDet (có clip betas).

    sqrt(alpha_bar) còn lại: t=249 -> 0,92 | t=499 -> 0,70 | t=749 -> 0,38.
    Linear chỉ còn 0,058 ở t=749 -> đo được cosine cho 3,70x AP.
    """
    x = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=dtype)
    ac = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = torch.clip(1 - (ac[1:] / ac[:-1]), 0, 0.999)
    return torch.cumprod(1.0 - betas, dim=0)


def q_sample(x_start, t, noise, alphas_cumprod):
    """x_t = sqrt(ab_t) x_0 + sqrt(1-ab_t) eps. `t` là MỘT giá trị cho cả ảnh."""
    ab = alphas_cumprod[t].to(x_start.dtype)
    return ab.sqrt() * x_start + (1 - ab).sqrt() * noise


def predict_noise_from_start(x_t, t, x_start, alphas_cumprod):
    """Suy ngược eps từ (x_t, x_0) — nghịch đảo giải tích của q_sample."""
    ab = alphas_cumprod[t].to(x_t.dtype)
    return ((1.0 / ab).sqrt() * x_t - x_start) / (1.0 / ab - 1).sqrt()


def ddim_time_pairs(num_timesteps=1000, sampling_steps=4):
    times = torch.linspace(-1, num_timesteps - 1, sampling_steps + 1)
    times = list(reversed(times.int().tolist()))
    return list(zip(times[:-1], times[1:]))


def make_placeholders(n, median_wh=None, valid_h=1.0, device="cpu", dtype=torch.float32,
                      generator=None):
    """Box giả trong hệ chuẩn cxcywh [0,1].

    Gốc DiffusionDet `randn/6 + 0.5` = N(0,5; 1/6) cho CẢ 4 chiều. Hai sửa:
      SỬA 1  — w/h theo log-normal quanh trung vị vật thật của CHÍNH ảnh đó
               (gốc cho 0,5 = nửa ảnh, to gấp 7,3x vật CE-130 median 0,0686).
      SỬA 1b — chặn cy trong vùng ảnh thật (13,7 % placeholder từng rơi vào pad).
    """
    def _randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    out = torch.empty(n, 4, device=device, dtype=dtype)
    out[:, 0] = _randn(n) / 6.0 + 0.5                                # cx: gốc
    out[:, 1] = (_randn(n) / 6.0 + 0.5).clamp(0.0, 1.0) * max(valid_h, 1e-6)  # SỬA 1b

    if median_wh is None:
        out[:, 2:] = (_randn(n, 2) / 6.0 + 0.5).clamp(min=1e-4)
    else:
        mw, mh = float(median_wh[0]), float(median_wh[1])
        sigma = 0.4
        out[:, 2] = (mw * torch.exp(_randn(n) * sigma)).clamp(1e-4, 1.0)
        out[:, 3] = (mh * torch.exp(_randn(n) * sigma)).clamp(1e-4, 1.0)
    return out


def prepare_diffusion_concat(gt_boxes, num_proposals, t, alphas_cumprod, snr_scale=2.0,
                             valid_h=1.0, adapt_placeholder=True, generator=None):
    """GT [M,4] cxcywh[0,1] -> (x_t [N,4], noise [N,4], is_gt [N] bool).

    Pad placeholder khi M<N, crop ngẫu nhiên khi M>N. N=100 cắt GT ở 8-11 % ảnh
    -> eval nên dùng N=300 (kiến trúc cho phép vì đã bỏ pos_emb theo chỉ số).
    """
    dev, dt = gt_boxes.device, gt_boxes.dtype
    m = gt_boxes.shape[0]

    if m == 0:
        x_start_norm = make_placeholders(num_proposals, None, valid_h, dev, dt, generator)
        is_gt = torch.zeros(num_proposals, dtype=torch.bool, device=dev)
    elif m < num_proposals:
        med = (gt_boxes[:, 2].median().item(), gt_boxes[:, 3].median().item()) \
            if adapt_placeholder else None
        ph = make_placeholders(num_proposals - m, med, valid_h, dev, dt, generator)
        x_start_norm = torch.cat([gt_boxes, ph], dim=0)
        is_gt = torch.zeros(num_proposals, dtype=torch.bool, device=dev)
        is_gt[:m] = True
    else:
        idx = torch.randperm(m, device=dev, generator=generator)[:num_proposals]
        x_start_norm = gt_boxes[idx]
        is_gt = torch.ones(num_proposals, dtype=torch.bool, device=dev)

    x_start = encode_diffusion(x_start_norm, snr_scale)
    noise = torch.randn(num_proposals, 4, device=dev, dtype=dt, generator=generator)
    x_t = q_sample(x_start, t, noise, alphas_cumprod).clamp(-snr_scale, snr_scale)
    return x_t, noise, is_gt

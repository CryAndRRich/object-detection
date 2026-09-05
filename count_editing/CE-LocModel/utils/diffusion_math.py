"""TORCH port of `diffusion_np.py` — MECHANICAL.

Round 1's four numerical bugs (each has a negative-control test on the numpy side):
  1. not clamping pred_x_start (1/sqrt(ab) reaches 20,291x at t=999)
  2. not RECOMPUTING pred_noise from the CLAMPED x_start
  3. scaling x_T by snr_scale (it must be N(0,I) with std 1.0)
  4. an epsilon loss under set matching (meaningless: the matcher permutes)
"""

import math

import torch

from .box_ops import encode_diffusion

__all__ = [
    "cosine_alphas_cumprod", "q_sample", "predict_noise_from_start",
    "ddim_time_pairs", "make_placeholders", "prepare_diffusion_concat",
]


def cosine_alphas_cumprod(num_timesteps=1000, s=0.008, dtype=torch.float64):
    """Identical to DiffusionDet's `cosine_beta_schedule` (betas clipped).

    sqrt(alpha_bar) remaining: t=249 -> 0.92 | t=499 -> 0.70 | t=749 -> 0.38.
    Linear leaves only 0.058 at t=749 -> measured cosine gives 3.70x the AP.
    """
    x = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=dtype)
    ac = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = torch.clip(1 - (ac[1:] / ac[:-1]), 0, 0.999)
    return torch.cumprod(1.0 - betas, dim=0)


def q_sample(x_start, t, noise, alphas_cumprod):
    """x_t = sqrt(ab_t) x_0 + sqrt(1-ab_t) eps. `t` is ONE scalar for the whole image."""
    ab = alphas_cumprod[t].to(x_start.dtype)
    return ab.sqrt() * x_start + (1 - ab).sqrt() * noise


def predict_noise_from_start(x_t, t, x_start, alphas_cumprod):
    """Recover eps from (x_t, x_0) — the analytic inverse of q_sample."""
    ab = alphas_cumprod[t].to(x_t.dtype)
    return ((1.0 / ab).sqrt() * x_t - x_start) / (1.0 / ab - 1).sqrt()


def ddim_time_pairs(num_timesteps=1000, sampling_steps=4):
    times = torch.linspace(-1, num_timesteps - 1, sampling_steps + 1)
    times = list(reversed(times.int().tolist()))
    return list(zip(times[:-1], times[1:]))


def make_placeholders(n, median_wh=None, valid_h=1.0, device="cpu", dtype=torch.float32,
                      generator=None):
    """Fake boxes in canonical cxcywh [0,1].

    DiffusionDet's original `randn/6 + 0.5` = N(0.5, 1/6) for ALL 4 dims. Two fixes:
      FIX 1  — w/h from a log-normal around the median real object size OF THAT
               SAME IMAGE (the original gives 0.5 = half the image, 7.3x larger
               than CE-130's median object of 0.0686).
      FIX 1b — keep cy inside the real image region (13.7 % of placeholders used
               to land in the padding).
    """
    def _randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    out = torch.empty(n, 4, device=device, dtype=dtype)
    out[:, 0] = _randn(n) / 6.0 + 0.5                                # cx: original
    out[:, 1] = (_randn(n) / 6.0 + 0.5).clamp(0.0, 1.0) * max(valid_h, 1e-6)  # FIX 1b

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

    Pads with placeholders when M<N, randomly crops when M>N. N=100 truncates GT
    on 8-11 % of images -> eval should use N=300 (the architecture allows it
    because index-based pos_emb was removed).
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

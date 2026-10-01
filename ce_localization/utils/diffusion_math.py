"""Toán khuếch tán dùng chung (torch): lịch cosine (DiffusionDet) / tuyến tính (CE-Loc gốc), q_sample,
suy ε từ x0, cặp bước DDIM. Phần ghép box / DDIM / DDPM trên box ở `engine/diffusion.py`."""

import math

import torch

__all__ = ["cosine_alphas_cumprod", "linear_alphas_cumprod", "q_sample", "predict_noise_from_start",
           "ddim_time_pairs"]


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


def linear_alphas_cumprod(num_timesteps=1000, beta_start=1e-4, beta_end=0.02, dtype=torch.float32):
    """Lịch β tuyến tính của DDPM / CE-Loc gốc (`refs/repos/Count-Editing/CE-LocModel/models/
    diffusion_module.py`: `torch.linspace(1e-4, 0.02, 1000)`, float32) -> ᾱ_t = Π (1 − β)."""
    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=dtype)
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

"""Toán khuếch tán: lịch β tuyến tính của CE-Loc gốc, q_sample."""

import torch

__all__ = ["linear_alphas_cumprod", "q_sample"]


def linear_alphas_cumprod(num_timesteps=1000, beta_start=1e-4, beta_end=0.02, dtype=torch.float32):
    """Lịch β tuyến tính của DDPM / CE-Loc gốc (`torch.linspace(1e-4, 0.02, T)`, float32) -> ᾱ_t = Π (1 − β)."""
    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=dtype)
    return torch.cumprod(1.0 - betas, dim=0)


def _ab(alphas_cumprod, t, x):
    """ᾱ_t theo dtype của x: `t` vô hướng hoặc [R] (một t mỗi hàng của x [R, ...])."""
    ab = alphas_cumprod[t].to(x.dtype)
    return ab.view(-1, *([1] * (x.dim() - 1))) if ab.dim() else ab


def q_sample(x_start, t, noise, alphas_cumprod):
    """x_t = sqrt(ab_t) x_0 + sqrt(1-ab_t) eps."""
    ab = _ab(alphas_cumprod, t, x_start)
    return ab.sqrt() * x_start + (1 - ab).sqrt() * noise

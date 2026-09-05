"""Diffusion math — pure numpy, NO torch dependency.

SOURCE OF TRUTH for `q_sample` / DDIM / `prepare_diffusion_concat`. The torch
port must be mechanical, with `allclose(torch_fn, np_fn)` tests.

ROUND-1'S FOUR NUMERICAL BUGS — each has a negative-control test in tests/:

  1. NOT clamping `pred_x_start` before use.
     The 1/sqrt(alpha_bar) factor reaches 20,291x at t=999 (>10x on 6.5 % of
     timesteps), so a box in [-1,1] flies out to huge values, wrecking both the
     cost matrix and the loss.
     (DiffusionDet clamps at detector.py:181)

  2. NOT recomputing `pred_noise` from the CLAMPED `x_start`.
     Skipping this raises DDIM error from ~2e-7 to ~5.4.
     (DiffusionDet: detector.py:182)

  3. Scaling the x_T initialisation by `snr_scale`.
     It must be N(0, I) with std 1.0 — at t=T-1, alpha_bar = 4e-5.
     (DiffusionDet uses a plain `randn`: detector.py:197)

  4. An epsilon loss under set matching.
     Meaningless: once the matcher assigns prediction p to GT g, no epsilon
     belongs to p and corresponds to g at the same time. `SetCriterionDynamicK`
     only has L1+GIoU on x0.

MEASURED LIMIT of the placeholder w/h fix (docs §(a)): the model sees
`x_t = q_sample(x_start)`, not `x_start`. Clamping pulls everything toward a
median of 0.5 at large t (t=999: decoded median w = 0.500; 79 % have w>0.3). So
the placeholder fix only helps at small t — still needed, but don't over-expect.
"""

import numpy as np

from .box_ops_np import decode_diffusion, encode_diffusion

__all__ = [
    "cosine_alphas_cumprod",
    "linear_alphas_cumprod",
    "q_sample",
    "predict_noise_from_start",
    "ddim_step",
    "ddim_time_pairs",
    "make_placeholders",
    "prepare_diffusion_concat",
]


# --------------------------------------------------------------------------
# Beta schedule
# --------------------------------------------------------------------------

def cosine_alphas_cumprod(num_timesteps=1000, s=0.008):
    """Cosine schedule (Nichol & Dhariwal), exactly what DiffusionDet uses.

    Measured in round 1: cosine gives 3.70x the AP of linear. Reason: at large t,
    linear leaves only 5.8 % signal (t=749), so boxes are near-pure noise, the
    matcher assigns arbitrarily, and gradients are 4.6x noisier.

        sqrt(alpha_bar) remaining   t=249    t=499    t=749
        cosine                      0.92     0.70     0.38
        linear                      0.72     0.28     0.058
    """
    t = np.arange(num_timesteps + 1, dtype=np.float64)
    f = np.cos(((t / num_timesteps) + s) / (1.0 + s) * np.pi / 2.0) ** 2
    ab = f / f[0]
    betas = np.clip(1.0 - (ab[1:] / ab[:-1]), 0.0, 0.999)
    return np.cumprod(1.0 - betas)


def linear_alphas_cumprod(num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
    """Linear schedule — for test comparison ONLY, never used for training."""
    betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    return np.cumprod(1.0 - betas)


# --------------------------------------------------------------------------
# Forward process
# --------------------------------------------------------------------------

def q_sample(x_start, t, noise, alphas_cumprod):
    """x_t = sqrt(ab_t) * x_0 + sqrt(1 - ab_t) * eps.

    `t` is ONE scalar for the whole image (not one t per box) — the N boxes are
    *a single* sample from a distribution over box sets; with a per-box t you
    could not reproduce the process at inference. DiffusionDet does the same:
    `torch.randint(..., (1,))`.
    """
    ab = float(alphas_cumprod[int(t)])
    return np.sqrt(ab) * np.asarray(x_start, dtype=np.float64) + np.sqrt(1.0 - ab) * np.asarray(
        noise, dtype=np.float64
    )


def predict_noise_from_start(x_t, t, x_start, alphas_cumprod):
    """Recover eps from (x_t, x_0). The analytic inverse of `q_sample`.

    This is why predicting x_0 loses NOTHING versus predicting eps: given x_t and
    t, the two quantities determine each other through one linear equation.
    """
    ab = float(alphas_cumprod[int(t)])
    sqrt_recip = np.sqrt(1.0 / ab)
    sqrt_recipm1 = np.sqrt(1.0 / ab - 1.0)
    return (sqrt_recip * np.asarray(x_t, dtype=np.float64) - np.asarray(x_start, dtype=np.float64)) / sqrt_recipm1


# --------------------------------------------------------------------------
# Reverse process (DDIM)
# --------------------------------------------------------------------------

def ddim_time_pairs(num_timesteps=1000, sampling_steps=4):
    """[(T-1, T-2), ..., (1, 0), (0, -1)] — same as DiffusionDet detector.py:191."""
    times = np.linspace(-1, num_timesteps - 1, sampling_steps + 1)
    times = list(reversed(times.astype(int).tolist()))
    return list(zip(times[:-1], times[1:]))


def ddim_step(x_t, x_start_raw, t, t_next, alphas_cumprod, snr_scale=2.0, eta=1.0, noise=None):
    """One DDIM step. Returns (x_next, clamped_x_start).

    MANDATORY order (bugs 1 and 2 at the top of this file):
      1. clamp x_start into the valid range
      2. RECOMPUTE pred_noise from the CLAMPED x_start
      3. only then take the DDIM step

    When `t_next < 0`, return x_start directly (final step, no noise added).
    """
    s = float(snr_scale)
    # (1) clamp — via decode/encode so there is exactly one definition of "valid"
    x_start = encode_diffusion(decode_diffusion(x_start_raw, s), s)

    if t_next < 0:
        return x_start, x_start

    # (2) RECOMPUTE pred_noise from the clamped x_start
    pred_noise = predict_noise_from_start(x_t, t, x_start, alphas_cumprod)

    ab_t = float(alphas_cumprod[int(t)])
    ab_next = float(alphas_cumprod[int(t_next)])
    sigma = eta * np.sqrt((1 - ab_t / ab_next) * (1 - ab_next) / (1 - ab_t))
    c = np.sqrt(max(1 - ab_next - sigma ** 2, 0.0))

    if noise is None:
        noise = np.random.standard_normal(np.shape(x_t))
    x_next = x_start * np.sqrt(ab_next) + c * pred_noise + sigma * noise
    return x_next, x_start


# --------------------------------------------------------------------------
# prepare_diffusion_concat — pad/crop GT up to N proposals
# --------------------------------------------------------------------------

def make_placeholders(n, median_wh=None, valid_h=1.0, rng=None):
    """Generate `n` fake boxes in canonical cxcywh [0,1].

    DiffusionDet's original: `randn/6 + 0.5` for ALL 4 dimensions — a Gaussian
    N(0.5, 1/6), whose own comment reads "3sigma = 1/2". Two fixes for CE-130:

    FIX 1 — data-driven w/h. The original gives w/h ~ 0.5 = half the image, while
      CE-130 objects have median 0.069, so placeholders are 7.3x too big. With
      N=100 and ~37.6 GT, 62 % of slots are placeholders, teaching the model a
      prior of "boxes are big and centred". Replaced with a log-normal around the
      median real object size OF THAT SAME IMAGE (7.3x -> ~0.8x).

    FIX 1b — keep cy inside the real image region. Measured 13.7 % of
      placeholders had centres landing in the padding (worst image: 92.2 %).
      Cheap because every CE-130 image is exactly 384px tall, so W>=H always,
      meaning padding is only ever at the BOTTOM -> one scalar threshold suffices.

    The cx centre keeps the original Gaussian (images always span the full canvas
    width).
    """
    rng = np.random.default_rng() if rng is None else rng
    out = np.empty((n, 4), dtype=np.float64)

    out[:, 0] = rng.standard_normal(n) / 6.0 + 0.5                      # cx: original
    out[:, 1] = np.clip(rng.standard_normal(n) / 6.0 + 0.5, 0.0, 1.0)   # cy: original
    out[:, 1] *= max(valid_h, 1e-6)                                      # FIX 1b

    if median_wh is None:                                                # DiffusionDet original
        out[:, 2:] = np.clip(rng.standard_normal((n, 2)) / 6.0 + 0.5, 1e-4, None)
    else:                                                                # FIX 1
        mw, mh = float(median_wh[0]), float(median_wh[1])
        sigma = 0.4  # log-normal spread, roughly one octave
        out[:, 2] = np.clip(mw * np.exp(rng.standard_normal(n) * sigma), 1e-4, 1.0)
        out[:, 3] = np.clip(mh * np.exp(rng.standard_normal(n) * sigma), 1e-4, 1.0)
    return out


def prepare_diffusion_concat(
    gt_boxes, num_proposals, t, alphas_cumprod, snr_scale=2.0,
    valid_h=1.0, adapt_placeholder=True, rng=None,
):
    """GT [M,4] cxcywh[0,1] -> (x_t [N,4], noise [N,4], is_gt [N] bool).

    Pads with placeholders when M < N, randomly crops when M > N. `is_gt` marks
    which slots are real GT (only those enter the coordinate loss via the matcher).

    N=100 truncates GT on 8-11 % of images (test: 11.2 %), so eval should use
    N=300. The architecture allows this because index-based pos_emb was removed.
    """
    rng = np.random.default_rng() if rng is None else rng
    gt = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    m = gt.shape[0]

    if m == 0:
        x_start_norm = make_placeholders(num_proposals, None, valid_h, rng)
        is_gt = np.zeros(num_proposals, dtype=bool)
    elif m < num_proposals:
        med = (np.median(gt[:, 2]), np.median(gt[:, 3])) if adapt_placeholder else None
        ph = make_placeholders(num_proposals - m, med, valid_h, rng)
        x_start_norm = np.concatenate([gt, ph], axis=0)
        is_gt = np.zeros(num_proposals, dtype=bool)
        is_gt[:m] = True
    else:
        idx = rng.permutation(m)[:num_proposals]
        x_start_norm = gt[idx]
        is_gt = np.ones(num_proposals, dtype=bool)

    x_start = encode_diffusion(x_start_norm, snr_scale)
    noise = rng.standard_normal((num_proposals, 4))
    x_t = np.clip(q_sample(x_start, t, noise, alphas_cumprod), -snr_scale, snr_scale)
    return x_t, noise, is_gt

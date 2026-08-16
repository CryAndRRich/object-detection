"""Diffusion process over decoder-query reference points.

This is the core of DiffuGroundingDINO. Instead of initialising the decoder's
reference points from the encoder's top-k proposals, we treat the reference
points as the latent variable of a DDPM and let the decoder act as the
denoiser, following DiffuDINO (DiffuDETR, ICLR 2026).

Everything is written from the published formulas:

  * forward process (Ho et al., NeurIPS 2020, eq. 4)
        q(x_t | x_0) = N(x_t; sqrt(a_t) x_0, (1 - a_t) I)
    with ``a_t`` the cumulative product of ``alpha``.
  * cosine schedule (Nichol & Dhariwal, ICML 2021, eq. 17)
        a_t = cos(((t/T) + s) / (1 + s) * pi/2)^2
  * DDIM update (Song et al., ICLR 2021, eq. 12)
        x_{t-1} = sqrt(a_{t-1}) x_0 + sqrt(1 - a_{t-1} - sigma^2) eps + sigma z

Coordinate spaces. Three distinct spaces are in play; mixing them up is the
single easiest way to break this model, so they are named explicitly:

  ``box``       cxcywh normalized to [0, 1]              -- what the dataset gives
  ``latent``    (box * 2 - 1) * snr_scale, roughly N(0,1) -- where diffusion lives
  ``unsigmoid`` logit(box)                                -- what the decoder wants

The signal-to-noise trick (``snr_scale``, DiffusionDet ICCV 2023) rescales the
latent so that its variance matches the unit Gaussian the schedule assumes.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from util.misc import force_fp32, inverse_sigmoid

SCHEDULES = ("cosine", "linear", "sqrt")
LOSS_WEIGHT_MODES = ("diffudino", "vlb", "none")
PAD_MODES = ("normal", "center", "sigmoid_normal")


def extract(buffer: Tensor, t: Tensor, ndim: int) -> Tensor:
    """Gather ``buffer[t]`` and reshape to (bs, 1, ..., 1) for broadcasting.

    ``t`` is (bs,) long; the result has ``ndim`` dimensions.
    """
    out = buffer.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (ndim - 1)))


def make_beta_schedule(name: str, timesteps: int, cosine_s: float = 0.008) -> Tensor:
    """Return the ``betas`` of a named schedule, shape (timesteps,), float64.

    DiffuDETR ablates exactly these three (Table 5); cosine wins, so it is the
    default. ``linear`` is scaled by ``1000/T`` so its behaviour is invariant to
    the number of timesteps, matching the DDPM defaults at T=1000.
    """
    assert name in SCHEDULES, f"unknown schedule {name!r}, expected one of {SCHEDULES}"

    if name == "cosine":
        steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(((steps / timesteps) + cosine_s) / (1 + cosine_s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]

    elif name == "linear":
        scale = 1000.0 / timesteps
        betas = torch.linspace(scale * 1e-4, scale * 2e-2, timesteps, dtype=torch.float64)

    else:  # "sqrt" -- Ting Chen, arXiv 2301.10972
        steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        alphas_cumprod = 1.0 - torch.sqrt(steps / timesteps + cosine_s)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]

    return torch.clip(betas, 0.0, 0.999)


class RefPointDiffusion(nn.Module):
    """DDPM over reference points, with a DDIM sampler for inference.

    Args:
        num_timesteps: length ``T`` of the forward process. The DiffuDETR paper
            uses ``T=100`` (§3.2, justified by the low dimensionality of the
            latent); their released code uses 1000, which is our default.
        sampling_timesteps: number of decoder evaluations at inference. 3 is the
            optimum in the DiffuDETR ablation (Table 6/7), *not* the
            ``SAMPLE_STEP=1`` default of DiffusionDet.
        snr_scale: latent rescaling. 2.0 matches both DiffusionDet's config
            default and DiffuDINO's ``self.scale``.
        ddim_eta: 1.0 keeps the stochastic DDPM-like sampler, 0.0 makes DDIM
            deterministic.
        loss_weight_mode: how the set-prediction loss is reweighted by ``t``.
            ``diffudino`` reproduces their released weight, ``vlb`` is the true
            Improved-DDPM VLB weight, ``none`` disables it.
        normalize_loss_weight: divide the weights by their mean so the overall
            loss magnitude (and hence the tuned ``bbox_loss_coef``) stays
            comparable to the non-diffusion baseline. Without this the box loss
            is scaled down ~5x on average, which silently changes the loss
            balance and makes an A/B against the baseline meaningless.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        sampling_timesteps: int = 3,
        snr_scale: float = 2.0,
        ddim_eta: float = 1.0,
        schedule: str = "cosine",
        cosine_s: float = 0.008,
        loss_weight_mode: str = "diffudino",
        normalize_loss_weight: bool = True,
        pad_mode: str = "normal",
        box_eps: float = 1e-3,
    ):
        super().__init__()
        assert 0 < sampling_timesteps <= num_timesteps
        assert loss_weight_mode in LOSS_WEIGHT_MODES, f"unknown loss_weight_mode {loss_weight_mode!r}"
        assert pad_mode in PAD_MODES, f"unknown pad_mode {pad_mode!r}"

        self.num_timesteps = int(num_timesteps)
        self.sampling_timesteps = int(sampling_timesteps)
        self.snr_scale = float(snr_scale)
        self.ddim_eta = float(ddim_eta)
        self.schedule = schedule
        self.loss_weight_mode = loss_weight_mode
        self.pad_mode = pad_mode
        self.box_eps = float(box_eps)

        betas = make_beta_schedule(schedule, self.num_timesteps, cosine_s)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        def register(name, value):
            self.register_buffer(name, value.to(torch.float32))

        register("betas", betas)
        register("alphas", alphas)
        register("alphas_cumprod", alphas_cumprod)
        register("alphas_cumprod_prev", alphas_cumprod_prev)

        # q(x_t | x_0)
        register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register("posterior_variance", posterior_variance)
        register("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        register("loss_weights", self._build_loss_weights(betas, alphas, alphas_cumprod, normalize_loss_weight))

    # ------------------------------------------------------------------ #
    # schedule-derived quantities
    # ------------------------------------------------------------------ #
    def _build_loss_weights(
        self, betas: Tensor, alphas: Tensor, alphas_cumprod: Tensor, normalize: bool
    ) -> Tensor:
        if self.loss_weight_mode == "none":
            return torch.ones_like(betas)

        if self.loss_weight_mode == "diffudino":
            # The weight actually used by the released DiffuDINO. Monotonically
            # decreasing in t: heavily noised samples contribute less.
            weights = 0.5 * torch.sqrt(alphas_cumprod) / (2.0 - alphas_cumprod)
        else:  # "vlb" -- Improved DDPM, the true variational bound weight
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            weights = betas**2 / (2.0 * posterior_variance.clamp(min=1e-20) * alphas * (1.0 - alphas_cumprod))

        # t=0 is degenerate for both forms (posterior variance -> 0).
        weights[0] = weights[1]
        if normalize:
            weights = weights / weights.mean()
        return weights

    def loss_weight(self, t: Tensor) -> Tensor:
        """Per-sample loss weight ``w(t)``, shape (bs,)."""
        return self.loss_weights.gather(0, t)

    # ------------------------------------------------------------------ #
    # space conversions
    # ------------------------------------------------------------------ #
    def boxes_to_latent(self, boxes: Tensor) -> Tensor:
        """``box`` space [0,1] -> ``latent`` space [-snr_scale, snr_scale]."""
        return (boxes * 2.0 - 1.0) * self.snr_scale

    def latent_to_boxes(self, x: Tensor) -> Tensor:
        """``latent`` -> ``box``, clamped strictly inside (0, 1).

        The final clamp keeps ``inverse_sigmoid`` finite: a reference point of
        exactly 0 or 1 maps to -/+inf and poisons the whole decoder.
        """
        x = torch.clamp(x, min=-self.snr_scale, max=self.snr_scale)
        boxes = ((x / self.snr_scale) + 1.0) / 2.0
        return boxes.clamp(min=self.box_eps, max=1.0 - self.box_eps)

    def latent_to_refpoints(self, x: Tensor) -> Tensor:
        """``latent`` -> ``unsigmoid`` reference points, ready for the decoder."""
        return inverse_sigmoid(self.latent_to_boxes(x))

    # ------------------------------------------------------------------ #
    # forward process
    # ------------------------------------------------------------------ #
    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """Sample ``x_t ~ q(x_t | x_0)``. ``t`` broadcasts over ``x_start[0]``."""
        if noise is None:
            noise = torch.randn_like(x_start)
        with force_fp32():
            a = extract(self.sqrt_alphas_cumprod, t, x_start.ndim)
            b = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.ndim)
            return a * x_start.float() + b * noise.float()

    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        with force_fp32():
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.ndim) * x_t.float()
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.ndim) * noise.float()
            )

    def predict_noise_from_start(self, x_t: Tensor, t: Tensor, x_start: Tensor) -> Tensor:
        """Invert ``predict_start_from_noise``: recover ``eps`` from a predicted ``x_0``."""
        with force_fp32():
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.ndim) * x_t.float() - x_start.float()
            ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.ndim)

    def sample_timesteps(self, batch_size: int, device) -> Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # training-time query construction
    # ------------------------------------------------------------------ #
    def _pad_boxes(self, num_pad: int, device, dtype) -> Tensor:
        """Filler boxes for the queries not covered by a ground-truth object.

        ``normal`` (DiffusionDet): N(0.5, 1/6) so that 3-sigma reaches the image
        border, keeping the fillers plausibly inside the frame. ``center`` and
        ``sigmoid_normal`` are the two variants used by DiffuDINO.
        """
        if self.pad_mode == "center":
            return torch.full((num_pad, 4), 0.5, device=device, dtype=dtype)
        if self.pad_mode == "sigmoid_normal":
            return torch.randn(num_pad, 4, device=device, dtype=dtype).sigmoid()

        pad = torch.randn(num_pad, 4, device=device, dtype=dtype) / 6.0 + 0.5
        pad[:, 2:] = pad[:, 2:].clip(min=1e-4)  # never a negative width/height
        return pad

    def prepare_diffusion_refpoints(
        self,
        gt_boxes: Tensor,
        num_queries: int,
        t: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Build the noised reference points for ONE image.

        Args:
            gt_boxes: (num_gt, 4) cxcywh normalized to [0, 1]. May be empty.
            num_queries: the decoder's fixed query count (900 for GroundingDINO).
            t: optional (1,) timestep; drawn uniformly if omitted.

        Returns:
            ``(refpoints_unsigmoid, noise, t)`` with shapes
            ``((num_queries, 4), (num_queries, 4), (1,))``. The caller stacks
            these over the batch.
        """
        device = self.betas.device
        dtype = torch.float32
        if t is None:
            t = self.sample_timesteps(1, device)

        gt_boxes = gt_boxes.to(device=device, dtype=dtype)
        num_gt = gt_boxes.shape[0]
        if num_gt == 0:
            # An image with no annotation still has to produce a valid latent.
            gt_boxes = torch.tensor([[0.5, 0.5, 1.0, 1.0]], device=device, dtype=dtype)
            num_gt = 1

        if num_gt < num_queries:
            x_start = torch.cat([gt_boxes, self._pad_boxes(num_queries - num_gt, device, dtype)], dim=0)
        elif num_gt > num_queries:
            keep = torch.randperm(num_gt, device=device)[:num_queries]
            x_start = gt_boxes[keep]
        else:
            x_start = gt_boxes

        x_start = self.boxes_to_latent(x_start)
        noise = torch.randn(num_queries, 4, device=device, dtype=dtype)
        x = self.q_sample(x_start=x_start, t=t.expand(num_queries), noise=noise)
        return self.latent_to_refpoints(x), noise, t

    def prepare_diffusion_refpoints_batch(
        self,
        gt_boxes_list: List[Tensor],
        num_queries: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Batched wrapper: one independent timestep per image.

        Returns ``(refpoints_unsigmoid (bs, nq, 4), noise (bs, nq, 4), t (bs,))``.
        """
        refpoints, noises, ts = [], [], []
        for gt_boxes in gt_boxes_list:
            refpoint, noise, t = self.prepare_diffusion_refpoints(gt_boxes, num_queries)
            refpoints.append(refpoint)
            noises.append(noise)
            ts.append(t)
        return torch.stack(refpoints), torch.stack(noises), torch.cat(ts)

    # ------------------------------------------------------------------ #
    # inference-time sampler
    # ------------------------------------------------------------------ #
    def ddim_time_pairs(self) -> List[Tuple[int, int]]:
        """``[(t, t_next), ...]`` from T-1 down to the -1 sentinel.

        ``len(pairs) == sampling_timesteps``, i.e. exactly that many decoder
        evaluations. The last pair has ``t_next < 0``, meaning "stop and return
        the predicted x_0 directly" rather than taking another DDIM step.
        """
        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        return list(zip(times[:-1], times[1:]))

    def init_latent(self, batch_size: int, num_queries: int, device) -> Tensor:
        """``x_T ~ N(0, I)`` in latent space."""
        return torch.randn(batch_size, num_queries, 4, device=device, dtype=torch.float32)

    def ddim_step(
        self,
        x: Tensor,
        x_start: Tensor,
        pred_noise: Tensor,
        time: int,
        time_next: int,
    ) -> Tensor:
        """One DDIM update in latent space.

        Args:
            x: current latent ``x_t`` (only its shape/device are used for the noise).
            x_start: the model's prediction of ``x_0``, in latent space.
            pred_noise: ``eps`` implied by ``(x_t, x_0)``.
            time / time_next: scalar timestep indices; ``time_next < 0`` returns
                ``x_start`` unchanged.
        """
        if time_next < 0:
            return x_start

        with force_fp32():
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).clamp(min=0).sqrt()

            noise = torch.randn_like(x) if self.ddim_eta > 0 else torch.zeros_like(x)
            return x_start.float() * alpha_next.sqrt() + c * pred_noise.float() + sigma * noise

    def extra_repr(self) -> str:
        return (
            f"T={self.num_timesteps}, sampling_timesteps={self.sampling_timesteps}, "
            f"snr_scale={self.snr_scale}, ddim_eta={self.ddim_eta}, schedule={self.schedule}, "
            f"loss_weight_mode={self.loss_weight_mode}, pad_mode={self.pad_mode}"
        )

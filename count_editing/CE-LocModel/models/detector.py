"""CE-Loc round 2 — a Diffusion Policy transformer body, borrowing DiffusionDet's
mechanism for generating N boxes at once.

    IMAGE -> CLIP ViT-B/16 FROZEN -> 1024 patch tokens -> Linear(768->256) ─┐
    TEXT  -> CLIP text    FROZEN  -> 1 token           -> Linear(768->256) ─┤
    t     -> sinusoidal                                                     ─┤
                                                          memory: 1026 tokens
    noisy boxes [B,N,4] -> sinPE(cx,cy,w,h) -> 6-layer decoder ─────────────┘
                             (self-attn among boxes + cross-attn into memory)
                                        |
                            boxes [B,N,4] + scores [B,N]

Difference from the original CE-Loc: memory goes from 2 tokens to 1026 POSITIONED
tokens. Round 1 measured that with 2 tokens all N boxes receive the SAME 256-d
vector, and gradients on unmatched boxes point in random directions (cosine
-0.0074 = a coin flip).

The output is DIRECT COORDINATES, not epsilon. DiffusionDet does the same
(`objective='pred_x0'`) — it predicts x_start via a delta and then DERIVES
pred_noise. Three reasons: (1) set matching requires it, (2) GIoU is only definable
on coordinates, (3) the 1/sqrt(ab) factor reaches 20,291x at t=999, so predicting
eps and deriving x0 would amplify the error by that much.
"""

import torch
import torch.nn as nn

from models.box_transformer import BoxTransformer
from models.clip_encoder import CLIPConditionEncoder
from utils.box_ops import decode_diffusion, encode_diffusion
from utils.diffusion_math import (
    cosine_alphas_cumprod, ddim_time_pairs, predict_noise_from_start,
    prepare_diffusion_concat,
)

__all__ = ["CELocDetector"]


def _check_generator(generator, dev):
    """`torch.randn(device=X, generator=g)` requires g.device == X, otherwise it
    raises a cryptic RuntimeError ("Expected a 'cuda' device type for generator
    but found 'cpu'"). This bit us once in the val loop -- fail early with a
    message that names the fix."""
    if generator is None:
        return
    gd, dd = torch.device(generator.device).type, torch.device(dev).type
    assert gd == dd, (
        f"generator is on '{gd}' but tensors are created on '{dd}'. "
        f"Fix: torch.Generator(device='{dd}').manual_seed(...)")


class CELocDetector(nn.Module):
    def __init__(self, clip_name="openai/clip-vit-base-patch16", d_model=256,
                 n_layer=6, n_head=8, image_size=512, num_timesteps=1000,
                 snr_scale=2.0, sampling_steps=4, dropout=0.1, freeze_clip=True):
        super().__init__()
        self.encoder = CLIPConditionEncoder(clip_name, d_model, image_size, freeze_clip)
        self.decoder = BoxTransformer(d_model, n_layer, n_head, dropout=dropout)

        self.num_timesteps = num_timesteps
        self.snr_scale = snr_scale
        self.sampling_steps = sampling_steps
        self.register_buffer("alphas_cumprod",
                             cosine_alphas_cumprod(num_timesteps).float(), persistent=False)

    # ------------------------------------------------------------------ train

    def build_inputs(self, targets, num_proposals, valid_h, generator=None):
        """Build x_t for the whole batch. `t` is ONE value per image (as in DiffusionDet)."""
        dev = self.alphas_cumprod.device
        _check_generator(generator, targets[0].device if targets else "cpu")
        t = int(torch.randint(0, self.num_timesteps, (1,), generator=generator).item())

        xs, gts = [], []
        for i, gt in enumerate(targets):
            x_t, _, is_gt = prepare_diffusion_concat(
                gt.to(dev), num_proposals, t, self.alphas_cumprod, self.snr_scale,
                valid_h=float(valid_h[i]), generator=generator,
            )
            xs.append(x_t)
            gts.append(is_gt)
        t_batch = torch.full((len(targets),), t, dtype=torch.long, device=dev)
        return torch.stack(xs), t_batch, torch.stack(gts)

    def forward(self, x_t, timesteps, pixel_values=None, texts=None,
                patch_raw=None, text_raw=None):
        """x_t [B,N,4] in diffusion space -> (boxes in [0,1], logits)."""
        memory = self.encoder(pixel_values, texts, patch_raw, text_raw)
        boxes_norm = decode_diffusion(x_t, self.snr_scale)
        return self.decoder(boxes_norm, timesteps, memory)

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def ddim_sample(self, num_proposals, pixel_values=None, texts=None,
                    patch_raw=None, text_raw=None, eta=1.0, generator=None):
        """Generate N boxes from pure noise.

        x_T ~ N(0, I) with std 1.0 — NOT scaled by snr_scale (round 1's bug 3).
        Each step: predict x_start -> CLAMP -> RECOMPUTE pred_noise from the
        clamped version.

        `eta=1.0` is DiffusionDet's default (`detector.py:97`) — DDIM degenerates
        into DDPM. A measured consequence, NOT a bug: on the first step
        (t=999 -> 749), sigma=0.925 and c=0.0000, so pred_noise is multiplied by
        zero and the whole step is `x_start*sqrt(ab_next) + noise`. Set eta=0 for
        deterministic DDIM.
        """
        memory = self.encoder(pixel_values, texts, patch_raw, text_raw)
        B, dev = memory.shape[0], memory.device

        _check_generator(generator, dev)
        img = torch.randn(B, num_proposals, 4, device=dev, generator=generator)
        boxes = logits = None

        for t, t_next in ddim_time_pairs(self.num_timesteps, self.sampling_steps):
            tb = torch.full((B,), t, dtype=torch.long, device=dev)
            boxes, logits = self.decoder(decode_diffusion(img, self.snr_scale), tb, memory)

            x_start = encode_diffusion(boxes, self.snr_scale)          # already in range
            if t_next < 0:
                break

            pred_noise = predict_noise_from_start(img, t, x_start, self.alphas_cumprod)
            a, a_next = self.alphas_cumprod[t], self.alphas_cumprod[t_next]
            sigma = eta * ((1 - a / a_next) * (1 - a_next) / (1 - a)).clamp(min=0).sqrt()
            c = (1 - a_next - sigma ** 2).clamp(min=0).sqrt()
            img = (x_start * a_next.sqrt() + c * pred_noise
                   + sigma * torch.randn(img.shape, device=dev, generator=generator))

        return boxes, logits

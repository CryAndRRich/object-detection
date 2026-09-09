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
    but found 'cpu'").

    This bit us TWICE in the val loop. The second time is the instructive one: the
    guard passed and the very next line still raised, because `randint` had no
    `device=` and so wanted a CPU generator while everything downstream wanted a
    CUDA one. A guard only covers calls that actually use the device it checks --
    so every RNG call under it must pass `device=dev` explicitly."""
    if generator is None:
        return
    gd, dd = torch.device(generator.device).type, torch.device(dev).type
    assert gd == dd, (
        f"generator is on '{gd}' but tensors are created on '{dd}'. "
        f"Fix: torch.Generator(device='{dd}').manual_seed(...)")


class CELocDetector(nn.Module):
    def __init__(self, clip_name="openai/clip-vit-base-patch16", d_model=256,
                 n_layer=6, n_head=8, image_size=512, num_timesteps=1000,
                 snr_scale=2.0, sampling_steps=4, dropout=0.1, freeze_clip=True,
                 roi_k=0, n_class=1, use_text=True, refine_rounds=0,
                 roi_to_tgt=True, score_roi=False):
        """`n_class`/`use_text` select EXPERIMENT A.2 (n_class=80, use_text=False):
        the class reaches the model through an 80-way OUTPUT head instead of the
        INPUT text. Defaults keep A/B byte-identical."""
        super().__init__()
        if (n_class > 1) != (not use_text):
            raise ValueError(
                f"n_class={n_class} and use_text={use_text} disagree. A/B are "
                f"(n_class=1, use_text=True); A.2 is (n_class=80, use_text=False). "
                f"A mix would leave the class reachable through BOTH paths (or "
                f"neither), which measures nothing.")
        self.n_class = n_class
        self.encoder = CLIPConditionEncoder(clip_name, d_model, image_size,
                                            freeze_clip, use_text=use_text)
        self.decoder = BoxTransformer(
            d_model, n_layer, n_head, dropout=dropout, roi_k=roi_k,
            roi_dim=self.encoder.vision.config.hidden_size, n_class=n_class,
            refine_rounds=refine_rounds,
            roi_to_tgt=roi_to_tgt, score_roi=score_roi)
        self.refine_rounds = refine_rounds

        self.num_timesteps = num_timesteps
        self.snr_scale = snr_scale
        self.sampling_steps = sampling_steps
        self.register_buffer("alphas_cumprod",
                             cosine_alphas_cumprod(num_timesteps).float(), persistent=False)

    # ------------------------------------------------------------------ train

    def build_inputs(self, targets, num_proposals, valid_h, generator=None):
        """Build x_t for the whole batch. `t` is ONE value per image (as in DiffusionDet)."""
        dev = self.alphas_cumprod.device
        _check_generator(generator, dev)
        # `device=dev` is REQUIRED, not cosmetic: without it randint creates a CPU
        # tensor and demands a CPU generator, while prepare_diffusion_concat below
        # creates CUDA tensors and demands a CUDA one. A single generator cannot
        # satisfy both, so one of the two always raises. Pinning both to `dev`
        # leaves exactly one device requirement.
        t = int(torch.randint(0, self.num_timesteps, (1,), device=dev,
                              generator=generator).item())

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
        """x_t [B,N,4] in diffusion space -> (boxes in [0,1], logits).

        With `refine_rounds > 0` (EXPERIMENT C1) this returns a LIST of that many
        (boxes, logits) pairs instead — one per refinement round, oldest first.
        The criterion supervises all of them; `ddim_sample` uses only the last.
        """
        need_raw = self.decoder.roi is not None
        out = self.encoder(pixel_values, texts, patch_raw, text_raw,
                           return_patch_raw=need_raw)
        memory, praw = out if need_raw else (out, None)
        boxes_norm = decode_diffusion(x_t, self.snr_scale)
        return self.decoder(boxes_norm, timesteps, memory, patch_raw=praw)

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def ddim_sample(self, num_proposals, pixel_values=None, texts=None,
                    patch_raw=None, text_raw=None, eta=1.0, generator=None,
                    return_all_rounds=False):
        """Generate N boxes from pure noise.

        x_T ~ N(0, I) with std 1.0 — NOT scaled by snr_scale (round 1's bug 3).
        Each step: predict x_start -> CLAMP -> RECOMPUTE pred_noise from the
        clamped version.

        `return_all_rounds=True` hands back every refinement round of the FINAL
        DDIM step instead of just the last one — used by
        tools/measure_box_quality.py to answer "does iterating help?".

        `eta=1.0` is DiffusionDet's default (`detector.py:97`) — DDIM degenerates
        into DDPM. A measured consequence, NOT a bug: on the first step
        (t=999 -> 749), sigma=0.925 and c=0.0000, so pred_noise is multiplied by
        zero and the whole step is `x_start*sqrt(ab_next) + noise`. Set eta=0 for
        deterministic DDIM.
        """
        need_raw = self.decoder.roi is not None
        out = self.encoder(pixel_values, texts, patch_raw, text_raw,
                           return_patch_raw=need_raw)
        memory, praw = out if need_raw else (out, None)
        B, dev = memory.shape[0], memory.device

        _check_generator(generator, dev)
        img = torch.randn(B, num_proposals, 4, device=dev, generator=generator)
        boxes = logits = None

        rounds = None
        for t, t_next in ddim_time_pairs(self.num_timesteps, self.sampling_steps):
            tb = torch.full((B,), t, dtype=torch.long, device=dev)
            out = self.decoder(decode_diffusion(img, self.snr_scale), tb,
                               memory, patch_raw=praw)
            # C1 returns a list of rounds. The LAST one is the prediction, exactly
            # as V-DETR does (`outputs = intermediate[-1]`, vdetr_transformer.py:445);
            # the earlier rounds exist for deep supervision during training.
            # Taking outs[0] by mistake would still produce valid boxes and only a
            # slightly worse AP -- no assertion would catch it, hence the test.
            if isinstance(out, list):
                rounds = out
                boxes, logits = out[-1]
            else:
                boxes, logits = out

            x_start = encode_diffusion(boxes, self.snr_scale)          # already in range
            if t_next < 0:
                break

            pred_noise = predict_noise_from_start(img, t, x_start, self.alphas_cumprod)
            a, a_next = self.alphas_cumprod[t], self.alphas_cumprod[t_next]
            sigma = eta * ((1 - a / a_next) * (1 - a_next) / (1 - a)).clamp(min=0).sqrt()
            c = (1 - a_next - sigma ** 2).clamp(min=0).sqrt()
            img = (x_start * a_next.sqrt() + c * pred_noise
                   + sigma * torch.randn(img.shape, device=dev, generator=generator))

        if return_all_rounds:
            if rounds is None:
                raise ValueError("return_all_rounds=True but the model has no "
                                 "refinement rounds (refine_rounds=0)")
            return rounds
        return boxes, logits


def build_model(cfg, dropout=None):
    """Config -> CELocDetector. ONE construction site for every entry point.

    Six scripts were each spelling out the same eleven arguments; adding
    EXPERIMENT A.2's `n_class`/`use_text` to all of them by hand is exactly how a
    tool ends up quietly building a DIFFERENT model than the one being trained.

    `dropout=0.0` is passed by eval/visualise/measure tools; training passes None
    to take the config's value.
    """
    m = cfg["model"]
    return CELocDetector(
        m["clip_name"], m["d_model"], m["n_layer"], m["n_head"],
        cfg["data"]["image_size"], cfg["diffusion"]["num_timesteps"],
        cfg["diffusion"]["snr_scale"], cfg["diffusion"]["sampling_steps"],
        m["dropout"] if dropout is None else dropout,
        m["freeze_clip"], roi_k=m.get("roi_k", 0),
        n_class=m.get("n_class", 1), use_text=m.get("use_text", True),
        refine_rounds=m.get("refine_rounds", 0),
        # Default True keeps every pre-E1 config (A, B, A.1, A.2, C1) meaning
        # exactly what it meant: with roi_k>0 that is EXPERIMENT B.
        roi_to_tgt=m.get("roi_to_tgt", True),
        score_roi=m.get("score_roi", False))

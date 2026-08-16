"""Timestep conditioning for the decoder-as-denoiser.

Two things live here:

  1. ``TimestepEncoder`` -- the classic sinusoidal positional encoding of ``t``
     followed by a small MLP, producing one embedding vector per image.
  2. The injectors that fold that embedding into the decoder queries. Two modes:

     ``film``  FiLM / adaLN-style scale-shift on the query, one block per decoder
               layer. This is what the released DiffuDINO actually does
               (``TimeStepBlock`` with ``emb_channels = 4 * d_model``).
     ``add``   Add a projection of ``t`` to ``query_pos``, so ``t`` reaches every
               sub-attention of the layer. Cheaper, closer to the paper's eq. 3
               (``MSDA(SA(q) + t)``).

Both are configurable; ``film`` is the default because it is the variant with
published numbers behind it.
"""

import math
from typing import Optional

import torch
from torch import Tensor, nn

INJECT_MODES = ("film", "add")


def sinusoidal_timestep_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Transformer sinusoidal encoding of a batch of timesteps.

    Args:
        t: (bs,) timesteps, any numeric dtype.
        dim: output width. Odd widths are zero-padded by one column.

    Returns:
        (bs, dim) float32.
    """
    t = t.float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEncoder(nn.Module):
    """``t -> sinusoidal -> Linear -> SiLU -> Linear``.

    Args:
        d_model: width of the sinusoidal encoding (the decoder's hidden size).
        out_dim: width of the produced embedding. FiLM wants ``4 * d_model``
            (it is consumed by a per-layer projection to ``2 * d_model``);
            ``add`` wants ``d_model``.
    """

    def __init__(self, d_model: int = 256, out_dim: Optional[int] = None, hidden_mult: int = 4):
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim if out_dim is not None else d_model
        hidden = d_model * hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """(bs,) -> (bs, out_dim)."""
        return self.mlp(sinusoidal_timestep_embedding(t, self.d_model))


class FiLMTimestepBlock(nn.Module):
    """Scale-shift conditioning of the queries on ``t``.

    ``h = x * (1 + scale) + shift`` with ``(scale, shift)`` projected from the
    timestep embedding.

    The output projection is zero-initialised, so at initialisation ``h == x``
    and the block is the identity. That matters here: we finetune from
    ``groundingdino_swint_ogc.pth``, and a randomly initialised modulation would
    corrupt the pretrained decoder on step 0.

    ``residual=True`` reproduces DiffuDINO's ``return x + h`` literally. Note
    that this returns ``2x`` at initialisation, which is fine when training from
    scratch (their setting) but destructive when finetuning (ours) -- hence the
    default is ``False``.
    """

    def __init__(self, d_model: int, emb_dim: int, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, 2 * d_model))
        nn.init.constant_(self.proj[1].weight, 0.0)
        nn.init.constant_(self.proj[1].bias, 0.0)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """``x``: (nq, bs, d) or (bs, nq, d). ``t_emb``: (bs, emb_dim)."""
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)  # (bs, d) each
        scale, shift = self._align(scale, x), self._align(shift, x)
        h = x * (1.0 + scale) + shift
        return x + h if self.residual else h

    @staticmethod
    def _align(v: Tensor, x: Tensor) -> Tensor:
        """Broadcast a per-image (bs, d) vector against ``x``.

        The decoder runs queries-first, ``(nq, bs, d)``, so the batch axis is 1.
        Accepting both layouts keeps this usable from either convention.
        """
        if x.ndim != 3:
            raise ValueError(f"expected a 3D query tensor, got shape {tuple(x.shape)}")
        bs = v.shape[0]
        if x.shape[1] == bs:  # (nq, bs, d)
            return v[None, :, :]
        if x.shape[0] == bs:  # (bs, nq, d)
            return v[:, None, :]
        raise ValueError(f"cannot align timestep embedding {tuple(v.shape)} with queries {tuple(x.shape)}")


class AddTimestepBlock(nn.Module):
    """Additive conditioning: ``x + W t``, zero-initialised (identity at init)."""

    def __init__(self, d_model: int, emb_dim: int):
        super().__init__()
        self.proj = nn.Linear(emb_dim, d_model)
        nn.init.constant_(self.proj.weight, 0.0)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        delta = self.proj(t_emb)
        return x + FiLMTimestepBlock._align(delta, x)


def build_timestep_modules(
    mode: str,
    d_model: int,
    num_layers: int,
    hidden_mult: int = 4,
    film_residual: bool = False,
    share_across_layers: bool = False,
    num_inject_points: int = 1,
):
    """Build ``(encoder, injectors)`` for a decoder with ``num_layers`` layers.

    ``injectors`` is a ``ModuleList`` of length ``num_layers`` (or 1 when
    ``share_across_layers``), applied inside each decoder layer.

    ``num_inject_points > 1`` (used for ``diff_time_inject_point="triple"``) builds
    that many *independently parameterised* injector sets instead of one -- the
    released DiffuDINO conditions self-attention, cross-attention and the FFN with
    three separate ``TimeStepBlock`` instances per layer, not one shared block
    reused three times. In that case ``injectors`` is a ``ModuleList`` of
    ``num_inject_points`` such per-layer ``ModuleList``s.
    """
    assert mode in INJECT_MODES, f"unknown timestep inject mode {mode!r}, expected one of {INJECT_MODES}"

    emb_dim = d_model * hidden_mult if mode == "film" else d_model
    encoder = TimestepEncoder(d_model=d_model, out_dim=emb_dim, hidden_mult=hidden_mult)

    count = 1 if share_across_layers else num_layers

    def make_injector_set():
        if mode == "film":
            return nn.ModuleList([FiLMTimestepBlock(d_model, emb_dim, residual=film_residual) for _ in range(count)])
        return nn.ModuleList([AddTimestepBlock(d_model, emb_dim) for _ in range(count)])

    if num_inject_points == 1:
        return encoder, make_injector_set()
    return encoder, nn.ModuleList([make_injector_set() for _ in range(num_inject_points)])

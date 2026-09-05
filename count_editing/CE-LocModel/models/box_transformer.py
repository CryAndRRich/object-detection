"""Decoder over N box tokens — Diffusion Policy's TransformerForDiffusion, with
3 changes so that boxes form a SET rather than an ordered sequence.

Each row [cx,cy,w,h] becomes ONE token. The same PE+Linear weights apply to every
row, so the only difference between boxes is their 4 coordinates -> PERMUTATION
EQUIVARIANT.

At each layer a box token does 3 things:
  1. self-attention over the N box tokens (including itself) — "35 sheep know
     about each other", which is where intra-category coherence lives
  2. cross-attention into the 1026 condition tokens — "a box reads the image at
     its own location", functionally RoIAlign but learned
  3. FFN

THREE MANDATORY CHANGES from the original `transformer_for_diffusion.py`:

  (a) DROP the learned `pos_emb` on box tokens. For actions, "the 3rd timestep"
      is meaningful; for boxes, "the 3rd box" is MEANINGLESS — the order is
      generated randomly by prepare_diffusion_concat and the matcher permutes
      freely. Keeping it teaches the network "slot 0 is usually real GT, slot 90
      is usually a placeholder" — exactly what it must NOT learn, since at
      inference every slot comes from randn.
      Position comes from SINUSOIDAL PE ON THE COORDINATES, not the array index.

  (b) `causal_attn=False`, dropping both tgt_mask and memory_mask. Box i must see
      every box j and the ENTIRE memory.

  (c) Dynamic `T`, `T_cond` -> N can differ freely between train (100) and eval
      (300). Possible precisely because index-based pos_emb is gone. `cond_pos_emb`
      is KEPT (memory IS ordered: patch 500 is always the same image region).

WHY SINUSOIDAL PE RATHER THAN A RAW Linear(4->D): Linear is linear, so position
enters the network as MAGNITUDE — a box at x=0.4 gives twice the vector of one at
x=0.2. Sinusoidal gives each position a SIGNATURE whose dot product decays with
distance, which is what attention actually needs. More importantly: ViT uses the
same mechanism for patch tokens, so boxes and patches SPEAK THE SAME LANGUAGE
about position. (DiffusionDet does not need this because RoIAlign samples using
the coordinates directly.)
"""

import math

import torch
import torch.nn as nn

__all__ = ["BoxTransformer", "SinusoidalCoordEmbedding"]


class SinusoidalCoordEmbedding(nn.Module):
    """Each coordinate -> `dim` sin/cos dims at several frequencies; 4 concatenated."""

    def __init__(self, dim=64, temperature=10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim, self.temperature = dim, temperature

    def forward(self, boxes):
        """[..., 4] in [0,1] -> [..., 4*dim]."""
        half = self.dim // 2
        freq = torch.arange(half, device=boxes.device, dtype=torch.float32)
        freq = self.temperature ** (2 * freq / self.dim)
        x = boxes.unsqueeze(-1) * 100.0 / freq          # scale 100: [0,1] -> useful range
        emb = torch.cat([x.sin(), x.cos()], dim=-1)
        return emb.flatten(-2)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard DDPM time embedding (same as `components.py::SinusoidalPosEmb`)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = math.log(10000) / (half - 1)
        f = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -f)
        a = t.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class BoxTransformer(nn.Module):
    def __init__(self, d_model=256, n_layer=6, n_head=8, coord_dim=64,
                 dim_feedforward=None, dropout=0.1, max_cond_len=1152):
        super().__init__()
        self.d_model = d_model

        self.coord_emb = SinusoidalCoordEmbedding(coord_dim)          # CHANGE (a)
        self.box_proj = nn.Linear(4 * coord_dim, d_model)
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.Mish(), nn.Linear(d_model * 4, d_model)
        )

        # memory IS ordered -> keep pos_emb for it (unlike box tokens).
        # Sized for the actual memory length (1 time token + patches), not a round
        # 4096: at 512px that is 1026, so a 4096 buffer left 75 % of the parameters
        # unused yet still owned by AdamW (3 state buffers) and still updated every
        # step. `max_cond_len` stays generous enough for a larger image size.
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, max_cond_len, d_model))
        nn.init.trunc_normal_(self.cond_pos_emb, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head,
            dim_feedforward=dim_feedforward or 4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,            # author's own comment: "important for stability"
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)

        self.box_head = nn.Linear(d_model, 4)      # DIRECT coordinates (not a delta)
        self.score_head = nn.Linear(d_model, 1)    # 1 dim: sigmoid == 2-dim softmax

    def forward(self, boxes_norm, timesteps, memory):
        """
        boxes_norm : [B, N, 4] cxcywh in [0,1]
        timesteps  : [B] long — ONE value per image
        memory     : [B, M, d_model] (text + patch tokens)
        -> (pred_boxes [B,N,4] in [0,1], logits [B,N])
        """
        tgt = self.box_proj(self.coord_emb(boxes_norm))               # CHANGE (a)

        t_tok = self.time_mlp(self.time_emb(timesteps)).unsqueeze(1)  # [B,1,D]
        mem = torch.cat([t_tok, memory], dim=1)                       # CHANGE (c): dynamic
        if mem.shape[1] > self.cond_pos_emb.shape[1]:
            raise ValueError(
                f"memory has {mem.shape[1]} tokens but cond_pos_emb holds only "
                f"{self.cond_pos_emb.shape[1]}; raise max_cond_len")
        mem = mem + self.cond_pos_emb[:, : mem.shape[1]]

        # CHANGE (b): NO masks at all
        h = self.ln_f(self.decoder(tgt=tgt, memory=mem))
        return self.box_head(h).sigmoid(), self.score_head(h).squeeze(-1)

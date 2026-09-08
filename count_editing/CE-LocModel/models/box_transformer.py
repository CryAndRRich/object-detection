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

  (d) EXPERIMENT C1, `refine_rounds > 0`: the decoder stops being a closed block.
      `nn.TransformerDecoder` IS a loop over `.layers` (verified: identical output
      to running the loop by hand), so C1 adds NO attention layer and no attention
      cost — it opens that loop to read a box after EVERY layer, and supervises all
      of them. `refine_rounds=0` keeps experiment A/B behaviour exactly.

      Two variables that must not be confused (silent bug — the model still runs,
      boxes stay in [0,1], loss still falls):
        anchor  the regression origin. FIXED for all rounds.
        box     the latest round's output. Only used to compute the RPE (C2).
      V-DETR assigns `proposal_center_normalized` ONCE before the loop
      (`vdetr_transformer.py:387`) and every layer regresses from that same origin,
      so errors do not compound across rounds. Our boxes start from PURE NOISE, so
      compounding six rounds from a random start is a real risk.

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

from models.roi_sampler import RoIFeatureSampler

__all__ = ["BoxTransformer", "SinusoidalCoordEmbedding", "update_box"]

MIN_WH = 0.005          # smallest real box in CE-130 is 0.0059 wide


def update_box(anchor, delta):
    """OBJECT-NORMALIZED update, exactly V-DETR's (`vdetr_transformer.py:271,278`):

        cx' = cx + d_cx * w          w' = w * exp(d_w)
        cy' = cy + d_cy * h          h' = h * exp(d_h)

    The centre shift is expressed in UNITS OF THE BOX'S OWN SIZE, so a small box is
    nudged gently and a large one can move far; the size is multiplicative, so it can
    never go negative and needs no clamping to stay positive. V-DETR measures +3.9
    AP50 for this over the plain version (Table 7).

    Chosen over `sigmoid(logit(box) + delta)`, which was the first draft: CE-130's
    median width is 0.069 -> logit -2.60, and its smallest box 0.0059 -> the
    derivative of logit is 1/(x(1-x)) = 170 right where the data is densest. That
    parameterisation has no counterpart in V-DETR either (a 3-D scene has no hard
    [0,1] border), so it would have been an untested invention inside an experiment
    that already has enough new parts.

    delta == 0 returns `anchor` EXACTLY (exp(0)=1, 0*w=0) — this is what makes
    C1 step 0 bit-identical to A.
    """
    cx, cy, w, h = anchor.unbind(-1)
    d_cx, d_cy, d_w, d_h = delta.unbind(-1)
    w2 = (w * d_w.exp()).clamp(min=MIN_WH)
    h2 = (h * d_h.exp()).clamp(min=MIN_WH)
    return torch.stack([(cx + d_cx * w).clamp(0.0, 1.0),
                        (cy + d_cy * h).clamp(0.0, 1.0), w2, h2], dim=-1)


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
                 dim_feedforward=None, dropout=0.1, max_cond_len=1152,
                 roi_k=0, roi_dim=768, n_class=1, refine_rounds=0):
        """`roi_k > 0` turns on EXPERIMENT B: each box additionally reads the frozen
        patch features sampled on a roi_k x roi_k grid INSIDE itself. roi_k=0 keeps
        experiment A's behaviour exactly.

        `n_class > 1` turns on EXPERIMENT A.2: the score head predicts WHICH of
        n_class categories the box holds, instead of "does this box match the one
        text I was given". A.2 also drops the text token from memory, so the class
        identity travels through the OUTPUT head rather than the INPUT text -- that
        one swap is the whole experiment. n_class=1 keeps A/B untouched.

        `refine_rounds > 0` turns on EXPERIMENT C1: read a box after every decoder
        layer instead of only after the last one, and supervise all of them.
        Must equal n_layer (one read per layer); 0 keeps A/B untouched.
        """
        super().__init__()
        self.d_model = d_model
        self.roi_k = roi_k

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
        # n_class=1 (A/B): sigmoid == a 2-dim softmax, "does this box match the text".
        # n_class=80 (A.2): one logit per COCO category, background = every logit low
        # -- exactly DiffusionDet's 80-not-81 convention.
        self.n_class = n_class
        self.score_head = nn.Linear(d_model, n_class)


        # EXPERIMENT B. Zero-initialised, so at step 0 this contributes nothing and
        # B is numerically identical to A -- the comparison changes one variable.
        #
        # Built LAST on purpose. Constructing it earlier would consume RNG draws and
        # shift every layer after it, so B and A would start from different weights
        # and "identical at step 0" would be false for a reason unrelated to the RoI
        # branch. Verified by tests/test_roi.py.
        self.roi = (RoIFeatureSampler(roi_dim, d_model, roi_k, dropout)
                    if roi_k > 0 else None)

        # EXPERIMENT C1. Built AFTER `self.roi` for the same reason `self.roi` is
        # built after the heads: `nn.Linear.__init__` CONSUMES RNG draws even when
        # the weights are zeroed straight afterwards (verified), so constructing
        # this earlier would shift every module built after it and "C1 step 0 ==
        # A" would be false for a reason unrelated to the experiment.
        self.refine_rounds = refine_rounds
        if refine_rounds:
            if refine_rounds != n_layer:
                raise ValueError(
                    f"refine_rounds ({refine_rounds}) must equal n_layer ({n_layer}): "
                    f"C1 reads a box after each EXISTING layer, it adds none")
            # SEPARATE head per round, as V-DETR does (`mlp_sep` defaults to True;
            # `vdetr_transformer.py:232` clones the heads num_layers+1 times).
            # Counted: 6 shared-vs-separate heads differ by 6,425 parameters =
            # 0.09 % of the 7.54M model, while EXPERIMENT B's 787K (10.4 %) is what
            # actually caused overfitting -- 102x apart, so "save parameters" is not
            # a reason here. Measured on a toy 6-round refine task, separate heads
            # also converge better (0.00087 vs 0.00134), and the features reaching
            # the head shift distribution across rounds (norm 1.44 -> 0.48), which a
            # single head would have to serve with one function.
            self.box_delta = nn.ModuleList([nn.Linear(d_model, 4)
                                            for _ in range(refine_rounds)])
            self.round_score = nn.ModuleList([nn.Linear(d_model, n_class)
                                              for _ in range(refine_rounds)])
            # ZERO-INIT the box heads (V-DETR `:170-173`). With object-normalized
            # updates delta=0 gives exp(0)=1 and cx + 0*w = cx, so round 1 returns
            # the anchor EXACTLY -> C1 at step 0 is bit-identical to A.
            # `round_score` is NOT zero-initialised: it takes no part in the update,
            # and zeroing it would make every score identical at step 0.
            for h in self.box_delta:
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)
        else:
            self.box_delta = self.round_score = None

    def forward(self, boxes_norm, timesteps, memory, patch_raw=None):
        """
        boxes_norm : [B, N, 4] cxcywh in [0,1]
        timesteps  : [B] long — ONE value per image
        memory     : [B, M, d_model] (text + patch tokens)
        patch_raw  : [B, P, d_in] raw patch tokens, only needed when roi_k > 0
        -> (pred_boxes [B,N,4] in [0,1], logits [B,N])
        """
        tgt = self.box_proj(self.coord_emb(boxes_norm))               # CHANGE (a)

        if self.roi is not None:
            if patch_raw is None:
                raise ValueError("roi_k > 0 needs patch_raw (the raw CLIP patch "
                                 "tokens); the model was built for experiment B")
            tgt = tgt + self.roi(patch_raw, boxes_norm)               # EXPERIMENT B

        t_tok = self.time_mlp(self.time_emb(timesteps)).unsqueeze(1)  # [B,1,D]
        mem = torch.cat([t_tok, memory], dim=1)                       # CHANGE (c): dynamic
        if mem.shape[1] > self.cond_pos_emb.shape[1]:
            raise ValueError(
                f"memory has {mem.shape[1]} tokens but cond_pos_emb holds only "
                f"{self.cond_pos_emb.shape[1]}; raise max_cond_len")
        mem = mem + self.cond_pos_emb[:, : mem.shape[1]]

        if self.refine_rounds:
            return self._forward_refine(tgt, mem, boxes_norm)

        # CHANGE (b): NO masks at all
        h = self.ln_f(self.decoder(tgt=tgt, memory=mem))
        logits = self.score_head(h)
        # [B,N] for n_class=1 so A/B keep their exact tensor shapes; [B,N,C] for A.2.
        return self.box_head(h).sigmoid(), (logits.squeeze(-1) if self.n_class == 1
                                            else logits)

    def _forward_refine(self, h, mem, anchor):
        """EXPERIMENT C1 — read a box after every layer. Returns a LIST of
        `refine_rounds` (boxes, logits) pairs, oldest first.

        `anchor` is the regression origin and NEVER changes; `box` is only what the
        latest round produced. Writing `box = update(box, delta)` instead would
        compound errors across six rounds from a purely random start — and nothing
        would flag it, which is why tests/test_refine.py checks that a CONSTANT
        non-zero delta yields six IDENTICAL outputs.
        """
        outs = []
        for r, layer in enumerate(self.decoder.layers):
            h = layer(h, mem)                       # the SAME 6 layers as A
            hn = self.ln_f(h)
            box = update_box(anchor, self.box_delta[r](hn))
            logits = self.round_score[r](hn)
            outs.append((box, logits.squeeze(-1) if self.n_class == 1 else logits))
        return outs

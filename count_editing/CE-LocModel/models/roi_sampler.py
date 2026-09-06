"""RoI sampling — give each box the image features INSIDE it (EXPERIMENT B).

WHY THIS EXISTS. Experiment A gave the decoder 1024 positioned patch tokens and
let cross-attention find the right one. Looking at its predictions showed it only
half worked: boxes land on the right REGION (marbles in the corner get boxes in
the corner) but at a near-constant SIZE -- too big for peas, too small for
elephants. Measured consequence: recall 0.217 on mid-sized objects but 0.041 when
objects are tiny.

The reason is that A asks the network to LEARN a mapping between two unrelated
coordinate systems: the box's sinusoidal PE of (cx,cy), and the patch tokens'
`cond_pos_emb`, which starts as random noise. Nothing ties them together; 1,911
images were not enough to learn it. DiffusionDet never learns this -- RoIAlign
crops features at the box coordinates directly, so the spatial relation is exact
by construction.

WHAT THIS DOES. Sample the frozen CLIP feature map at a k x k grid of points laid
out INSIDE each box. Because the sample locations scale with (w,h), the resulting
feature depends on the box's size, which is exactly the signal A was missing.

MEASURED, on real CE-130 test images (frozen CLIP ViT-B/16 @512):

  grid    AUC object/background    AUC correct-size vs 2x-too-big
  1x1              0.990                        0.000
  3x3              0.989                        0.896

  The 1x1 column is the point: sampling only the centre gives the SAME vector
  however large the box is, so it carries no size information at all (AUC 0.000
  is not noise -- it is a proof of indifference). A 3x3 grid recovers that signal
  without giving up any object/background separation.

WHY k=3 AND NOT DiffusionDet's 7x7: CE-130 objects have a median size of about
2.0 x 1.7 patches on the 32x32 grid. Putting 49 sample points inside a 2-patch
box oversamples the same handful of features at 5.4x the cost.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RoIFeatureSampler", "box_grid_points"]


def box_grid_points(boxes_norm, k=3):
    """[B,N,4] cxcywh in [0,1] -> [B,N,k*k,2] sample points in [0,1] image coords.

    Points sit at cell CENTRES of a k x k grid spanning the box, i.e. offsets
    (i+0.5)/k - 0.5 for i in 0..k-1. Cell centres, not corners, so no sample sits
    exactly on the box edge where it would straddle object and background.
    """
    dev, dt = boxes_norm.device, boxes_norm.dtype
    off = (torch.arange(k, device=dev, dtype=dt) + 0.5) / k - 0.5      # [k]
    gy, gx = torch.meshgrid(off, off, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)        # [k*k, 2]

    cx, cy, w, h = boxes_norm.unbind(-1)                               # [B,N] each
    x = cx.unsqueeze(-1) + pts[:, 0] * w.unsqueeze(-1)                 # [B,N,k*k]
    y = cy.unsqueeze(-1) + pts[:, 1] * h.unsqueeze(-1)
    return torch.stack([x, y], dim=-1)                                 # [B,N,k*k,2]


class RoIFeatureSampler(nn.Module):
    """Frozen patch tokens + boxes -> one d_model vector per box.

    Layout follows the "project first, then mix" option: a shared Linear(d_in ->
    d_model) is applied to each of the k*k sampled positions, then one
    Linear(k*k*d_model -> d_model) mixes them. That is 0.79M parameters at k=3
    versus 1.77M for a single Linear(k*k*d_in -> d_model), and it keeps what
    matters -- the centre and the edges stay separate, which is the signal that
    tells a box it is too big. Pooling the k*k samples first would be cheaper
    still but would average exactly that away.

    The output projection is ZERO-INITIALISED. At step 0 this module contributes
    exactly nothing, so experiment B starts as a bit-exact copy of A and the
    comparison isolates one variable. The branch is not frozen: for y = Wx + b
    with W = 0, dL/dW = delta * x^T is still non-zero, so it starts learning on
    the first step. Watching ||W|| grow is also a cheap early read on whether the
    RoI signal is being used at all.
    """

    def __init__(self, d_in=768, d_model=256, k=3, dropout=0.0):
        super().__init__()
        self.k = k
        self.proj_point = nn.Linear(d_in, d_model)
        self.out = nn.Linear(k * k * d_model, d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, patch_tokens, boxes_norm, grid=None):
        """
        patch_tokens : [B, P, d_in] raw CLIP patch tokens (P must be a square)
        boxes_norm   : [B, N, 4] cxcywh in [0,1]
        grid         : optional int, the patch grid side; inferred from P if None
        -> [B, N, d_model]
        """
        B, P, d_in = patch_tokens.shape
        g = grid or int(round(P ** 0.5))
        assert g * g == P, f"{P} patch tokens is not a square grid"

        fmap = patch_tokens.transpose(1, 2).reshape(B, d_in, g, g)

        pts = box_grid_points(boxes_norm, self.k)                      # [B,N,k*k,2]
        # grid_sample wants [-1,1] with align_corners=False matching the
        # pixel-centre convention the patch grid already uses.
        samp = F.grid_sample(fmap, pts * 2.0 - 1.0, mode="bilinear",
                             padding_mode="border", align_corners=False)
        # [B, d_in, N, k*k] -> [B, N, k*k, d_in]
        samp = samp.permute(0, 2, 3, 1)

        h = self.proj_point(samp.to(self.proj_point.weight.dtype))     # [B,N,k*k,d_model]
        h = self.drop(h.flatten(-2))                                   # [B,N,k*k*d_model]
        return self.out(h)                                             # [B,N,d_model]

    def branch_norm(self):
        """||W|| of the zero-initialised output layer.

        Reported every epoch: if it is still ~0 after a few dozen epochs, the
        network is saying the RoI features are useless -- stop early rather than
        burn 300 epochs finding out.
        """
        return float(self.out.weight.detach().norm())

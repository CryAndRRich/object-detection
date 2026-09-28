"""STAGE 1 -- the minimal training-free pipeline.

    SD2 self-attention -> affinity A -> p-Laplacian from a grid of one-hot
    prompts -> ONE threshold -> connected components -> boxes

What stage 1 deliberately does NOT have (all of it is stage 2):
  - KL clustering of the propagated maps
  - six granularity levels
  - multi-level NMS

Why split at all: the only question stage 1 has to answer is whether the
mechanism produces anything at all on CE-130, and in particular whether p=1.6
beats p=2.0. Adding clustering first would mean two new mechanisms failing or
succeeding together with no way to tell which did what.

Run (on the server -- see README):
    python tools/check_plaplacian_vs_p2.py --config config/stage1.py --split val
    python tools/run_stage1.py            --config config/stage1.py --split val
"""

from .base import Diffu2SegConfig

# Everything already defaults to stage 1 in base.py; this file exists so the
# --config argument names an explicit, quotable configuration rather than
# "the defaults".
cfg = Diffu2SegConfig(stage=1).validate()

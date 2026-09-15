r"""One image -> a list of boxes. The whole training-free pipeline, assembled.

    image (canvas x canvas uint8)
      |
      | SD2 VAE encode, ONE denoising step, hook every self-attention
      v
    A  (N, N) row-stochastic graph over patch tokens          [attention, affinity]
      |
      | K one-hot seeds on a regular grid, padding cells dropped    [prompts]
      | p-Laplacian propagation, all K in parallel               [plaplacian]
      v
    f  (K, N) soft object maps
      |
      | threshold -> connected component containing the seed         [masks]
      | outer-edge box -> filter -> dedup                            [boxes]
      v
    boxes (M, 4) cxcywh in [0, 1]

NOT A MODEL. There is no parameter anywhere in this path -- the masks are the
solution of an optimisation problem on a graph, reached by fixed-point
iteration. SD2 is frozen and is used only to supply the graph.

STAGE 1 vs STAGE 2: stage 1 takes a single threshold and splits by spatial
connectivity. Stage 2 replaces that readout with KL clustering over six levels
plus multi-level NMS, and changes nothing before it. Splitting them this way
means gate 1 measures the propagation mechanism alone; if clustering came first,
a null result would not say which half was responsible.
"""

import numpy as np
import torch

from .affinity import blend_timesteps, to_affinity
from .boxes import dedup_boxes, filter_boxes, masks_to_boxes
from .masks import extract_masks
from .plaplacian import plaplacian_propagate
from .prompts import build_prompt_grid, f0_onehot

__all__ = ["build_affinity", "segment_image"]


def build_affinity(image, cfg, aggregator):
    """image -> (N, N) row-stochastic affinity on the aggregator's device."""
    attns = aggregator.extract_attention(image, timesteps=list(cfg.timesteps))
    if len(attns) > 1:
        attn = blend_timesteps(attns, list(cfg.timestep_weights))
    else:
        attn = attns[0]
    return to_affinity(attn, tau_att=cfg.tau_att, dtype=torch.float32)


def segment_image(image, valid_h, cfg, aggregator=None, A=None):
    """Segment one padded canvas image.

    Args:
        image:      (canvas, canvas, 3) uint8.
        valid_h:    fraction of canvas height that is real image; the rest is
                    padding and must not be seeded.
        aggregator: StableDiffusion2AttentionAggregator. Optional if `A` given.
        A:          precomputed (N, N) affinity. Passing it lets one SD2 forward
                    serve a whole parameter sweep -- A is 0.07 GB at r=64 and
                    cannot be cached to disk across 908 images (63 GB), so a
                    sweep that re-extracted per setting would pay for SD2 again
                    every time.

    Returns a dict with `boxes` plus the diagnostics the gate checks:
    n_iter (did propagation converge, or did it hit the cap), f_max (the anchor
    is weak and the solution can decay), and the filter counts (a large
    n_in_padding means the padding logic is broken, not that the image is odd).
    """
    if A is None:
        assert aggregator is not None, "pass either an aggregator or a precomputed A"
        A = build_affinity(image, cfg, aggregator)

    r = cfg.grid_r
    assert A.shape == (r * r, r * r), \
        f"affinity is {tuple(A.shape)}, expected {(r * r, r * r)} for grid_r={r}"

    cells = build_prompt_grid(r, cfg.prompt_stride_cells, valid_h=valid_h,
                             min_valid_frac=cfg.min_valid_frac)
    if len(cells) == 0:
        return _empty(r, "no prompt survived the padding filter")

    f0 = f0_onehot(cells, r, device=A.device, dtype=A.dtype)
    f, n_iter, _ = plaplacian_propagate(
        A, f0, p=cfg.p, lam=cfg.lam, tau_prop=cfg.tau_prop,
        max_iter=cfg.max_iter, g_eps=cfg.g_eps,
    )

    f_np = f.detach().float().cpu().numpy()
    masks, kept_cells = extract_masks(
        f_np, cells, r, mode=cfg.mask_threshold_mode, quantile=cfg.mask_quantile,
        connectivity=cfg.connectivity, min_component_cells=cfg.min_component_cells,
        rel_floor=cfg.mask_rel_floor,
    )

    raw = masks_to_boxes(masks, r, cfg.canvas)
    kept, info = filter_boxes(raw, r, cfg.canvas, min_box_cells=cfg.min_box_cells,
                              max_area_frac=cfg.max_box_area_frac, valid_h=valid_h)
    boxes, _ = dedup_boxes(kept, iou_thr=cfg.dedup_iou)

    return {
        "boxes": boxes,
        "masks": masks,
        "kept_cells": kept_cells,
        "n_prompts": int(len(cells)),
        "n_masks": int(len(masks)),
        "n_boxes": int(len(boxes)),
        "n_iter": int(n_iter),
        "converged": bool(n_iter < cfg.max_iter),
        "f_max": float(f_np.max()) if f_np.size else 0.0,
        "filter_info": info,
    }


def _empty(r, reason):
    return {
        "boxes": np.zeros((0, 4)),
        "masks": np.zeros((0, r, r), dtype=bool),
        "kept_cells": np.zeros((0, 2), dtype=np.int64),
        "n_prompts": 0, "n_masks": 0, "n_boxes": 0,
        "n_iter": 0, "converged": True, "f_max": 0.0,
        "filter_info": {"n_in": 0, "n_kept": 0, "n_degenerate": 0,
                        "n_too_large": 0, "n_in_padding": 0},
        "reason": reason,
    }

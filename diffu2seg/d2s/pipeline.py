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

STAGE 1 vs STAGE 2 -- A DIFFERENT READOUT, NOT A BETTER ONE:

  stage 1   each map thresholded at a quantile -> connected component around
            its own seed -> box. Independent blobs. NOT in the paper; it exists
            so gate 1 can measure the propagation mechanism by itself.
  stage 2   Algorithm 2 of the paper: normalise maps to distributions, cluster
            prompts by symmetric KL, cut the dendrogram at 6 log-spaced heights,
            average within cluster, UPSAMPLE, then ARGMAX ACROSS CLUSTERS ->
            a PARTITION -> connected components -> area-descending NMS.

Stage 2 never thresholds a map. Reading stage 1 as "stage 2 without the
clustering" is wrong: the readouts differ in kind, which is why the paper's
numbers are only comparable to stage 2.
"""

import numpy as np
import torch

from .affinity import blend_timesteps, to_affinity
from .boxes import dedup_boxes, filter_boxes, masks_to_boxes
from .masks import extract_masks
from .merging import merge_maps_to_masks
from .plaplacian import plaplacian_propagate
from .prompts import build_prompt_grid, f0_onehot

__all__ = ["build_affinity", "segment_image", "masks_full_to_boxes"]


def build_affinity(image, cfg, aggregator):
    """image -> (N, N) row-stochastic affinity on the aggregator's device."""
    attns = aggregator.extract_attention(image, timesteps=list(cfg.timesteps))
    if len(attns) > 1:
        attn = blend_timesteps(attns, list(cfg.timestep_weights))
    else:
        attn = attns[0]
    return to_affinity(attn, tau_att=cfg.tau_att, dtype=torch.float32)


def segment_image(image, valid_h, cfg, aggregator=None, A=None, valid_w=1.0,
                  orig_hw=None):
    """Segment one padded canvas image.

    Args:
        image:      (canvas, canvas, 3) uint8.
        valid_h:    fraction of canvas height that is real image; the rest is
                    padding and must not be seeded.
        valid_w:    same for width. 1.0 for CE-130 (W >= H always, so padding
                    only ever lands at the bottom); COCO portrait images pad on
                    the right and need this.
        orig_hw:    (H, W) of the ORIGINAL image. Required when cfg.stage >= 2:
                    Algorithm 2 upsamples to that size before the argmax, and
                    its a_min = 100 is counted in original pixels.
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
                             min_valid_frac=cfg.min_valid_frac, valid_w=valid_w)
    if len(cells) == 0:
        return _empty(r, "no prompt survived the padding filter", orig_hw)

    f0 = f0_onehot(cells, r, device=A.device, dtype=A.dtype)
    f, n_iter, _ = plaplacian_propagate(
        A, f0, p=cfg.p, lam=cfg.lam, tau_prop=cfg.tau_prop,
        max_iter=cfg.max_iter, g_eps=cfg.g_eps,
    )

    f_np = f.detach().float().cpu().numpy()

    if cfg.stage >= 2:
        # ALGORITHM 2. Needs the ORIGINAL image size, because the paper
        # upsamples before the argmax and filters by area in original pixels.
        assert orig_hw is not None, \
            "stage 2 needs orig_hw=(H, W): Algorithm 2 upsamples to the original " \
            "resolution before argmax, and a_min = 100 is in original pixels"
        H, W = orig_hw
        masks_full, merge_info = merge_maps_to_masks(
            f_np, cfg, H, W, valid_w=valid_w, valid_h=valid_h)
        masks_full = np.asarray(masks_full, dtype=bool) if masks_full \
            else np.zeros((0, H, W), dtype=bool)
        boxes = masks_full_to_boxes(masks_full, W, H, valid_w, valid_h)
        return {
            "boxes": boxes,
            "masks_full": masks_full,
            "masks": np.zeros((0, r, r), dtype=bool),
            "kept_cells": cells,
            "n_prompts": int(len(cells)),
            "n_masks": int(len(masks_full)),
            "n_boxes": int(len(boxes)),
            "n_iter": int(n_iter),
            "converged": bool(n_iter < cfg.max_iter),
            "f_max": float(f_np.max()) if f_np.size else 0.0,
            "merge_info": merge_info,
            "filter_info": {"n_in": merge_info["nms"]["n_in"],
                            "n_kept": merge_info["nms"]["n_kept"],
                            "n_degenerate": 0, "n_too_large": 0,
                            "n_in_padding": 0},
        }

    masks, kept_cells = extract_masks(
        f_np, cells, r, mode=cfg.mask_threshold_mode, quantile=cfg.mask_quantile,
        connectivity=cfg.connectivity, min_component_cells=cfg.min_component_cells,
        rel_floor=cfg.mask_rel_floor,
    )

    raw = masks_to_boxes(masks, r, cfg.canvas)
    kept, info = filter_boxes(raw, r, cfg.canvas, min_box_cells=cfg.min_box_cells,
                              max_area_frac=cfg.max_box_area_frac, valid_h=valid_h,
                              valid_w=valid_w)
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


def masks_full_to_boxes(masks, W, H, valid_w=1.0, valid_h=1.0):
    """(M, H, W) bool at original resolution -> (M, 4) cxcywh in [0, 1] on canvas.

    Stage 2 produces masks directly at image resolution, so the box comes from
    the mask's own extent rather than from latent cells -- no outer-edge rule
    and no 8 px quantisation.
    """
    masks = np.asarray(masks, dtype=bool)
    if len(masks) == 0:
        return np.zeros((0, 4))
    out = np.zeros((len(masks), 4))
    for i, m in enumerate(masks):
        ys, xs = np.where(m)
        if not len(ys):
            continue
        # +1 on the far edge: a pixel spans [x, x+1).
        x1, x2 = xs.min() / W * valid_w, (xs.max() + 1) / W * valid_w
        y1, y2 = ys.min() / H * valid_h, (ys.max() + 1) / H * valid_h
        out[i] = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    return out


def _empty(r, reason, orig_hw=None):
    H, W = orig_hw if orig_hw else (1, 1)
    return {
        "boxes": np.zeros((0, 4)),
        "masks": np.zeros((0, r, r), dtype=bool),
        "masks_full": np.zeros((0, H, W), dtype=bool),
        "merge_info": {"n_prompts_used": 0, "n_levels": 0, "per_level": [],
                       "nms": {"n_in": 0, "n_kept": 0, "n_suppressed": 0,
                               "n_over_cap": 0}},
        "kept_cells": np.zeros((0, 2), dtype=np.int64),
        "n_prompts": 0, "n_masks": 0, "n_boxes": 0,
        "n_iter": 0, "converged": True, "f_max": 0.0,
        "filter_info": {"n_in": 0, "n_kept": 0, "n_degenerate": 0,
                        "n_too_large": 0, "n_in_padding": 0},
        "reason": reason,
    }

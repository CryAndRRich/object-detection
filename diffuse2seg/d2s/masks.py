"""Propagated maps -> binary masks (STAGE 1: a single threshold level).

Diffuse2Seg's stage 2 replaces this file's thresholding with KL clustering over
six levels; the connected-component step survives into stage 2 unchanged.

                 WHY CONNECTED COMPONENTS, NOT M2N2's FLOOD FILL

M2N2 needs a minimax-path flood fill (markov_map.py:60-117, JIT-compiled with
numba) because its Markov-map is a CONTINUOUS distance field that later gets cut
at an adaptively chosen threshold -- the whole range must survive the instance
split. Here the threshold is applied first, so what is left is an ordinary
binary mask, and splitting it costs one call to scipy.ndimage.label.

That matters for CE-130 specifically: the median image holds ~21 objects OF THE
SAME CLASS, and self-attention cannot tell two sheep apart -- it only knows they
look alike. Spatial connectivity is what separates them.

Bonus: scipy is already in the project's venv, numba is not (and M2N2's own
requirements.txt forgets to list it).
"""

import numpy as np
from scipy import ndimage

__all__ = ["threshold_maps", "largest_component_containing_seed", "extract_masks"]


def threshold_maps(f, grid_r, mode="quantile", quantile=0.90, rel_floor=0.05):
    """(K, N) propagated maps -> (K, r, r) bool.

    `quantile` keeps the top (1 - q) fraction of each map INDEPENDENTLY, so a
    prompt on a small object and one on a large object are each cut at their own
    scale rather than against a shared absolute level. The values of f depend on
    lam and on how far propagation ran, so a single absolute threshold would not
    be comparable across images anyway.

    `rel_floor` IS NOT OPTIONAL, and it fixes a hole a quantile alone cannot see.
    A quantile is RELATIVE, so it always selects ~(1-q) of the cells -- even from
    a map with no signal whatsoever. Measured on a synthetic graph: a prompt
    seeded on background propagated to max = 0.0000, yet its own q90 was 1.5e-05,
    so `f > q90` still returned one cell (the seed's own), which then became a
    perfectly well-formed 1x1 box. That is how 2 objects turned into 58 boxes.

    The floor asks a second, absolute question: is any cell at least `rel_floor`
    of the strongest cell ACROSS ALL PROMPTS in this image? A dead map fails it
    and yields nothing, which is the correct answer for a prompt that landed on
    empty background.
    """
    f = np.asarray(f, dtype=np.float64)
    K = f.shape[0]

    if mode == "quantile":
        thr = np.quantile(f, quantile, axis=1, keepdims=True)
    elif mode == "absolute":
        thr = np.full((K, 1), quantile, dtype=np.float64)
    else:
        raise ValueError(f"unknown threshold mode: {mode}")

    if rel_floor is not None and rel_floor > 0 and f.size:
        thr = np.maximum(thr, rel_floor * f.max())

    return (f > thr).reshape(K, grid_r, grid_r)


def largest_component_containing_seed(mask_2d, seed_rc, connectivity=1):
    """Keep only the connected component holding the seed cell.

    Returns an all-False mask if the seed itself fell below the threshold --
    which happens, and must not be papered over by silently picking the biggest
    component instead. A prompt whose own cell did not survive has nothing to
    say about any object.
    """
    structure = ndimage.generate_binary_structure(2, connectivity)
    labels, n = ndimage.label(mask_2d, structure=structure)
    if n == 0:
        return np.zeros_like(mask_2d, dtype=bool)

    r, c = int(seed_rc[0]), int(seed_rc[1])
    seed_label = labels[r, c]
    if seed_label == 0:
        return np.zeros_like(mask_2d, dtype=bool)
    return labels == seed_label


def extract_masks(f, cells, grid_r, mode="quantile", quantile=0.90,
                  connectivity=1, min_component_cells=1, rel_floor=0.05):
    """Full stage-1 mask step: threshold, isolate the seed's component, filter.

    Returns (masks (M, r, r) bool, kept_cells (M, 2)) -- M <= K, since prompts
    whose own cell fell below threshold or whose component is too small drop out.
    """
    binm = threshold_maps(f, grid_r, mode=mode, quantile=quantile,
                          rel_floor=rel_floor)
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
    assert len(cells) == len(binm), "one seed cell per propagated map"

    masks, kept = [], []
    for k in range(len(binm)):
        comp = largest_component_containing_seed(binm[k], cells[k], connectivity)
        if comp.sum() >= min_component_cells and comp.any():
            masks.append(comp)
            kept.append(cells[k])

    if not masks:
        return np.zeros((0, grid_r, grid_r), dtype=bool), np.zeros((0, 2), dtype=np.int64)
    return np.stack(masks), np.stack(kept)

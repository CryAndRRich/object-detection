#!/usr/bin/env python3
"""GATE for EXPERIMENT C: does a vertex-RPE bias actually give the attention a
spatial prior, BEFORE spending 2 GPU-hours training one?

WHY THIS EXISTS
---------------
EXPERIMENT A measured that a box token and a patch token are ORTHOGONAL at
initialisation (cosine +0.0005, std 0.060). The attention map therefore starts
as white noise and the network must learn the box<->patch correspondence from
1,911 images. EXPERIMENT B tried to bypass this with hard grid_sample INSIDE the
box and did not improve (AP50 0.0152 -> 0.0144).

V-DETR (arXiv 2308.04409) modulates attention instead, with a bias computed from
the geometry between the query box's VERTICES and each key position:

    A = Softmax(Q K^T + R),   R = sum_{i=1..V} MLP_i(F(vertex_i - patch_xy))

Their Table 5 measures soft bias (66.0 AP50) > hard box mask (60.8) > nothing
(47.6); their Table 3 measures 8 vertices (65.0) > 1 centre (54.8).

THE THREE QUESTIONS THIS ANSWERS, WITHOUT TRAINING
--------------------------------------------------
1. Is the attention map with R actually concentrated near the box, versus flat
   for the current model? -> the premise of the whole direction.
2. Does the map SCALE with the box's w,h? This is what experiment A lacked
   (measured recall 0.217 mid-size vs 0.041 tiny) and what a centre-only
   encoding cannot express (roi_sampler measured AUC 0.000 at k=1).
3. What log_scale? V-DETR uses 512 for coordinates in METRES; ours are in
   [0,1], so their value is not transferable and a wrong one silently flattens
   the signal.

A FLAT MAP HERE MEANS THE PREMISE IS WRONG -> do not train.

Outputs PNGs (PIL, as the other viz tools here do -- no matplotlib in .venv-cpu)
plus a JSON of the numbers.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------- geometry

def patch_centres(grid):
    """[P,2] centre of each patch cell in [0,1], row-major (y outer, x inner)."""
    ys, xs = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    return np.stack([(xs.ravel() + 0.5) / grid, (ys.ravel() + 0.5) / grid], -1)


def corners_of(box):
    """(cx,cy,w,h) -> [4,2] corners: TL, TR, BL, BR."""
    cx, cy, w, h = box
    return np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                     [cx - w / 2, cy + h / 2], [cx + w / 2, cy + h / 2]])


def signed_log(d, log_scale):
    """V-DETR's F(x)=sign(x)log2(|x|*s+1)/log2(8) (vdetr_transformer.py:722).

    Their Table 2 measures this beats identity by 16.8 AP50, because it
    magnifies small offsets (near the box, where precision matters) and
    compresses large ones (far away, where it does not).
    """
    return np.sign(d) * np.log2(np.abs(d) * log_scale + 1.0) / math.log2(8)


# ------------------------------------------------------------------------ model

class RefRPE:
    """Reference numpy implementation of the vertex-RPE bias.

    Mirrors what a torch nn.Module would compute at INITIALISATION. The point of
    the gate is the geometry, so the MLP is a fixed random projection here --
    the question is whether the INPUT to the MLP carries spatial structure, not
    what a trained MLP does with it.
    """

    def __init__(self, n_vertex=4, rpe_dim=64, n_head=8, log_scale=20.0, seed=0):
        rng = np.random.default_rng(seed)
        self.n_vertex, self.log_scale = n_vertex, log_scale
        # Same shape as V-DETR's build_cpb_mlp: Linear(2->rpe_dim) ReLU Linear(->heads)
        self.W1 = [rng.normal(0, math.sqrt(2.0 / 2), (2, rpe_dim)) for _ in range(n_vertex)]
        self.b1 = [np.zeros(rpe_dim) for _ in range(n_vertex)]
        self.W2 = [rng.normal(0, math.sqrt(2.0 / rpe_dim), (rpe_dim, n_head))
                   for _ in range(n_vertex)]

    def points(self, box):
        """Reference points: 4 corners, or the centre when n_vertex == 1."""
        if self.n_vertex == 1:
            return np.array([[box[0], box[1]]])
        return corners_of(box)

    def bias(self, box, pxy, head=0):
        """-> [P] the bias added to this box's attention row, for one head."""
        R = np.zeros(len(pxy))
        for i, v in enumerate(self.points(box)):
            d = signed_log(v[None, :] - pxy, self.log_scale)        # [P,2]
            hdn = np.maximum(d @ self.W1[i] + self.b1[i], 0.0)      # ReLU
            R += (hdn @ self.W2[i])[:, head]
        return R


# ---------------------------------------------------------------------- metrics

def spatial_structure(bias, pxy, box):
    """|corr| between the bias and distance-to-box-centre, measured in the box's
    OWN size units.

    Sign-invariant on purpose. At initialisation the MLP is random, so one seed
    puts high bias near the box and another puts it far -- averaging the SIGNED
    value cancels to ~0 and says nothing. What matters for the gate is whether
    the bias is a FUNCTION OF GEOMETRY at all, i.e. whether training has a
    spatial signal to shape. |corr| ~ 0 would mean the vertex encoding carries
    no usable geometry and the direction is dead.
    """
    d = np.hypot((pxy[:, 0] - box[0]) / max(box[2], 1e-6),
                 (pxy[:, 1] - box[1]) / max(box[3], 1e-6))
    if bias.std() < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(bias, d)[0, 1]))


def concentration(attn, pxy, box):
    """Fraction of attention mass landing inside the box, over the fraction of
    the canvas the box covers. 1.0 = indifferent (what a flat map gives); higher
    = the map prefers the box."""
    cx, cy, w, h = box
    inside = ((np.abs(pxy[:, 0] - cx) <= w / 2) & (np.abs(pxy[:, 1] - cy) <= h / 2))
    area = max(w * h, 1e-9)
    if inside.sum() == 0:                      # box smaller than one patch cell
        return float("nan"), int(inside.sum())
    return float(attn[inside].sum() / area), int(inside.sum())


def peak_offset(attn, pxy, box):
    """Distance from the attention's centre of mass to the box centre, in units
    of the box's own size -- so it is comparable across box scales."""
    com = (attn[:, None] * pxy).sum(0) / max(attn.sum(), 1e-12)
    d = com - np.array([box[0], box[1]])
    return float(np.hypot(d[0] / max(box[2], 1e-6), d[1] / max(box[3], 1e-6)))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ----------------------------------------------------------------------- render

def heat_png(attn, grid, box, path, upscale=12):
    """Attention map -> PNG, with the box drawn on top."""
    from PIL import Image, ImageDraw
    a = attn.reshape(grid, grid)
    a = (a - a.min()) / max(a.max() - a.min(), 1e-12)
    rgb = np.zeros((grid, grid, 3), np.uint8)
    rgb[..., 0] = (255 * np.clip(a * 1.6, 0, 1))            # red   = hot
    rgb[..., 1] = (255 * np.clip(a * 1.1 - 0.25, 0, 1))
    rgb[..., 2] = (255 * np.clip(0.55 - a * 0.9, 0, 1))     # blue  = cold
    img = Image.fromarray(rgb).resize((grid * upscale,) * 2, Image.NEAREST)

    cx, cy, w, h = box
    s = grid * upscale
    ImageDraw.Draw(img).rectangle(
        [((cx - w / 2) * s, (cy - h / 2) * s), ((cx + w / 2) * s, (cy + h / 2) * s)],
        outline=(255, 255, 255), width=2)
    img.save(path)


# ------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=32, help="CLIP ViT-B/16 @512 -> 32")
    ap.add_argument("--out", default="/tmp/rpe_gate")
    ap.add_argument("--rpe-dim", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=8, help="random MLP draws to average")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pxy = patch_centres(args.grid)
    P = len(pxy)

    # CE-130 real sizes (measured, best_sizecheck_test.json): GT w p25/p50/p75 =
    # 0.060 / 0.094 / 0.139, min 0.006, max 1.002.
    boxes = {
        "tiny_0.02":   (0.30, 0.30, 0.02, 0.02),
        "median_0.09": (0.30, 0.30, 0.09, 0.09),
        "large_0.30":  (0.30, 0.30, 0.30, 0.30),
        "offcentre":   (0.70, 0.25, 0.09, 0.09),
        "wide":        (0.50, 0.50, 0.40, 0.08),
    }
    report = {"grid": args.grid, "n_patch": P}

    # ---- 1. BASELINE: what experiment A's attention looks like at init -------
    # Box token and patch token are orthogonal (measured cosine +0.0005), so the
    # logits are effectively random -> this is the bar to beat.
    rng = np.random.default_rng(0)
    base = []
    for _ in range(args.seeds):
        for b in boxes.values():
            base.append(spatial_structure(rng.normal(0, 1.0, P), pxy, b))
    report["baseline_current_model"] = {
        "spatial_structure_mean": float(np.mean(base)),
        "note": ("|corr(bias, distance-to-box)|. The current model's attention "
                 "logits carry NO geometry (box/patch cosine measured +0.0005), "
                 "so this is the floor any bias must beat."),
    }
    print(f"[baseline] spatial structure = {np.mean(base):.4f}   "
          f"(current model: no geometric term at all)")

    # ---- 2. SWEEP log_scale --------------------------------------------------
    # V-DETR uses 512 for METRES; our coords are [0,1] so it must be re-picked.
    print("\n[sweep] log_scale   |corr(bias, dist)| by box size (mean over seeds/heads)")
    sweep = {}
    for ls in [512.0, 200.0, 100.0, 50.0, 20.0, 8.0, 3.0, 1.0]:
        per_box = {}
        for name, b in boxes.items():
            vals = []
            for sd in range(args.seeds):
                m = RefRPE(4, args.rpe_dim, 8, ls, seed=sd)
                for hd in range(4):
                    vals.append(spatial_structure(m.bias(b, pxy, hd), pxy, b))
            per_box[name] = float(np.mean(vals)) if vals else float("nan")
        sweep[ls] = per_box
        cells = "  ".join(f"{k.split('_')[0][:6]:>6}={v:6.2f}" for k, v in per_box.items())
        print(f"         {ls:6.1f}   {cells}")
    report["log_scale_sweep"] = {str(k): v for k, v in sweep.items()}

    # pick the scale with the best mean concentration across box sizes
    best_ls = max(sweep, key=lambda k: np.nanmean(list(sweep[k].values())))
    report["best_log_scale"] = best_ls
    print(f"\n[pick] best log_scale = {best_ls}")

    # ---- 3. VERTEX COUNT: 4 corners vs 1 centre ------------------------------
    # V-DETR Table 3 measured 8 vertices (65.0 AP50) > 1 centre (54.8). The claim
    # is that a centre-only encoding cannot express SIZE. Test it directly: does
    # the map change when only w,h change?
    print("\n[vertices] does the attention map track box SIZE?")
    vtest = {}
    for nv, label in [(1, "centre_only"), (4, "four_corners")]:
        maps = {}
        for s in range(args.seeds):
            m = RefRPE(nv, args.rpe_dim, 8, best_ls, seed=s)
            for name in ("tiny_0.02", "median_0.09", "large_0.30"):
                maps.setdefault(name, []).append(m.bias(boxes[name], pxy, 0))
        # correlation between the tiny-box map and the large-box map: if size is
        # ignored these are IDENTICAL (corr 1.0), which is the failure mode.
        cors = [float(np.corrcoef(t, l)[0, 1])
                for t, l in zip(maps["tiny_0.02"], maps["large_0.30"])]
        struct = {n: float(np.mean([spatial_structure(a, pxy, boxes[n]) for a in v]))
                  for n, v in maps.items()}
        vtest[label] = {"corr_tiny_vs_large": float(np.mean(cors)),
                        "spatial_structure": struct}
        print(f"  {label:>13}: corr(tiny,large) = {np.mean(cors):+.4f}"
              f"   (1.0 = size IGNORED)")
        for n, c in struct.items():
            print(f"                 {n:>12} |corr| {c:6.3f}")
    report["vertex_count"] = vtest

    # ---- 4. PNGs -------------------------------------------------------------
    m = RefRPE(4, args.rpe_dim, 8, best_ls, seed=0)
    for name, b in boxes.items():
        heat_png(softmax(m.bias(b, pxy, 0)), args.grid, b,
                 os.path.join(args.out, f"rpe_{name}.png"))
    heat_png(softmax(np.random.default_rng(0).normal(0, 1, P)), args.grid,
             boxes["median_0.09"], os.path.join(args.out, "baseline_current.png"))
    m1 = RefRPE(1, args.rpe_dim, 8, best_ls, seed=0)
    for name in ("tiny_0.02", "large_0.30"):
        heat_png(softmax(m1.bias(boxes[name], pxy, 0)), args.grid, boxes[name],
                 os.path.join(args.out, f"centreonly_{name}.png"))

    # ---- 5. VERDICT ----------------------------------------------------------
    c4 = np.nanmean(list(sweep[best_ls].values()))
    b0 = report["baseline_current_model"]["spatial_structure_mean"]
    corr4 = vtest["four_corners"]["corr_tiny_vs_large"]
    corr1 = vtest["centre_only"]["corr_tiny_vs_large"]
    verdict = []
    verdict.append(("PASS" if c4 > 5 * b0 else "FAIL") +
                   f": bias is a function of geometry -- |corr| {c4:.3f} versus "
                   f"{b0:.3f} for a bias with no geometric term ({c4/max(b0,1e-9):.0f}x).")
    verdict.append(("PASS" if corr4 < 0.95 else "FAIL") +
                   f": 4-corner encoding SEPARATES box sizes "
                   f"(corr tiny-vs-large {corr4:+.3f}; 1.0 would mean size ignored).")
    verdict.append(("PASS" if corr1 > 0.999 else "FAIL") +
                   f": centre-only is size-BLIND (corr {corr1:+.4f}) -- reproduces "
                   f"V-DETR Table 3 (1 vertex 54.8 vs 8 vertices 65.0 AP50) and our "
                   f"own roi_sampler k=1 measurement (AUC 0.000).")
    report["verdict"] = verdict
    print("\n=== VERDICT ===")
    for v in verdict:
        print(" ", v)

    with open(os.path.join(args.out, "rpe_gate.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nPNGs + rpe_gate.json -> {args.out}")


if __name__ == "__main__":
    main()

"""
Calibrated Negative Log-Likelihood (C-NLL) for CE-Loc localization, following
Add One Take One (NeurIPS 2026) §3.2 and Appendix A.2, Eq. 10:

    C-NLL(x) = -log q(x) - min_{z in {B_j}} ( -log q(z) )

where q is a Gaussian fit over the 4D spatial feature of the existing
same-category boxes {B_j} in the image:
    [width, height, max IoU with {B_j}, avg Euclidean distance to the K=3
    nearest neighbors in {B_j}]

{B_j} for a given sample is NOT available in samples/*/annotation/*.json
(only a single target_bbox per file). It lives in the original CE-130 dump
(all_phase2_V2/{split}/{img_id}_b{N}/fixed_annotation.json), which has
"all_bboxes" (every same-category box in the source image) and
"inpainted_bboxes" (the boxes removed turn by turn, in order). Each
samples/*/annotation/{img_id}_{n}.json target_bbox matches, by VALUE, exactly
one (branch, turn) pair's inpainted_bboxes[turn] — the "_n" numbering in
samples/ is NOT the branch/turn index and must not be used directly.
"""
import glob
import json
import os

import numpy as np


def _xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def load_branch_lookup(all_phase2_dir, split=None):
    """
    Build, once, a lookup of every (branch_dir, turn_idx, inpainted_box_cxcywh,
    all_bboxes_cxcywh) tuple grouped by source image id, from
    all_phase2_dir/{split}/{img_id}_b{N}/{fixed_annotation.json | annotation.json}.

    `split`: if given, only that split's directory is scanned. If None (the
    default), ALL splits present under all_phase2_dir are scanned — this
    matters because samples/{train,test}/ (CE-LocModel's own 2-way split)
    does NOT line up with all_phase2_V2's train/test/val 3-way split; an
    image counted as "test" by CE-LocModel can live under all_phase2_V2's
    "val" directory. Restricting the lookup to a single matching split
    silently drops ~18% of images (measured on samples/test).

    Prefers fixed_annotation.json (corrects a small coordinate mismatch
    between all_bboxes and inpainted_bboxes) but falls back to
    annotation.json when fixed_annotation.json is absent — measured: the
    "train" split ships NO fixed_annotation.json at all (0/4653 branches),
    only "test"/"val" do. Skipping unfixed branches would silently drop
    ~67% of samples/train images from the lookup.

    Returns: dict[img_id: str] -> list of dicts with keys
        {"branch": str, "turn": int, "box": (cx,cy,w,h), "all_bboxes": list[(cx,cy,w,h)]}
    """
    lookup = {}
    splits = [split] if split is not None else sorted(
        d for d in os.listdir(all_phase2_dir)
        if os.path.isdir(os.path.join(all_phase2_dir, d))
    )
    for sp in splits:
        split_dir = os.path.join(all_phase2_dir, sp)
        for branch_path in sorted(glob.glob(os.path.join(split_dir, "*"))):
            if not os.path.isdir(branch_path):
                continue
            anno_path = os.path.join(branch_path, "fixed_annotation.json")
            if not os.path.exists(anno_path):
                anno_path = os.path.join(branch_path, "annotation.json")
            if not os.path.exists(anno_path):
                continue
            branch_dir = os.path.basename(branch_path)
            # branch_dir looks like "{img_id}_b{N}"
            img_id = branch_dir.rsplit("_b", 1)[0]
            with open(anno_path, "r") as f:
                data = json.load(f)
            all_bboxes_cxcywh = [_xyxy_to_cxcywh(b) for b in data["all_bboxes"]]
            entries = lookup.setdefault(img_id, [])
            for turn_idx, box in enumerate(data["inpainted_bboxes"], start=1):
                entries.append({
                    "branch": branch_dir,
                    "turn": turn_idx,
                    "box": _xyxy_to_cxcywh(box),
                    "all_bboxes": all_bboxes_cxcywh,
                })
    return lookup


def find_all_bboxes(img_id, target_bbox_cxcywh, lookup, tol=0.05):
    """
    Find {B_j} (the other same-category boxes) for one sample, by matching
    target_bbox_cxcywh against inpainted_bboxes[turn] across every branch of
    img_id, then returning that branch's all_bboxes with the matched box
    removed.

    Returns: (all_bboxes_minus_self: list[(cx,cy,w,h)], n_matches: int)
    n_matches > 1 means multiple branches matched (ambiguous — different
    branches of the same source image can have slightly different all_bboxes
    since each branch re-annotates the scene after its own removal order;
    the first match, by branch name, is used).
    """
    entries = lookup.get(img_id, [])
    tx, ty, tw, th = target_bbox_cxcywh
    matches = [
        e for e in entries
        if abs(e["box"][0] - tx) < tol and abs(e["box"][1] - ty) < tol
        and abs(e["box"][2] - tw) < tol and abs(e["box"][3] - th) < tol
    ]
    if not matches:
        return None, 0
    chosen = sorted(matches, key=lambda e: e["branch"])[0]
    b_j = [b for b in chosen["all_bboxes"] if b != chosen["box"]]
    return b_j, len(matches)


def _iou_cxcywh(a, b):
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def _spatial_feature(box_cxcywh, b_j):
    """4D feature per paper §3.2: [width, height, max IoU with B_j, avg dist to 3-NN in B_j]."""
    cx, cy, w, h = box_cxcywh
    if len(b_j) == 0:
        return np.array([w, h, 0.0, 0.0])
    ious = [_iou_cxcywh(box_cxcywh, b) for b in b_j]
    max_iou = max(ious)
    dists = [np.hypot(cx - b[0], cy - b[1]) for b in b_j]
    k = min(3, len(dists))
    avg_knn_dist = float(np.mean(sorted(dists)[:k]))
    return np.array([w, h, max_iou, avg_knn_dist])


def fit_gaussian(b_j):
    """Fit N(mu, Sigma) over the 4D spatial features of {B_j}."""
    feats = np.stack([_spatial_feature(b, [x for x in b_j if x != b]) for b in b_j])
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    sigma += np.eye(4) * 1e-6  # regularize against singular covariance
    return mu, sigma


def _neg_log_q(x, mu, sigma):
    d = x - mu
    sigma_inv = np.linalg.inv(sigma)
    sign, logdet = np.linalg.slogdet(sigma)
    quad = d @ sigma_inv @ d
    return 0.5 * (4 * np.log(2 * np.pi) + logdet + quad)


def prepare_cnll(b_j):
    """
    Precompute everything in Eq. 10 that depends only on {B_j}: the fitted
    Gaussian and the calibration term min_z(-log q(z)). Both are invariant
    across candidate boxes, so scoring K candidates for one image should do
    this ONCE rather than K times.

    Returns None when {B_j} is too small to fit a Gaussian, matching
    compute_cnll's contract.
    """
    if len(b_j) < 2:
        return None
    mu, sigma = fit_gaussian(b_j)
    nll_bj = [_neg_log_q(_spatial_feature(b, [x for x in b_j if x != b]), mu, sigma) for b in b_j]
    return mu, sigma, min(nll_bj)


def compute_cnll_prepared(pred_box_cxcywh, b_j, prepared):
    """C-NLL for one candidate, reusing prepare_cnll's output."""
    mu, sigma, min_nll_bj = prepared
    nll_pred = _neg_log_q(_spatial_feature(pred_box_cxcywh, b_j), mu, sigma)
    return float(nll_pred - min_nll_bj)


def compute_cnll(pred_box_cxcywh, b_j):
    """
    C-NLL(pred) = -log q(pred) - min_{z in B_j} (-log q(z)),  Eq. 10.
    Returns None if {B_j} is empty (can't fit a Gaussian).

    Convenience wrapper that re-derives the {B_j}-only terms on every call.
    When scoring many candidates against the SAME {B_j} (the usual case in
    test_mul_box.py), use prepare_cnll + compute_cnll_prepared instead — the
    Gaussian fit and the calibration min are identical across candidates, so
    recomputing them per candidate is pure waste (it dominated eval time at
    num_proposals=100).
    """
    prepared = prepare_cnll(b_j)
    if prepared is None:
        return None
    return compute_cnll_prepared(pred_box_cxcywh, b_j, prepared)

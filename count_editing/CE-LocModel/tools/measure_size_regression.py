#!/usr/bin/env python3
"""Does the model regress box SIZE at all, or does it emit a near-constant size?

WHY THIS EXISTS
---------------
EXPERIMENT B measured (test split, N=300):

    recall@0.10 = 0.373      <- the model FINDS 37 % of the objects
    recall@0.50 = 0.119      <- only 12 % are tight enough to count
    ratio       = 3.14

A ratio that high is the signature of a REGRESSION failure, not a DETECTION
failure: boxes land on objects but do not fit them. The images from EXPERIMENT A
already suggested why -- box centres looked right, box sizes looked constant --
but "looked constant" is not a measurement. This script turns it into numbers.

THE ONE NUMBER THAT DECIDES IT
------------------------------
    size_ratio = std(pred w) / std(GT w)

  ~0.0  the model emits ONE size regardless of the object. Every architectural
        idea aimed at localisation is beside the point; the fix is the loss.
  ~1.0  the model does vary size, it is just inaccurate -> a genuinely different
        problem, and this script's diagnosis is WRONG. Say so and stop.

  Careful: std alone can be inflated by a handful of outlier boxes, so IQR and
  the per-image std are reported next to it. Per-image matters most: the model
  could vary size ACROSS images (easy -- one global cue per image) while being
  constant WITHIN an image (the hard part, and the part that drives IoU).

THE SECOND QUESTION: WOULD A BETTER SIZE ACTUALLY HELP?
-------------------------------------------------------
Assuming "constant size" is confirmed, it is still not obvious that fixing it
buys much -- maybe the centres are too far off for any size to reach IoU 0.5.
So this also computes ORACLE CEILINGS on the already-matched pairs:

  IoU_oracle_size    keep the predicted centre, substitute the GT w/h
  IoU_oracle_centre  keep the predicted w/h,     substitute the GT centre

Whichever recovers more recall@0.50 names the bottleneck. If oracle-size lifts
recall from 0.12 to (say) 0.45 then the loss rebalance is worth a run; if it
lifts it to 0.15, size is NOT the bottleneck and EXPERIMENT C must go elsewhere.

WHY THE L1 TERM IS SUSPECT (the hypothesis being tested)
--------------------------------------------------------
CE-130 objects have median w ~ 0.069 of the image. L1 on [cx,cy,w,h] therefore
sees w and h on a scale ~14x smaller than cx,cy: getting w COMPLETELY wrong
(predicting 0 instead of 0.069) costs 0.069 of L1, while a 0.069 centre shift
costs the same -- yet the first destroys IoU and the second barely dents it.
With weights 5.0*L1 vs 2.0*GIoU, the gradient the model actually feels is
dominated by centres. Training bore this out: loss_l1 fell to 0.078 while
IoU_matched stalled at 0.37. `l1_share_wh` below quantifies the imbalance.

MATCHING: identical to eval.py -- top-k (never an absolute threshold, a
non-discriminating focal head converges below any fixed one), class-agnostic
NMS, then greedy score-ordered assignment. Reusing the eval path is deliberate:
a second, subtly different matcher would make these numbers incomparable to the
AP already reported.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import normalize_for_clip  # noqa: E402
from data.factory import build_dataset  # noqa: E402
from models.detector import build_model  # noqa: E402
from utils.box_ops_np import box_iou, cxcywh_to_xyxy  # noqa: E402
from eval import nms_class_agnostic, scores_and_classes  # noqa: E402


def describe(x, name):
    """Spread of a 1-D sample. `std` is the headline; IQR guards against the case
    where a few outliers manufacture a std that looks like real variation."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"name": name, "n": 0}
    q1, q50, q3 = np.percentile(x, [25, 50, 75])
    return {"name": name, "n": int(x.size), "mean": float(x.mean()),
            "std": float(x.std()), "min": float(x.min()), "max": float(x.max()),
            "p25": float(q1), "p50": float(q50), "p75": float(q3),
            "iqr": float(q3 - q1)}


def greedy_match(pred_xyxy, scores, gt_xyxy, iou_thr=0.0):
    """Score-ordered greedy assignment, EXACTLY as eval.py counts a TP.

    Returns (pred_idx, gt_idx, iou) for pairs above `iou_thr`. `iou_thr=0.0`
    keeps every pair with any overlap, which is what the size analysis wants: it
    must describe the boxes the model actually put on objects, not only the ones
    that already passed 0.5 (conditioning on success would bias every statistic
    towards good boxes and hide the failure being measured).
    """
    pi, gi, ious = [], [], []
    if len(pred_xyxy) == 0 or len(gt_xyxy) == 0:
        return np.array(pi, int), np.array(gi, int), np.array(ious)
    used = np.zeros(len(gt_xyxy), dtype=bool)
    iou_full = box_iou(pred_xyxy, gt_xyxy)[0]
    for i in np.argsort(-scores):
        iou = np.where(used, -1.0, iou_full[i])
        j = int(np.argmax(iou))
        if iou[j] > iou_thr:
            used[j] = True
            pi.append(int(i)); gi.append(j); ious.append(float(iou[j]))
    return np.array(pi, int), np.array(gi, int), np.array(ious)


def iou_cxcywh(a, b):
    """Row-wise IoU between two [K,4] cxcywh arrays."""
    if len(a) == 0:
        return np.zeros(0)
    ax, bx = cxcywh_to_xyxy(a), cxcywh_to_xyxy(b)
    inter_w = np.maximum(0, np.minimum(ax[:, 2], bx[:, 2]) - np.maximum(ax[:, 0], bx[:, 0]))
    inter_h = np.maximum(0, np.minimum(ax[:, 3], bx[:, 3]) - np.maximum(ax[:, 1], bx[:, 1]))
    inter = inter_w * inter_h
    ua = (a[:, 2] * a[:, 3]) + (b[:, 2] * b[:, 3]) - inter
    return inter / np.maximum(ua, 1e-9)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_b.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-proposals", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="JSON path; default <ckpt>_sizecheck_<split>.json")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    N = a.num_proposals or cfg["diffusion"]["num_proposals_eval"]
    out_path = a.out or (os.path.splitext(a.ckpt)[0] + f"_sizecheck_{a.split}.json")

    ds = build_dataset(cfg, a.split)
    if a.limit:
        ds.items = ds.items[: a.limit]

    exp = cfg.get("experiment", "?")
    print("=" * 78, flush=True)
    print(f"  SIZE REGRESSION CHECK — EXPERIMENT {exp}", flush=True)
    print("-" * 78, flush=True)
    for k, v in [("timestamp", datetime.now().isoformat(timespec="seconds")),
                 ("ckpt", os.path.abspath(a.ckpt)), ("split", a.split),
                 ("N", N), ("device", str(dev)),
                 ("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                 ("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES")),
                 ("command", " ".join(sys.argv)),
                 ("dataset", ds.stats())]:
        print(f"  {k:22s} {v}", flush=True)
    print("=" * 78, flush=True)

    model = build_model(cfg, dropout=0.0).to(dev)
    sd = torch.load(a.ckpt, map_location=dev)
    w = sd["model"] if "model" in sd else sd
    missing, unexpected = model.load_state_dict(w, strict=False)
    missing = [k for k in missing
               if not (k.startswith("encoder.vision.") or k.startswith("encoder.text."))]
    assert not missing and not unexpected, \
        f"checkpoint mismatch: missing={missing} unexpected={unexpected}"
    model.eval()

    topk, nms_iou = cfg["eval"]["topk"], cfg["eval"]["nms_iou"]
    S = cfg["data"]["image_size"]

    pred_w, pred_h, gt_w, gt_h = [], [], [], []      # matched pairs only
    all_pred_w, all_pred_h = [], []                  # every kept box
    per_img_pred_std, per_img_gt_std = [], []
    iou_real, iou_osize, iou_ocentre = [], [], []
    l1_c, l1_wh = [], []
    n_gt_total = 0
    per_image = []

    t0 = time.time()
    for i in range(len(ds)):
        m = ds[i]
        px = torch.from_numpy(normalize_for_clip(m["image"])).unsqueeze(0).to(dev)
        boxes, logits = model.ddim_sample(N, pixel_values=px, texts=[m["text"]])
        b = boxes[0].cpu().numpy()                      # cxcywh [0,1]
        # SAME reader as eval.py. A raw sigmoid on A.2's [N,80] leaves a 2-D array
        # and NMS then fails on 2-D indices. The size statistics below are
        # class-agnostic by design -- they describe box GEOMETRY, which does not
        # depend on what the box is called -- so only the score is needed here.
        s, _ = scores_and_classes(logits[0])

        keep = np.argsort(-s)[:topk]
        b_k, s_k = b[keep], s[keep]
        k2 = nms_class_agnostic(cxcywh_to_xyxy(b_k) * S, s_k, nms_iou)
        b_f, s_f = b_k[k2], s_k[k2]                     # final predictions

        g = m["boxes"]                                  # cxcywh [0,1]
        n_gt_total += len(g)
        all_pred_w.append(b_f[:, 2]); all_pred_h.append(b_f[:, 3])

        pi, gi, iou = greedy_match(cxcywh_to_xyxy(b_f) * S, s_f,
                                   cxcywh_to_xyxy(g) * S, iou_thr=0.0)
        if len(pi) == 0:
            continue
        p_m, g_m = b_f[pi], g[gi]

        pred_w.append(p_m[:, 2]); pred_h.append(p_m[:, 3])
        gt_w.append(g_m[:, 2]);   gt_h.append(g_m[:, 3])

        # WITHIN-image spread: the discriminating statistic. Varying size across
        # images only needs one global cue; varying it per object is the skill.
        if len(pi) >= 3:
            per_img_pred_std.append(float(p_m[:, 2].std()))
            per_img_gt_std.append(float(g_m[:, 2].std()))

        # Oracle substitutions on the SAME matched pairs -> directly comparable.
        osize = np.concatenate([p_m[:, :2], g_m[:, 2:]], axis=1)
        ocent = np.concatenate([g_m[:, :2], p_m[:, 2:]], axis=1)
        iou_real.append(iou / 1.0)
        iou_osize.append(iou_cxcywh(osize, g_m))
        iou_ocentre.append(iou_cxcywh(ocent, g_m))

        l1_c.append(np.abs(p_m[:, :2] - g_m[:, :2]).sum(1))
        l1_wh.append(np.abs(p_m[:, 2:] - g_m[:, 2:]).sum(1))

        per_image.append({
            "image_id": m["image_id"], "class": m["text"], "n_gt": int(len(g)),
            "n_pred": int(len(b_f)), "n_matched": int(len(pi)),
            "pred_w_std": float(p_m[:, 2].std()), "gt_w_std": float(g_m[:, 2].std()),
            "pred_w_mean": float(p_m[:, 2].mean()), "gt_w_mean": float(g_m[:, 2].mean()),
            "iou_mean": float(iou.mean()),
        })

        if (i + 1) % max(len(ds) // 10, 1) == 0 or i == len(ds) - 1:
            el = time.time() - t0
            print(f"  [{i+1:5d}/{len(ds)}] {100*(i+1)/len(ds):5.1f}% | "
                  f"{1000*el/(i+1):.0f}ms/img | elapsed {el:.0f}s | "
                  f"ETA {el/(i+1)*(len(ds)-i-1):.0f}s", flush=True)

    cat = lambda L: np.concatenate(L) if L else np.zeros(0)
    pw, ph, gw, gh = cat(pred_w), cat(pred_h), cat(gt_w), cat(gt_h)
    apw, aph = cat(all_pred_w), cat(all_pred_h)
    ir, ios, ioc = cat(iou_real), cat(iou_osize), cat(iou_ocentre)
    lc, lwh = cat(l1_c), cat(l1_wh)

    ratio_w = float(pw.std() / gw.std()) if gw.std() > 0 else float("nan")
    ratio_h = float(ph.std() / gh.std()) if gh.std() > 0 else float("nan")
    ratio_iqr_w = float(describe(pw, "")["iqr"] / describe(gw, "")["iqr"]) if len(gw) else float("nan")
    pis, gis = np.array(per_img_pred_std), np.array(per_img_gt_std)
    ratio_within = float(pis.mean() / gis.mean()) if len(gis) and gis.mean() > 0 else float("nan")

    # Oracle recall: recall@0.50 counts a GT as found when its matched pair
    # clears 0.50, so substituting one half of the box and recounting gives the
    # ceiling that half is responsible for.
    rec = lambda x: float((x >= 0.5).sum() / max(n_gt_total, 1))
    res = {
        "size_ratio_w": ratio_w, "size_ratio_h": ratio_h,
        "size_ratio_iqr_w": ratio_iqr_w, "size_ratio_within_image_w": ratio_within,
        "pred_w": describe(pw, "pred_w"), "gt_w": describe(gw, "gt_w"),
        "pred_h": describe(ph, "pred_h"), "gt_h": describe(gh, "gt_h"),
        "pred_w_all_boxes": describe(apw, "pred_w_all"),
        "pred_h_all_boxes": describe(aph, "pred_h_all"),
        "iou_matched_mean": float(ir.mean()) if len(ir) else 0.0,
        "iou_oracle_size_mean": float(ios.mean()) if len(ios) else 0.0,
        "iou_oracle_centre_mean": float(ioc.mean()) if len(ioc) else 0.0,
        "recall50_real": rec(ir), "recall50_oracle_size": rec(ios),
        "recall50_oracle_centre": rec(ioc),
        "l1_centre_mean": float(lc.mean()) if len(lc) else 0.0,
        "l1_wh_mean": float(lwh.mean()) if len(lwh) else 0.0,
        "l1_share_wh": float(lwh.mean() / max(lc.mean() + lwh.mean(), 1e-9)) if len(lc) else 0.0,
        "n_matched_pairs": int(len(pw)), "n_gt_total": int(n_gt_total),
    }

    print("\n" + "=" * 78, flush=True)
    print(f"RESULTS — SIZE REGRESSION, EXPERIMENT {exp}, {a.split}, N={N}", flush=True)
    print("-" * 78, flush=True)
    print(f"  matched pairs            {res['n_matched_pairs']} / {n_gt_total} GT", flush=True)
    print(f"  pred w   mean {pw.mean():.4f}  std {pw.std():.4f}  "
          f"[{pw.min():.4f}, {pw.max():.4f}]", flush=True)
    print(f"  GT   w   mean {gw.mean():.4f}  std {gw.std():.4f}  "
          f"[{gw.min():.4f}, {gw.max():.4f}]", flush=True)
    print(f"  SIZE RATIO std(pred w)/std(GT w)      {ratio_w:.4f}", flush=True)
    print(f"  size ratio, IQR                       {ratio_iqr_w:.4f}", flush=True)
    print(f"  size ratio, WITHIN image (the hard one) {ratio_within:.4f}", flush=True)
    print("-" * 78, flush=True)
    print(f"  IoU matched (real)                    {res['iou_matched_mean']:.4f}", flush=True)
    print(f"  IoU with GT w,h  (oracle size)        {res['iou_oracle_size_mean']:.4f}", flush=True)
    print(f"  IoU with GT cx,cy (oracle centre)     {res['iou_oracle_centre_mean']:.4f}", flush=True)
    print(f"  recall@0.50  real / oracle-size / oracle-centre  "
          f"{res['recall50_real']:.4f} / {res['recall50_oracle_size']:.4f} / "
          f"{res['recall50_oracle_centre']:.4f}", flush=True)
    print("-" * 78, flush=True)
    print(f"  L1 centre {res['l1_centre_mean']:.4f} | L1 w,h {res['l1_wh_mean']:.4f} "
          f"| w,h share of L1 {100*res['l1_share_wh']:.1f}%", flush=True)
    print("-" * 78, flush=True)

    # Verdict. Stated as a rule so it cannot be reverse-engineered from the number
    # after the fact.
    v = []
    if ratio_within < 0.25:
        v.append(f"CONFIRMED: within-image size ratio {ratio_within:.3f} < 0.25 -- the "
                 f"model emits a near-constant size per image.")
    elif ratio_within < 0.60:
        v.append(f"PARTIAL: within-image size ratio {ratio_within:.3f} -- some size "
                 f"variation, well below GT.")
    else:
        v.append(f"REFUTED: within-image size ratio {ratio_within:.3f} >= 0.60 -- the "
                 f"model DOES vary size. The constant-size hypothesis is wrong; do "
                 f"not rebalance the loss on this evidence.")

    gain_s = res["recall50_oracle_size"] - res["recall50_real"]
    gain_c = res["recall50_oracle_centre"] - res["recall50_real"]
    if gain_s > gain_c * 1.5:
        v.append(f"SIZE is the bottleneck: fixing w,h recovers {gain_s:.3f} recall vs "
                 f"{gain_c:.3f} for centres.")
    elif gain_c > gain_s * 1.5:
        v.append(f"CENTRE is the bottleneck: fixing cx,cy recovers {gain_c:.3f} recall "
                 f"vs {gain_s:.3f} for size. A loss rebalance towards w,h would NOT "
                 f"be the right EXPERIMENT C.")
    else:
        v.append(f"BOTH halves are comparably wrong (size {gain_s:.3f}, centre "
                 f"{gain_c:.3f}) -- no single-variable fix is indicated.")
    res["verdict"] = v
    for line in v:
        print(f"  {line}", flush=True)
    print("=" * 78, flush=True)

    with open(out_path, "w") as f:
        json.dump({"summary": res, "settings": {"N": N, "split": a.split,
                   "topk": topk, "nms_iou": nms_iou, "ckpt": os.path.abspath(a.ckpt),
                   "config": a.config},
                   "environment": {"timestamp": datetime.now().isoformat(timespec="seconds"),
                                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                   "command": " ".join(sys.argv)},
                   "per_image": per_image}, f, indent=1)
    print(f"  full metrics: {out_path}", flush=True)
    print(f"  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

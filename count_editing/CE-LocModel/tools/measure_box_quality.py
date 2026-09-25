#!/usr/bin/env python3
"""Box quality and score quality, measured SEPARATELY. AP cannot tell them apart.

WHY THIS EXISTS
---------------
EXPERIMENT A.1 scored AP50 0.0015 — a number that reads as total failure. But its
recall@0.10 was 0.4975, essentially identical to A.2's 0.4961: the two models FIND
the same objects. The 16x gap in AP came from ranking, not from box quality.

A controlled simulation makes the problem exact. Same boxes, only the scores change:

    box GOOD + score GOOD      AP50 0.4453
    box GOOD + score STUCK     AP50 0.0725      <- 6x drop, identical boxes
    box GOOD + score RANDOM    AP50 0.0542
    box BAD  + score GOOD      AP50 0.0070
    box BAD  + score STUCK     AP50 0.0066

"good boxes, broken score" (0.0725) barely outranks "bad boxes, good score" (0.0070),
so a low AP does not say WHICH half is broken — and the two need opposite fixes.

THE THREE METRICS, AND THE PROOF THEY SEPARATE
----------------------------------------------
Measured on the same simulation:

    scenario                 oracle_recall  mean_bestIoU  score_AUC
    box GOOD + score GOOD         0.710         0.592       0.884
    box GOOD + score STUCK        0.745         0.604       0.530
    box GOOD + score RANDOM       0.675         0.587       0.530
    box BAD  + score GOOD         0.160         0.293       0.557
    box BAD  + score STUCK        0.220         0.330       0.505

Read down the columns: the first two DO NOT MOVE when the score changes; the third
DOES NOT MOVE when the boxes change. Two independent axes.

  oracle_recall  fraction of GT with at least one box at IoU >= 0.5, IGNORING score
                 entirely. It is the CEILING a perfect score head could reach, so
                 `oracle_recall - recall` is exactly the price of the score head.
  mean_bestIoU   best IoU per GT, averaged. Pure geometry.
  score_AUC      how well the score ranks matched boxes above unmatched ones.
                 0.5 == a coin flip. An ABSOLUTE reference, not a relative one.

    (high, high)  genuinely good
    (high, ~0.5)  boxes fine, SCORE HEAD broken      <- A.1 lives here
    (low,  high)  boxes broken, score fine
    (low,  ~0.5)  both broken

PER-ROUND MODE (`--per-round`) is what EXPERIMENT C1 needs. C1 opens the decoder
into 6 refinement rounds; the question "does iterating help?" is answered by whether
these metrics IMPROVE from round 1 to round 6. A flat curve means iteration buys
nothing, and that verdict must be visible before C2 is built on top of it.

Run it on the EXISTING A / B / A.1 / A.2 checkpoints first: without those baselines
C1's numbers cannot be read.
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
from eval import scores_and_classes  # noqa: E402
# Định nghĩa oracle_recall / score_AUC nằm ở MỘT chỗ, dùng chung với eval.py.
from utils.metrics_np import (nms_class_agnostic, quality, roc_auc,  # noqa: E402,F401
                              summarise)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-proposals", type=int, default=None)
    ap.add_argument("--per-round", action="store_true",
                    help="report metrics for EVERY refinement round (EXPERIMENT C1). "
                         "Ignored by models without rounds.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    N = a.num_proposals or cfg["diffusion"]["num_proposals_eval"]
    out_path = a.out or (os.path.splitext(a.ckpt)[0] + f"_boxquality_{a.split}.json")

    ds = build_dataset(cfg, a.split)
    if a.limit:
        ds.items = ds.items[: a.limit]

    exp = cfg.get("experiment", "?")
    print("=" * 78, flush=True)
    print(f"  BOX QUALITY — EXPERIMENT {exp}", flush=True)
    print("-" * 78, flush=True)
    for k, v in [("timestamp", datetime.now().isoformat(timespec="seconds")),
                 ("ckpt", os.path.abspath(a.ckpt)), ("split", a.split), ("N", N),
                 ("per_round", a.per_round), ("device", str(dev)),
                 ("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
                 ("command", " ".join(sys.argv)), ("dataset", ds.stats())]:
        print(f"  {k:18s} {v}", flush=True)
    print("=" * 78, flush=True)

    model = build_model(cfg, dropout=0.0).to(dev)
    sd = torch.load(a.ckpt, map_location=dev)
    missing, unexpected = model.load_state_dict(sd.get("model", sd), strict=False)
    missing = [k for k in missing
               if not k.startswith(("encoder.vision.", "encoder.text."))]
    assert not missing and not unexpected, \
        f"checkpoint mismatch: missing={missing} unexpected={unexpected}"
    model.eval()

    topk, nms_iou, S = cfg["eval"]["topk"], cfg["eval"]["nms_iou"], cfg["data"]["image_size"]
    n_rounds = cfg["model"].get("refine_rounds", 0) if a.per_round else 0

    # One accumulator per round; index -1 is the final prediction (what eval.py uses).
    acc = {r: {"best": [], "hit": 0, "n_gt": 0, "auc": [], "scored_hit": 0}
           for r in (list(range(n_rounds)) if n_rounds else [0])}
    t0 = time.time()

    for i in range(len(ds)):
        m = ds[i]
        px = torch.from_numpy(normalize_for_clip(m["image"])).unsqueeze(0).to(dev)
        res = model.ddim_sample(N, pixel_values=px, texts=[m["text"]],
                                return_all_rounds=bool(n_rounds)) if n_rounds else \
              model.ddim_sample(N, pixel_values=px, texts=[m["text"]])
        rounds = res if n_rounds else [res]

        for r, (boxes, logits) in enumerate(rounds):
            b = boxes[0].cpu().numpy()
            s, cls = scores_and_classes(logits[0])
            keep = np.argsort(-s)[:topk]
            b_k, s_k = b[keep], s[keep]
            k2 = nms_class_agnostic(cxcywh_to_xyxy(b_k) * S, s_k, nms_iou)
            b_f, s_f = b_k[k2], s_k[k2]
            c_f = None if cls is None else cls[keep][k2]

            best, hit, ngt, auc = quality(b_f, s_f, m["boxes"], S,
                                          pred_cls=c_f, gt_cls=m.get("labels")
                                          if cls is not None else None)
            A = acc[r]
            A["best"].append(best); A["hit"] += hit; A["n_gt"] += ngt
            if not np.isnan(auc):
                A["auc"].append(auc)

            # recall the way eval.py counts it (greedy, score-ordered) so the gap
            # against oracle_recall is directly the score head's cost
            if len(b_f) and ngt:
                g_xyxy = cxcywh_to_xyxy(m["boxes"]) * S
                mm = box_iou(cxcywh_to_xyxy(b_f) * S, g_xyxy)[0]
                if c_f is not None and m.get("labels") is not None:
                    mm = np.where(c_f[:, None] == np.asarray(m["labels"])[None, :], mm, 0.0)
                used = np.zeros(ngt, dtype=bool)
                for pi in np.argsort(-s_f):
                    row = np.where(used, -1.0, mm[pi])
                    j = int(np.argmax(row))
                    if row[j] >= 0.5:
                        used[j] = True
                A["scored_hit"] += int(used.sum())

        if (i + 1) % max(len(ds) // 10, 1) == 0 or i == len(ds) - 1:
            el = time.time() - t0
            print(f"  [{i+1:5d}/{len(ds)}] {100*(i+1)/len(ds):5.1f}% | "
                  f"{1000*el/(i+1):.0f}ms/img | ETA {el/(i+1)*(len(ds)-i-1):.0f}s",
                  flush=True)

    results = {}
    for r, A in acc.items():
        results[f"round_{r+1}" if n_rounds else "final"] = summarise(
            A["best"], A["hit"], A["n_gt"], A["auc"],
            recall_scored=A["scored_hit"] / max(A["n_gt"], 1))

    print("\n" + "=" * 78, flush=True)
    print(f"RESULTS — EXPERIMENT {exp}, {a.split}, N={N}", flush=True)
    print("-" * 78, flush=True)
    print(f"  {'':10s} {'oracle_rec':>11s} {'recall':>9s} {'cost':>8s} "
          f"{'bestIoU':>9s} {'score_AUC':>10s}", flush=True)
    for name, v in results.items():
        print(f"  {name:10s} {v['oracle_recall']:11.4f} {v['recall_scored']:9.4f} "
              f"{v['score_head_cost']:8.4f} {v['mean_bestIoU']:9.4f} "
              f"{v['score_AUC']:10.4f}", flush=True)
    print("-" * 78, flush=True)

    # Verdicts, stated as rules so they cannot be reverse-engineered from the number.
    last = results[list(results)[-1]]
    v = []
    if last["score_AUC"] < 0.60:
        v.append(f"SCORE HEAD is near-random (AUC {last['score_AUC']:.3f} vs 0.5 = coin "
                 f"flip) -> AP understates the boxes. Read oracle_recall, not AP.")
    elif last["score_AUC"] > 0.75:
        v.append(f"Score head ranks well (AUC {last['score_AUC']:.3f}); AP is a fair "
                 f"summary here.")
    else:
        v.append(f"Score head is weak but not dead (AUC {last['score_AUC']:.3f}).")
    v.append(f"Score head costs {last['score_head_cost']:.3f} recall "
             f"({last['oracle_recall']:.3f} reachable vs {last['recall_scored']:.3f} taken).")

    if n_rounds:
        first, final = results["round_1"], results[f"round_{n_rounds}"]
        d = final["oracle_recall"] - first["oracle_recall"]
        if d > 0.02:
            v.append(f"REFINEMENT WORKS: oracle_recall {first['oracle_recall']:.3f} -> "
                     f"{final['oracle_recall']:.3f} (+{d:.3f}) across {n_rounds} rounds.")
        elif d < -0.02:
            v.append(f"REFINEMENT HURTS: {first['oracle_recall']:.3f} -> "
                     f"{final['oracle_recall']:.3f} ({d:+.3f}). Later rounds undo earlier "
                     f"ones — check that `anchor` was not confused with `box`.")
        else:
            v.append(f"REFINEMENT IS FLAT: {first['oracle_recall']:.3f} -> "
                     f"{final['oracle_recall']:.3f} ({d:+.3f}). Iterating buys nothing; "
                     f"do NOT build C2 on this without rethinking the diagnosis.")
    for line in v:
        print(f"  {line}", flush=True)
    print("=" * 78, flush=True)

    with open(out_path, "w") as f:
        json.dump({"results": results, "verdict": v,
                   "settings": {"N": N, "split": a.split, "topk": topk,
                                "nms_iou": nms_iou, "per_round": a.per_round,
                                "n_rounds": n_rounds,
                                "ckpt": os.path.abspath(a.ckpt), "config": a.config},
                   "environment": {"timestamp": datetime.now().isoformat(timespec="seconds"),
                                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                                   "command": " ".join(sys.argv)}}, f, indent=1)
    print(f"  full metrics: {out_path}\n  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

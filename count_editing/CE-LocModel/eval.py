#!/usr/bin/env python3
"""Eval CE-Loc round 2 — COCO-style AP + P/R expressed as % of the CEILING.

TWO THINGS TO REMEMBER WHEN READING THE NUMBERS:

1. THE STRUCTURAL PRECISION CEILING = min(M, N)/N. With N=100 and M~37.6 GT the
   ceiling is only 0.376 — a variant that emits more boxes is penalised PURELY for
   emitting more boxes, even if every box is perfect. Comparing raw P/R across
   variants with different N is MEANINGLESS.

2. THE ANNOTATIONS MISS OBJECTS. Verified by eye: image train/1074_b2 has ~8
   buffalo but all_bboxes lists only 6. A missed object that the model correctly
   detects counts as a FALSE POSITIVE -> measured precision is LOWER than true
   precision. Do not conclude the model is bad without looking at the images.

TOP-K, NOT an absolute threshold: focal with a non-discriminating head converges
to a constant (round 1: 0.263 < 0.5) -> every box is filtered out -> argmax keeps
exactly 1 box.
"""

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ce130_dataset import CE130Detection, normalize_for_clip  # noqa: E402
from models.detector import CELocDetector  # noqa: E402
from utils.box_ops_np import box_iou, cxcywh_to_xyxy  # noqa: E402


def nms_class_agnostic(boxes_xyxy, scores, iou_thr=0.5):
    """NMS does not need scores to remove duplicates in principle, but it does need
    them to decide WHICH box to keep within an overlapping group."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        iou = box_iou(boxes_xyxy[i:i + 1], boxes_xyxy[order[1:]])[0][0]
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)


def ap_from_pr(rec, prec):
    """COCO-style AP: make precision monotonically decreasing, then integrate."""
    m_rec = np.concatenate([[0.0], rec, [1.0]])
    m_pre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(m_pre) - 2, -1, -1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    idx = np.where(m_rec[1:] != m_rec[:-1])[0]
    return float(np.sum((m_rec[idx + 1] - m_rec[idx]) * m_pre[idx + 1]))


def evaluate(predictions, iou_thr=0.5):
    """predictions: list[(boxes_xyxy [K,4], scores [K], gt_xyxy [M,4])] -> dict."""
    records, total_gt = [], 0
    for boxes, scores, gt in predictions:
        total_gt += len(gt)
        if len(boxes) == 0:
            continue
        order = np.argsort(-scores)
        used = np.zeros(len(gt), dtype=bool)
        for i in order:
            if len(gt) == 0:
                records.append((scores[i], 0))
                continue
            iou = box_iou(boxes[i:i + 1], gt)[0][0]
            j = int(np.argmax(iou))
            if iou[j] >= iou_thr and not used[j]:
                used[j] = True
                records.append((scores[i], 1))
            else:
                records.append((scores[i], 0))

    if not records or total_gt == 0:
        return {"AP": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pred": 0}

    records.sort(key=lambda x: -x[0])
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([1 - r[1] for r in records])
    rec = tp / total_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    p, r = float(prec[-1]), float(rec[-1])
    return {
        "AP": ap_from_pr(rec, prec), "precision": p, "recall": r,
        "f1": 2 * p * r / max(p + r, 1e-9), "n_pred": len(records),
        "n_gt": total_gt,
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-proposals", type=int, default=None)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    N = a.num_proposals or cfg["diffusion"]["num_proposals_eval"]

    ds = CE130Detection(cfg["data"]["root"], a.split, cfg["data"]["image_size"])
    if a.limit:
        ds.items = ds.items[: a.limit]

    exp_name = cfg.get("experiment", "?")
    env = {
        "experiment": exp_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": str(dev), "torch": torch.__version__,
        "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "command": " ".join(sys.argv), "ckpt": os.path.abspath(a.ckpt),
    }
    print("=" * 78, flush=True)
    print(f"  EVAL — EXPERIMENT {exp_name}", flush=True)
    print("-" * 78, flush=True)
    for k, v in env.items():
        print(f"  {k:22s} {v}", flush=True)
    print(f"  {'split':22s} {a.split} | N={N} | topk={cfg['eval']['topk']} "
          f"| nms={cfg['eval']['nms_iou']} | sampling_steps={cfg['diffusion']['sampling_steps']}",
          flush=True)
    print(f"  {'dataset':22s} {ds.stats()}", flush=True)
    print("=" * 78, flush=True)

    model = CELocDetector(
        cfg["model"]["clip_name"], cfg["model"]["d_model"], cfg["model"]["n_layer"],
        cfg["model"]["n_head"], cfg["data"]["image_size"],
        cfg["diffusion"]["num_timesteps"], cfg["diffusion"]["snr_scale"],
        cfg["diffusion"]["sampling_steps"], 0.0, cfg["model"]["freeze_clip"],
    ).to(dev)
    sd = torch.load(a.ckpt, map_location=dev)
    w = sd["model"] if "model" in sd else sd
    # The checkpoint holds trainable parameters only; frozen CLIP is reloaded from
    # HuggingFace.
    missing, unexpected = model.load_state_dict(w, strict=False)
    missing = [k for k in missing
               if not (k.startswith("encoder.vision.") or k.startswith("encoder.text."))]
    assert not missing and not unexpected, \
        f"checkpoint mismatch: missing={missing} unexpected={unexpected}"
    model.eval()

    topk, nms_iou = cfg["eval"]["topk"], cfg["eval"]["nms_iou"]
    predictions, all_scores = [], []

    t0 = time.time()
    per_image = []
    for i in range(len(ds)):
        t_i = time.time()
        m = ds[i]
        px = torch.from_numpy(normalize_for_clip(m["image"])).unsqueeze(0).to(dev)
        boxes, logits = model.ddim_sample(N, pixel_values=px, texts=[m["text"]])

        b = boxes[0].cpu().numpy()
        s = torch.sigmoid(logits[0]).cpu().numpy()
        all_scores.append(s)

        keep = np.argsort(-s)[:topk]                       # TOP-K, no threshold
        b_xyxy = cxcywh_to_xyxy(b[keep]) * cfg["data"]["image_size"]
        s_k = s[keep]
        k2 = nms_class_agnostic(b_xyxy, s_k, nms_iou)

        gt = cxcywh_to_xyxy(m["boxes"]) * cfg["data"]["image_size"]
        predictions.append((b_xyxy[k2], s_k[k2], gt))

        per_image.append({"image_id": m["image_id"], "class": m["text"],
                          "n_gt": len(gt), "n_after_topk": len(keep),
                          "n_after_nms": len(k2),
                          "score_max": float(s.max()), "score_min": float(s.min()),
                          "seconds": time.time() - t_i})
        if (i + 1) % max(len(ds) // 20, 1) == 0 or i == len(ds) - 1:
            el = time.time() - t0
            eta = el / (i + 1) * (len(ds) - i - 1)
            print(f"  [{i+1:5d}/{len(ds)}] {100*(i+1)/len(ds):5.1f}% | "
                  f"{1000*el/(i+1):.0f}ms/img | elapsed {el:.0f}s | ETA {eta:.0f}s",
                  flush=True)

    # AP at several IoU thresholds (AP50/AP75 + the COCO-style average)
    res = evaluate(predictions, 0.5)
    ap_by_thr = {f"AP{int(100*t)}": evaluate(predictions, t)["AP"]
                 for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]}
    ap_coco = float(np.mean(list(ap_by_thr.values())))

    ceiling = float(np.mean([min(len(g), len(b)) / max(len(b), 1) for b, _, g in predictions]))
    scores = np.concatenate(all_scores)
    n_box = [len(b) for b, _, _ in predictions]
    total_time = time.time() - t0

    print("\n" + "=" * 78, flush=True)
    print(f"RESULTS — EXPERIMENT {exp_name}", flush=True)
    print("-" * 78, flush=True)
    print(f"  AP (COCO, IoU .50:.95)   {ap_coco:.4f}", flush=True)
    print(f"  AP50                     {ap_by_thr['AP50']:.4f}", flush=True)
    print(f"  AP75                     {ap_by_thr['AP75']:.4f}", flush=True)
    print(f"  precision                {res['precision']:.4f}", flush=True)
    print(f"  recall                   {res['recall']:.4f}", flush=True)
    print(f"  f1                       {res['f1']:.4f}", flush=True)
    print("-" * 78, flush=True)
    print(f"  AP by threshold: " + "  ".join(
        f"{k} {v:.4f}" for k, v in ap_by_thr.items()), flush=True)
    print("-" * 78, flush=True)
    print(f"  predicted boxes  {res['n_pred']} ({np.mean(n_box):.1f}/img, "
          f"before NMS {np.mean([x['n_after_topk'] for x in per_image]):.1f})", flush=True)
    print(f"  GT boxes         {res.get('n_gt', 0)} "
          f"({res.get('n_gt', 0)/max(len(ds),1):.1f}/img)", flush=True)
    print(f"  precision ceiling {ceiling:.4f}  (= min(M,N)/N — raw P/R across "
          f"different N is MEANINGLESS)", flush=True)
    print(f"  precision / ceiling {res['precision']/max(ceiling,1e-9):.4f}  "
          f"<- COMPARE THIS ONE", flush=True)
    print("-" * 78, flush=True)
    print(f"  score  mu {scores.mean():.4f}  sd {scores.std():.4f}  "
          f"[{scores.min():.4f}, {scores.max():.4f}]  "
          f"p50 {np.percentile(scores,50):.4f}  p99 {np.percentile(scores,99):.4f}", flush=True)
    print(f"  time {total_time:.0f}s ({1000*total_time/max(len(ds),1):.0f}ms/img)", flush=True)

    warnings = []
    if scores.std() < 0.05:
        warnings.append("score sd < 0.05 — the head is stuck at a constant, so AP is "
                        "nearly meaningless (ranking is random)")
    if np.mean(n_box) < 2:
        warnings.append(f"only {np.mean(n_box):.1f} boxes/img after NMS — check topk/NMS")
    if res["recall"] < 0.01:
        warnings.append(f"recall {res['recall']:.4f} is very low")
    for w in warnings:
        print(f"  [!] {w}", flush=True)
    print("=" * 78, flush=True)

    out_path = os.path.splitext(a.ckpt)[0] + f"_eval_{a.split}_N{N}.json"
    with open(out_path, "w") as f:
        json.dump({
            "summary": {**res, "AP_coco": ap_coco, "precision_ceiling": ceiling,
                        "precision_over_ceiling": res["precision"] / max(ceiling, 1e-9),
                        "warnings": warnings},
            "ap_by_threshold": ap_by_thr,
            "score": {"mean": float(scores.mean()), "std": float(scores.std()),
                      "min": float(scores.min()), "max": float(scores.max()),
                      **{f"p{q}": float(np.percentile(scores, q))
                         for q in [1, 25, 50, 75, 99]}},
            "boxes_per_image": {"mean": float(np.mean(n_box)), "min": int(np.min(n_box)),
                                "max": int(np.max(n_box))},
            "settings": {"N": N, "split": a.split, "topk": topk, "nms_iou": nms_iou,
                         "sampling_steps": cfg["diffusion"]["sampling_steps"]},
            "environment": env,
            "dataset": ds.stats(),
            "total_seconds": total_time,
            "per_image": per_image,      # per-image, to find which ones fail
        }, f, indent=2, ensure_ascii=False)
    print(f"  full metrics (incl. per-image): {out_path}", flush=True)


if __name__ == "__main__":
    main()

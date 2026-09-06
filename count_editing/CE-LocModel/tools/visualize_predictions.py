#!/usr/bin/env python3
"""Draw GT and predicted boxes on the same image, to see WHY the score is low.

The metrics say the model is only ~2x better than a random baseline: 67 % of
objects have no box within IoU 0.1, yet the model emits 53.8 boxes/image against
48.5 GT — enough boxes, in the wrong places. Numbers cannot distinguish "learned
a weak prior" from "a geometry bug in eval", so look at the pixels.

Image selection is DELIBERATELY MIXED, not "the 30 best". Best-only images hide
the failure that dominates the score. Each image is labelled with which bucket it
came from:

  best   - highest recall@0.5. Shows what the model does when it works.
  worst  - lowest recall despite many GT. Shows the dominant failure mode.
  few    - images with <15 GT, where the model over-predicts 3.7x.
  many   - images with >50 GT, which hold 67 % of all test objects.
  random - unbiased sample, so the picture is not curated.

Colors:
  green  = ground truth
  orange = predicted box that matched a GT at IoU>=0.5  (a hit)
  red    = predicted box that matched nothing           (a false positive)
  yellow line = padding boundary

  python3 tools/run_on_free_gpu.py -- tools/visualize_predictions.py \
      --ckpt checkpoints/experiment_a/best.pth --split test --n 30 --out viz_pred
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection, normalize_for_clip  # noqa: E402
from models.detector import CELocDetector  # noqa: E402
from utils.box_ops_np import box_iou, cxcywh_to_xyxy  # noqa: E402
from eval import nms_class_agnostic  # noqa: E402

GREEN, ORANGE, RED, YELLOW = (0, 230, 60), (255, 150, 0), (235, 40, 40), (255, 255, 0)


def draw_one(img_u8, gt_xyxy, pred_xyxy, scores, iou_thr=0.5, show_score=True):
    """Return (PIL image, n_hit). A prediction is a 'hit' if it takes a GT at iou_thr,
    matched greedily by score — the same rule eval.py scores with, so the picture
    and the number agree."""
    img = Image.fromarray(img_u8).convert("RGB")
    dr = ImageDraw.Draw(img)

    for b in gt_xyxy:
        dr.rectangle(b.tolist(), outline=GREEN, width=2)

    hit = np.zeros(len(pred_xyxy), dtype=bool)
    if len(gt_xyxy) and len(pred_xyxy):
        used = np.zeros(len(gt_xyxy), dtype=bool)
        for i in np.argsort(-scores):                     # greedy by score, as in eval
            iou = box_iou(pred_xyxy[i:i + 1], gt_xyxy)[0][0]
            j = int(np.argmax(iou))
            if iou[j] >= iou_thr and not used[j]:
                used[j] = True
                hit[i] = True

    for i, b in enumerate(pred_xyxy):
        dr.rectangle(b.tolist(), outline=ORANGE if hit[i] else RED, width=2)
        if show_score and hit[i]:
            dr.text((b[0] + 2, b[1] + 2), f"{scores[i]:.2f}", fill=ORANGE)
    return img, int(hit.sum())


def pick(stats, n):
    """Spread the sample across buckets so the montage shows failure, not just success."""
    order_recall = sorted(stats, key=lambda s: -s["recall"])
    many = [s for s in stats if s["n_gt"] > 50]
    few = [s for s in stats if s["n_gt"] < 15]
    rng = np.random.default_rng(0)

    buckets = [
        ("best", order_recall[: max(n // 5, 1)]),
        ("worst", [s for s in sorted(many, key=lambda s: s["recall"])][: max(n // 5, 1)]),
        ("few", few[: max(n // 5, 1)]),
        ("many", sorted(many, key=lambda s: -s["n_gt"])[: max(n // 5, 1)]),
    ]
    chosen, seen = [], set()
    for name, items in buckets:
        for s in items:
            if s["image_id"] not in seen:
                seen.add(s["image_id"])
                chosen.append((name, s))
    rest = [s for s in stats if s["image_id"] not in seen]
    for s in rng.permutation(len(rest))[: max(n - len(chosen), 0)]:
        chosen.append(("random", rest[int(s)]))
    return chosen[:n]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scan", type=int, default=200,
                    help="how many images to score before picking the sample")
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    N = cfg["diffusion"]["num_proposals_eval"]
    topk = a.topk or cfg["eval"]["topk"]
    nms_iou = cfg["eval"]["nms_iou"]
    os.makedirs(a.out, exist_ok=True)

    ds = CE130Detection(cfg["data"]["root"], a.split, cfg["data"]["image_size"])
    print(f"[data] {a.split}: {ds.stats()}", flush=True)

    model = CELocDetector(
        cfg["model"]["clip_name"], cfg["model"]["d_model"], cfg["model"]["n_layer"],
        cfg["model"]["n_head"], cfg["data"]["image_size"],
        cfg["diffusion"]["num_timesteps"], cfg["diffusion"]["snr_scale"],
        cfg["diffusion"]["sampling_steps"], 0.0, cfg["model"]["freeze_clip"]).to(dev)
    sd = torch.load(a.ckpt, map_location=dev)
    missing, unexpected = model.load_state_dict(sd.get("model", sd), strict=False)
    missing = [k for k in missing
               if not (k.startswith("encoder.vision.") or k.startswith("encoder.text."))]
    assert not missing and not unexpected, f"checkpoint mismatch: {missing} {unexpected}"
    model.eval()
    print(f"[model] loaded {a.ckpt} (epoch {sd.get('epoch','?')}, "
          f"val_loss {sd.get('loss', float('nan')):.4f})", flush=True)

    def run(i):
        m = ds[i]
        px = torch.from_numpy(normalize_for_clip(m["image"])).unsqueeze(0).to(dev)
        boxes, logits = model.ddim_sample(N, pixel_values=px, texts=[m["text"]])
        b = boxes[0].cpu().numpy()
        s = torch.sigmoid(logits[0]).cpu().numpy()
        keep = np.argsort(-s)[:topk]
        p = cxcywh_to_xyxy(b[keep]) * cfg["data"]["image_size"]
        sc = s[keep]
        k2 = nms_class_agnostic(p, sc, nms_iou)
        gt = cxcywh_to_xyxy(m["boxes"]) * cfg["data"]["image_size"]
        return m, gt, p[k2], sc[k2]

    # pass 1: score a scan window so the sample can be chosen by behaviour
    n_scan = min(a.scan, len(ds))
    step = max(len(ds) // n_scan, 1)
    idxs = list(range(0, len(ds), step))[:n_scan]
    stats = []
    print(f"[scan] scoring {len(idxs)} images to pick a spread...", flush=True)
    for c, i in enumerate(idxs):
        m, gt, p, sc = run(i)
        _, n_hit = draw_one(m["image"], gt, p, sc)
        stats.append({"idx": i, "image_id": m["image_id"], "class": m["text"],
                      "n_gt": len(gt), "n_pred": len(p),
                      "n_hit": n_hit,
                      "recall": n_hit / max(len(gt), 1),
                      "precision": n_hit / max(len(p), 1)})
        if (c + 1) % max(len(idxs) // 10, 1) == 0:
            print(f"  {c+1}/{len(idxs)}", flush=True)

    # pass 2: draw only the chosen ones
    chosen = pick(stats, a.n)
    print(f"\n[draw] {len(chosen)} images -> {a.out}/", flush=True)
    print(f"  {'bucket':8s} {'image':>8s} {'class':16s} {'GT':>5s} {'pred':>5s} "
          f"{'hit':>5s} {'recall':>7s} {'prec':>7s}", flush=True)
    rows = []
    for bucket, s in chosen:
        m, gt, p, sc = run(s["idx"])
        img, n_hit = draw_one(m["image"], gt, p, sc)
        nh = int(round(m["valid_h"] * cfg["data"]["image_size"]))
        if nh < cfg["data"]["image_size"] - 1:
            ImageDraw.Draw(img).line([(0, nh), (cfg["data"]["image_size"], nh)],
                                     fill=YELLOW, width=3)
        name = (f"{bucket}_{m['image_id']}_{m['text'].replace(' ', '-')}"
                f"_gt{len(gt)}_hit{n_hit}.png")
        img.save(os.path.join(a.out, name))
        r, pr = n_hit / max(len(gt), 1), n_hit / max(len(p), 1)
        print(f"  {bucket:8s} {m['image_id']:>8s} {m['text'][:16]:16s} {len(gt):5d} "
              f"{len(p):5d} {n_hit:5d} {r:7.3f} {pr:7.3f}", flush=True)
        rows.append({"bucket": bucket, "file": name, **s})

    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump({"scanned": stats, "drawn": rows,
                   "settings": {"N": N, "topk": topk, "nms_iou": nms_iou,
                                "split": a.split}}, f, indent=2)

    rec = np.array([s["recall"] for s in stats])
    npred = np.array([s["n_pred"] for s in stats])
    if npred.mean() < 3:
        print(f"\n[!] only {npred.mean():.1f} boxes/image survive NMS. With a score head "
              f"stuck near a constant, NMS collapses everything into one box — check "
              f"that the checkpoint actually trained (score std should be > 0.05).",
              flush=True)
    print(f"\n[summary over {len(stats)} scanned images]", flush=True)
    print(f"  recall@0.5 per image: mean {rec.mean():.3f}, median {np.median(rec):.3f}, "
          f"max {rec.max():.3f}", flush=True)
    print(f"  images with recall 0: {int((rec == 0).sum())}/{len(rec)}", flush=True)
    print(f"  index + per-image stats: {os.path.join(a.out, 'index.json')}", flush=True)


if __name__ == "__main__":
    main()

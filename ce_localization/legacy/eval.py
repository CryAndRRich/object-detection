#!/usr/bin/env python3
"""Eval CE-Loc GỐC (bài add) trên samples/test: mỗi ảnh sinh N box (mặc định 30 như
`test_mul_box.py`), so với target_bbox.

Hai sampler: `mock` (vòng lặp gốc, 100 bước "x -= eps/100" — số so được với bài) và `ddpm`
(1000 bước đúng công thức). Chỉ số mỗi sampler:
  best_iou        trung bình IoU tốt nhất trong N box (ORACLE: chọn box bằng GT — giao thức của bài)
  hit50           tỉ lệ ảnh có ít nhất 1 box IoU >= 0,5
  mean_iou        IoU trung bình của MỘT box bất kỳ (không oracle)
  best_iou_orig   best_iou tính bằng `calculate_iou` gốc trên toạ độ chuẩn hoá (công thức SAI
                  của bài, chỉ để đối chiếu số đã công bố)
Mốc `prior`: N target_bbox lấy ngẫu nhiên từ samples/train, KHÔNG nhìn ảnh — model phải hơn nó.

  python ../tools/run_on_free_gpu.py -- legacy/eval.py --ckpt checkpoints/celoc_density/best.pt \\
      --out /mnt/disk1/aiotlab/haitn/output/celoc_density_test.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.legacy.celoc_data import ObjectPlacementDataset  # noqa: E402
from ce_localization.legacy.celoc_model import (  # noqa: E402
    iou_original_formula, iou_pixels, load_policy, sample_ddpm, sample_mock)
from ce_localization.legacy.celoc_vision import TARGET  # noqa: E402
from ce_localization.utils.log import fmt_time  # noqa: E402

SAMPLERS = ("mock", "ddpm")


@torch.no_grad()
def run_eval(model, loader, n_samples=30, samplers=SAMPLERS, seed=0, log_every=0):
    """-> {sampler: {"best", "mean", "best_orig": list theo ảnh}}, gt [M, 4]."""
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed)
    out = {s: {"best": [], "mean": [], "best_orig": []} for s in samplers}
    gts, t0, done = [], time.time(), 0
    for k, batch in enumerate(loader):
        den = batch["density_map"].to(dev) if "density_map" in batch else None
        cond = model.condition(batch["pixel_values"].to(dev), den, list(batch["text"]))
        gt = batch["bbox"].numpy()[:, None]                              # [B, 1, 4]
        gts.append(gt[:, 0])
        for s in samplers:
            fn = sample_mock if s == "mock" else sample_ddpm
            boxes = fn(model, cond, n_samples, generator=g).cpu().numpy()  # [B, N, 4]
            iou = iou_pixels(boxes, gt)
            out[s]["best"] += iou.max(1).tolist()
            out[s]["mean"] += iou.mean(1).tolist()
            out[s]["best_orig"] += iou_original_formula(boxes, gt).max(1).tolist()
        done += len(gt)
        if log_every and (k + 1) % log_every == 0:
            el = time.time() - t0
            print(f"  eval {done}/{len(loader.dataset)} | {fmt_time(el)} | ETA "
                  f"{fmt_time(el / done * (len(loader.dataset) - done))}", flush=True)
    return out, np.concatenate(gts)


def summarize(out):
    return {s: {"best_iou": float(np.mean(v["best"])), "hit50": float(np.mean(np.array(v["best"]) >= 0.5)),
                "mean_iou": float(np.mean(v["mean"])), "best_iou_orig": float(np.mean(v["best_orig"]))}
            for s, v in out.items()}


def prior_boxes(train_dir):
    """Mọi target_bbox của train, chuẩn hoá như dataset gốc (cần kích thước ảnh: đọc header)."""
    ds = ObjectPlacementDataset(train_dir, use_density=False)
    boxes = []
    for f in ds.files:
        W, H = Image.open(os.path.join(ds.image_dir, f)).size
        s = min(TARGET / W, TARGET / H)
        _, b = ds.parse_annotation(f)
        boxes.append([(v * s / TARGET) * 2 - 1 for v in b])
    return np.array(boxes, np.float64)


def prior_baseline(prior, gt, n_samples, seed=0):
    rng = np.random.default_rng(seed)
    boxes = prior[rng.integers(0, len(prior), size=(len(gt), n_samples))]   # [M, N, 4]
    iou = iou_pixels(boxes, gt[:, None])
    return {"best_iou": float(iou.max(1).mean()), "hit50": float((iou.max(1) >= 0.5).mean()),
            "mean_iou": float(iou.mean()), "best_iou_orig": float(iou_original_formula(boxes, gt[:, None]).max(1).mean())}


def print_table(summary, title=""):
    print(f"\n{title}\n{'':>8} {'best_iou':>9} {'hit50':>7} {'mean_iou':>9} {'best_orig':>10}")
    for s, r in summary.items():
        print(f"{s:>8} {r['best_iou']:9.4f} {r['hit50']:7.4f} {r['mean_iou']:9.4f} {r['best_iou_orig']:10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.pt / last.pt của legacy/train.py, hoặc best_model.pth gốc")
    ap.add_argument("--data", default="../data/samples/test")
    ap.add_argument("--prior-from", default="../data/samples/train", help="'' = bỏ mốc prior")
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--samplers", nargs="+", default=list(SAMPLERS), choices=SAMPLERS)
    ap.add_argument("--limit", type=int, default=0, help="0 = cả tập")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    model, ck = load_policy(args.ckpt, dev)
    print(f"[{fmt_time(time.time() - t0)}] {args.ckpt}: epoch {ck.get('epoch')} loss {ck.get('loss')} "
          f"density {model.use_density} | config {ck.get('config', ck.get('args'))}", flush=True)
    ds = ObjectPlacementDataset(args.data, use_density=model.use_density)
    if args.limit:
        ds = Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"{len(ds)} ảnh, {args.n_samples} box/ảnh, sampler {args.samplers}", flush=True)

    out, gt = run_eval(model, loader, args.n_samples, args.samplers, args.seed, log_every=10)
    summary = summarize(out)
    if args.prior_from:
        summary["prior"] = prior_baseline(prior_boxes(args.prior_from), gt, args.n_samples, args.seed)
    print_table(summary, f"TEST ({len(ds)} ảnh) — best_iou/hit50 là ORACLE best-of-{args.n_samples}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(args=vars(args), epoch=ck.get("epoch"), use_density=model.use_density,
                       summary=summary, per_image={s: v for s, v in out.items()}), f)
    print(f"[{fmt_time(time.time() - t0)}] xong -> {args.out}")


if __name__ == "__main__":
    main()

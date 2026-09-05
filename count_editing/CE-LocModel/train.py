#!/usr/bin/env python3
"""Train CE-Loc round 2.

THREE TRACKING METRICS (as important as the loss — round 1 lacked them and stayed
blind through 5 rounds of fixes):

  1. % of matched pairs PRESERVED between epochs — tells you whether the
     score<->coordinate feedback loop has been broken. Round 1 sat at ~55 %, i.e.
     more than half the labels changed every epoch, so the score head could never
     learn anything.
  2. std of sigmoid(score) — < 0.05 means the head is stuck at a constant (focal
     with alpha=0.25 and a non-discriminating head converges to a fixed value).
  3. mean IoU of matched pairs — separate from the loss, so it is easy to read.

Run in the background, logging to a file (on the server:
/mnt/disk1/aiotlab/haitn/log/):
  nohup python3 train.py --config config/experiment_a.yaml > <log> 2>&1 & echo $!
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ce130_dataset import CE130Detection, PatchCache, normalize_for_clip  # noqa: E402
from models.detector import CELocDetector  # noqa: E402
from models.criterion import SetCriterion  # noqa: E402


class TorchWrap(Dataset):
    """Wraps CE130Detection (numpy) as a torch Dataset.

    With a `cache`, it returns precomputed patch/text tokens and DROPS the image
    entirely — measured on an A30: CLIP takes 76.8 % of per-batch time, so the
    cache gives ~4.3x speedup.
    """

    def __init__(self, ds, cache=None):
        self.ds = ds
        self.cache = cache

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        m = self.ds[i]
        out = {
            "boxes": torch.from_numpy(m["boxes"]).float(),
            "text": m["text"],
            "valid_h": m["valid_h"],
            "image_id": m["image_id"],
        }
        if self.cache is None:
            out["pixel_values"] = torch.from_numpy(normalize_for_clip(m["image"]))
        else:
            patch, text = self.cache.get(m["image_id"], m["text"], m["flipped"])
            out["patch_raw"] = torch.from_numpy(patch)
            out["text_raw"] = torch.from_numpy(text)
        return out


def collate(batch):
    """Box count varies per image -> keep them as a list, do not pad here."""
    out = {
        "boxes": [b["boxes"] for b in batch],
        "text": [b["text"] for b in batch],
        "valid_h": [b["valid_h"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
    }
    for k in ("pixel_values", "patch_raw", "text_raw"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def fmt_time(seconds):
    """3661 -> '1h01m01s'. Used for both elapsed time and ETA."""
    seconds = int(max(seconds, 0))
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def array_stats(x):
    """Full distribution of an array — so it can be re-read when something breaks."""
    if x is None or len(x) == 0:
        return {}
    x = np.asarray(x, dtype=np.float64)
    q = np.percentile(x, [1, 25, 50, 75, 99])
    return {"mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "max": float(x.max()),
            "p1": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p99": float(q[4])}


def label_stability(before, after):
    """% of (image_id, pred_idx) -> gt_idx pairs preserved between two epochs."""
    if not before:
        return float("nan")
    shared = set(before) & set(after)
    if not shared:
        return 0.0
    return sum(before[k] == after[k] for k in shared) / len(shared)


def write_json(save_dir, env, cfg, history, best, ds_train, ds_val):
    """Write EVERY metric to history.json after EACH epoch.

    Written every epoch (not just at the end) so a job that dies mid-run is still
    readable. It holds enough to diagnose without re-running: environment, the
    full config, dataset statistics, and every per-epoch metric with its
    distribution (mean/std/min/max/percentiles), not just the mean.
    """
    # Compute best FROM history, not from the `best` argument — this function is
    # called BEFORE the training loop updates `best`, so using it would be off by
    # one epoch.
    top = min(history, key=lambda e: e["val"]["loss"]) if history else None
    summary = {
        "epochs_completed": len(history),
        "best_val_loss": top["val"]["loss"] if top else None,
        "best_epoch": top["epoch"] if top else None,
        "total_time": fmt_time(history[-1]["elapsed_sec"]) if history else "0s",
        "epochs_with_warnings": [e["epoch"] for e in history if e["warnings"]],
    }
    if len(history) >= 2:
        v = [e["val"]["loss"] for e in history]
        summary["val_loss_first_last"] = [v[0], v[-1]]
        summary["val_rising_streak"] = sum(
            1 for i in range(len(v) - 1, 0, -1) if v[i] > v[i - 1]) if v[-1] > v[-2] else 0

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump({
            "summary": summary,
            "environment": env,
            "config": cfg,
            "dataset": {"train": ds_train.stats(), "val": ds_val.stats()},
            "epochs": history,
        }, f, indent=2, ensure_ascii=False)


def model_inputs(batch, dev):
    """Return model kwargs: raw images, or cached tokens."""
    if "patch_raw" in batch:
        return {"patch_raw": batch["patch_raw"].to(dev),
                "text_raw": batch["text_raw"].to(dev)}
    return {"pixel_values": batch["pixel_values"].to(dev), "texts": batch["text"]}


@torch.no_grad()
def run_val(model, loader, crit, N, dev):
    """Validation loss — with 1,911 images and 3 splits whose classes are DISJOINT,
    without val you cannot tell when overfitting starts. A fixed seed for `t` makes
    epochs comparable (a random t makes val loss noisy and the trend unreadable)."""
    model.eval()
    keys = ["loss", "loss_l1", "loss_giou", "loss_ce", "iou_matched", "n_matched"]
    total, nb, scores = {k: 0.0 for k in keys}, 0, []
    t0 = time.time()
    # The generator MUST be on the same device as the tensors it creates
    # (torch.randn(device='cuda', generator=<cpu gen>) -> RuntimeError). GT is
    # already on `dev`, so placeholders are created on `dev` too.
    g = torch.Generator(device=dev).manual_seed(1234)
    for batch in loader:
        tg = [b.to(dev) for b in batch["boxes"]]
        x_t, tt, _ = model.build_inputs(tg, N, batch["valid_h"], generator=g)
        pb, lg = model(x_t, tt, **model_inputs(batch, dev))
        _, st, _ = crit(pb, lg, tg)
        for k in keys:
            total[k] += st[k]
        scores.append(lg.sigmoid().cpu().numpy().ravel())
        nb += 1
    model.train()
    out = {k: v / max(nb, 1) for k, v in total.items()}
    out["score"] = array_stats(np.concatenate(scores)) if scores else {}
    out["seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="shrink the dataset for a smoke test")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache", default=None,
                    help="patch-token cache dir (tools/build_cache.py). Enabling it "
                         "removes the CLIP forward from training — measured ~4.3x faster.")
    ap.add_argument("--log-every-n-batch", type=int, default=None,
                    help="print progress every N batches; default is 5 times per epoch")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if a.epochs:
        cfg["training"]["epochs"] = a.epochs
    if a.batch_size:
        cfg["training"]["batch_size"] = a.batch_size

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg["training"]["seed"])

    exp_name = cfg.get("experiment", "?")
    env = {
        "experiment": exp_name,
        "description": cfg.get("description", ""),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(dev),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "hostname": socket.gethostname(),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
    }
    print("=" * 78, flush=True)
    print(f"  TRAIN — EXPERIMENT {exp_name}", flush=True)
    print("-" * 78, flush=True)
    for k, v in env.items():
        print(f"  {k:22s} {v}", flush=True)
    print("-" * 78, flush=True)
    for section in ["data", "model", "diffusion", "loss", "matcher", "training"]:
        print(f"  {section:10s} {json.dumps(cfg[section], ensure_ascii=False)}", flush=True)
    print("=" * 78, flush=True)

    ds = CE130Detection(cfg["data"]["root"], "train",
                        cfg["data"]["image_size"], cfg["data"]["flip_prob"],
                        seed=cfg["training"]["seed"])
    if a.limit:
        ds.items = ds.items[: a.limit]
    print(f"[data] {ds.stats()}", flush=True)

    cache_tr = cache_va = None
    if a.cache:
        cache_tr = PatchCache(a.cache, "train")
        cache_va = PatchCache(a.cache, "val")
        print(f"[cache] using {a.cache} — CLIP forward skipped during training "
              f"({cache_tr.n_ver} versions/image)", flush=True)

    loader = DataLoader(TorchWrap(ds, cache_tr), batch_size=cfg["training"]["batch_size"],
                        shuffle=True, num_workers=cfg["data"]["num_workers"],
                        collate_fn=collate, drop_last=False)

    # val: NO flipping (no augmentation at evaluation time)
    ds_val = CE130Detection(cfg["data"]["root"], "val", cfg["data"]["image_size"])
    if a.limit:
        ds_val.items = ds_val.items[: max(a.limit // 2, 1)]
    val_loader = DataLoader(TorchWrap(ds_val, cache_va), batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=cfg["data"]["num_workers"],
                            collate_fn=collate)
    print(f"[val ] {ds_val.stats()}", flush=True)

    model = CELocDetector(
        cfg["model"]["clip_name"], cfg["model"]["d_model"], cfg["model"]["n_layer"],
        cfg["model"]["n_head"], cfg["data"]["image_size"],
        cfg["diffusion"]["num_timesteps"], cfg["diffusion"]["snr_scale"],
        cfg["diffusion"]["sampling_steps"], cfg["model"]["dropout"],
        cfg["model"]["freeze_clip"],
    ).to(dev)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] trainable parameters: {sum(p.numel() for p in trainable)/1e6:.2f}M "
          f"/ total {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    crit = SetCriterion(cfg["matcher"]["method"],
                        **({"use_center_prior": cfg["matcher"]["use_center_prior"],
                            "radius_ratio": cfg["matcher"]["center_radius"]}
                           if cfg["matcher"]["method"] == "simota" else {}))
    opt = torch.optim.AdamW(trainable, lr=float(cfg["training"]["lr"]),
                            weight_decay=float(cfg["training"]["weight_decay"]))

    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    N = cfg["diffusion"]["num_proposals_train"]
    if a.log_every_n_batch is None:
        a.log_every_n_batch = max(len(loader) // 5, 1) if len(loader) >= 10 else 0
    best, history, prev_labels = float("inf"), [], {}
    t_start = time.time()

    for ep in range(cfg["training"]["epochs"]):
        model.train()
        t0 = time.time()
        total = {"loss": 0.0, "loss_l1": 0.0, "loss_giou": 0.0, "loss_ce": 0.0,
                 "iou_matched": 0.0, "n_matched": 0}
        labels_now, scores, nb = {}, [], 0
        grad_norms, n_gt, t_batch = [], [], []

        for batch in loader:
            t_b = time.time()
            tg = [b.to(dev) for b in batch["boxes"]]

            x_t, tt, _ = model.build_inputs(tg, N, batch["valid_h"])
            pb, lg = model(x_t, tt, **model_inputs(batch, dev))
            loss, st, idx = crit(pb, lg, tg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            for k in total:
                total[k] += st[k]
            nb += 1
            grad_norms.append(float(gn))
            n_gt += [len(b) for b in batch["boxes"]]
            t_batch.append(time.time() - t_b)
            scores.append(lg.detach().sigmoid().cpu().numpy().ravel())

            # In-epoch progress — with 1,911 images an epoch takes several minutes,
            # so it should not stay silent. Print 5 times per epoch (no tqdm, so the
            # log stays readable in a file).
            if a.log_every_n_batch and nb % a.log_every_n_batch == 0:
                el = time.time() - t0
                print(f"      ... batch {nb}/{len(loader)} | loss {st['loss']:.4f} | "
                      f"{1000*el/nb:.0f}ms/batch | {fmt_time(el)} | "
                      f"left {fmt_time(el/nb*(len(loader)-nb))}", flush=True)
            for i, (pi, gi) in enumerate(idx):                 # metric 1
                iid = batch["image_id"][i]
                for p, g in zip(pi.tolist(), gi.tolist()):
                    labels_now[(iid, p)] = g

        tr = {k: v / max(nb, 1) for k, v in total.items()}
        stability = label_stability(prev_labels, labels_now)
        prev_labels = labels_now
        score_stats = array_stats(np.concatenate(scores)) if scores else {}
        std_score = score_stats.get("std", 0.0)
        train_sec = time.time() - t0

        val = run_val(model, val_loader, crit, N, dev)
        epoch_sec = time.time() - t0
        elapsed = time.time() - t_start
        remaining = cfg["training"]["epochs"] - (ep + 1)
        eta = (elapsed / (ep + 1)) * remaining

        # --- LOG: 3 fixed lines per epoch, printing everything readable ---
        print(f"[ep {ep+1:4d}/{cfg['training']['epochs']}] "
              f"train {tr['loss']:8.4f} (l1 {tr['loss_l1']:.4f} giou {tr['loss_giou']:.4f} "
              f"ce {tr['loss_ce']:.4f})   val {val['loss']:8.4f} "
              f"(l1 {val['loss_l1']:.4f} giou {val['loss_giou']:.4f} ce {val['loss_ce']:.4f})",
              flush=True)
        print(f"           IoU train {tr['iou_matched']:.4f} / val {val['iou_matched']:.4f} | "
              f"matched {tr['n_matched']:.1f}/{N} ({100*tr['n_matched']/N:.0f}%) | "
              f"GT/img {np.mean(n_gt):.1f} | label_stability {stability:.3f} | "
              f"lr {opt.param_groups[0]['lr']:.2e} | grad {np.mean(grad_norms):.3f}",
              flush=True)
        print(f"           score mu {score_stats.get('mean', 0):.4f} sd {std_score:.4f} "
              f"[{score_stats.get('min', 0):.3f}, {score_stats.get('max', 0):.3f}] "
              f"p50 {score_stats.get('p50', 0):.4f} | "
              f"{fmt_time(train_sec)}+{fmt_time(val['seconds'])} "
              f"({1000*np.mean(t_batch):.0f}ms/batch) | "
              f"elapsed {fmt_time(elapsed)} | ETA {fmt_time(eta)}", flush=True)

        warnings = []
        if std_score < 0.05:
            warnings.append("std_score < 0.05 — the score head may be stuck at a constant")
        if not np.isnan(stability) and stability < 0.4:
            warnings.append(f"label_stability {stability:.2f} < 0.40 — labels change too much per epoch")
        if np.mean(grad_norms) > 100:
            warnings.append(f"grad norm {np.mean(grad_norms):.1f} is very large")
        for w in warnings:
            print(f"           [!] {w}", flush=True)

        history.append({
            "epoch": ep + 1,
            "train": {**tr, "score": score_stats,
                      "grad_norm": array_stats(grad_norms),
                      "gt_per_image": array_stats(n_gt),
                      "ms_per_batch": array_stats([1000 * x for x in t_batch]),
                      "n_batches": nb, "seconds": train_sec},
            "val": val,
            "label_stability": stability,
            "lr": opt.param_groups[0]["lr"],
            "epoch_sec": epoch_sec,
            "elapsed_sec": elapsed,
            "eta_sec": eta,
            "warnings": warnings,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        write_json(save_dir, env, cfg, history, best, ds, ds_val)

        # pick best by VAL loss, not train loss
        if val["loss"] < best:
            best = val["loss"]
            # Save ONLY the TRAINABLE parameters (~8.3M). Saving frozen CLIP too
            # made the checkpoint 698 MB, of which 98 % is weights re-downloadable
            # from HuggingFace, and it would force an exact CLIP version match at
            # load time.
            trainable_sd = {k: v for k, v in model.state_dict().items()
                            if not (k.startswith("encoder.vision.") or k.startswith("encoder.text."))}
            torch.save({"epoch": ep, "model": trainable_sd, "optimizer": opt.state_dict(),
                        "loss": best, "cfg": cfg, "trainable_only": True},
                       os.path.join(save_dir, "best.pth"))
            print(f"  -> saved best (val_loss {best:.4f})", flush=True)

    total_time = time.time() - t_start
    print("=" * 78, flush=True)
    print(f"DONE — EXPERIMENT {exp_name} — {fmt_time(total_time)} ({len(history)} epochs, "
          f"{fmt_time(total_time/max(len(history),1))}/epoch)", flush=True)
    if history:
        top = min(history, key=lambda e: e["val"]["loss"])
        print(f"  best: epoch {top['epoch']} | val_loss {top['val']['loss']:.4f} | "
              f"val_IoU {top['val']['iou_matched']:.4f}", flush=True)
        print(f"  val_loss: {history[0]['val']['loss']:.4f} -> "
              f"{history[-1]['val']['loss']:.4f}", flush=True)
        if top["epoch"] == len(history):
            print("  [!] best landed on the LAST epoch — not saturated yet, train longer "
                  "(round 1 hit this 4 times in a row)", flush=True)
        warned = [e["epoch"] for e in history if e["warnings"]]
        if warned:
            print(f"  [!] {len(warned)}/{len(history)} epochs had warnings: "
                  f"{warned[:10]}{'...' if len(warned) > 10 else ''}", flush=True)
    print(f"  full metrics: {os.path.join(save_dir, 'history.json')}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()

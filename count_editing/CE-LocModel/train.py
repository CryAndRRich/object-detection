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

from data.ce130_dataset import PatchCache, normalize_for_clip  # noqa: E402
from data.factory import build_dataset  # noqa: E402
from models.detector import build_model  # noqa: E402
from models.criterion import SetCriterion  # noqa: E402
from utils.box_ops import box_iou, cxcywh_to_xyxy  # noqa: E402


class TorchWrap(Dataset):
    """Wraps a numpy dataset (CE-130 or COCO, see data/factory.py) as a torch Dataset.

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
        # With a cache the image is never used, so do not decode it (16.9 ms/image).
        m = self.ds.__getitem__(i, need_image=self.cache is None)
        out = {
            "boxes": torch.from_numpy(m["boxes"]).float(),
            # Row-aligned with "boxes"; only A.2's C-way head reads it, but every
            # dataset supplies it so the loop needs no per-experiment branch.
            "labels": torch.from_numpy(m["labels"]).long(),
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
        # Parallel to "boxes" row for row. Needed only by A.2's C-way head, but
        # carried always so the loop has no per-experiment branch.
        "labels": [b["labels"] for b in batch],
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
    # `select_loss` is the loss of the round inference actually uses (C1) and falls
    # back to `val.loss` for A/B/A.1/A.2 — the SAME rule the checkpoint block uses,
    # so history.json's "best" always names the epoch that was actually saved.
    sel = lambda e: e.get("select_loss", e["val"]["loss"])          # noqa: E731
    top = min(history, key=sel) if history else None
    # With select_metric=oracle_recall (C1c) `select_loss` holds the NEGATED
    # coverage so that "lower is better" still holds for min()/val_rising_streak.
    # Report it un-negated so nobody reads "best_val_loss: -0.133" as a loss.
    sel_name = (history[-1].get("select_metric", "loss_final") if history
                else "loss_final")
    is_gain = sel_name == "oracle_recall"
    summary = {
        "epochs_completed": len(history),
        "select_metric": sel_name,
        "best_score": (-sel(top) if is_gain else sel(top)) if top else None,
        "best_val_loss": (None if is_gain else (sel(top) if top else None)),
        "best_val_loss_mean_over_rounds": top["val"]["loss"] if top else None,
        "best_epoch": top["epoch"] if top else None,
        "total_time": fmt_time(history[-1]["elapsed_sec"]) if history else "0s",
        "epochs_with_warnings": [e["epoch"] for e in history if e["warnings"]],
    }
    if len(history) >= 2:
        # Same quantity the checkpoint is chosen on: for C1 that is the LAST
        # refinement round (what ddim_sample returns), not the mean over six.
        # Reading `val.loss` here would make the overfitting signal track a number
        # no forward pass ever produces.
        v = [e.get("select_loss", e["val"]["loss"]) for e in history]
        summary["val_loss_first_last"] = ([-v[0], -v[-1]] if is_gain
                                          else [v[0], v[-1]])
        summary["select_loss_is"] = ("last refinement round"
                                     if history[-1].get("val_loss_final") is not None
                                     else "single forward")
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
    """Return model kwargs: raw images, or cached tokens.

    `non_blocking=True` is what makes the loader's `pin_memory` worth anything: on
    page-locked memory the host->device copy is asynchronous and overlaps with
    compute. Without it, pin_memory only adds a staging copy.
    """
    nb = dev.type == "cuda"
    if "patch_raw" in batch:
        return {"patch_raw": batch["patch_raw"].to(dev, non_blocking=nb),
                "text_raw": batch["text_raw"].to(dev, non_blocking=nb)}
    return {"pixel_values": batch["pixel_values"].to(dev, non_blocking=nb),
            "texts": batch["text"]}


@torch.no_grad()
def oracle_recall(pred_cxcywh, gt_cxcywh, iou_thr=0.5):
    """Fraction of GT covered by AT LEAST ONE predicted box, ignoring the score
    entirely and ignoring the matcher entirely.

    This is the quantity `loss_final` cannot see. `iou_matched` only averages over
    the pairs Hungarian already chose (a fixed 268.2 of them every round), so a
    round that tightens the boxes it already had while LOSING coverage of other GT
    scores BETTER on the loss and WORSE in reality. Measured on C1's test set that
    is exactly what happens: iou_matched rises 0.3426 -> 0.3528 over the six rounds
    while oracle_recall falls 0.138 -> 0.133.

    Same definition as tools/measure_box_quality.py::quality so the numbers printed
    here are directly comparable with the ones that tool reports.
    """
    # Return n_gt even when there are NO predictions: those GT are uncovered, not
    # absent. Returning 0 would shrink the denominator and report coverage as if
    # the hard images had never been in the split.
    if gt_cxcywh.numel() == 0:
        return 0.0, 0
    if pred_cxcywh.numel() == 0:
        return 0.0, int(gt_cxcywh.shape[0])
    iou = box_iou(cxcywh_to_xyxy(pred_cxcywh), cxcywh_to_xyxy(gt_cxcywh))  # [P,G]
    if isinstance(iou, tuple):
        iou = iou[0]
    best = iou.max(dim=0).values                       # best over predictions, per GT
    return float((best >= iou_thr).sum()), int(gt_cxcywh.shape[0])


# `no_grad` belongs HERE, on run_val -- not only on oracle_recall above.
# It was lost once by inserting a new decorated function directly above this
# one: the decorator stayed with the new function and run_val silently kept
# building a graph, which crashed at `.numpy()` and held the activations.
@torch.no_grad()
def run_val(model, loader, crit, N, dev):
    """Validation loss — with 1,911 images and 3 splits whose classes are DISJOINT,
    without val you cannot tell when overfitting starts. A fixed seed for `t` makes
    epochs comparable (a random t makes val loss noisy and the trend unreadable)."""
    model.eval()
    keys = ["loss", "loss_l1", "loss_giou", "loss_ce", "iou_matched", "n_matched"]
    # EXPERIMENT C1 adds per-round stats. They must survive into `val`, because the
    # round that INFERENCE uses is the last one -- `loss` here is the mean over six
    # rounds, of which round 1 regresses straight from the noisy anchor and is always
    # bad. Measured on a controlled example: mean 6.22 vs final 3.95 (1.58x), and
    # IoU mean 0.732 vs final 0.990. Dropping the per-round keys would leave the
    # checkpoint chosen, and A compared, on a number no forward pass ever produces.
    total, nb, scores = {k: 0.0 for k in keys}, 0, []
    extra, per_round = {}, {}
    # SCORE-FREE, MATCHER-FREE coverage, accumulated as raw counts (hits, n_gt) and
    # divided ONCE at the end. Averaging per-batch ratios instead would weight a
    # 3-GT image the same as a 500-GT one, which on CE-130 (1..1229 GT per image)
    # is a different number entirely.
    orec_hits, orec_tot = {}, 0
    t0 = time.time()
    # The generator MUST be on the same device as the tensors it creates
    # (torch.randn(device='cuda', generator=<cpu gen>) -> RuntimeError). GT is
    # already on `dev`, so placeholders are created on `dev` too.
    g = torch.Generator(device=dev).manual_seed(1234)
    for batch in loader:
        tg = [b.to(dev) for b in batch["boxes"]]
        tl = [l.to(dev) for l in batch["labels"]]
        x_t, tt, _ = model.build_inputs(tg, N, batch["valid_h"], generator=g)
        out = model(x_t, tt, **model_inputs(batch, dev))
        if isinstance(out, list):
            _, st, _ = crit(out, tg, labels=tl)
            lg = out[-1][1]
            rounds = [r[0] for r in out]
        else:
            pb, lg = out
            _, st, _ = crit(pb, lg, tg, labels=tl)
            rounds = [pb]
        # Coverage per round. `rounds[r][i]` is [N,4] cxcywh for image i.
        for r, br in enumerate(rounds):
            h = orec_hits.setdefault(r, 0.0)
            for i, gt in enumerate(tg):
                hit, _n = oracle_recall(br[i], gt)
                h += hit
            orec_hits[r] = h
        orec_tot += int(sum(int(g.shape[0]) for g in tg))
        for k in keys:
            total[k] += st[k]
        for k, v in st.items():
            if k.endswith("_final"):
                extra[k] = extra.get(k, 0.0) + v
            elif k.endswith("_per_round"):
                if k not in per_round:
                    per_round[k] = [0.0] * len(v)
                for r, x in enumerate(v):
                    per_round[k][r] += x
        # A.2's logits are [B,N,C]: take the max over classes so the reported
        # distribution keeps meaning "confidence that this slot holds an object",
        # comparable with A/B's 1-D score rather than diluted by 79 negatives.
        scores.append((lg if lg.dim() == 2 else lg.max(-1).values)
                      .sigmoid().cpu().numpy().ravel())
        nb += 1
    model.train()
    out = {k: v / max(nb, 1) for k, v in total.items()}
    out.update({k: v / max(nb, 1) for k, v in extra.items()})
    out.update({k: [x / max(nb, 1) for x in v] for k, v in per_round.items()})
    if orec_tot:
        oc = [orec_hits[r] / orec_tot for r in sorted(orec_hits)]
        out["oracle_recall_per_round"] = oc
        out["oracle_recall_final"] = oc[-1]
        out["oracle_recall_best_round"] = int(np.argmax(oc)) + 1
        out["oracle_recall_max"] = max(oc)
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

    ds = build_dataset(cfg, "train", cfg["data"]["flip_prob"],
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

    # Measured on an A30 with the cache enabled: the DataLoader, not the GPU, is the
    # bottleneck. At 4 workers the step waited 740 ms (89 %) with std 1612 ms —
    # worker starvation. Raising workers to 8 took the whole step from 1012 to 91 ms.
    #   persistent_workers : 8 processes are rebuilt every epoch otherwise, and with
    #                        300 epochs that startup cost is paid 300 times.
    #   prefetch_factor    : each worker keeps 4 batches ready, absorbing the spikes
    #                        that showed up as max 5172 ms.
    #   pin_memory         : page-locked staging so the 12.6 MB/batch host->device
    #                        copy can overlap compute.
    dl_kw = dict(collate_fn=collate, num_workers=cfg["data"]["num_workers"],
                 pin_memory=(dev.type == "cuda"))
    if cfg["data"]["num_workers"] > 0:
        dl_kw.update(persistent_workers=True, prefetch_factor=4)

    loader = DataLoader(TorchWrap(ds, cache_tr), batch_size=cfg["training"]["batch_size"],
                        shuffle=True, drop_last=False, **dl_kw)

    # val: NO flipping (no augmentation at evaluation time)
    ds_val = build_dataset(cfg, "val")     # flip_prob 0.0: never augment at eval
    if a.limit:
        ds_val.items = ds_val.items[: max(a.limit // 2, 1)]
    val_loader = DataLoader(TorchWrap(ds_val, cache_va), batch_size=cfg["training"]["batch_size"],
                            shuffle=False, **dl_kw)
    print(f"[val ] {ds_val.stats()}", flush=True)

    model = build_model(cfg, dropout=None).to(dev)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] trainable parameters: {sum(p.numel() for p in trainable)/1e6:.2f}M "
          f"/ total {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    roi = model.decoder.roi
    if roi is not None:
        # The SAME sampler serves B and E1; only where its output goes differs, so
        # the banner must say which one is running or a log reader will mislabel
        # the run. E1 = score head consumes it; B = decoder input consumes it.
        which = ("EXPERIMENT E1 (score head reads the PREDICTED box)"
                 if model.decoder.score_roi
                 else "EXPERIMENT B (RoI added to the decoder input)")
        print(f"[roi  ] {which}: {roi.k}x{roi.k} grid inside each box, "
              f"{sum(p.numel() for p in roi.parameters())/1e6:.3f}M params, "
              f"zero-init (branch_norm {roi.branch_norm():.4f} -> "
              f"{'boxes == A exactly' if model.decoder.score_roi else 'B == A at step 0'})",
              flush=True)
    else:
        print("[roi  ] no RoI branch (experiment A behaviour)", flush=True)

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
        tr_iou_round = []                    # C1: mean IoU of each refinement round

        for batch in loader:
            t_b = time.time()
            tg = [b.to(dev) for b in batch["boxes"]]
            tl = [l.to(dev) for l in batch["labels"]]

            x_t, tt, _ = model.build_inputs(tg, N, batch["valid_h"])
            out = model(x_t, tt, **model_inputs(batch, dev))
            # C1 returns a list of rounds; criterion dispatches on that.
            if isinstance(out, list):
                loss, st, idx = crit(out, tg, labels=tl)
                lg = out[-1][1]                   # the round eval.py will use
            else:
                pb, lg = out
                loss, st, idx = crit(pb, lg, tg, labels=tl)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            for k in total:
                total[k] += st[k]
            # EXPERIMENT C1: the per-round IoU curve is THE metric — a flat curve
            # means iterating buys nothing, and that has to be visible before C2 is
            # built on top of it.
            if "iou_matched_per_round" in st:
                if not tr_iou_round:
                    tr_iou_round.extend([0.0] * len(st["iou_matched_per_round"]))
                for r, v in enumerate(st["iou_matched_per_round"]):
                    tr_iou_round[r] += v
            nb += 1
            grad_norms.append(float(gn))
            n_gt += [len(b) for b in batch["boxes"]]
            t_batch.append(time.time() - t_b)
            scores.append((lg if lg.dim() == 2 else lg.max(-1).values)
                          .detach().sigmoid().cpu().numpy().ravel())

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
        # n_matched is summed over the whole batch (so is the loss denominator —
        # that part is correct and matches DiffusionDet). For the log it has to be
        # divided by batch size, otherwise it reads as "258/100 (258 %)", which is
        # impossible and looks like a bug.
        bs = cfg["training"]["batch_size"]
        stability = label_stability(prev_labels, labels_now)
        prev_labels = labels_now
        score_stats = array_stats(np.concatenate(scores)) if scores else {}
        std_score = score_stats.get("std", 0.0)
        train_sec = time.time() - t0

        val = run_val(model, val_loader, crit, N, dev)
        # The loss of the round inference actually uses. For A/B/A.1/A.2 there is
        # only one round, so this IS `val["loss"]`; for C1 it is the last of six,
        # because `ddim_sample` returns `outs[-1]`. Everything downstream --
        # console log, history, checkpoint selection, val_rising_streak -- reads
        # THIS, so all four agree on one quantity.
        val_sel = val.get("loss_final", val["loss"])
        # SELECTION CRITERION. Default stays `loss_final` so every earlier run
        # (A/B/A.1/A.2/C1) keeps meaning exactly what it meant.
        #
        # `oracle_recall` is offered because on C1 the two disagree: `loss_final`
        # improved monotonically for 300 epochs while coverage of GT went DOWN over
        # the six rounds. Loss is computed only over Hungarian-matched pairs, so it
        # is blind to GT that no box reaches at all -- which is the failure mode
        # here. Selecting on coverage optimises the quantity C2 is supposed to move.
        #
        # NOTE it is a GAIN, not a loss: the comparison direction flips.
        sel_metric = cfg["training"].get("select_metric", "loss_final")
        if sel_metric == "oracle_recall":
            if "oracle_recall_final" not in val:
                raise ValueError("select_metric=oracle_recall but run_val produced no "
                                 "oracle_recall_final (no GT in the val split?)")
            val_sel = -val["oracle_recall_final"]   # negate -> lower is still better
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
              f"matched {tr['n_matched']/bs:.1f}/{N} per img "
              f"({100*tr['n_matched']/(bs*N):.0f}% of slots) | "
              f"GT/img {np.mean(n_gt):.1f} | label_stability {stability:.3f} | "
              f"lr {opt.param_groups[0]['lr']:.2e} | grad {np.mean(grad_norms):.3f}",
              flush=True)
        if tr_iou_round:
            pr = [v / max(nb, 1) for v in tr_iou_round]
            vr = val.get("iou_matched_per_round")
            print(f"           IoU/round tr " + " ".join(f"{v:.3f}" for v in pr) +
                  (("  va " + " ".join(f"{v:.3f}" for v in vr)) if vr else ""),
                  flush=True)
            # The VAL curve is the one that matters: it is what the checkpoint is
            # chosen on, and C1's whole question ("does iterating help?") has to be
            # answered on data the model did not train on.
            d_tr = pr[-1] - pr[0]
            d_va = (vr[-1] - vr[0]) if vr else d_tr
            print(f"           round1->{len(pr)}  train {d_tr:+.3f}  val {d_va:+.3f}"
                  f" | val loss mean {val['loss']:.4f} final {val_sel:.4f}"
                  f"{'   [!] FLAT on val: iterating is not helping' if abs(d_va) < 0.005 else ''}"
                  f"{'   [!] DECREASING on val: later rounds undo earlier ones' if d_va < -0.005 else ''}",
                  flush=True)
        # COVERAGE per round -- the metric `loss_final` is blind to. On C1 this line
        # would have shown coverage FALLING across the six rounds for 300 epochs
        # while every loss number rose.
        oc = val.get("oracle_recall_per_round")
        if oc:
            print("           oracle_recall/round " + " ".join(f"{v:.4f}" for v in oc) +
                  f" | best round {int(np.argmax(oc)) + 1}/{len(oc)}" +
                  ("   [!] LAST round is not the best -- inference uses the last one"
                   if int(np.argmax(oc)) != len(oc) - 1 else ""),
                  flush=True)
        roi_norm = roi.branch_norm() if roi is not None else None
        if roi_norm is not None:
            print(f"           roi_branch_norm {roi_norm:.4f}"
                  f"{'  (still ~0: the net is not using RoI features)' if roi_norm < 1e-3 else ''}",
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
        # The 0.40 threshold came from round 1, whose architecture kept `t` far more
        # stable. Here `t ~ U(0, num_timesteps)` is redrawn every batch, so slot i in
        # epoch n and epoch n+1 hold genuinely different boxes and there is no reason
        # for them to match the same GT. Simulated with a PERFECT (oracle) model:
        #   same t        -> 100 %      t=50 vs 60   -> 50 %
        #   t=50 vs 300   ->  20 %      t=50 vs 900  ->  6 %
        # With t uniform, even a flawless model lands near 1/mean_GT (~2.7 %). So a
        # low value here measures the timestep schedule, not model quality — warn
        # only if it is far below even that floor, which would mean the matching is
        # actively anti-correlated.
        if not np.isnan(stability) and stability < 0.005:
            warnings.append(f"label_stability {stability:.3f} below the random floor "
                            f"(~1/GT_per_image) — matching may be broken")
        if np.mean(grad_norms) > 100:
            warnings.append(f"grad norm {np.mean(grad_norms):.1f} is very large")
        # A zero-initialised branch that never grows is the network saying the RoI
        # features are useless. Cheap early read: stop at ~30 epochs instead of 300.
        if roi_norm is not None and ep >= 10 and roi_norm < 1e-3:
            warnings.append(f"roi_branch_norm {roi_norm:.5f} still ~0 after {ep+1} "
                            f"epochs — the RoI branch is not being used")
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
            "roi_branch_norm": roi_norm,
            # BOTH curves: train tells whether the mechanism works, val whether it
            # generalises -- and val is what the checkpoint is chosen on.
            "iou_per_round": [v / max(nb, 1) for v in tr_iou_round] if tr_iou_round else None,
            "val_iou_per_round": val.get("iou_matched_per_round"),
            "val_loss_per_round": val.get("loss_per_round"),
            "val_loss_final": val.get("loss_final"),
            "val_iou_final": val.get("iou_matched_final"),
            "select_loss": val_sel,
            "select_metric": sel_metric,
            "oracle_recall_per_round": val.get("oracle_recall_per_round"),
            "oracle_recall_final": val.get("oracle_recall_final"),
            "lr": opt.param_groups[0]["lr"],
            "epoch_sec": epoch_sec,
            "elapsed_sec": elapsed,
            "eta_sec": eta,
            "warnings": warnings,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        write_json(save_dir, env, cfg, history, best, ds, ds_val)

        # Pick best by the VAL loss OF THE ROUND INFERENCE ACTUALLY USES.
        #
        # For C1 `val["loss"]` is the mean over six refinement rounds, but
        # `ddim_sample` returns only the last one, and round 1 regresses straight
        # from the noisy anchor so it is always poor. Measured: mean 6.22 vs final
        # 3.95 (1.58x apart). Selecting on the mean can therefore save the WRONG
        # checkpoint -- a model with rounds [9.5,7.0,6.4,5.6,4.8,2.0] (final 2.0)
        # loses on the mean to one with [4.0,...,3.5] (final 3.5), even though the
        # first is strictly better at inference.
        #
        # It also makes "C1 val_loss 2.9 vs A's 2.51" a comparison of two different
        # quantities. `select_loss` is the like-for-like one.
        if val_sel < best:
            best = val_sel
            # Save ONLY the TRAINABLE parameters (~8.3M). Saving frozen CLIP too
            # made the checkpoint 698 MB, of which 98 % is weights re-downloadable
            # from HuggingFace, and it would force an exact CLIP version match at
            # load time.
            trainable_sd = {k: v for k, v in model.state_dict().items()
                            if not (k.startswith("encoder.vision.") or k.startswith("encoder.text."))}
            torch.save({"epoch": ep, "model": trainable_sd, "optimizer": opt.state_dict(),
                        "loss": best, "cfg": cfg, "trainable_only": True},
                       os.path.join(save_dir, "best.pth"))
            print(f"  -> saved best ({sel_metric} "
                  f"{-best if sel_metric == 'oracle_recall' else best:.4f})", flush=True)

    total_time = time.time() - t_start
    print("=" * 78, flush=True)
    print(f"DONE — EXPERIMENT {exp_name} — {fmt_time(total_time)} ({len(history)} epochs, "
          f"{fmt_time(total_time/max(len(history),1))}/epoch)", flush=True)
    if history:
        # SAME rule as write_json and the checkpoint block: `select_loss` is the
        # round inference actually uses. These three lines are the only thing most
        # people read when a job ends, so if they used the six-round MEAN they would
        # name a different epoch than the one `best.pth` holds -- and the "not
        # saturated" warning below, computed from `top`, would fire on the wrong
        # epoch too. That warning matters: round 1 of this project hit it four times
        # in a row.
        sel_of = lambda e: e.get("select_loss", e["val"]["loss"])      # noqa: E731
        iou_of = lambda e: e["val"].get("iou_matched_final",           # noqa: E731
                                        e["val"]["iou_matched"])
        top = min(history, key=sel_of)
        print(f"  best: epoch {top['epoch']} | val_loss {sel_of(top):.4f} | "
              f"val_IoU {iou_of(top):.4f}", flush=True)
        print(f"  val_loss: {sel_of(history[0]):.4f} -> "
              f"{sel_of(history[-1]):.4f}", flush=True)
        if history[-1].get("val_loss_final") is not None:
            print(f"  (val_loss is the LAST refinement round, the one ddim_sample "
                  f"returns; the 6-round mean was {top['val']['loss']:.4f})",
                  flush=True)
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

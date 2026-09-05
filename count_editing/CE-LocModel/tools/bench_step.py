#!/usr/bin/env python3
"""Break one training step into stages and time each — on the GPU, with real data.

Why this exists: the first server run measured 615 ms/batch WITHOUT the cache and
1799 ms/batch WITH it. The cache is supposed to remove the CLIP forward, which
profiling said was 76.8 % of the step — so enabling it should have been ~4x
FASTER, not 2.9x slower. Something is wrong in a way that guessing will not find,
and every hypothesis checked locally (I/O volume, matcher cost, IPC size) fails to
explain a 95x gap between the ~19 ms of theoretical GPU work and what was observed.

So: measure, do not guess. This times each stage separately, with
torch.cuda.synchronize() around each (without it, CUDA is asynchronous and every
measurement is meaningless — a plausible reason the original profiling was wrong).

Run BOTH ways and compare:
  python3 tools/run_on_free_gpu.py -- tools/bench_step.py
  python3 tools/run_on_free_gpu.py -- tools/bench_step.py --cache ../../data/cache_clip
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection, PatchCache  # noqa: E402
from models.criterion import SetCriterion  # noqa: E402
from models.detector import CELocDetector  # noqa: E402
from train import TorchWrap, collate, model_inputs  # noqa: E402


class Timer:
    """Wall-clock per stage, with CUDA sync so the numbers mean something."""

    def __init__(self, dev):
        self.dev, self.acc = dev, {}

    def __call__(self, name):
        self.name = name
        return self

    def __enter__(self):
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        self.acc.setdefault(self.name, []).append(time.perf_counter() - self.t0)
        return False

    def report(self, n_skip=2):
        rows = []
        for k, v in self.acc.items():
            v = v[n_skip:] or v
            rows.append((k, 1000 * np.mean(v), 1000 * np.std(v), 1000 * np.max(v)))
        total = sum(r[1] for r in rows if r[0] != "TOTAL")
        print(f"\n{'stage':28s} {'mean ms':>9s} {'std':>7s} {'max':>8s}  {'% of step':>9s}")
        print("-" * 70)
        for k, m, s, mx in rows:
            pct = "" if k == "TOTAL" else f"{100*m/max(total,1e-9):8.1f}%"
            print(f"{k:28s} {m:9.1f} {s:7.1f} {mx:8.1f}  {pct:>9s}")
        print("-" * 70)
        print(f"{'sum of stages':28s} {total:9.1f}")
        return {k: 1000 * np.mean(v[n_skip:] or v) for k, v in self.acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    B = a.batch_size or cfg["training"]["batch_size"]
    nw = cfg["data"]["num_workers"] if a.num_workers is None else a.num_workers
    N = cfg["diffusion"]["num_proposals_train"]

    print("=" * 70)
    print(f"  BENCH — device={dev} batch={B} workers={nw} "
          f"cache={'yes' if a.cache else 'no'}")
    if dev.type == "cuda":
        print(f"  gpu={torch.cuda.get_device_name(0)}  "
              f"torch={torch.__version__}  threads={torch.get_num_threads()}")
    print("=" * 70, flush=True)

    ds = CE130Detection(cfg["data"]["root"], "train", cfg["data"]["image_size"],
                        cfg["data"]["flip_prob"], seed=cfg["training"]["seed"])
    cache = PatchCache(a.cache, "train") if a.cache else None
    loader = DataLoader(TorchWrap(ds, cache), batch_size=B, shuffle=True,
                        num_workers=nw, collate_fn=collate, drop_last=True)

    model = CELocDetector(
        cfg["model"]["clip_name"], cfg["model"]["d_model"], cfg["model"]["n_layer"],
        cfg["model"]["n_head"], cfg["data"]["image_size"],
        cfg["diffusion"]["num_timesteps"], cfg["diffusion"]["snr_scale"],
        cfg["diffusion"]["sampling_steps"], cfg["model"]["dropout"],
        cfg["model"]["freeze_clip"]).to(dev)
    model.train()
    crit = SetCriterion(cfg["matcher"]["method"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)

    T = Timer(dev)
    it = iter(loader)
    t_prev = time.perf_counter()

    for step in range(a.steps):
        # Time spent WAITING for the DataLoader = the true data-pipeline cost as
        # seen by the training loop (workers prefetch, so this is what is exposed).
        with T("1 dataloader wait"):
            batch = next(it)

        with T("2 batch -> gpu"):
            tg = [b.to(dev, non_blocking=True) for b in batch["boxes"]]
            kw = model_inputs(batch, dev)

        with T("3 build_inputs (diffusion)"):
            x_t, tt, _ = model.build_inputs(tg, N, batch["valid_h"])

        with T("4 encoder (CLIP or proj)"):
            mem = model.encoder(kw.get("pixel_values"), kw.get("texts"),
                                kw.get("patch_raw"), kw.get("text_raw"))

        with T("5 decoder forward"):
            from utils.box_ops import decode_diffusion
            pb, lg = model.decoder(decode_diffusion(x_t, model.snr_scale), tt, mem)

        with T("6 criterion (matcher, CPU)"):
            loss, st, _ = crit(pb, lg, tg)

        with T("7 backward"):
            opt.zero_grad(set_to_none=True)
            loss.backward()

        with T("8 optimizer step"):
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

        now = time.perf_counter()
        if step >= 2:
            T.acc.setdefault("TOTAL", []).append(now - t_prev)
        t_prev = now

    res = T.report()

    print(f"\nreal wall-clock per step: {res.get('TOTAL', 0):.0f} ms")
    if dev.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ------------------------------------------------------------- diagnosis
    print("\n" + "=" * 70)
    print("READING THIS")
    print("=" * 70)
    dl = res.get("1 dataloader wait", 0)
    total = res.get("TOTAL", 1)
    enc = res.get("4 encoder (CLIP or proj)", 0)
    gpu = sum(res.get(k, 0) for k in
              ["5 decoder forward", "7 backward", "8 optimizer step"])
    cpu = res.get("6 criterion (matcher, CPU)", 0) + res.get("3 build_inputs (diffusion)", 0)

    if dl > 0.4 * total:
        print(f"* DATA-BOUND: {100*dl/total:.0f} % of the step is spent waiting for the")
        print("  DataLoader. Raise num_workers, or set persistent_workers=True /")
        print("  prefetch_factor higher. With a cache the workers do almost no work,")
        print("  so a high wait means IPC or disk, not compute.")
    if enc > 0.3 * total:
        print(f"* ENCODER-BOUND: {100*enc/total:.0f} % in the encoder. Without a cache this")
        print("  is the CLIP forward (expected). WITH a cache it should be ~0 — if it")
        print("  is not, the cache is not actually being used.")
    if cpu > 0.25 * total:
        print(f"* CPU-BOUND: {100*cpu/total:.0f} % in matcher/diffusion, which run on CPU")
        print("  per image while the GPU idles.")
    if gpu > 0.5 * total:
        print(f"* GPU-BOUND: {100*gpu/total:.0f} % in decoder+backward — the healthy case.")
        print("  Speed up by raising batch_size until memory is used.")
    print("\nCompare the two runs (with and without --cache): the encoder row should")
    print("collapse to near zero with the cache. If the TOTAL does not drop by a")
    print("similar amount, the extra time has moved somewhere else — and this table")
    print("says where.")
    print("=" * 70)


if __name__ == "__main__":
    main()

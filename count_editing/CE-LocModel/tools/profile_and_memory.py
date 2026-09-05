#!/usr/bin/env python3
"""Measure real GPU memory + the % of time spent in the CLIP forward.
RUN THIS BEFORE ANY LONG TRAINING.

Two questions that must be answered with NUMBERS, not guesses:

1. MEMORY. Hand calculation gives ~0.8 GB decoder + ~1 GB CLIP (sdpa) on a 24GB
   A30. If the measurement exceeds 5 GB, a `no_grad` is missing somewhere —
   backward through the ViT costs ~19 GB.

2. IS A CACHE NEEDED. Lessons §6 measured the detect branch spending only
   0.1-0.2 % in the DataLoader -> "no cache needed". But that was with ResNet18;
   CLIP ViT-B/16 @512 is far more expensive (27x the attention cost of 224px), so
   the balance MAY have flipped. Only build the cache if CLIP > 30 % of the time —
   do not build something before knowing it is needed.

  python3 tools/profile_and_memory.py --batch-size 8 --steps 10
"""

import argparse
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection, normalize_for_clip  # noqa: E402
from models.detector import CELocDetector  # noqa: E402
from models.criterion import SetCriterion  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--num-proposals", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    N = a.num_proposals or cfg["diffusion"]["num_proposals_train"]
    print(f"[cfg] device={dev} batch={a.batch_size} N={N}", flush=True)

    ds = CE130Detection(cfg["data"]["root"], "train", cfg["data"]["image_size"])
    model = CELocDetector(
        cfg["model"]["clip_name"], cfg["model"]["d_model"], cfg["model"]["n_layer"],
        cfg["model"]["n_head"], cfg["data"]["image_size"],
        cfg["diffusion"]["num_timesteps"], cfg["diffusion"]["snr_scale"],
        cfg["diffusion"]["sampling_steps"], cfg["model"]["dropout"],
        cfg["model"]["freeze_clip"],
    ).to(dev)
    model.train()
    crit = SetCriterion(cfg["matcher"]["method"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)

    print(f"[model] trainable {sum(p.numel() for p in trainable)/1e6:.2f}M "
          f"/ total {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    samples = [ds[i] for i in range(a.batch_size)]
    px = torch.stack([torch.from_numpy(normalize_for_clip(m["image"])) for m in samples]).to(dev)
    tg = [torch.from_numpy(m["boxes"]).float().to(dev) for m in samples]
    txt = [m["text"] for m in samples]
    vh = [m["valid_h"] for m in samples]

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_clip = t_total = 0.0
    for i in range(a.steps):
        t0 = time.time()

        t1 = time.time()
        with torch.no_grad():
            patch_raw = model.encoder.encode_image_raw(px)
            text_raw = model.encoder.encode_text_raw(txt, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt_clip = time.time() - t1

        x_t, tt, _ = model.build_inputs(tg, N, vh)
        pb, lg = model(x_t, tt, patch_raw=patch_raw, text_raw=text_raw)
        loss, st, _ = crit(pb, lg, tg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()

        if i >= 2:                      # skip 2 warm-up steps
            t_clip += dt_clip
            t_total += time.time() - t0

    n = max(a.steps - 2, 1)
    ratio = 100 * t_clip / max(t_total, 1e-9)
    print(f"\n=== TIME (mean over {n} steps) ===")
    print(f"  CLIP forward : {1000*t_clip/n:7.1f} ms  ({ratio:.1f} %)")
    print(f"  everything else: {1000*(t_total-t_clip)/n:7.1f} ms")
    print(f"  total / step : {1000*t_total/n:7.1f} ms")
    print(f"\n  -> {'BUILD the cache' if ratio > 30 else 'NO cache needed'} "
          f"(threshold 30 %)")

    if dev.type == "cuda":
        gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n=== MEMORY ===\n  peak: {gb:.2f} GB")
        print("  " + ("[!] > 5 GB — a no_grad may be missing somewhere" if gb > 5
                      else "[ok] matches the estimate (~0.8 GB decoder + ~1 GB CLIP)"))
    else:
        print("\n(CPU — GPU memory cannot be measured here; re-run on the server)")


if __name__ == "__main__":
    main()

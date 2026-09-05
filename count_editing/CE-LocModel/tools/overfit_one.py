#!/usr/bin/env python3
"""Overfit ONE image — the gate before any long training run.

If the loss does not approach zero, the matcher or the loss is broken: STOP AND
FIX, do not keep training. Round 1 trained 5 times in a row on broken code purely
because this step was missing.

Use a small fixed t to isolate the question: at large t the input boxes are almost
pure noise, so overfitting is impossible, and that is NOT a bug.

  python3 tools/overfit_one.py --steps 300 --device cpu
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection, normalize_for_clip  # noqa: E402
from models.detector import CELocDetector  # noqa: E402
from models.criterion import SetCriterion  # noqa: E402
from utils.diffusion_math import prepare_diffusion_concat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--t", type=int, default=50, help="fixed (small) timestep")
    ap.add_argument("--num-proposals", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)

    ds = CE130Detection(cfg["data"]["root"], "train", cfg["data"]["image_size"])
    m = ds[a.index]
    px = torch.from_numpy(normalize_for_clip(m["image"])).unsqueeze(0).to(dev)
    gt = torch.from_numpy(m["boxes"]).float().to(dev)
    print(f"[image] {m['image_id']} '{m['text']}' | {len(gt)} GT | N={a.num_proposals} "
          f"| t={a.t}", flush=True)

    model = CELocDetector(
        cfg["model"]["clip_name"], cfg["model"]["d_model"], cfg["model"]["n_layer"],
        cfg["model"]["n_head"], cfg["data"]["image_size"],
        cfg["diffusion"]["num_timesteps"], cfg["diffusion"]["snr_scale"],
        cfg["diffusion"]["sampling_steps"], 0.0, cfg["model"]["freeze_clip"],
    ).to(dev)
    model.train()
    crit = SetCriterion(cfg["matcher"]["method"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr)

    with torch.no_grad():                          # CLIP frozen -> encode once
        patch_raw = model.encoder.encode_image_raw(px)
        text_raw = model.encoder.encode_text_raw([m["text"]], dev)

    g = torch.Generator(device="cpu").manual_seed(0)
    x_t, _, _ = prepare_diffusion_concat(gt.cpu(), a.num_proposals, a.t,
                                         model.alphas_cumprod.cpu(),
                                         cfg["diffusion"]["snr_scale"],
                                         valid_h=m["valid_h"], generator=g)
    x_t = x_t.unsqueeze(0).to(dev)
    tt = torch.full((1,), a.t, dtype=torch.long, device=dev)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    first_loss, history = None, []
    for i in range(a.steps):
        pb, lg = model(x_t, tt, patch_raw=patch_raw, text_raw=text_raw)
        loss, st, _ = crit(pb, lg, [gt])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        history.append((st["loss"], st["iou_matched"]))
        if first_loss is None:
            first_loss = st["loss"]
        if i % max(a.steps // 10, 1) == 0 or i == a.steps - 1:
            print(f"  [{i:4d}] loss {st['loss']:7.4f} | l1 {st['loss_l1']:.4f} "
                  f"giou {st['loss_giou']:.4f} ce {st['loss_ce']:.4f} | "
                  f"IoU {st['iou_matched']:.4f}", flush=True)

    # Judge by the BEST IoU and the mean of the last 10 steps, NOT by the single
    # final step: with a fixed LR the final step is just a random sample of the
    # oscillation (seen in practice: IoU reached 0.69 at step 210 then swung back
    # to 0.47 at step 299).
    best_iou = max(x[1] for x in history)
    iou_last10 = float(np.mean([x[1] for x in history[-10:]]))
    loss_last10 = float(np.mean([x[0] for x in history[-10:]]))
    ratio = loss_last10 / max(first_loss, 1e-9)

    print(f"\nloss {first_loss:.4f} -> {loss_last10:.4f} "
          f"({100*ratio:.1f} % remaining, mean of last 10 steps)")
    print(f"IoU_matched: best {best_iou:.4f} | last 10 steps {iou_last10:.4f}")

    if best_iou > 0.6 and ratio < 0.35:
        print("[PASS] matcher and loss work correctly, safe to continue training")
        if iou_last10 < best_iou - 0.1:
            print("  (IoU oscillates at the end — normal when overfitting 1 image, "
                  "not a bug)")
    elif ratio < 0.5:
        print("[PARTIAL] loss clearly drops but IoU is not high. Check before long training:")
        print("  run tools/visualize_data.py to see whether GT boxes bound the objects.")
    else:
        print("[FAIL] the matcher or the loss is broken. STOP, do not train long.")


if __name__ == "__main__":
    main()

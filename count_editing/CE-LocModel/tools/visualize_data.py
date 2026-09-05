#!/usr/bin/env python3
"""Draw boxes through the EXACT 6-step chain onto the padded canvas. PIL + numpy only.

Run this BEFORE writing the model. In round 1, visualisation caught 3 bugs that
the entire test suite and 3 code review passes had missed — looking at images is
the only check that catches errors about the NATURE of the data (the loss is only
computed on matched pairs, so it never sees the 62 % placeholders).

  python3 tools/visualize_data.py --n 20 --out <directory>
  python3 tools/visualize_data.py --split test --placeholder --n 8
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection  # noqa: E402
from utils.box_ops_np import cxcywh_to_xyxy, decode_diffusion, encode_diffusion  # noqa: E402
from utils.diffusion_np import cosine_alphas_cumprod, prepare_diffusion_concat  # noqa: E402

GREEN, RED, YELLOW, ORANGE = (0, 255, 0), (255, 60, 60), (255, 255, 0), (255, 150, 0)


def draw(sample, snr_scale=2.0, draw_placeholder=False, num_proposals=100, t=None, rng=None):
    """Return the drawn PIL image. Boxes go through encode -> decode to test the chain."""
    img = Image.fromarray(sample["image"])
    dr = ImageDraw.Draw(img)
    T = img.size[0]

    if draw_placeholder:
        ab = cosine_alphas_cumprod(1000)
        tt = int(rng.integers(0, 1000)) if t is None else t
        x_t, _, is_gt = prepare_diffusion_concat(
            sample["boxes"], num_proposals, tt, ab, snr_scale,
            valid_h=sample["valid_h"], rng=rng,
        )
        for b, real in zip(cxcywh_to_xyxy(decode_diffusion(x_t, snr_scale)) * T, is_gt):
            dr.rectangle(b.tolist(), outline=GREEN if real else ORANGE, width=2)
    else:
        # GT through the whole chain: cxcywh[0,1] -> encode -> decode -> xyxy px
        back = decode_diffusion(encode_diffusion(sample["boxes"], snr_scale), snr_scale)
        for b in cxcywh_to_xyxy(back) * T:
            dr.rectangle(b.tolist(), outline=GREEN, width=2)

    nh = int(round(sample["valid_h"] * T))
    if nh < T - 1:                                   # padding boundary
        dr.line([(0, nh), (T, nh)], fill=YELLOW, width=3)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../../data/all_phase2_V2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=20, help="number of images; -1 = all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--placeholder", action="store_true", help="also draw fake boxes")
    ap.add_argument("--num-proposals", type=int, default=100)
    ap.add_argument("--t", type=int, default=None, help="fixed timestep")
    ap.add_argument("--snr-scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    ds = CE130Detection(a.root, a.split)
    print(f"{a.split}: {ds.stats()}")

    idx = range(len(ds)) if a.n < 0 else np.linspace(0, len(ds) - 1, min(a.n, len(ds))).astype(int)
    rows = []
    for i in idx:
        m = ds[int(i)]
        img = draw(m, a.snr_scale, a.placeholder, a.num_proposals, a.t, rng)
        name = f"{a.split}_{m['image_id']}_{m['text'].replace(' ', '-')}.png"
        img.save(os.path.join(a.out, name))
        rows.append(f"{name},{len(m['boxes'])},{m['valid_h']:.4f},{m['orig_size'][0]}x{m['orig_size'][1]}")

    with open(os.path.join(a.out, "index.csv"), "w") as f:
        f.write("file,n_boxes,valid_h,original_size\n" + "\n".join(rows) + "\n")
    print(f"saved {len(rows)} images to {a.out}")


if __name__ == "__main__":
    main()

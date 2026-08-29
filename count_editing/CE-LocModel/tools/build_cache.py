"""Cache the resize+pad result of every sample into one flat uint8 memmap.

Profiling on the server showed the training loop is 92.6% DataLoader (28.3 of
30.6 min/epoch), and a local split showed PNG decode is 83% of that per-sample
cost. Decoding the same 20k PNGs identically on every one of 200 epochs is pure
waste: resize+pad is deterministic, so it can be done once.

Layout (per split): cache_{size}.u8    -- N * (3+1) * S * S bytes, C-order
                    cache_{size}.json  -- file order + per-sample scale + meta
Read back with np.memmap; a batch is then a memcpy, no decode, no PIL.

Run once per split, then train with `--use_cache`.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ObjectPlacementDataset


def build(root, size, out_dir=None):
    out_dir = out_dir or root
    os.makedirs(out_dir, exist_ok=True)
    ds = ObjectPlacementDataset(root, target_size=(size, size))
    n = len(ds.files)
    bin_path = os.path.join(out_dir, f"cache_{size}.u8")
    meta_path = os.path.join(out_dir, f"cache_{size}.json")

    per = 4 * size * size
    print(f"{root}: {n} mẫu -> {bin_path} ({n * per / 1e9:.1f} GB)", flush=True)

    mm = np.memmap(bin_path, dtype=np.uint8, mode="w+", shape=(n, 4, size, size))
    scales, sizes = [], []
    t0 = time.monotonic()

    for i, fname in enumerate(ds.files):
        img = Image.open(os.path.join(ds.image_dir, fname)).convert("RGB")
        dname = os.path.splitext(fname)[0] + ".png"
        den = Image.open(os.path.join(ds.density_dir, dname)).convert("L")
        orig_size = img.size
        pimg, pden, scale = ds.resize_and_pad(img, den)

        mm[i, :3] = np.asarray(pimg, dtype=np.uint8).transpose(2, 0, 1)
        mm[i, 3] = np.asarray(pden, dtype=np.uint8)
        scales.append(float(scale))
        sizes.append([int(orig_size[0]), int(orig_size[1])])

        if (i + 1) % 2000 == 0 or i + 1 == n:
            el = time.monotonic() - t0
            eta = el / (i + 1) * (n - i - 1)
            print(f"  {i+1}/{n}  elapsed {el/60:.1f}m  ETA {eta/60:.1f}m", flush=True)

    mm.flush()
    del mm
    with open(meta_path, "w") as f:
        json.dump({"files": ds.files, "scales": scales, "sizes": sizes,
                   "size": size, "n": n}, f)
    print(f"  xong trong {(time.monotonic()-t0)/60:.1f} phút -> {meta_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True,
                   help="các split cần cache, vd ../../data/samples/train ../../data/samples/test")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--out_dir", default=None, help="mặc định: ghi ngay trong thư mục split")
    a = p.parse_args()
    for r in a.roots:
        build(r, a.size, a.out_dir)

#!/usr/bin/env python3
"""Cache uint8 cho bài ADD (GAMMA2, Kaggle): ảnh inpaint + density của mẫu đã letterbox kiểu bài, [T,T,4] mỗi mẫu, trong MỘT
file — `zlib` (mặc định, `data.zlib`, ~0,41 MiB / mẫu: train samples/ ≈ 8 GB, vừa output Kaggle nên lưu lại giữa các phiên) hoặc
`raw` (memmap `data.u8`, ~1 MiB / mẫu) — + `meta.json` — `CE130AddDataset(cache=AddCache(...))` đọc thẳng, `/255` ra đúng từng bit đầu vào kiểu bài
(có test). Bỏ nghẽn giải mã PNG (Kaggle 4 vCPU: lần train lại CE-Loc trước 8,8 -> 2,9 phút / epoch).
~15–30 phút với 4 worker:
  python tools/build_add_cache.py --turn-index ../data/turn_index.json --samples ../data/samples --split-source samples \\
      --splits train --out /kaggle/temp/add_cache
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.turns import SPLIT_SOURCES, TurnIndex, build_add_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-index", default="../data/turn_index.json")
    ap.add_argument("--samples", default="../data/samples")
    ap.add_argument("--split-source", default="samples", choices=SPLIT_SOURCES)
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--format", default="zlib", choices=["zlib", "raw"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    index = TurnIndex(a.turn_index)
    keys = sorted({k for s in a.splits for k in index.keys(s, a.split_source)})
    raw_gb = len(keys) * a.image_size ** 2 * 4 / 2 ** 30
    print(f"[cache] {len(keys)} mẫu ({a.split_source}: {a.splits}) -> {a.out}, {a.format} "
          f"(~{raw_gb * (0.41 if a.format == 'zlib' else 1):.1f} GB)", flush=True)
    n = build_add_cache(index, a.samples, keys, a.out, a.image_size, a.workers, log=lambda m: print(m, flush=True),
                        fmt=a.format)
    print(f"[cache] xong {n} mẫu | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

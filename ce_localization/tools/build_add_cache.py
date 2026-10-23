#!/usr/bin/env python3
"""Cache uint8 ảnh train (Kaggle): ảnh inpaint + density của mẫu đã letterbox kiểu bài, [T,T,4] mỗi mẫu trong MỘT file — `zlib` (~0,41 MiB /
mẫu, train ≈ 8 GB, vừa output Kaggle nên giữ được giữa các phiên) hoặc `raw` (memmap, ~1 MiB / mẫu, đọc không giải nén) + `meta.json`.
`/255` ra đúng từng bit đầu vào kiểu bài (có test).
  python tools/build_add_cache.py --turn-index ../data/turn_index.json --samples ../data/samples --splits train --out /kaggle/temp/add_cache
  python tools/build_add_cache.py --from-cache /kaggle/working/add_cache --out /kaggle/temp/add_cache_raw     # zlib -> raw
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.turns import TurnIndex, build_add_cache, convert_add_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-index", default="../data/turn_index.json")
    ap.add_argument("--samples", default="../data/samples")
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--format", default="zlib", choices=["zlib", "raw"])
    ap.add_argument("--from-cache", default=None, help="bung cache zlib có sẵn thành raw (bỏ qua dựng từ PNG)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    if a.from_cache:
        n = convert_add_cache(a.from_cache, a.out, a.workers, log=lambda m: print(m, flush=True))
        print(f"[cache] raw xong {n} mẫu -> {a.out} | {time.time() - t0:.0f}s", flush=True)
        return
    index = TurnIndex(a.turn_index)
    keys = sorted({k for s in a.splits for k in index.keys(s)})
    raw_gb = len(keys) * a.image_size ** 2 * 4 / 2 ** 30
    print(f"[cache] {len(keys)} mẫu ({a.splits}) -> {a.out}, {a.format} "
          f"(~{raw_gb * (0.41 if a.format == 'zlib' else 1):.1f} GB)", flush=True)
    n = build_add_cache(index, a.samples, keys, a.out, a.image_size, a.workers, log=lambda m: print(m, flush=True),
                        fmt=a.format)
    print(f"[cache] xong {n} mẫu | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

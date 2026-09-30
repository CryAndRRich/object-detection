#!/usr/bin/env python3
"""Dựng cache uint8 (memmap) cho `celoc_paper/train.py --cache-dir` — xem `celoc_data.build_cache`.

  python celoc_paper/build_cache.py --data ../data/samples/train --out <thư mục cache> [--no-density]
Cache CÓ density dùng được cho cả hai bản (bản không density chỉ đọc images.u8).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.celoc_paper.celoc_data import build_cache  # noqa: E402
from ce_localization.utils.log import fmt_time  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-density", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    build_cache(args.data, args.out, use_density=not args.no_density, workers=args.workers,
                log=lambda s: print(f"[{fmt_time(time.time() - t0)}] {s}", flush=True))


if __name__ == "__main__":
    main()

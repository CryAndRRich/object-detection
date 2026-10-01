#!/usr/bin/env python3
"""Cửa G0 của EXPERIMENT GAMMA (docs/EXPERIMENT_GAMMA.md mục 2.2): dựng `data/turn_index.json` MỘT lần.

Khớp mỗi (nhánh, lượt) của `all_phase2_V2` (`inpainted_turn_t.png`) với đúng một file `samples/*/images/` bằng
hash pixel, kiểm `target_bbox` (cxcywh) của sample = lỗ lượt t (`inpainted_bboxes[t-1]`, xyxy, lệch <= 1 px) và
class, gán lượt bị xoá cho từng vật `all_bboxes`. In báo cáo; KHÔNG một-một (26.120 / 26.120) hay lệch bất kỳ
thì KHÔNG ghi chỉ mục, thoát mã 1 (thêm `--allow-mismatch` để vẫn ghi, chỉ khi đã hiểu vì sao lệch).

Giải mã ~52k PNG (26k inpaint + 26k samples) trên đĩa dùng chung, 8 worker: ước 15–40 phút ⇒ chạy nền:
  cd /mnt/disk1/aiotlab/haitn/object-detection/ce_localization
  LOG=/mnt/disk1/aiotlab/haitn/log/gamma/turn_index_$(date +%m%d_%H%M).log
  mkdir -p $(dirname $LOG)
  nohup python tools/build_turn_index.py --out ../data/turn_index.json --workers 8 > $LOG 2>&1 &
  echo "PID $! -> $LOG"
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.turns import build_turn_index  # noqa: E402
from ce_localization.utils.log import fmt_time  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce130", default="../data/all_phase2_V2")
    ap.add_argument("--samples", default="../data/samples")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--allow-mismatch", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda *s: print(f"[{fmt_time(time.time() - t0)}]", *s, flush=True)  # noqa: E731
    idx = build_turn_index(a.ce130, a.samples, workers=a.workers, log=log)
    r = idx["report"]
    log(f"nhánh {r['n_branches']} | (nhánh, lượt) {r['n_turns']} | file samples {r['n_samples']} | ghép {r['n_matched']}")
    log(f"theo split: {r['per_split']}")
    log(f"lỗ không ghép được vật all_bboxes (IoU < 0,5): {r['n_holes_unassigned']}")
    for k in ("turn_count_mismatch", "match_issues", "unmatched_samples", "target_mismatch", "class_mismatch"):
        log(f"{k}: {len(r[k])}" + (f" — vd. {r[k][:5]}" if r[k] else ""))
    if not r["ok"] and not a.allow_mismatch:
        log("❌ KHÔNG một-một hoặc có lệch -> không ghi chỉ mục. Gửi log để soát.")
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(idx, f)
    log(f"{'✅' if r['ok'] else '⚠️  (--allow-mismatch)'} -> {a.out} ({os.path.getsize(a.out) / 2 ** 20:.1f} MB)")


if __name__ == "__main__":
    main()

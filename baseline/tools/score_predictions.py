#!/usr/bin/env python3
"""Chấm lại một file dump dự đoán (của `predict.py` / `gdino/predict.py`) bằng bộ chấm chung
`baseline/scoring.py` — cùng thước đo với ALPHA / BETA và docs/SCORE.md.

Chỉ đọc annotation + JSON, CPU. Quét annotation 1 split trên đĩa dùng chung mất vài phút ⇒ chạy nền:

  cd /mnt/disk1/aiotlab/haitn/object-detection/baseline
  LOG=/mnt/disk1/aiotlab/haitn/log/score_<tên>_$(date +%m%d_%H%M).log
  nohup python tools/score_predictions.py --pred /mnt/disk1/aiotlab/haitn/output/baselines/<dump>.json \\
      --budgets 200 all > $LOG 2>&1 &
  echo "PID $! -> $LOG"

Ghi `<dump>_metrics.json` cạnh dump (hoặc --out) và in hàng markdown dán vào docs/SCORE.md.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.scoring import budget_tag, load_gt, print_summary, read_dump, score_pred, score_row  # noqa: E402


def parse_budgets(values):
    return [None if str(v).lower() == "all" else int(v) for v in values]


def score_dump_file(pred_path, data_root, budgets, out=None, log=print, gt=None):
    """Chấm một dump với nhiều ngân sách box -> dict kết quả (đã ghi JSON). `gt`: bảng `load_gt` đã quét
    sẵn (tránh quét lại annotation khi chấm nhiều dump cùng split)."""
    meta, pred = read_dump(pred_path)
    t = time.time()
    gt = gt if gt is not None else load_gt(data_root, meta["split"])
    if meta.get("limit"):                          # dump chạy --limit: chấm đúng các ảnh đã dump
        gt = {k: v for k, v in gt.items() if k in pred}   # (không --limit: thiếu ảnh nào là lỗi)
    log(f"[score] {pred_path} | split {meta['split']} | {len(gt)} ảnh | quét GT {time.time() - t:.0f} s")
    out_all = {"meta": meta, "results": {}, "score_rows": {}}
    max_boxes = max((len(p["scores"]) for p in pred.values()), default=0)
    done = set()
    for b in budgets:
        eff = None if b is None or b >= max_boxes else b          # ngân sách >= số box dump == giữ hết
        if eff in done:
            continue
        done.add(eff)
        res = score_pred(pred, gt, budget=eff)
        tag = budget_tag(eff)
        print_summary(f"{meta.get('run', '?')} {tag}", res, log)
        row = score_row({**meta, "budget": eff}, res)
        log(f"  SCORE.md: {row}")
        out_all["results"][tag] = res
        out_all["score_rows"][tag] = row
    out = out or os.path.splitext(pred_path)[0] + "_metrics.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(out_all, f, indent=1, ensure_ascii=False, default=float)
    log(f"  -> {out}")
    return out_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="file dump .json")
    ap.add_argument("--data-root", default="../data/all_phase2_V2", help="thư mục all_phase2_V2/")
    ap.add_argument("--budgets", nargs="+", default=["200", "all"],
                    help="số box điểm cao nhất mỗi ảnh vào bộ chấm; 'all' = mọi box đã dump")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    score_dump_file(a.pred, a.data_root, parse_budgets(a.budgets), a.out)


if __name__ == "__main__":
    main()

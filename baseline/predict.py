#!/usr/bin/env python3
"""Chạy một baseline ZETA trên CE-130 test_all hoặc CE-CoCount review5: mỗi ảnh K box (mặc định 30) pixel ảnh gốc -> `<out>_raw.jsonl`,
rồi chấm bằng code chấm CE-Loc (`score.py`) -> `<out>.json` + `<out>_boxes.json`. Chỉ số chính: C-NLL box tốt nhất / ảnh (TB, trung vị,
bỏ 1 %). `--ckpt` = `best.pth` / `snapshots/epNNN.pth` của finetune, hay weight công bố (hàng zero-shot). OracleLoc (`zeta7.yaml`): không
`--ckpt`, chạy CPU, mỗi tiêu chí một bộ `<out>_<tiêu chí>.json` (+ `_raw.jsonl`, `_boxes.json`).

  cd /mnt/disk1/aiotlab/haitn/object-detection/baseline
  python ../tools/run_on_free_gpu.py -- predict.py --config config/zeta1.yaml --ckpt ../weights/add/zeta1/best.pth \\
      --dataset ce130 --out /mnt/disk1/aiotlab/haitn/output/add/zeta1/ce130_test_all/best.json
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from baseline.methods import NO_WEIGHTS, load_model, predict  # noqa: E402
from baseline.score import add_data_args, load_data, run_score  # noqa: E402
from baseline.scoring import read_boxes, write_jsonl  # noqa: E402
from ce_localization.utils.config import load_config  # noqa: E402
from ce_localization.utils.log import fmt_time  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="bắt buộc trừ method không weight (OracleLoc)")
    add_data_args(ap)
    ap.add_argument("--out", required=True, help="X.json; box thô -> X_raw.jsonl")
    ap.add_argument("--k", type=int, default=None, help="số box / ảnh (mặc định predict.k = 30)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    if a.k:
        cfg["predict"]["k"] = a.k
    no_w = cfg["method"] in NO_WEIGHTS
    if not no_w and not a.ckpt:
        ap.error(f"{cfg['method']}: cần --ckpt")
    dev = torch.device("cpu" if no_w else a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    samples, index, geo = load_data(a)
    print(f"[predict] {cfg['method']} | {a.ckpt} | {a.dataset}: {len(samples)} mẫu | K = {cfg['predict']['k']} | {dev}", flush=True)
    t = time.time()
    log = lambda m: print(m, flush=True)  # noqa: E731
    if no_w:
        runs = {f"_{c}": r for c, r in predict(cfg, None, samples, log=log, geo_fn=geo).items()}
    else:
        runs = {"": predict(cfg, load_model(cfg, a.ckpt, dev), samples, dev=dev, seed=a.seed)}
    sec = time.time() - t
    print(f"[predict] suy luận {fmt_time(sec)} ({sec / max(len(samples), 1) * 1000:.0f} ms/ảnh)", flush=True)
    stem = os.path.splitext(a.out)[0]
    for suf, rows in runs.items():
        raw, a.out = f"{stem}{suf}_raw.jsonl", f"{stem}{suf}.json"
        write_jsonl(raw, rows)
        print(f"[predict] -> {raw}", flush=True)
        run_score(a, samples, index, geo, read_boxes(raw),
                  {"method": cfg["method"], "config": a.config, "ckpt": a.ckpt, "predict": cfg["predict"], "raw": raw, "predict_sec": sec,
                   "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None, "date": time.strftime("%Y-%m-%d %H:%M"),
                   **({"criterion": suf[1:]} if suf else {})})
    print(f"[predict] xong {fmt_time(time.time() - t0)}", flush=True)


if __name__ == "__main__":
    main()

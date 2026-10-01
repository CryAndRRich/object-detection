#!/usr/bin/env python3
"""BASELINE3.1 / 3.2 — dump dự đoán Grounding DINO trên CE-130 (prompt = tên lớp CỦA TỪNG ẢNH) rồi chấm
bằng bộ chấm chung (`scoring.py`). 900 query / ảnh, không NMS nội bộ; bộ chấm lấy 200 box điểm cao nhất
(N = 200 của docs/SCORE.md) và mọi box ('all').

Nhiều `--weights` + `--select-out`: chạy lần lượt (thường trên val) rồi ghi checkpoint có `oracle_recall`
cao nhất (200 box) — cách chọn checkpoint của BASELINE3.2 (cạm bẫy 3: chọn bằng chỉ số matcher không thấy).

Từ object-detection/baseline/, `export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache` (BERT), ~10–30 phút ⇒ nền:

  LOG=/mnt/disk1/aiotlab/haitn/log/baselines/baseline3_1_predict_$(date +%m%d_%H%M).log
  nohup python ../tools/run_on_free_gpu.py -- gdino/predict.py --config configs/baseline3_1_gdino_zeroshot.yaml \\
      --split test --out-dir /mnt/disk1/aiotlab/haitn/output/baselines --where "zero-shot" > $LOG 2>&1 &
  echo "PID $! -> $LOG"
"""

import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image  # noqa: E402

from baseline.gdino.runtime import (build_gdino, load_config, load_weights, make_transform,  # noqa: E402
                                    phrase_positive_map, predict_image, prompt_of, resolve)
from baseline.scoring import load_gt, write_dump  # noqa: E402
from baseline.tools.score_predictions import parse_budgets, score_dump_file  # noqa: E402


def fmt_s(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", nargs="+", default=None, help="mặc định: weights của config (weight chính thức)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--budgets", nargs="+", default=["200", "all"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default="../data/all_phase2_V2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--select-out", default=None, help="ghi checkpoint có oracle_recall (200 box) cao nhất")
    ap.add_argument("--where", default="—", help="nơi chạy / GPU lúc TRAIN (3.2) hoặc 'zero-shot' (3.1)")
    ap.add_argument("--train-time", default="—")
    a = ap.parse_args()

    import torch

    cfg = load_config(a.config)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = [resolve(w) if not os.path.isabs(w) and not os.path.exists(w) else w
               for w in (a.weights or [cfg["weights"]])]
    for w in weights:
        if not os.path.exists(w):
            raise FileNotFoundError(f"thiếu {w}" + (f" — tải: wget -O {w} {cfg['weights_url']}"
                                                  if a.weights is None else ""))
    t0 = time.time()
    model, args, shim = build_gdino(cfg, device)
    transform = make_transform(args)
    gt = load_gt(a.data_root, a.split, a.limit)
    print(f"[gdino predict] {cfg['name']} | {a.split} {len(gt)} ảnh | {device} | op "
          f"{'PyTorch thuần (shim)' if shim else 'CUDA'} | khởi động {fmt_s(time.time() - t0)}", flush=True)

    pos_cache, by_weights = {}, {}
    for w in weights:
        epoch = load_weights(model, w)
        pred, t = {}, time.time()
        for i, (iid, g) in enumerate(sorted(gt.items()), 1):
            caption = prompt_of(g["text"], cfg["predict"]["prompt"])
            if caption not in pos_cache:
                pos_cache[caption] = phrase_positive_map(model.tokenizer, caption, prompt_of(g["text"], "{name}"))
            with Image.open(g["img_path"]) as im:
                boxes, scores = predict_image(model, transform, im.convert("RGB"), caption, pos_cache[caption], device)
            pred[iid] = {"boxes_xyxy": boxes, "scores": scores}
            if i % max(len(gt) // 10, 1) == 0 or i == len(gt):
                el = time.time() - t
                print(f"    [predict {i:5d}/{len(gt)}] {el / i * 1000:.0f} ms/ảnh | {fmt_s(el)} | "
                      f"còn ~{fmt_s(el / i * (len(gt) - i))}", flush=True)
        ep = "—" if epoch is None else f"epoch {epoch + 1}"
        meta = {"run": cfg["name"], "config": os.path.join("baseline", os.path.relpath(os.path.abspath(a.config),
                                                                                       resolve("."))),
                "arch": "GroundingDINO-SwinT", "weights": w, "iter": ep, "split": a.split, "steps": "—",
                "batch": cfg["finetune"]["batch_total"] if cfg.get("finetune") else "—", "where": a.where,
                "train_time": a.train_time, "date": datetime.date.today().isoformat(),
                "predict_sec": round(time.time() - t, 1), "prompt": cfg["predict"]["prompt"],
                "msda": "pytorch-shim" if shim else "cuda", "limit": a.limit}
        tag = f"{cfg['name']}_{a.split}" + ("" if epoch is None else f"_{os.path.splitext(os.path.basename(w))[0]}")
        path = os.path.join(a.out_dir, f"{tag}.json")
        write_dump(path, meta, pred)
        print(f"  dump -> {path} ({fmt_s(time.time() - t)})", flush=True)
        res = score_dump_file(path, a.data_root, parse_budgets(a.budgets), gt=gt)
        first = next(iter(res["results"].values()))
        by_weights[w] = first["nms_first"]["oracle_recall"]

    if a.select_out:
        best = max(by_weights, key=by_weights.get)
        os.makedirs(os.path.dirname(os.path.abspath(a.select_out)), exist_ok=True)
        with open(a.select_out, "w", encoding="utf-8") as f:
            json.dump({"split": a.split, "metric": "oracle_recall (200 box)", "best": best,
                       "by_weights": by_weights}, f, indent=1)
        print(f"[gdino predict] chọn {best} (oracle_recall {by_weights[best]:.4f}) -> {a.select_out}", flush=True)


if __name__ == "__main__":
    main()

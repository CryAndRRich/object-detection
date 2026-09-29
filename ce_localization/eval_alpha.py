#!/usr/bin/env python3
"""Eval EXPERIMENT ALPHA: DDIM từ nhiễu thuần -> AP + các chỉ số tách box khỏi xếp hạng.

Giao thức (docs/EXPERIMENT_ALPHA.md mục 6.2): test, N=200, top-k 100, NMS 0,5.
  oracle_recall / score_AUC / mean_bestIoU trên TOÀN BỘ box trước NMS ; AP trên top-k sau NMS.
  --oracle-score : trần khi score = IoU thật với GT (cùng box) — chênh do box hay do xếp hạng.
  --steps 1 4    : 1 bước (chính) và 4 bước (renewal + ensemble; chạy batch 1).
  --attn-diag K  : chẩn đoán attention cross-attn trên K batch (mục 6.3).
Config lấy từ checkpoint (an toàn hơn), trừ khi truyền --config.

  cd object-detection/ce_localization
  LOG=/mnt/disk1/aiotlab/haitn/log/alpha0_eval_$(date +%m%d_%H%M).log
  nohup python ../tools/run_on_free_gpu.py -- eval_alpha.py --ckpt checkpoints/alpha0/best.pth \\
      --split test --num-proposals 200 --top-k 100 --nms --oracle-score --steps 1 4 --attn-diag 20 \\
      --out /mnt/disk1/aiotlab/haitn/output/alpha0_test_N200.json > $LOG 2>&1 &
  echo "PID $! -> $LOG"
"""

import argparse
import json
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ce_localization.alpha.data import AlphaCE130, collate  # noqa: E402
from ce_localization.alpha.evaluate import attention_diagnostics, predict, score  # noqa: E402
from ce_localization.alpha.model import build_model  # noqa: E402
from ce_localization.utils.log import fmt_time  # noqa: E402
import ce_localization.train_alpha as train_alpha  # noqa: E402

KEYS = ("AP50", "AP75", "AP_coco", "precision", "recall", "recall10", "recall30",
        "oracle_recall", "score_AUC", "mean_bestIoU", "score_head_cost")


def print_results(tag, res):
    print(f"\n  === {tag} ===")
    for k in KEYS:
        print(f"  {k:16s} {res[k]:.4f}")
    print(f"  {'recall/stage':16s} {' '.join(f'{v:.3f}' for v in res['oracle_recall_per_stage'])}")
    print(f"  {'box giữ/ảnh':16s} {res['kept_per_image']:.1f}  (hậu xử lý {res['postprocess']})")
    for name, v in res["size_recall"].items():
        print(f"  {'recall ' + name:16s} {v['oracle_recall']:.4f}  (n_gt {v['n_gt']})")
    for name, v in res["density_recall"].items():
        print(f"  {'ảnh ' + name:16s} oracle_recall {v['oracle_recall']:.4f} | box giữ lại phủ "
              f"{v['kept_recall']:.4f}  ({v['n_img']} ảnh, n_gt {v['n_gt']})")
    if "oracle_score" in res:
        o = res["oracle_score"]
        print("  TRẦN khi score = IoU thật (cùng box, cùng top-k/NMS):")
        for k in ("AP50", "AP75", "AP_coco", "recall"):
            print(f"  {k:16s} thật {res[k]:.4f} | trần {o[k]:.4f} ({res[k] / max(o[k], 1e-9) * 100:.0f} % trần)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None, help="mặc định dùng config LƯU TRONG checkpoint")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", default=None, help="ghi đè data.root (vd. Kaggle)")
    ap.add_argument("--num-proposals", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--nms", action="store_true")
    ap.add_argument("--nms-thr", type=float, default=0.5)
    ap.add_argument("--oracle-score", action="store_true")
    ap.add_argument("--steps", type=int, nargs="+", default=[1])
    ap.add_argument("--no-renewal", action="store_true", help="tắt box renewal khi nhiều bước")
    ap.add_argument("--attn-diag", type=int, default=0, help="số batch cho chẩn đoán attention; 0 = tắt")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="chỉ cho 1 bước (nhiều bước luôn batch 1); @1024 batch 8 OOM trên GPU dùng chung")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="file .json trong /mnt/disk1/aiotlab/haitn/output/")
    a = ap.parse_args()

    t0 = time.time()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if a.config:
        with open(a.config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = ck["config"]
    if a.data_root:
        cfg["data"]["root"] = a.data_root
    n_prop = a.num_proposals or cfg["diffusion"]["num_proposals"]
    top_k = a.top_k or cfg["eval"]["top_k"]
    nms_thr = a.nms_thr if a.nms else None

    ds = AlphaCE130(cfg["data"]["root"], a.split, cfg["data"]["image_size"])
    if a.limit:
        ds.items = ds.items[: a.limit]
    text_table = train_alpha.build_text_table(ds.classes(), cfg, dev)
    model = build_model(cfg, pretrained_backbone=False).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[eval] {a.ckpt} (iter {ck.get('iter')}) | memory={cfg['model']['memory']} | split={a.split} "
          f"({len(ds)} ảnh) | N={n_prop} | top_k={top_k} | nms={nms_thr} | steps={a.steps} | "
          f"khởi động {fmt_time(time.time() - t0)}", flush=True)

    out = {"ckpt": a.ckpt, "iter": ck.get("iter"), "split": a.split, "n_proposals": n_prop,
           "top_k": top_k, "nms_thr": nms_thr, "memory": cfg["model"]["memory"], "results": {}}
    for steps in a.steps:
        bs = a.batch_size if steps == 1 else 1
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=a.num_workers,
                            collate_fn=collate)
        t = time.time()
        rec, stage = predict(model, loader, text_table, n_prop, steps=steps, top_k=top_k,
                             nms_thr=nms_thr, renewal=not a.no_renewal, seed=a.seed,
                             log_every=max(len(loader) // 10, 1))
        el = time.time() - t
        # có NMS thì báo CẢ HAI thứ tự hậu xử lý (người dùng chốt 2026-09-29): khoá cũ `steps{k}` =
        # top-k trước (so được với bảng cũ), `steps{k}_nmsfirst` = NMS trước như DiffusionDet
        orders = ["topk_first", "nms_first"] if nms_thr is not None else ["topk_first"]
        for order in orders:
            res = score(rec, stage, top_k, nms_thr, oracle=a.oracle_score, order=order)
            res["eval_sec"] = el
            key = f"steps{steps}" + ("_nmsfirst" if order == "nms_first" else "")
            print_results(f"{steps} bước, {order} ({fmt_time(el)})", res)
            out["results"][key] = res

    if a.attn_diag:
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                            collate_fn=collate)
        t = time.time()
        diag = attention_diagnostics(model, loader, text_table, n_prop, max_batches=a.attn_diag,
                                     seed=a.seed)
        print(f"\n  === chẩn đoán attention ({fmt_time(time.time() - t)}) — khối lượng TB theo stage ===")
        for tt, d in diag.items():
            for k, v in d.items():
                print(f"  t={tt:4d} {k:16s} {' '.join('—' if x is None else f'{x:.3f}' for x in v)}")
        out["attention"] = diag

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=float)
        print(f"  -> {a.out}")
    print(f"[eval] xong {fmt_time(time.time() - t0)}", flush=True)


if __name__ == "__main__":
    main()

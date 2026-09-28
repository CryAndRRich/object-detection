#!/usr/bin/env python3
"""Eval CE-Loc: suy luận DDIM từ nhiễu thuần -> AP + các chỉ số tách box khỏi xếp hạng.

Mọi thứ sau model là numpy, chấm bằng `utils/metrics_np.py` — cùng giao thức với bảng
vòng 1 và baseline D.1. SỐ ĐỂ SO VỚI D.1 (AP50 58,13):
    --split test --num-proposals 300 --top-k 100 --nms

Đọc số:
  oracle_recall    GT được ít nhất một trong N box phủ — chất lượng BOX, không dùng score.
  score_AUC        box khớp GT có score cao hơn box còn lại không — chất lượng XẾP HẠNG.
  --oracle-score   thay score bằng IoU thật với GT, CÙNG box -> trần AP khi xếp hạng hoàn
                   hảo. Trần >> thật: nút thắt là xếp hạng; trần cũng thấp: là box.
  N=30 (mặc định config) có trần oracle_recall chỉ 83,8/82,4/76,1 % (train/val/test).

Chạy (~1–3 phút, trực tiếp được):
  cd object-detection/ce_localization
  python ../tools/run_on_free_gpu.py -- eval.py --ckpt checkpoints/<tên>/best.pt \\
      --split test --num-proposals 300 --top-k 100 --nms --oracle-score \\
      --out /mnt/disk1/aiotlab/haitn/output/<tên>_test_N300.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ce_localization.data.ce130_dataset import CE130Detection  # noqa: E402
from ce_localization.data.loader import make_loader, model_inputs  # noqa: E402
from ce_localization.models.detector import build_model  # noqa: E402
from ce_localization.utils.box_ops_np import box_iou, cxcywh_to_xyxy  # noqa: E402
from ce_localization.utils.log import fmt_time  # noqa: E402
from ce_localization.utils.metrics_np import (COCO_THR, evaluate,  # noqa: E402
                                              nms_class_agnostic, oracle_hits,
                                              quality, summarise)

# Mốc để in cạnh kết quả (test, N=300, top-k 100, NMS 0,5).
REFERENCES = [("BASELINE D.1 (DiffusionDet)", 0.5813, 0.6734, 0.9371, 0.5974),
              ("A vòng 2 (last.pt)", 0.1036, 0.4045, 0.6148, 0.4120)]


def postprocess(boxes_cxcywh, scores, top_k, nms_thr=None):
    """Một ảnh -> chỉ số box giữ lại: TOP-K theo score, rồi NMS nếu bật (thứ tự như vòng 1)."""
    keep = np.argsort(-np.asarray(scores), kind="stable")[:top_k]
    if nms_thr is not None and len(keep):
        keep = keep[nms_class_agnostic(cxcywh_to_xyxy(boxes_cxcywh[keep]), scores[keep],
                                       nms_thr)]
    return keep


@torch.no_grad()
def predict(model, loader, n_prop, dev, top_k, nms_thr=None):
    """DDIM từ nhiễu thuần -> (bản ghi numpy mỗi ảnh, oracle_recall theo tầng).

    Bản ghi giữ CẢ N box (cho oracle_recall / score_AUC) lẫn `keep` sau top-k/NMS (cho AP).
    """
    model.eval()
    records, layer_hit, n_gt = [], None, 0
    gen = torch.Generator(device=dev.type).manual_seed(0)
    n_img, t0 = len(loader.dataset), time.time()

    for bi, batch in enumerate(loader):
        vh = torch.as_tensor(batch["valid_h"], dtype=torch.float32, device=dev)
        layers = model.ddim_sample(n_prop, valid_h=vh, generator=gen,
                                   return_all_layers=True, **model_inputs(batch, dev))
        if layer_hit is None:
            layer_hit = np.zeros(len(layers))
        boxes_f, logits_f = layers[-1]

        for i, gt in enumerate(batch["boxes"]):
            gt = gt.numpy()
            n_gt += len(gt)
            for li, (lb, _) in enumerate(layers):
                layer_hit[li] += oracle_hits(lb[i].float().cpu().numpy(), gt)[0]
            b = boxes_f[i].float().cpu().numpy()
            sc = logits_f[i].float().sigmoid().cpu().numpy()
            records.append({"image_id": batch["image_id"][i], "boxes": b, "scores": sc,
                            "keep": postprocess(b, sc, top_k, nms_thr), "gt": gt})

        done = len(records)
        if bi % max(len(loader) // 10, 1) == 0 or done == n_img:
            el = time.time() - t0
            print(f"  [{done:5d}/{n_img}] {el / done * 1000:.0f} ms/ảnh | "
                  f"đã chạy {fmt_time(el)} | còn ~{fmt_time(el / done * (n_img - done))}",
                  flush=True)
    return records, (layer_hit / max(n_gt, 1)).tolist()


def with_oracle_scores(records, top_k, nms_thr=None):
    """Score = IoU thật lớn nhất với GT, rồi top-k/NMS lại. Box giữ nguyên từng bit."""
    out = []
    for r in records:
        if len(r["gt"]) and len(r["boxes"]):
            sc = box_iou(cxcywh_to_xyxy(r["boxes"]), cxcywh_to_xyxy(r["gt"]))[0].max(axis=1)
        else:
            sc = np.zeros(len(r["boxes"]))
        out.append({**r, "scores": sc, "keep": postprocess(r["boxes"], sc, top_k, nms_thr)})
    return out


def score_records(records):
    """Bản ghi numpy -> mọi chỉ số. Hàm thuần, test được."""
    preds = [(cxcywh_to_xyxy(r["boxes"][r["keep"]]), r["scores"][r["keep"]],
              cxcywh_to_xyxy(r["gt"])) for r in records]
    ap_by_thr = {f"AP{int(round(100 * t))}": evaluate(preds, t)["AP"] for t in COCO_THR}
    at50 = evaluate(preds, 0.5)

    best_all, hits, n_gt, aucs = [], 0, 0, []
    for r in records:
        best, hit, ng, auc = quality(r["boxes"], r["scores"], r["gt"], size=1)
        best_all.append(best)
        hits, n_gt = hits + hit, n_gt + ng
        aucs.append(auc)
    res = summarise(best_all, hits, n_gt, aucs, recall_scored=at50["recall"])

    res.update(ap_by_thr)
    res["AP_coco"] = float(np.mean(list(ap_by_thr.values())))
    res.update({k: at50[k] for k in ("precision", "recall", "f1", "n_pred", "n_gt")})
    # Ngưỡng IoU thấp (không gộp vào AP_coco): recall tăng vọt khi hạ ngưỡng = box nằm
    # trên vật nhưng chưa khít; đứng yên = bỏ sót vật hẳn.
    for t in (0.1, 0.3):
        res[f"recall{int(100 * t)}"] = evaluate(preds, t)["recall"]
    return res


def print_results(res, rec_layer, n_prop, oracle=None):
    print()
    for k in ("AP50", "AP75", "AP_coco", "precision", "recall", "recall10", "recall30",
              "oracle_recall", "score_AUC", "mean_bestIoU", "score_head_cost"):
        print(f"  {k:16s} {res[k]:.4f}")
    print(f"  {'recall/tầng':16s} {' '.join(f'{v:.3f}' for v in rec_layer)}")

    print(f"\n  {'mốc (test, N=300)':28s} {'AP50':>7} {'or_recall':>10} {'AUC':>7} {'bestIoU':>8}")
    for name, ap, orc, auc, biou in REFERENCES:
        print(f"  {name:28s} {ap:7.4f} {orc:10.4f} {auc:7.4f} {biou:8.4f}")
    if n_prop == 30:
        print("  ⚠️ N=30: trần oracle_recall chỉ 83,8/82,4/76,1 % (train/val/test).")

    if oracle is not None:
        print("\n  TRẦN khi score = IoU thật với GT (cùng box, cùng top-k/NMS):")
        for k in ("AP50", "AP75", "AP_coco", "recall"):
            print(f"  {k:16s} thật {res[k]:.4f}  |  trần {oracle[k]:.4f}  "
                  f"({res[k] / max(oracle[k], 1e-9) * 100:.0f} % trần)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None,
                    help="mặc định dùng config LƯU TRONG checkpoint — an toàn hơn")
    ap.add_argument("--cache", default="../data/cache_clip_1024")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-proposals", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--nms", action="store_true", help="NMS không phân lớp sau top-k")
    ap.add_argument("--nms-thr", type=float, default=0.5)
    ap.add_argument("--oracle-score", action="store_true",
                    help="in thêm TRẦN AP khi score = IoU thật (không chạy lại model)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="file .json trong /mnt/disk1/aiotlab/haitn/output/")
    a = ap.parse_args()

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if a.config:
        with open(a.config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = ckpt["config"]
    n_prop = a.num_proposals or cfg["diffusion"]["num_proposals_eval"]
    top_k = a.top_k or cfg["eval"]["top_k"]
    nms_thr = a.nms_thr if a.nms else None

    ds = CE130Detection.from_config(cfg, a.split)
    if a.limit:
        ds.items = ds.items[: a.limit]
    loader = make_loader(ds, a.cache, a.split, a.batch_size, cfg["data"]["num_workers"],
                         dev, train=False)
    model = build_model(cfg, dropout=0.0).to(dev)
    model.load_state_dict(ckpt["model"])
    print(f"[eval] ckpt epoch={ckpt.get('epoch')} | split={a.split} ({len(ds)} ảnh) | "
          f"N={n_prop} | top_k={top_k} | nms={nms_thr}", flush=True)

    records, rec_layer = predict(model, loader, n_prop, dev, top_k, nms_thr)
    res = score_records(records)
    res["oracle_recall_per_layer"] = rec_layer
    oracle = score_records(with_oracle_scores(records, top_k, nms_thr)) if a.oracle_score else None
    if oracle is not None:
        res["oracle_score"] = oracle
    print_results(res, rec_layer, n_prop, oracle)

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"ckpt": a.ckpt, "epoch": ckpt.get("epoch"), "split": a.split,
                       "n_proposals": n_prop, "top_k": top_k, "nms_thr": nms_thr,
                       "results": res}, f, indent=2, ensure_ascii=False)
        print(f"  -> {a.out}")


if __name__ == "__main__":
    main()

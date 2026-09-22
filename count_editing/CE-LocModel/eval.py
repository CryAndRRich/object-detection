#!/usr/bin/env python3
"""Eval EXPERIMENT A (vòng 2) — AP50 + bộ chỉ số tách bạch box với score.

TÁI DÙNG, KHÔNG CHÉP: `nms_class_agnostic`, `scores_and_classes`, `ap_from_pr`,
`evaluate` lấy nguyên từ `eval.py`. Chúng chỉ đụng tới box và điểm số, không dính kiến
trúc.

BA ĐIỀU PHẢI NHỚ KHI ĐỌC SỐ
---------------------------
1. `oracle_recall` là chỉ số CHÍNH, không phải `iou_matched`. `iou_matched` mù với GT mà
   không box nào chạm tới và ĐÃ ĐÁNH LỪA HAI LẦN ở vòng 1.

2. `--num-proposals` mặc định lấy từ config (vòng 2 là **30**). Trần `oracle_recall` ở
   N=30 chỉ là **83,8 / 82,4 / 76,1 %** (train/val/test) vì GT bị cắt cụt trên 34-50 %
   số ảnh. **KHÔNG so số này với vòng 1 (N=300, trần ~97 %)** — khác thang đo.
   Muốn số để báo cáo thì chạy lại chính checkpoint này với `--num-proposals 300`;
   đổi tự do vì kiến trúc không còn pos_emb theo chỉ số.

3. TOP-K, không phải ngưỡng tuyệt đối 0,5: focal với alpha=0,25 và head không phân biệt
   hội tụ về HẰNG SỐ (vòng 1 rơi vào 0,263 < 0,5 -> lọc sạch mọi box, và triệu chứng
   "ảnh chỉ vẽ đúng một box" từng bị chẩn đoán nhầm là lỗi công cụ vẽ).

CHẠY TRÊN SERVER
----------------
  python tools/run_on_free_gpu.py -- eval.py \
      --ckpt checkpoints/round2_a/best.pt \
      --cache ../../data/cache_clip --split val \
      --out /mnt/disk1/aiotlab/haitn/output/round2_a_eval_val.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ce130_dataset import PatchCache
from data.factory import build_dataset
from models.detector import build_model
from train import TorchWrap, collate, model_inputs, oracle_recall
from utils.box_ops import box_iou, cxcywh_to_xyxy
from torch.utils.data import DataLoader


def nms_class_agnostic(boxes_xyxy, scores, iou_thr=0.5):
    """NMS does not need scores to remove duplicates in principle, but it does need
    them to decide WHICH box to keep within an overlapping group."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        iou = box_iou(boxes_xyxy[i:i + 1], boxes_xyxy[order[1:]])[0][0]
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)


def scores_and_classes(logits_1):
    """[N] or [N,C] logits for ONE image -> (score [N], class [N] or None).

    A/B have a 1-D head, so the score is the sigmoid and there is no class.
    A.2 has a C-way head: the score is the BEST class's confidence and the class
    is its index -- the standard detector reading, and the one that keeps AP
    comparable, because a proposal's confidence must not be diluted by the 79
    classes it is not.
    """
    if logits_1.dim() == 1:
        return torch.sigmoid(logits_1).cpu().numpy(), None
    p = torch.sigmoid(logits_1)
    best = p.max(dim=-1)
    return best.values.cpu().numpy(), best.indices.cpu().numpy()


def ap_from_pr(rec, prec):
    """COCO-style AP: make precision monotonically decreasing, then integrate."""
    m_rec = np.concatenate([[0.0], rec, [1.0]])
    m_pre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(m_pre) - 2, -1, -1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    idx = np.where(m_rec[1:] != m_rec[:-1])[0]
    return float(np.sum((m_rec[idx + 1] - m_rec[idx]) * m_pre[idx + 1]))


def evaluate(predictions, iou_thr=0.5):
    """predictions: list[(boxes_xyxy [K,4], scores [K], gt_xyxy [M,4])] -> dict."""
    records, total_gt = [], 0
    for boxes, scores, gt in predictions:
        total_gt += len(gt)
        if len(boxes) == 0:
            continue
        order = np.argsort(-scores)
        used = np.zeros(len(gt), dtype=bool)
        for i in order:
            if len(gt) == 0:
                records.append((scores[i], 0))
                continue
            iou = box_iou(boxes[i:i + 1], gt)[0][0]
            j = int(np.argmax(iou))
            if iou[j] >= iou_thr and not used[j]:
                used[j] = True
                records.append((scores[i], 1))
            else:
                records.append((scores[i], 0))

    if not records or total_gt == 0:
        return {"AP": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pred": 0}

    records.sort(key=lambda x: -x[0])
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([1 - r[1] for r in records])
    rec = tp / total_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    p, r = float(prec[-1]), float(rec[-1])
    return {
        "AP": ap_from_pr(rec, prec), "precision": p, "recall": r,
        "f1": 2 * p * r / max(p + r, 1e-9), "n_pred": len(records),
        "n_gt": total_gt,
    }


@torch.no_grad()

@torch.no_grad()
def predict(model, loader, n_prop, dev, top_k, use_nms, nms_thr):
    """-> (danh sách dự đoán cho `evaluate`, thống kê theo tầng)."""
    model.eval()
    preds, layer_rec, best_ious = [], [], []
    gen = torch.Generator(device=dev.type).manual_seed(0)

    for batch in loader:
        targets = [b.to(dev) for b in batch["boxes"]]
        vh = torch.as_tensor(batch["valid_h"], dtype=torch.float32, device=dev)
        layers = model.ddim_sample(n_prop, valid_h=vh, generator=gen,
                                   return_all_layers=True, **model_inputs(batch, dev))
        boxes_f, logits_f = layers[-1]

        layer_rec.append([
            sum(oracle_recall(b[i].cpu(), targets[i].cpu())[0] for i in range(len(targets)))
            / max(sum(len(g) for g in targets), 1)
            for b, _ in layers])

        for i, gt in enumerate(targets):
            b = boxes_f[i].cpu()
            sc, cls = scores_and_classes(logits_f[i].cpu())
            b_xyxy = cxcywh_to_xyxy(b)
            keep = torch.argsort(sc, descending=True)[:top_k]
            if use_nms:
                keep = keep[nms_class_agnostic(b_xyxy[keep], sc[keep], nms_thr)]
            preds.append({"boxes": b_xyxy[keep], "scores": sc[keep],
                          "classes": cls[keep],
                          "gt": cxcywh_to_xyxy(gt.cpu())})
            if gt.numel():
                iou = box_iou(b_xyxy, cxcywh_to_xyxy(gt.cpu()))
                iou = iou[0] if isinstance(iou, tuple) else iou
                best_ious.append(float(iou.max(dim=0).values.mean()))

    return preds, {
        "oracle_recall_per_layer": np.mean(layer_rec, axis=0).tolist(),
        "mean_bestIoU": float(np.mean(best_ious)) if best_ious else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="mặc định lấy config đã lưu TRONG checkpoint — an toàn hơn")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="../../data/cache_clip")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-proposals", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--nms", action="store_true")
    ap.add_argument("--nms-thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)

    # Ưu tiên config TRONG checkpoint: dựng model khác với model đã train là cách yên
    # lặng nhất để báo cáo sai số.
    if a.config:
        with open(a.config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = ckpt["config"]

    n_prop = a.num_proposals or cfg["diffusion"]["num_proposals_eval"]
    top_k = a.top_k or cfg["eval"]["top_k"]

    ds = build_dataset(cfg, a.split)
    if a.limit:
        ds.items = ds.items[: a.limit]
    cache = PatchCache(a.cache, a.split) if a.cache else None
    loader = DataLoader(TorchWrap(ds, cache), batch_size=a.batch_size, shuffle=False,
                        num_workers=cfg["data"]["num_workers"], collate_fn=collate,
                        pin_memory=dev.type == "cuda")

    # dropout=0.0: eval không được lấy mẫu ngẫu nhiên trong mạng.
    model = build_model(cfg, dropout=0.0).to(dev)
    model.load_state_dict(ckpt["model"])

    print(f"[eval round2-A] ckpt epoch={ckpt.get('epoch')} | split={a.split} "
          f"({len(ds)} ảnh) | N={n_prop} | top_k={top_k} | nms={a.nms}", flush=True)

    preds, layer_stats = predict(model, loader, n_prop, dev, top_k, a.nms, a.nms_thr)
    res = evaluate(preds)
    res.update(layer_stats)

    n_gt = sum(len(p["gt"]) for p in preds)
    hit = sum(oracle_recall(
        torch.as_tensor(np.stack([
            (p["boxes"][:, 0] + p["boxes"][:, 2]) / 2,
            (p["boxes"][:, 1] + p["boxes"][:, 3]) / 2,
            p["boxes"][:, 2] - p["boxes"][:, 0],
            p["boxes"][:, 3] - p["boxes"][:, 1]], axis=-1)) if len(p["boxes"]) else
        torch.zeros(0, 4),
        torch.as_tensor(np.stack([
            (p["gt"][:, 0] + p["gt"][:, 2]) / 2, (p["gt"][:, 1] + p["gt"][:, 3]) / 2,
            p["gt"][:, 2] - p["gt"][:, 0], p["gt"][:, 3] - p["gt"][:, 1]], axis=-1))
        if len(p["gt"]) else torch.zeros(0, 4))[0] for p in preds)
    res["oracle_recall"] = hit / max(n_gt, 1)

    print()
    for k in ("AP50", "AP", "oracle_recall", "mean_bestIoU"):
        if k in res:
            print(f"  {k:16s} {res[k]:.4f}")
    print(f"  {'recall/tầng':16s} "
          f"{' '.join(f'{v:.3f}' for v in res['oracle_recall_per_layer'])}")
    print()
    print(f"  ⚠️ N={n_prop}: trần oracle_recall là 83,8/82,4/76,1 % (train/val/test) "
          f"nếu N=30.")
    print(f"     KHÔNG so với vòng 1 (N=300). Chạy lại --num-proposals 300 để lấy số "
          f"báo cáo.")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"ckpt": a.ckpt, "split": a.split, "n_proposals": n_prop,
                       "top_k": top_k, "nms": a.nms, "results": res},
                      f, indent=2, ensure_ascii=False)
        print(f"  -> {a.out}")


if __name__ == "__main__":
    main()

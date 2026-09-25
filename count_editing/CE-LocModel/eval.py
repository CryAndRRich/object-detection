#!/usr/bin/env python3
"""Eval EXPERIMENT A (vòng 2) — AP50 + bộ chỉ số tách bạch box với score.

MỌI THỨ SAU MODEL LÀ NUMPY, chấm bằng `utils/metrics_np.py` — cùng giao thức và cùng
định nghĩa chỉ số với bảng vòng 1 (top-k -> NMS -> ghép tham lam -> AP; `oracle_recall`
và `score_AUC` tính trên TẤT CẢ N box). Bản trước 2026-09-25 trộn torch/numpy và vỡ ngay
lần chạy đầu (`torch.argsort` trên mảng numpy); không test nào chạy trọn luồng.

SỐ ĐỂ SO VỚI BASELINE D.1 (DiffusionDet trên CE-130, AP50 58,13) và E1 (6,36):
`--split test --num-proposals 300 --top-k 100 --nms`.

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
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ce130_dataset import PatchCache  # noqa: E402
from data.factory import build_dataset  # noqa: E402
from models.detector import build_model  # noqa: E402
from train import TorchWrap, collate, fmt_time, model_inputs  # noqa: E402
from utils.box_ops_np import box_iou, cxcywh_to_xyxy  # noqa: E402
from utils.metrics_np import (COCO_THR, evaluate, nms_class_agnostic,  # noqa: E402,F401
                              quality, summarise)

# `nms_class_agnostic` được import lại ở đây để các tool cũ vẫn `from eval import ...`.


def scores_and_classes(logits_1):
    """[N] hoặc [N,C] logit của MỘT ảnh -> (score [N] numpy, class [N] numpy hoặc None).

    Head 1 chiều: score = sigmoid, không có lớp. Head C lớp (A.2): score là độ tin của
    lớp TỐT NHẤT và lớp là chỉ số của nó.
    """
    logits_1 = logits_1.detach().float().cpu()
    if logits_1.dim() == 1:
        return torch.sigmoid(logits_1).numpy(), None
    best = torch.sigmoid(logits_1).max(dim=-1)
    return best.values.numpy(), best.indices.numpy()


def postprocess(boxes_cxcywh, scores, top_k, nms_thr=None):
    """Một ảnh, numpy -> chỉ số box giữ lại: TOP-K theo score, rồi NMS nếu bật.

    Top-k chứ không ngưỡng tuyệt đối: focal với head không phân biệt hội tụ về hằng số
    (vòng 1: 0,263 < 0,5 -> lọc sạch mọi box). Thứ tự top-k rồi NMS giống vòng 1.
    """
    keep = np.argsort(-np.asarray(scores), kind="stable")[:top_k]
    if nms_thr is not None and len(keep):
        k2 = nms_class_agnostic(cxcywh_to_xyxy(boxes_cxcywh[keep]), scores[keep], nms_thr)
        keep = keep[k2]
    return keep


@torch.no_grad()
def predict(model, loader, n_prop, dev, top_k, nms_thr=None):
    """Chạy DDIM từ nhiễu thuần -> list bản ghi numpy mỗi ảnh + recall theo tầng.

    Bản ghi giữ CẢ N box (để tính `oracle_recall`/`score_AUC` đúng định nghĩa) lẫn chỉ số
    `keep` sau top-k/NMS (để tính AP).
    """
    model.eval()
    records, layer_hit, n_gt_tot = [], None, 0
    gen = torch.Generator(device=dev.type).manual_seed(0)
    n_img, t0 = len(loader.dataset), time.time()

    for bi, batch in enumerate(loader):
        vh = torch.as_tensor(batch["valid_h"], dtype=torch.float32, device=dev)
        layers = model.ddim_sample(n_prop, valid_h=vh, generator=gen,
                                   return_all_layers=True, **model_inputs(batch, dev))
        if layer_hit is None:
            layer_hit = np.zeros(len(layers))

        for i, gt in enumerate(batch["boxes"]):
            gt = gt.numpy()
            n_gt_tot += len(gt)
            for li, (lb, _) in enumerate(layers):
                layer_hit[li] += quality(lb[i].float().cpu().numpy(), np.zeros(len(lb[i])),
                                         gt, size=1)[1]
            boxes_f, logits_f = layers[-1]
            b = boxes_f[i].float().cpu().numpy()
            sc, cls = scores_and_classes(logits_f[i])
            records.append({"image_id": batch["image_id"][i], "boxes": b, "scores": sc,
                            "classes": cls, "keep": postprocess(b, sc, top_k, nms_thr),
                            "gt": gt})

        done = len(records)
        if bi % max(len(loader) // 10, 1) == 0 or done == n_img:
            el = time.time() - t0
            print(f"  [{done:5d}/{n_img}] {el / max(done, 1) * 1000:.0f} ms/ảnh | "
                  f"đã chạy {fmt_time(el)} | còn ~{fmt_time(el / done * (n_img - done))}",
                  flush=True)

    return records, (layer_hit / max(n_gt_tot, 1)).tolist()


def with_oracle_scores(records, top_k, nms_thr=None):
    """Thay score của mỗi box bằng IoU THẬT lớn nhất của nó với GT, rồi top-k/NMS lại.

    -> TRẦN AP với ĐÚNG bộ box hiện tại, nếu score head xếp hạng hoàn hảo. Tách hai lỗi:
    trần cao mà AP thật thấp -> nút thắt là XẾP HẠNG (sửa score head); trần cũng thấp ->
    nút thắt là BOX (sửa score vô ích). Không chạy lại model, box giữ nguyên từng bit.
    """
    out = []
    for r in records:
        if len(r["gt"]) and len(r["boxes"]):
            sc = box_iou(cxcywh_to_xyxy(r["boxes"]), cxcywh_to_xyxy(r["gt"]))[0].max(axis=1)
        else:
            sc = np.zeros(len(r["boxes"]))
        out.append({**r, "scores": sc, "keep": postprocess(r["boxes"], sc, top_k, nms_thr)})
    return out


def score_records(records):
    """Bản ghi numpy -> mọi chỉ số, cùng định nghĩa với bảng vòng 1. Hàm thuần, test được."""
    preds = [(cxcywh_to_xyxy(r["boxes"][r["keep"]]), r["scores"][r["keep"]],
              cxcywh_to_xyxy(r["gt"])) for r in records]
    ap_by_thr = {f"AP{int(round(100 * t))}": evaluate(preds, t)["AP"] for t in COCO_THR}
    at50 = evaluate(preds, 0.5)

    best_all, hits, n_gt, aucs = [], 0, 0, []
    for r in records:
        best, hit, ng, auc = quality(r["boxes"], r["scores"], r["gt"], size=1)
        best_all.append(best)
        hits += hit
        n_gt += ng
        aucs.append(auc)
    res = summarise(best_all, hits, n_gt, aucs, recall_scored=at50["recall"])

    res.update(ap_by_thr)
    res["AP_coco"] = float(np.mean(list(ap_by_thr.values())))
    res.update({k: at50[k] for k in ("precision", "recall", "f1", "n_pred", "n_gt")})
    # Chẩn đoán ở ngưỡng IoU THẤP (không gộp vào AP_coco): recall tăng vọt khi hạ ngưỡng
    # nghĩa là box nằm trên vật nhưng chưa khít (vấn đề hồi quy); đứng yên nghĩa là bỏ
    # sót vật hẳn (vấn đề phát hiện).
    for t in (0.1, 0.3):
        res[f"recall{int(100 * t)}"] = evaluate(preds, t)["recall"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="mặc định lấy config đã lưu TRONG checkpoint — an toàn hơn")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="../../data/cache_clip")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-proposals", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--nms", action="store_true", help="NMS không phân lớp sau top-k (E1 có)")
    ap.add_argument("--nms-thr", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--oracle-score", action="store_true",
                    help="chấm thêm TRẦN AP khi score = IoU thật với GT (không chạy lại model)")
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
    nms_thr = a.nms_thr if a.nms else None

    ds = build_dataset(cfg, a.split)
    if a.limit:
        ds.items = ds.items[: a.limit]
    cache = PatchCache(a.cache, a.split) if a.cache else None
    loader = DataLoader(TorchWrap(ds, cache), batch_size=a.batch_size, shuffle=False,
                        num_workers=cfg["data"]["num_workers"], collate_fn=collate,
                        pin_memory=dev.type == "cuda")

    model = build_model(cfg, dropout=0.0).to(dev)     # dropout=0: eval không lấy mẫu
    model.load_state_dict(ckpt["model"])
    print(f"[eval round2-A] ckpt epoch={ckpt.get('epoch')} | split={a.split} "
          f"({len(ds)} ảnh) | N={n_prop} | top_k={top_k} | nms={nms_thr}", flush=True)

    records, rec_layer = predict(model, loader, n_prop, dev, top_k, nms_thr)
    res = score_records(records)
    res["oracle_recall_per_layer"] = rec_layer

    print()
    for k in ("AP50", "AP75", "AP_coco", "precision", "recall", "recall10", "recall30",
              "oracle_recall", "score_AUC", "mean_bestIoU", "score_head_cost"):
        print(f"  {k:16s} {res[k]:.4f}")
    print(f"  {'recall/tầng':16s} {' '.join(f'{v:.3f}' for v in rec_layer)}")
    # BASELINE là D.1 (DiffusionDet trên CE-130), KHÔNG phải E1 — E1 chỉ là mốc nội bộ
    # của CE-Loc vòng 1. Cùng split test, N=300.
    print("\n  mốc (test, N=300)     AP50    oracle_recall  score_AUC  mean_bestIoU")
    print("  BASELINE D.1        0,5813      0,6734       0,9371      0,5974")
    print("  E1 (CE-Loc vòng 1)  0,0636      0,2559       0,6852      0,3086")
    if n_prop == 30:
        print("  ⚠️ N=30: trần oracle_recall chỉ 83,8/82,4/76,1 % (train/val/test).")

    if a.oracle_score:
        orc = score_records(with_oracle_scores(records, top_k, nms_thr))
        res["oracle_score"] = orc
        print("\n  TRẦN khi score = IoU thật với GT (cùng box, cùng top-k/NMS):")
        for k in ("AP50", "AP75", "AP_coco", "recall"):
            print(f"  {k:16s} thật {res[k]:.4f}  |  trần {orc[k]:.4f}  "
                  f"({res[k] / max(orc[k], 1e-9) * 100:.0f} % trần)")
        print("  -> trần >> thật: nút thắt là XẾP HẠNG. trần cũng thấp: nút thắt là BOX.")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"ckpt": a.ckpt, "epoch": ckpt.get("epoch"), "split": a.split,
                       "n_proposals": n_prop, "top_k": top_k, "nms_thr": nms_thr,
                       "results": res}, f, indent=2, ensure_ascii=False)
        print(f"  -> {a.out}")


if __name__ == "__main__":
    main()

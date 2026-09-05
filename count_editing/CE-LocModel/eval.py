#!/usr/bin/env python3
"""Eval CE-Loc vòng 2 — AP kiểu COCO + P/R quy về % của TRẦN.

HAI ĐIỂM PHẢI NHỚ KHI ĐỌC SỐ:

1. TRẦN PRECISION CẤU TRÚC = min(M, N)/N. Với N=100 và M~37,6 GT thì trần chỉ
   0,376 — variant sinh nhiều box bị phạt nặng hơn THUẦN TUÝ vì sinh nhiều box,
   kể cả khi mọi box đều hoàn hảo. So P/R thô giữa các variant khác N là VÔ NGHĨA.

2. ANNOTATION BỎ SÓT VẬT. Verify bằng mắt: ảnh train/1074_b2 có ~8 con trâu mà
   all_bboxes chỉ 6. Vật bị sót mà model phát hiện đúng sẽ tính là FALSE POSITIVE
   -> precision đo được THẤP HƠN precision thật. Đừng kết luận model kém khi chưa
   nhìn ảnh.

TOP-K, KHÔNG ngưỡng tuyệt đối: focal với head không phân biệt hội tụ về hằng số
(vòng 1: 0,263 < 0,5) -> mọi box bị lọc -> argmax giữ đúng 1 box.
"""

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ce130_dataset import CE130Detection, normalize_for_clip  # noqa: E402
from models.detector import CELocDetector  # noqa: E402
from utils.box_ops_np import box_iou, cxcywh_to_xyxy  # noqa: E402


def nms_class_agnostic(boxes_xyxy, scores, iou_thr=0.5):
    """NMS KHÔNG cần score để khử trùng lặp về mặt nguyên tắc, nhưng cần score để
    biết GIỮ box nào trong nhóm chồng nhau."""
    order = np.argsort(-scores)
    giu = []
    while len(order):
        i = order[0]
        giu.append(i)
        if len(order) == 1:
            break
        iou = box_iou(boxes_xyxy[i:i + 1], boxes_xyxy[order[1:]])[0][0]
        order = order[1:][iou <= iou_thr]
    return np.array(giu, dtype=int)


def ap_tu_pr(rec, prec):
    """AP kiểu COCO: nội suy precision đơn điệu giảm rồi tích phân."""
    m_rec = np.concatenate([[0.0], rec, [1.0]])
    m_pre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(m_pre) - 2, -1, -1):
        m_pre[i] = max(m_pre[i], m_pre[i + 1])
    idx = np.where(m_rec[1:] != m_rec[:-1])[0]
    return float(np.sum((m_rec[idx + 1] - m_rec[idx]) * m_pre[idx + 1]))


def danh_gia(du_doan, iou_thr=0.5):
    """du_doan: list[(boxes_xyxy [K,4], scores [K], gt_xyxy [M,4])] -> dict."""
    ban_ghi, tong_gt = [], 0
    for boxes, scores, gt in du_doan:
        tong_gt += len(gt)
        if len(boxes) == 0:
            continue
        order = np.argsort(-scores)
        da_dung = np.zeros(len(gt), dtype=bool)
        for i in order:
            if len(gt) == 0:
                ban_ghi.append((scores[i], 0))
                continue
            iou = box_iou(boxes[i:i + 1], gt)[0][0]
            j = int(np.argmax(iou))
            if iou[j] >= iou_thr and not da_dung[j]:
                da_dung[j] = True
                ban_ghi.append((scores[i], 1))
            else:
                ban_ghi.append((scores[i], 0))

    if not ban_ghi or tong_gt == 0:
        return {"AP": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n_pred": 0}

    ban_ghi.sort(key=lambda x: -x[0])
    tp = np.cumsum([r[1] for r in ban_ghi])
    fp = np.cumsum([1 - r[1] for r in ban_ghi])
    rec = tp / tong_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    p, r = float(prec[-1]), float(rec[-1])
    return {
        "AP": ap_tu_pr(rec, prec), "precision": p, "recall": r,
        "f1": 2 * p * r / max(p + r, 1e-9), "n_pred": len(ban_ghi),
        "n_gt": tong_gt,
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-proposals", type=int, default=None)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    N = a.num_proposals or cfg["diffusion"]["num_proposals_eval"]

    ds = CE130Detection(cfg["data"]["root"], a.split, cfg["data"]["image_size"])
    if a.limit:
        ds.items = ds.items[: a.limit]

    ten_tn = cfg.get("experiment", "?")
    moi_truong = {
        "experiment": ten_tn,
        "thoi_diem": datetime.now().isoformat(timespec="seconds"),
        "device": str(dev), "torch": torch.__version__,
        "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "lenh": " ".join(sys.argv), "ckpt": os.path.abspath(a.ckpt),
    }
    print("=" * 78, flush=True)
    print(f"  EVAL — EXPERIMENT {ten_tn}", flush=True)
    print("-" * 78, flush=True)
    for k, v in moi_truong.items():
        print(f"  {k:22s} {v}", flush=True)
    print(f"  {'split':22s} {a.split} | N={N} | topk={cfg['eval']['topk']} "
          f"| nms={cfg['eval']['nms_iou']} | sampling_steps={cfg['diffusion']['sampling_steps']}",
          flush=True)
    print(f"  {'dataset':22s} {ds.thong_ke()}", flush=True)
    print("=" * 78, flush=True)

    model = CELocDetector(
        cfg["model"]["clip_name"], cfg["model"]["d_model"], cfg["model"]["n_layer"],
        cfg["model"]["n_head"], cfg["data"]["image_size"],
        cfg["diffusion"]["num_timesteps"], cfg["diffusion"]["snr_scale"],
        cfg["diffusion"]["sampling_steps"], 0.0, cfg["model"]["freeze_clip"],
    ).to(dev)
    sd = torch.load(a.ckpt, map_location=dev)
    w = sd["model"] if "model" in sd else sd
    # Checkpoint chỉ chứa tham số học được; CLIP frozen tải lại từ HuggingFace.
    thieu, thua = model.load_state_dict(w, strict=False)
    thieu = [k for k in thieu
             if not (k.startswith("encoder.vision.") or k.startswith("encoder.text."))]
    assert not thieu and not thua, f"checkpoint không khớp: thiếu={thieu} thừa={thua}"
    model.eval()

    topk, nms_iou = cfg["eval"]["topk"], cfg["eval"]["nms_iou"]
    du_doan, moi_diem = [], []

    t0 = time.time()
    moi_anh = []
    for i in range(len(ds)):
        t_i = time.time()
        m = ds[i]
        px = torch.from_numpy(normalize_for_clip(m["image"])).unsqueeze(0).to(dev)
        boxes, logits = model.ddim_sample(N, pixel_values=px, texts=[m["text"]])

        b = boxes[0].cpu().numpy()
        s = torch.sigmoid(logits[0]).cpu().numpy()
        moi_diem.append(s)

        keep = np.argsort(-s)[:topk]                       # TOP-K, không ngưỡng
        b_xyxy = cxcywh_to_xyxy(b[keep]) * cfg["data"]["image_size"]
        s_k = s[keep]
        k2 = nms_class_agnostic(b_xyxy, s_k, nms_iou)

        gt = cxcywh_to_xyxy(m["boxes"]) * cfg["data"]["image_size"]
        du_doan.append((b_xyxy[k2], s_k[k2], gt))

        moi_anh.append({"image_id": m["image_id"], "class": m["text"],
                        "n_gt": len(gt), "n_sau_topk": len(keep),
                        "n_sau_nms": len(k2),
                        "score_max": float(s.max()), "score_min": float(s.min()),
                        "giay": time.time() - t_i})
        if (i + 1) % max(len(ds) // 20, 1) == 0 or i == len(ds) - 1:
            da = time.time() - t0
            eta = da / (i + 1) * (len(ds) - i - 1)
            print(f"  [{i+1:5d}/{len(ds)}] {100*(i+1)/len(ds):5.1f}% | "
                  f"{1000*da/(i+1):.0f}ms/ảnh | đã chạy {da:.0f}s | ETA {eta:.0f}s",
                  flush=True)

    # AP ở nhiều ngưỡng IoU (AP50/AP75 + AP trung bình kiểu COCO)
    kq = danh_gia(du_doan, 0.5)
    ap_theo_nguong = {f"AP{int(100*t)}": danh_gia(du_doan, t)["AP"]
                      for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]}
    ap_coco = float(np.mean(list(ap_theo_nguong.values())))

    tran = float(np.mean([min(len(g), len(b)) / max(len(b), 1) for b, _, g in du_doan]))
    diem = np.concatenate(moi_diem)
    n_box = [len(b) for b, _, _ in du_doan]
    tong_tg = time.time() - t0

    print("\n" + "=" * 78, flush=True)
    print(f"KẾT QUẢ — EXPERIMENT {ten_tn}", flush=True)
    print("-" * 78, flush=True)
    print(f"  AP (COCO, IoU .50:.95)   {ap_coco:.4f}", flush=True)
    print(f"  AP50                     {ap_theo_nguong['AP50']:.4f}", flush=True)
    print(f"  AP75                     {ap_theo_nguong['AP75']:.4f}", flush=True)
    print(f"  precision                {kq['precision']:.4f}", flush=True)
    print(f"  recall                   {kq['recall']:.4f}", flush=True)
    print(f"  f1                       {kq['f1']:.4f}", flush=True)
    print("-" * 78, flush=True)
    print(f"  AP theo ngưỡng: " + "  ".join(
        f"{k} {v:.4f}" for k, v in ap_theo_nguong.items()), flush=True)
    print("-" * 78, flush=True)
    print(f"  box dự đoán      {kq['n_pred']} ({np.mean(n_box):.1f}/ảnh, "
          f"trước NMS {np.mean([x['n_sau_topk'] for x in moi_anh]):.1f})", flush=True)
    print(f"  box GT           {kq.get('n_gt', 0)} ({kq.get('n_gt', 0)/max(len(ds),1):.1f}/ảnh)",
          flush=True)
    print(f"  trần precision   {tran:.4f}  (= min(M,N)/N — so P/R thô giữa các N là VÔ NGHĨA)",
          flush=True)
    print(f"  precision / trần {kq['precision']/max(tran,1e-9):.4f}  <- SO CÁI NÀY", flush=True)
    print("-" * 78, flush=True)
    print(f"  score  μ {diem.mean():.4f}  σ {diem.std():.4f}  "
          f"[{diem.min():.4f}, {diem.max():.4f}]  "
          f"p50 {np.percentile(diem,50):.4f}  p99 {np.percentile(diem,99):.4f}", flush=True)
    print(f"  thời gian {tong_tg:.0f}s ({1000*tong_tg/max(len(ds),1):.0f}ms/ảnh)", flush=True)

    canh_bao = []
    if diem.std() < 0.05:
        canh_bao.append("score σ < 0,05 — head kẹt ở hằng số, AP gần như vô nghĩa "
                        "(xếp hạng ngẫu nhiên)")
    if np.mean(n_box) < 2:
        canh_bao.append(f"chỉ {np.mean(n_box):.1f} box/ảnh sau NMS — kiểm lại topk/NMS")
    if kq["recall"] < 0.01:
        canh_bao.append(f"recall {kq['recall']:.4f} rất thấp")
    for c in canh_bao:
        print(f"  ⚠ {c}", flush=True)
    print("=" * 78, flush=True)

    duong_dan = os.path.splitext(a.ckpt)[0] + f"_eval_{a.split}_N{N}.json"
    with open(duong_dan, "w") as f:
        json.dump({
            "tom_tat": {**kq, "AP_coco": ap_coco, "tran_precision": tran,
                        "precision_tren_tran": kq["precision"] / max(tran, 1e-9),
                        "canh_bao": canh_bao},
            "ap_theo_nguong": ap_theo_nguong,
            "score": {"mean": float(diem.mean()), "std": float(diem.std()),
                      "min": float(diem.min()), "max": float(diem.max()),
                      **{f"p{q}": float(np.percentile(diem, q))
                         for q in [1, 25, 50, 75, 99]}},
            "box_moi_anh": {"mean": float(np.mean(n_box)), "min": int(np.min(n_box)),
                            "max": int(np.max(n_box))},
            "cau_hinh": {"N": N, "split": a.split, "topk": topk, "nms_iou": nms_iou,
                         "sampling_steps": cfg["diffusion"]["sampling_steps"]},
            "moi_truong": moi_truong,
            "dataset": ds.thong_ke(),
            "thoi_gian_giay": tong_tg,
            "moi_anh": moi_anh,          # per-ảnh, để tìm ảnh nào hỏng
        }, f, indent=2, ensure_ascii=False)
    print(f"  số liệu đầy đủ (kèm per-ảnh): {duong_dan}", flush=True)


if __name__ == "__main__":
    main()

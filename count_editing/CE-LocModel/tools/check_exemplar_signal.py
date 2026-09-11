#!/usr/bin/env python3
"""Cửa chặn cho hướng EXEMPLAR — đo trên CLIP FROZEN, KHÔNG train gì, không cần GPU.

Câu hỏi: CE-130 mỗi ảnh chỉ có MỘT class (3.598/3.598) và trung vị 20-30 vật/ảnh, nên các
vật khác trong CÙNG ảnh có thể làm "template" nội tại. Lấy 1 vật làm exemplar, correlate
feature CLIP của nó với lưới patch 32x32 -- bản đồ tương quan có định vị được CÁC VẬT CÒN
LẠI không?

Vì sao đáng đo: đường box<->patch của EXPERIMENT A đo được TRỰC GIAO lúc khởi tạo
(cosine +0,0005) và score_AUC sau train vẫn chỉ 0,4965-0,4988 (tung đồng xu). Nếu đường
patch<->patch (exemplar) cho AUC cao NGAY KHI CHƯA TRAIN thì đó là tín hiệu miễn phí mà
kiến trúc hiện tại đang bỏ qua hoàn toàn.

⚠️ ĐỌC ĐÚNG KẾT QUẢ: phép đo này chứng minh "tín hiệu TỒN TẠI", KHÔNG chứng minh
"train xong sẽ tốt hơn" -- đúng như hạn chế đã ghi cho cửa chặn của C2
(tools/check_vertex_rpe.py). Nó cũng dùng ô lưới chứa TÂM box, tức bỏ qua kích thước.

Chạy:  .venv-cpu/bin/python tools/check_exemplar_signal.py [--split test] [--limit 30]
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection, normalize_for_clip  # noqa: E402
from models.detector import build_model  # noqa: E402


def auc(scores, labels):
    """AUC bằng thứ hạng (Mann-Whitney). Không cần sklearn."""
    order = np.argsort(-scores)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    return (rank[neg].mean() - rank[pos].mean()) / len(labels) + 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--config", default="config/experiment_a.yaml")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    enc = build_model(cfg, dropout=0.0).eval().encoder
    ds = CE130Detection("../../data/all_phase2_V2", a.split)
    g = cfg["data"]["image_size"] // 16          # ViT-B/16 -> 32

    def cell(b):
        cx, cy = b[0] * g, b[1] * g
        return int(np.clip(cy, 0, g - 1)) * g + int(np.clip(cx, 0, g - 1))

    scores = []
    for i in range(min(a.limit, len(ds.items))):
        it = ds[i]
        gt = it["boxes"]
        if len(gt) < 4:
            continue
        px = torch.from_numpy(normalize_for_clip(it["image"])).unsqueeze(0)
        with torch.no_grad():
            patch = enc.encode_image_raw(px)[0]
        F = torch.nn.functional.normalize(patch, dim=-1)

        # exemplar = vật ĐẦU TIÊN; nhãn dương = ô chứa tâm các vật CÒN LẠI
        sim = (F @ F[cell(gt[0])]).numpy()
        lab = np.zeros(g * g)
        for b in gt[1:]:
            lab[cell(b)] = 1
        s = auc(sim, lab)
        if s is not None:
            scores.append(s)

    s = np.array(scores)
    print(f"CỬA CHẶN EXEMPLAR — CLIP frozen, CHƯA train gì  (split={a.split}, n={len(s)} ảnh)")
    print(f"  AUC định vị vật khác từ 1 exemplar: mean {s.mean():.4f}  median {np.median(s):.4f}")
    print(f"  số ảnh AUC > 0,70: {(s > 0.7).sum()}/{len(s)}")
    print(f"  tung đồng xu = 0,50")
    print()
    print("  Đối chiếu số đã có của dự án:")
    print("    score_AUC của A/B/C1 SAU KHI TRAIN : 0,4965 - 0,4988")
    print("    cosine(box token, patch token) init: +0,0005 (trực giao)")
    ok = s.mean() > 0.70
    print()
    print(f"  => {'ĐẠT' if ok else 'KHÔNG ĐẠT'}: tín hiệu exemplar "
          f"{'tồn tại và mạnh' if ok else 'không đủ mạnh'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""ĐỘ PHÂN GIẢI LƯỚI có phải nút thắt không? — ngoại suy TRƯỚC khi build cache 1024px.

VÌ SAO TỒN TẠI
--------------
Chẩn đoán (`gate_delta_ablation.py`, 2026-09-23) kết luận: trần của đặc trưng nằm quanh
**0,30**, và nguyên nhân là ĐỘ PHÂN GIẢI chứ không phải kiến trúc —

    k=5 bó trong box : 0,042      k=5 nới 2,0x : 0,288
    k=7 bó trong box : 0,164      k=7 nới 2,0x : 0,328

Box CE-130 trung vị rộng 1,96 ô lưới. Lấy 5 điểm trải trên 1,96 ô = 0,39 ô/điểm, DÀY HƠN
một ô ⇒ `grid_sample` nội suy ra 5 giá trị gần trùng nhau. Nới rộng giúp vì các điểm bắt
đầu chạm những ô KHÁC nhau. Tức là bottleneck là số ô mỗi box, không phải số điểm.

Hướng sửa: nâng ảnh 512 -> 1024px, lưới 32x32 -> 64x64, box trung vị 1,96 -> 3,92 ô.
Nhưng build cache 1024px tốn ~4x dung lượng và ViT attention tốn 16x (4097 token).
Trước khi trả giá đó, ngoại suy bằng cách đi NGƯỢC LẠI: hạ lưới hiện có 32 -> 16 -> 8
bằng average-pool, đo trần tụt bao nhiêu.

  - Trần tụt MẠNH khi hạ  ⇒ độ phân giải ĐÚNG là nút thắt ⇒ nâng lên 64 đáng làm.
  - Trần tụt ÍT khi hạ    ⇒ thông tin không nằm ở độ phân giải ⇒ 1024px cũng vô ích,
                             ĐỪNG build cache, phải đổi hướng khác.

Phép đo này KHÔNG chứng minh 64x64 sẽ đạt 0,5 — nó chỉ chặn trường hợp chắc chắn vô ích.
Chiều tăng không bắt buộc đối xứng với chiều giảm (ViT pretrain ở 14x14, nội suy pos_embed
càng xa càng kém tin cậy), nên kết quả đọc theo hướng CHẶN, không phải hướng hứa hẹn.

~3 phút. Đọc cache có sẵn, KHÔNG cần build gì.

CHẠY TRÊN SERVER
----------------
  LOG=/mnt/disk1/aiotlab/haitn/log/gate_grid_$(date +%m%d_%H%M).log
  nohup python tools/run_on_free_gpu.py -- tools/gate_grid_resolution.py \\
      --split val --out /mnt/disk1/aiotlab/haitn/output/round2_gate_grid.json \\
      > $LOG 2>&1 &
  echo "PID $! -> $LOG"
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection, PatchCache            # noqa: E402
from tools.gate_delta_ablation import (_cos, _cos_se, grid_points,   # noqa: E402
                                       probe_mlp)
import tools.gate_delta_direction as _gdd                           # noqa: E402
from tools.gate_delta_direction import fmt, perturb, set_grid, true_delta  # noqa: E402


@torch.no_grad()
def sample_at_grid(praw, boxes, k, scale, g_new):
    """Hạ lưới CLIP xuống `g_new` x `g_new` bằng average-pool rồi lấy mẫu RoI.

    average-pool chứ không phải lấy thưa (stride): pool giữ lại thông tin trung bình của
    vùng, đúng như một ViT có patch to hơn sẽ thấy. Lấy thưa sẽ vứt hẳn 3/4 dữ liệu và
    làm phép so sánh bi quan giả tạo.
    """
    B, P, d_in = praw.shape
    g = int(round(P ** 0.5))
    fmap = praw.transpose(1, 2).reshape(B, d_in, g, g)
    if g_new != g:
        assert g % g_new == 0, f"{g} không chia hết cho {g_new}"
        fmap = F.avg_pool2d(fmap, g // g_new)
    pts = grid_points(boxes.to(praw.device), k, scale)
    s = F.grid_sample(fmap, pts * 2.0 - 1.0, mode="bilinear",
                      padding_mode="border", align_corners=False)
    return s.permute(0, 2, 3, 1)[0].flatten(-2)


@torch.no_grad()
def collect(ds, cache, d_cells, seed, dev, max_img, combos):
    rng = np.random.default_rng(seed)
    feats = {c: [] for c in combos}
    D, W = [], []
    for i in range(min(len(ds), max_img)):
        s_ = ds.__getitem__(i, need_image=False)
        gt = torch.as_tensor(s_["boxes"], dtype=torch.float32)
        if gt.numel() == 0:
            continue
        patch, _ = cache.get(s_["image_id"], s_["text"], False)
        praw = torch.from_numpy(patch).unsqueeze(0).to(dev)
        box = perturb(gt, d_cells, rng)
        for (gn, k, sc) in combos:
            feats[(gn, k, sc)].append(sample_at_grid(praw, box, k, sc, gn).cpu())
        D.append(true_delta(box, gt))
        W.append(gt[:, 2] * _gdd.GRID)
    return ({c: torch.cat(v) for c, v in feats.items()},
            torch.cat(D), torch.cat(W))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--cache", default="../../data/cache_clip")
    ap.add_argument("--split", default="val")
    ap.add_argument("--d-cells", type=float, default=1.0)
    ap.add_argument("--max-images", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="gate_grid.json")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = CE130Detection(cfg["data"]["root"], a.split, cfg["data"]["image_size"])
    if not os.path.exists(os.path.join(a.cache, f"{a.split}_meta.json")):
        raise SystemExit(f"KHÔNG THẤY CACHE: {a.cache}/{a.split}_meta.json")
    cache = PatchCache(a.cache, a.split)
    g_cache = set_grid(cache.meta["shape"][2])

    # Hai cách lấy mẫu MẠNH NHẤT từ chẩn đoán, để so trên cùng điều kiện.
    grids = [g for g in (8, 16, 32, 64) if g <= g_cache]
    combos = [(g, 3, 1.0) for g in grids] + [(g, 7, 2.0) for g in grids]

    t0 = time.time()
    print(f"split={a.split}  ảnh={min(len(ds), a.max_images)}  d={a.d_cells} ô  t=-1  "
          f"thiết bị={dev}", flush=True)
    print(f"Lưới cache {g_cache}x{g_cache}. Hạ xuống {grids} bằng average-pool, "
          f"đo trần mỗi mức.", flush=True)
    feats, d, w = collect(ds, cache, a.d_cells, a.seed, dev, a.max_images, combos)
    n = len(d)
    print(f"  [xong {fmt(time.time()-t0)}] {n} box", flush=True)

    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(n, generator=g)
    cut = int(n * 0.7)
    tr, te = perm[:cut], perm[cut:]
    dte = d[te]
    res = {"config": vars(a), "n_box": n, "rows": []}

    print(f"\n  {'lưới':>6} {'ô/box':>7} {'lấy mẫu':<16} {'cos tâm':>8} {'±se':>6} "
          f"{'dx':>6} {'dy':>6}", flush=True)
    for (gn, k, sc) in combos:
        X = feats[(gn, k, sc)]
        pred = probe_mlp(X[tr], d[tr], X[te], dte, dev, a.seed, 1024, 2)
        r = {"grid": gn, "k": k, "scale": sc, "d_in": X.shape[1],
             "o_tren_box": 1.96 * gn / 32,      # 1,96 ô là số đo Ở LƯỚI 32
             "cos_tam": _cos(pred, dte), "se": _cos_se(pred, dte),
             "cos_dx": _cos(pred[:, 0:1], dte[:, 0:1], False),
             "cos_dy": _cos(pred[:, 1:2], dte[:, 1:2], False)}
        res["rows"].append(r)
        print(f"  {f'{gn}x{gn}':>6} {r['o_tren_box']:7.2f} {f'k={k}, nới {sc}':<16} "
              f"{r['cos_tam']:8.3f} ±{r['se']:5.3f} {r['cos_dx']:6.3f} "
              f"{r['cos_dy']:6.3f}", flush=True)

    # Ngoại suy: hồi quy tuyến tính cos theo log2(lưới), cho mỗi cách lấy mẫu.
    print("\n  NGOẠI SUY (hồi quy cos theo log2 lưới, chỉ để CHẶN, không phải lời hứa):",
          flush=True)
    res["extrapolation"] = []
    for k, sc in [(3, 1.0), (7, 2.0)]:
        rows = [r for r in res["rows"] if r["k"] == k]
        x = np.log2([r["grid"] for r in rows])
        y = np.array([r["cos_tam"] for r in rows])
        slope, intercept = np.polyfit(x, y, 1)
        pred64 = slope * np.log2(2 * g_cache) + intercept
        res["extrapolation"].append({"k": k, "scale": sc, "slope_per_doubling": float(slope),
                                     "luoi_du_bao": 2 * g_cache, "du_bao": float(pred64)})
        print(f"    k={k}, nới {sc}: mỗi lần GẤP ĐÔI lưới -> {slope:+.3f} cos  "
              f"|  dự báo ở lưới {2*g_cache}: {pred64:.3f}", flush=True)

    res["total_sec"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print("\n  ĐỌC KẾT QUẢ:")
    print("    - Dốc LỚN (> +0,06/lần gấp đôi) ⇒ độ phân giải ĐÚNG là nút thắt,")
    print("      build cache 1024px đáng làm.")
    print("    - Dốc ~0 hoặc ÂM ⇒ thông tin KHÔNG nằm ở độ phân giải; 1024px sẽ không")
    print("      cứu được. ĐỪNG build cache, phải đổi hướng.")
    print(f"    - 'dự báo ở lưới {2*g_cache}' là ngoại suy tuyến tính — dùng để CHẶN")
    print("      (nếu nó còn dưới 0,5 thì ngay cả kịch bản lạc quan cũng trượt),")
    print("      KHÔNG dùng để kết luận sẽ đạt.")
    print(f"  tổng {fmt(res['total_sec'])}  ->  {a.out}")


if __name__ == "__main__":
    main()

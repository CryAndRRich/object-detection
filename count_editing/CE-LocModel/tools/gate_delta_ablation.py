#!/usr/bin/env python3
"""CHẨN ĐOÁN cửa chặn — vì sao `cos2304` chỉ đạt 0,266 ở hàng phán quyết?

VÌ SAO TỒN TẠI
--------------
`gate_delta_direction.py` (chạy 2026-09-23, sau khi sửa 2 lỗi đo) cho:

    t=-1, d=1.0 :  cos2304 = 0,266   lstsq = 0,222   xáo = -0,010   r=0 = 0,007

Đối chứng SẠCH (xáo/r=0 ~ 0) ⇒ toàn bộ 0,266 đến TỪ ẢNH, không phải prior. Tín hiệu có
thật nhưng dưới tiêu chí 0,5. Trước khi bỏ EXPERIMENT A phải loại trừ khả năng
**PHÉP ĐO còn yếu**, chứ không phải đặc trưng yếu — vì `lstsq ~ cosine` mới chỉ chứng
minh chạm trần TUYẾN TÍNH, không nói gì về trần phi tuyến.

7 GIẢ THUYẾT, mỗi cái đổi được quyết định:

  1. probe   — `Linear` quá yếu? -> MLP 1-2 lớp, vài bề rộng.
  2. bottleneck — `proj_point` (768->256) cũng đóng băng ngẫu nhiên, là nút thắt THỨ HAI
                  chưa ai gỡ. -> chấm thẳng trên 9x768 CLIP THÔ.
  3. k       — lưới 3x3 quá thưa? -> k = 1,3,5,7.
  4. context — chỉ lấy mẫu TRONG box nên không thấy biên vật; muốn biết "lệch sang trái"
               thì phải thấy cả bên ngoài. -> nới lưới 1,5x / 2,0x.
  5. kênh    — cosine 2 kênh tâm có thể che một kênh tốt. -> tách dx, dy, dw, dh.
  6. TRẦN    — quan trọng nhất: k-NN phi tham số + oracle, trả lời "đặc trưng CÓ CHỨA
               thông tin không" tách khỏi "mô hình nào rút được".
  7. ảnh?    — so với đặc trưng VỊ TRÍ thuần (không ảnh): nếu ngang nhau thì CLIP không
               đóng góp gì ngoài việc mã hoá toạ độ.

CHỈ chạy ở hàng phán quyết (t=-1) — các hàng t>=0 đã chứng minh là hỏi sai câu hỏi.

KHÔNG train model thật. Đọc cache, không cần CLIP.

CHẠY TRÊN SERVER (~15 phút, chạy nền)
-------------------------------------
  LOG=/mnt/disk1/aiotlab/haitn/log/gate_ablation_$(date +%m%d_%H%M).log
  nohup python tools/run_on_free_gpu.py -- tools/gate_delta_ablation.py \\
      --split val --out /mnt/disk1/aiotlab/haitn/output/round2_gate_ablation.json \\
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
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection, PatchCache          # noqa: E402
import tools.gate_delta_direction as _gdd                           # noqa: E402
from tools.gate_delta_direction import fmt, perturb, set_grid, true_delta  # noqa: E402

NGUONG = 0.5                    # tiêu chí của cửa chặn


# ---------------------------------------------------------------------------
# lấy mẫu
# ---------------------------------------------------------------------------

def grid_points(boxes, k, scale=1.0):
    """[n,4] cxcywh -> [1,n,k*k,2] điểm lấy mẫu, `scale` nới rộng lưới ra NGOÀI box.

    `scale > 1` là giả thuyết 4: muốn biết box lệch sang trái thì phải thấy được phần
    vật nằm ngoài mép phải, mà lưới bó trong box thì không bao giờ thấy.
    """
    dev, dt = boxes.device, boxes.dtype
    off = ((torch.arange(k, device=dev, dtype=dt) + 0.5) / k - 0.5) * scale
    gy, gx = torch.meshgrid(off, off, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    cx, cy, w, h = boxes.unbind(-1)
    x = cx.unsqueeze(-1) + pts[:, 0] * w.unsqueeze(-1)
    y = cy.unsqueeze(-1) + pts[:, 1] * h.unsqueeze(-1)
    return torch.stack([x, y], dim=-1).unsqueeze(0)


@torch.no_grad()
def sample_raw(praw, boxes, k, scale=1.0):
    """-> [n, k*k*768] CLIP THÔ, KHÔNG qua `proj_point`.

    Khác `sample_roi` ở chỗ bỏ hẳn nút thắt 768->256 (giả thuyết 2).
    """
    B, P, d_in = praw.shape
    g = int(round(P ** 0.5))
    fmap = praw.transpose(1, 2).reshape(B, d_in, g, g)
    pts = grid_points(boxes.to(praw.device), k, scale)
    s = F.grid_sample(fmap, pts * 2.0 - 1.0, mode="bilinear",
                      padding_mode="border", align_corners=False)
    return s.permute(0, 2, 3, 1)[0].flatten(-2)                    # [n, k*k*768]


@torch.no_grad()
def collect(ds, cache, d_cells, seed, dev, max_img, ks, scales, sj=0.0):
    """Thu đặc trưng cho MỌI (k, scale) trong một lượt đọc cache.

    Đọc cache là phần đắt nhất (1m07s cho 400 ảnh), nên đọc MỘT lần rồi lấy mẫu nhiều
    kiểu, thay vì lặp lại mỗi cấu hình.
    """
    rng = np.random.default_rng(seed)
    feats = {(k, s): [] for k in ks for s in scales}
    D, W, POS = [], [], []
    for i in range(min(len(ds), max_img)):
        s_ = ds.__getitem__(i, need_image=False)
        gt = torch.as_tensor(s_["boxes"], dtype=torch.float32)
        if gt.numel() == 0:
            continue
        patch, _ = cache.get(s_["image_id"], s_["text"], False)
        praw = torch.from_numpy(patch).unsqueeze(0).to(dev)
        box = perturb(gt, d_cells, rng)                     # t=-1: KHÔNG nhiễu
        if sj > 0:
            f = torch.from_numpy(np.exp(rng.uniform(-sj, sj, (len(box), 2)))).float()
            box = torch.cat([box[:, :2], box[:, 2:] * f], dim=1)
        for k in ks:
            for sc in scales:
                feats[(k, sc)].append(sample_raw(praw, box, k, sc).cpu())
        D.append(true_delta(box, gt))
        W.append(gt[:, 2] * _gdd.GRID)
        POS.append(box)                                     # giả thuyết 7
    return ({kk: torch.cat(v) for kk, v in feats.items()},
            torch.cat(D), torch.cat(W), torch.cat(POS))


# ---------------------------------------------------------------------------
# các probe
# ---------------------------------------------------------------------------

def _cos(pred, d, dim2=True):
    a, b = (pred[:, :2], d[:, :2]) if dim2 else (pred, d)
    return float(F.cosine_similarity(a, b, dim=-1).mean())


def _cos_se(pred, d):
    """Sai số chuẩn của cosine trung bình. Không có nó thì không biết chênh lệch giữa
    hai hàng là thật hay là nhiễu lấy mẫu."""
    c = F.cosine_similarity(pred[:, :2], d[:, :2], dim=-1)
    return float(c.std() / max(len(c) ** 0.5, 1.0))


def probe_linear(Xtr, dtr, Xte, dte, dev, seed, epochs=400, lr=1e-3):
    torch.manual_seed(seed)
    head = nn.Linear(Xtr.shape[1], 4).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    Xd, dd = Xtr.to(dev), dtr.to(dev)
    for _ in range(epochs):
        opt.zero_grad(); F.mse_loss(head(Xd), dd).backward(); opt.step()
    with torch.no_grad():
        return head(Xte.to(dev)).cpu()


def probe_mlp(Xtr, dtr, Xte, dte, dev, seed, hidden=512, layers=2,
              epochs=3000, lrs=(3e-4, 1e-3, 3e-3), bs=4096, patience=15):
    """MLP có minibatch + early stop + QUÉT lr + CHUẨN HOÁ đặc trưng.

    Ba thứ sửa sau lần chạy @1024 (2026-09-23), nơi `mlp 3 lớp` cho 0,077 còn
    `k=7 nới 2,0x` cho 0,030 — THẤP HƠN CẢ MỨC SÀN k=1 (0,019). Không thể là kết luận
    về đặc trưng; đó là lỗi tối ưu:

      1. CHUẨN HOÁ (z-score theo từng chiều, thống kê lấy TỪ TẬP TRAIN): đặc trưng CLIP
         thô ở 37.632 chiều có thang rất lệch, gradient bước đầu lớn -> phân kỳ ngay.
      2. QUÉT lr rồi lấy cấu hình tốt nhất theo tập val riêng: một lr cố định không thể
         hợp cho cả 768-d lẫn 37.632-d.
      3. PATIENCE 15 thay vì 8: mạng sâu hơn cần nhiều epoch hơn mới thoát vùng phẳng.

    Vẫn là early stop theo val TÁCH TỪ TRAIN, không đụng tập test — nếu chọn lr theo
    test thì con số cuối là rò rỉ, không còn là trần trung thực.
    """
    n = len(Xtr)
    cut = int(n * 0.85)
    g = torch.Generator().manual_seed(seed)
    pm = torch.randperm(n, generator=g)
    tr, va = pm[:cut], pm[cut:]

    # Chuẩn hoá bằng thống kê CỦA TẬP TRAIN, áp cho cả val lẫn test.
    mu = Xtr[tr].mean(0, keepdim=True)
    sd = Xtr[tr].std(0, keepdim=True).clamp(min=1e-5)
    Xd = ((Xtr[tr] - mu) / sd).to(dev)
    Xv = ((Xtr[va] - mu) / sd).to(dev)
    Xt = ((Xte - mu) / sd).to(dev)
    dd, dv = dtr[tr].to(dev), dtr[va].to(dev)

    tot_best, tot_pred = -2.0, None
    for lr in lrs:
        torch.manual_seed(seed)
        mods, d_in = [], Xtr.shape[1]
        for _ in range(layers - 1):
            mods += [nn.Linear(d_in, hidden), nn.GELU()]
            d_in = hidden
        mods += [nn.Linear(d_in, 4)]
        net = nn.Sequential(*mods).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

        best, best_state, bad = -2.0, None, 0
        steps_per_ep = max(1, len(Xd) // bs)
        for _ in range(epochs // steps_per_ep + 1):
            idx = torch.randperm(len(Xd), device=dev)
            for j in range(steps_per_ep):
                bidx = idx[j * bs:(j + 1) * bs]
                opt.zero_grad()
                F.mse_loss(net(Xd[bidx]), dd[bidx]).backward()
                opt.step()
            with torch.no_grad():
                c = _cos(net(Xv).cpu(), dv.cpu())
            if c > best + 1e-4:
                best, bad = c, 0
                best_state = {k: v.detach().clone()
                              for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None and best > tot_best:
            net.load_state_dict(best_state)
            tot_best = best
            with torch.no_grad():
                tot_pred = net(Xt).cpu()
    return tot_pred


def probe_knn(Xtr, dtr, Xte, dev, k=10, chunk=512):
    """k-NN PHI THAM SỐ — TRẦN THẬT của đặc trưng.

    Không có tham số nào để train, nên không thể đổ lỗi 'mô hình yếu'. Nếu k-NN cũng
    thấp thì đặc trưng THỰC SỰ không chứa thông tin hướng dịch, và mọi kiến trúc xây
    trên nó đều vô vọng. Chuẩn hoá L2 trước: khoảng cách cosine hợp với đặc trưng CLIP
    hơn khoảng cách Euclid trên độ lớn thô.
    """
    A = F.normalize(Xtr.to(dev), dim=1)
    B = F.normalize(Xte.to(dev), dim=1)
    dt = dtr.to(dev)
    out = []
    for i in range(0, len(B), chunk):
        sim = B[i:i + chunk] @ A.T
        _, nb = sim.topk(min(k, len(A)), dim=1)
        out.append(dt[nb].mean(1).cpu())
    return torch.cat(out)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--cache", default="../../data/cache_clip")
    ap.add_argument("--split", default="val")
    ap.add_argument("--d-cells", type=float, default=1.0,
                    help="hàng phán quyết của cửa chặn")
    ap.add_argument("--scale-jitter", type=float, default=0.0,
                    help="nhân kích thước box với exp(U(-j,j)). Mặc định 0 = giữ nguyên "
                         "như cửa chặn, khi đó delta_w/h luôn 0 và hai cột dw/dh vô nghĩa. "
                         "Đặt 0.3 để hỏi thêm 'box có biết mình to/nhỏ sai không'.")
    ap.add_argument("--max-images", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="gate_ablation.json")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = CE130Detection(cfg["data"]["root"], a.split, cfg["data"]["image_size"])
    meta = os.path.join(a.cache, f"{a.split}_meta.json")
    if not os.path.exists(meta):
        raise SystemExit(f"KHÔNG THẤY CACHE: {meta}")
    cache = PatchCache(a.cache, a.split)
    g = set_grid(cache.meta["shape"][2])

    ks, scales = [1, 3, 5, 7], [1.0, 1.5, 2.0]
    t0 = time.time()
    print(f"split={a.split}  ảnh={min(len(ds), a.max_images)}  d={a.d_cells} ô  "
          f"t=-1 (KHÔNG nhiễu)  lưới {g}x{g}  thiết bị={dev}", flush=True)
    print(f"Thu đặc trưng cho {len(ks)}x{len(scales)} cách lấy mẫu trong MỘT lượt "
          f"đọc cache...", flush=True)

    feats, d, w, pos = collect(ds, cache, a.d_cells, a.seed, dev, a.max_images,
                               ks, scales, a.scale_jitter)
    n = len(d)
    print(f"  [xong {fmt(time.time()-t0)}] {n} box", flush=True)

    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(n, generator=g)
    cut = int(n * 0.7)
    tr, te = perm[:cut], perm[cut:]
    dte = d[te]
    res = {"config": vars(a), "n_box": n, "rows": []}

    co_wh = a.scale_jitter > 0        # không jitter thì delta_w/h luôn 0, cột vô nghĩa

    def ghi(nhom, ten, pred, extra=None):
        r = {"nhom": nhom, "ten": ten,
             "cos_tam": _cos(pred, dte),
             "se": _cos_se(pred, dte),
             "cos_dx": _cos(pred[:, 0:1], dte[:, 0:1], False),
             "cos_dy": _cos(pred[:, 1:2], dte[:, 1:2], False)}
        if co_wh:
            r["cos_dw"] = _cos(pred[:, 2:3], dte[:, 2:3], False)
            r["cos_dh"] = _cos(pred[:, 3:4], dte[:, 3:4], False)
        if extra:
            r.update(extra)
        res["rows"].append(r)
        wh = (f" {r['cos_dw']:6.3f} {r['cos_dh']:6.3f}" if co_wh else "")
        flag = "  <== ĐẠT" if r["cos_tam"] - 2 * r["se"] > NGUONG else ""
        print(f"  {ten:<34} {r['cos_tam']:7.3f} ±{r['se']:5.3f} | "
              f"{r['cos_dx']:6.3f} {r['cos_dy']:6.3f}{wh}{flag}", flush=True)
        return r

    hdr = (f"  {'cấu hình':<34} {'cos tâm':>7} {'±se':>6} | {'dx':>6} {'dy':>6}"
           + (f" {'dw':>6} {'dh':>6}" if co_wh else ""))

    # -- GT 1+2: probe mạnh hơn, trên đặc trưng THÔ (không nút thắt nào) -------
    X = feats[(3, 1.0)]                                  # k=3, không nới: như model thật
    print(f"\n[1+2] PROBE MẠNH DẦN trên CLIP THÔ 9x768 (bỏ cả 2 nút thắt)\n{hdr}",
          flush=True)
    ghi("probe", "linear (như cửa chặn)", probe_linear(X[tr], d[tr], X[te], dte, dev, a.seed))
    for hid, lay in [(512, 2), (1024, 2), (1024, 3)]:
        ghi("probe", f"mlp {lay} lớp, ẩn {hid}",
            probe_mlp(X[tr], d[tr], X[te], dte, dev, a.seed, hid, lay))

    # -- GT 6: TRẦN phi tham số ------------------------------------------------
    print(f"\n[6] TRẦN PHI THAM SỐ — k-NN (không có gì để train, không đổ lỗi được)\n{hdr}",
          flush=True)
    for kk in [1, 10, 50]:
        ghi("tran", f"k-NN k={kk}", probe_knn(X[tr], d[tr], X[te], dev, kk))

    # -- GT 3+4: hình học lấy mẫu ---------------------------------------------
    print(f"\n[3+4] LƯỚI LẤY MẪU: k x k, `scale` nới ra ngoài box (probe = mlp 2x1024)\n{hdr}",
          flush=True)
    for k in ks:
        for sc in scales:
            Xk = feats[(k, sc)]
            ghi("luoi", f"k={k}, nới {sc:.1f}x  ({k*k*768}-d)",
                probe_mlp(Xk[tr], d[tr], Xk[te], dte, dev, a.seed, 1024, 2),
                {"k": k, "scale": sc, "d_in": Xk.shape[1]})

    # -- GT 7: ảnh có đóng góp gì không? --------------------------------------
    print(f"\n[7] ĐỐI CHỨNG — đặc trưng KHÔNG ẢNH (chỉ toạ độ box)\n{hdr}", flush=True)
    P = pos
    Ppe = torch.cat([P, torch.sin(P * 6.28), torch.cos(P * 6.28),
                     torch.sin(P * 25.1), torch.cos(P * 25.1)], dim=1)
    ghi("doi_chung", "chỉ toạ độ box (4-d)",
        probe_mlp(P[tr], d[tr], P[te], dte, dev, a.seed, 1024, 2))
    ghi("doi_chung", "toạ độ + PE sin/cos (20-d)",
        probe_mlp(Ppe[tr], d[tr], Ppe[te], dte, dev, a.seed, 1024, 2))
    Xzero = torch.zeros_like(X)
    ghi("doi_chung", "đặc trưng = 0 (prior thuần)",
        probe_mlp(Xzero[tr], d[tr], Xzero[te], dte, dev, a.seed, 1024, 2))
    sh = X[tr][torch.randperm(cut, generator=g)]
    ghi("doi_chung", "xáo đặc trưng",
        probe_mlp(sh, d[tr], X[te], dte, dev, a.seed, 1024, 2))

    res["total_sec"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    best = max(res["rows"], key=lambda r: r["cos_tam"])
    print(f"\n  CAO NHẤT: {best['ten']} -> cos tâm {best['cos_tam']:.3f}")
    print(f"  TIÊU CHÍ cửa chặn: > {NGUONG} (tính theo biên dưới cos-2se)")
    print("\n  ĐỌC KẾT QUẢ:")
    print("    - Nếu k-NN [6] cũng thấp: đặc trưng THỰC SỰ không chứa hướng dịch.")
    print("      Không phải lỗi mô hình -> EXPERIMENT A phải thiết kế lại.")
    print("    - Nếu mlp >> linear: cửa chặn đo bằng mô hình quá yếu, không phải")
    print("      đặc trưng yếu -> chỉ cần đổi `box_delta` thành MLP.")
    print("    - Nếu 'nới 2.0x' >> 'nới 1.0x': phải lấy mẫu CẢ NGOÀI box.")
    print("    - Nếu [7] 'chỉ toạ độ' ~ cột ảnh: CLIP không đóng góp gì ngoài mã hoá")
    print("      toạ độ -> RoI vô nghĩa, đây là kết luận NẶNG nhất.")
    print(f"  tổng {fmt(res['total_sec'])}  ->  {a.out}")


if __name__ == "__main__":
    main()

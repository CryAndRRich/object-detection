#!/usr/bin/env python3
"""CỬA CHẶN cho hướng SOFT-ARGMAX CỤC BỘ (docs/bai-toan-map-box-anh-box-va-plan-research.md
§5sexies). CLIP FROZEN, KHÔNG train gì, không cần GPU.

Ý tưởng đang kiểm: thay vì lấy kỳ vọng trên CẢ lưới (SpatialSoftmax gốc -> nhiều vật thì
rơi vào trọng tâm chung, vô dụng cho detect), lấy soft-argmax trong một CỬA SỔ quanh TỪNG
box. Mỗi box một cửa sổ -> N toạ độ độc lập.

    B1.  S      = cosine(patch, exemplar)                       [g,g]
    B2.  S_n    = crop(S, tâm=(cx,cy), kích thước=kappa*(w,h))  [k,k]
    B3.  a_n    = softmax(tau * S_n);  (dx,dy) = sum a_n*(u,v)
    B4.  cx += dx * w/2 * kappa   (object-normalized, dạng delta2bbox)

⚠️ VÌ SAO PHẢI CÓ VÒNG LẶP (§5sexies.5bis): chạy MỘT lần lên x_T (nhiễu thuần) thì cửa sổ
rơi vào chỗ ngẫu nhiên -> S_n phẳng -> (dx,dy) ~ 0 -> KHÔNG dịch đi đâu. Cơ chế chỉ có
nghĩa khi box đã tạm đúng. Nhưng "đọc box sau mỗi layer" ĐÃ TỒN TẠI và ĐÃ TRAIN -- là C1,
và C1 TỆ DẦN (oracle_recall 0,1384 -> 0,1328). Nên phép đo (2) dưới đây mới là phép phân
biệt thật: cùng cấu trúc vòng lặp, khác NGUỒN của bước cập nhật.

BA PHÉP ĐO, chạy theo thứ tự:
  (1) --mode curve  : đường cong theo t, MỘT vòng, từ GT nhiễu hoá -> tìm ngưỡng t*.
                      t nhỏ mà KHÔNG cải thiện => ý tưởng sai TỪ GỐC, dừng, khỏi viết model.
  (2) --mode loop   : lặp R vòng với kappa(t) giảm dần, xuất phát t=T (nhiễu thuần).
                      oracle_recall tăng đơn điệu hay tệ dần như C1?
  (3) --mode modes  : đếm box trùng đỉnh -> định lượng rủi ro SỤP MODE (§5sexies.7 mục 5).

⚠️ ĐỌC ĐÚNG KẾT QUẢ (cùng hạn chế như tools/check_exemplar_signal.py và
tools/check_vertex_rpe.py): phép đo này nói "cơ chế có kéo box về đúng chỗ không", KHÔNG
nói "train xong sẽ tốt hơn". Nó cũng CHỈ sửa TÂM -- w,h giữ nguyên (§5sexies.7 mục 4).

Chạy:
  .venv-cpu/bin/python tools/check_local_softargmax.py --mode curve
  .venv-cpu/bin/python tools/check_local_softargmax.py --mode loop
  .venv-cpu/bin/python tools/check_local_softargmax.py --mode modes
  ... thêm --exemplar text để đo rủi ro (2): exemplar lấy đâu lúc inference
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as Fnn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection, normalize_for_clip  # noqa: E402
from models.detector import build_model  # noqa: E402
from utils.diffusion_math import cosine_alphas_cumprod  # noqa: E402


# ----------------------------------------------------------------------------- đo
def oracle_recall(pred, gt, thr=0.5):
    """Cùng định nghĩa với train.py::oracle_recall và tools/measure_box_quality.py.
    KHÔNG dùng score, KHÔNG dùng matcher -- đúng thứ `loss_final` không nhìn thấy."""
    if gt.numel() == 0:
        return 0.0, 0
    if pred.numel() == 0:
        return 0.0, int(gt.shape[0])
    iou = _iou(_xyxy(pred), _xyxy(gt))                       # [P,G]
    return float((iou.max(dim=0).values >= thr).sum()), int(gt.shape[0])


def mean_best_iou(pred, gt):
    if gt.numel() == 0 or pred.numel() == 0:
        return 0.0
    return float(_iou(_xyxy(pred), _xyxy(gt)).max(dim=0).values.mean())


def _xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def _iou(a, b):
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


# ------------------------------------------------------------------- cơ chế B1-B4
def sim_map(patch, exemplar_vec):
    """B1: bản đồ tương đồng [g,g] trên lưới patch. patch đã chuẩn hoá L2."""
    g = int(round(patch.shape[0] ** 0.5))
    return (patch @ exemplar_vec).reshape(g, g)


def soft_argmax_local(S, boxes, kappa, tau, k=7):
    """B2-B3: cắt cửa sổ kappa*(w,h) quanh TỪNG box rồi lấy kỳ vọng trong cửa sổ đó.

    S     : [g,g]  bản đồ tương đồng
    boxes : [N,4]  cxcywh trong [0,1]
    ->      [N,2]  (dx,dy) TƯƠNG ĐỐI trong cửa sổ, mỗi giá trị trong [-1,1]

    Dùng grid_sample nên KHẢ VI theo cx,cy,w,h -- điểm khác biệt với việc cắt bằng
    chỉ số nguyên (không có gradient về toạ độ).
    """
    N = boxes.shape[0]
    cx, cy, w, h = boxes.unbind(-1)
    # nửa-chiều cửa sổ trong hệ chuẩn hoá [-1,1] của grid_sample
    hw = (kappa * w).clamp(min=1e-3)
    hh = (kappa * h).clamp(min=1e-3)

    lin = torch.linspace(-1, 1, k, dtype=S.dtype)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")               # [k,k]
    # [-1,1] của grid_sample ứng với toàn ảnh; box ở [0,1] -> 2*c-1
    px = (2 * cx - 1)[:, None, None] + gx[None] * hw[:, None, None]
    py = (2 * cy - 1)[:, None, None] + gy[None] * hh[:, None, None]
    grid = torch.stack([px, py], dim=-1)                           # [N,k,k,2]

    win = Fnn.grid_sample(
        S[None, None].expand(N, -1, -1, -1), grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    )[:, 0]                                                        # [N,k,k]

    a = torch.softmax((tau * win).reshape(N, -1), dim=-1).reshape(N, k, k)
    dx = (a * gx[None]).sum((1, 2))
    dy = (a * gy[None]).sum((1, 2))

    # HỆ SỐ TIN CẬY -- BẮT BUỘC, không phải tinh chỉnh.
    # Bản đồ PHẲNG trong cửa sổ (không có đỉnh nào) vẫn cho (dx,dy) khác 0 chỉ vì
    # nhiễu, và bước dịch đó PHÁ box đang đúng. Đo được ở lần chạy đầu: t=0 (box = GT,
    # oracle_recall 0,9960) bị kéo xuống 0,3470 -- cơ chế phá chính đầu vào hoàn hảo.
    # `conf` = chênh lệch đỉnh-so-với-trung-bình trong cửa sổ, chuẩn hoá theo std của
    # S trên TOÀN ảnh: cửa sổ phẳng -> conf ~ 0 -> không dịch; cửa sổ có đỉnh rõ ->
    # conf ~ 1 -> dịch hết. Đây chính là thứ SpatialSoftmax gốc KHÔNG có (nó luôn trả
    # về một toạ độ, kể cả khi bản đồ vô nghĩa).
    spread = win.amax(dim=(1, 2)) - win.mean(dim=(1, 2))
    conf = (spread / (S.std() + 1e-6)).clamp(0, 1)
    return torch.stack([dx, dy], -1) * conf[:, None], win


def step(S, boxes, kappa, tau, k=7):
    """B4: cập nhật TÂM, object-normalized (dạng delta2bbox của V-DETR/Plain-DETR,
    đã verify tương đương update_box của C1, diff 0,0). w,h GIỮ NGUYÊN."""
    d, win = soft_argmax_local(S, boxes, kappa, tau, k)
    out = boxes.clone()
    out[:, 0] = (boxes[:, 0] + d[:, 0] * kappa * boxes[:, 2]).clamp(0, 1)
    out[:, 1] = (boxes[:, 1] + d[:, 1] * kappa * boxes[:, 3]).clamp(0, 1)
    return out, win


# ------------------------------------------------------------------------ dữ liệu
def load(cfg_path, split, limit, min_gt=4):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    enc = build_model(cfg, dropout=0.0).eval().encoder
    ds = CE130Detection("../../data/all_phase2_V2", split)
    g = cfg["data"]["image_size"] // 16

    out = []
    for i in range(len(ds.items)):
        if len(out) >= limit:
            break
        it = ds[i]
        gt = it["boxes"]
        if len(gt) < min_gt:
            continue
        px = torch.from_numpy(normalize_for_clip(it["image"])).unsqueeze(0)
        with torch.no_grad():
            patch = enc.encode_image_raw(px)[0]
        F = torch.nn.functional.normalize(patch.float(), dim=-1)
        out.append((F, torch.as_tensor(gt, dtype=torch.float32), it.get("text", ""), enc, g))
    return out, g


def exemplar_vec(F, gt, g, kind, enc):
    """Vector để correlate. `gt` = vật ĐẦU TIÊN làm mẫu (rủi ro (2): lúc inference
    chưa có box đúng nào -> đo thêm biến thể text)."""
    if kind == "text":
        raise SystemExit("--exemplar text: cần text tower, chạy riêng (chưa nối ở bản này)")
    b = gt[0]
    cx, cy = int(np.clip(b[0] * g, 0, g - 1)), int(np.clip(b[1] * g, 0, g - 1))
    return F[cy * g + cx]


def noisy(gt, t, alphas, scale, rng):
    """Nhiễu hoá GT đúng cách q_sample của dự án: đưa về [-scale,scale] rồi thêm nhiễu."""
    a = alphas[t]
    x0 = (gt * 2.0 - 1.0) * scale
    x = a.sqrt() * x0 + (1 - a).sqrt() * torch.randn(gt.shape, generator=rng) * scale
    x = x.clamp(-scale, scale)
    return ((x / scale) + 1) / 2.0


# ------------------------------------------------------------------------- 3 phép
def mode_curve(data, args, alphas):
    print("PHÉP (1) — ĐƯỜNG CONG THEO t, MỘT vòng, từ GT nhiễu hoá")
    print("  Câu hỏi: ở t nào thì cơ chế còn kéo box về đúng chỗ? (ngưỡng t*)")
    print("  KHÔNG ĐẠT nếu t NHỎ mà không cải thiện => sai từ gốc, dừng.\n")
    print(f"  {'t':>5} | {'oracle_recall':^21} | {'mean_bestIoU':^21}")
    print(f"  {'':>5} | {'trước':>9} {'sau':>9} | {'trước':>9} {'sau':>9}")
    print("  " + "-" * 54)

    rows = []
    for t in args.timesteps:
        rng = torch.Generator().manual_seed(args.seed)
        c0 = c1 = n = 0
        i0 = i1 = 0.0
        for F, gt, _, enc, g in data:
            S = sim_map(F, exemplar_vec(F, gt, g, args.exemplar, enc))
            x = noisy(gt, t, alphas, args.snr_scale, rng)
            y, _ = step(S, x, args.kappa, args.tau, args.k)
            a, ng = oracle_recall(x, gt)
            b, _ = oracle_recall(y, gt)
            c0 += a; c1 += b; n += ng
            i0 += mean_best_iou(x, gt); i1 += mean_best_iou(y, gt)
        m = len(data)
        rows.append((t, c0 / n, c1 / n, i0 / m, i1 / m))
        print(f"  {t:>5} | {c0/n:>9.4f} {c1/n:>9.4f} | {i0/m:>9.4f} {i1/m:>9.4f}")

    small = [r for r in rows if r[0] <= args.t_small]
    ok = any(r[2] > r[1] + 1e-4 for r in small)
    print(f"\n  => {'ĐẠT' if ok else 'KHÔNG ĐẠT'}: ở t <= {args.t_small}, cơ chế "
          f"{'CÓ' if ok else 'KHÔNG'} cải thiện oracle_recall")
    if not ok:
        print("     (KHÔNG ĐẠT = ý tưởng sai TỪ GỐC, không phải chỉnh tham số)")
    return ok


def mode_loop(data, args, alphas):
    print("PHÉP (2) — LẶP nhiều vòng, kappa(t) GIẢM DẦN, xuất phát t=T (nhiễu thuần)")
    print("  Đây là phép PHÂN BIỆT với C1: cùng cấu trúc vòng lặp, khác NGUỒN cập nhật.")
    print(f"  C1 đã train: oracle_recall TỆ DẦN 0,1384 -> 0,1328 qua 6 vòng.\n")
    print(f"  {'vòng':>5} | {'kappa':>6} | {'oracle_recall':>13} | {'mean_bestIoU':>12}")
    print("  " + "-" * 48)

    rng = torch.Generator().manual_seed(args.seed)
    T = len(alphas) - 1
    states = []
    for F, gt, _, enc, g in data:
        S = sim_map(F, exemplar_vec(F, gt, g, args.exemplar, enc))
        states.append((S, noisy(gt, T, alphas, args.snr_scale, rng), gt))

    hist = []
    for r in range(args.rounds + 1):
        c = n = 0
        iou = 0.0
        for S, x, gt in states:
            a, ng = oracle_recall(x, gt)
            c += a; n += ng; iou += mean_best_iou(x, gt)
        hist.append(c / n)
        kap = args.kappa_max - (args.kappa_max - args.kappa) * r / max(args.rounds, 1)
        print(f"  {r:>5} | {kap:>6.2f} | {c/n:>13.4f} | {iou/len(states):>12.4f}"
              + ("   <- khởi tạo (t=T, nhiễu thuần)" if r == 0 else ""))
        if r == args.rounds:
            break
        states = [(S, step(S, x, kap, args.tau, args.k)[0], gt) for S, x, gt in states]

    up = hist[-1] > hist[0] + 1e-4
    mono = all(hist[i + 1] >= hist[i] - 1e-4 for i in range(len(hist) - 1))
    print(f"\n  tăng tổng thể: {'CÓ' if up else 'KHÔNG'}   đơn điệu: {'CÓ' if mono else 'KHÔNG'}")
    print(f"  => {'ĐẠT' if up else 'KHÔNG ĐẠT'}: vòng lặp "
          f"{'hội tụ về phía GT' if up else 'KHÔNG hội tụ (giống C1 tệ dần)'}")
    return up


def mode_modes(data, args, alphas):
    print("PHÉP (3) — SỤP MODE (rủi ro §5sexies.7 mục 5)")
    print("  soft-argmax chỉ kéo box về ĐỈNH GẦN NHẤT; CE-130 có 20-30 vật CÙNG class")
    print("  => nhiều box có thể trượt về CÙNG một đỉnh. Chưa có matcher/NMS nào chống.\n")
    print(f"  {'vòng':>5} | {'#tâm phân biệt':>14} | {'/ #box':>8} | {'tỉ lệ':>7}")
    print("  " + "-" * 44)

    rng = torch.Generator().manual_seed(args.seed)
    T = len(alphas) - 1
    states = []
    for F, gt, _, enc, g in data:
        S = sim_map(F, exemplar_vec(F, gt, g, args.exemplar, enc))
        states.append((S, noisy(gt, T, alphas, args.snr_scale, rng), gt, g))

    for r in range(args.rounds + 1):
        uniq = tot = 0
        for S, x, gt, g in states:
            cell = (x[:, 1] * g).long().clamp(0, g - 1) * g + (x[:, 0] * g).long().clamp(0, g - 1)
            uniq += len(torch.unique(cell)); tot += x.shape[0]
        print(f"  {r:>5} | {uniq:>14} | {tot:>8} | {uniq/tot:>7.3f}"
              + ("   <- khởi tạo" if r == 0 else ""))
        if r == args.rounds:
            break
        kap = args.kappa_max - (args.kappa_max - args.kappa) * r / max(args.rounds, 1)
        states = [(S, step(S, x, kap, args.tau, args.k)[0], gt, g) for S, x, gt, g in states]
    print("\n  Tỉ lệ giảm mạnh qua các vòng = SỤP MODE. Đây KHÔNG bác ý tưởng, nhưng")
    print("  nghĩa là soft-argmax không thể là cơ chế duy nhất -- cần box<->box (C3).")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["curve", "loop", "modes"], default="curve")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--exemplar", choices=["object", "text"], default="object")
    ap.add_argument("--kappa", type=float, default=1.5, help="nửa-chiều cửa sổ / kích thước box, ở t NHỎ")
    ap.add_argument("--kappa-max", type=float, default=6.0, help="ở t LỚN (cửa sổ rộng ~ toàn cục)")
    ap.add_argument("--tau", type=float, default=30.0)
    ap.add_argument("--k", type=int, default=5,
                    help="số điểm mẫu mỗi chiều trong cửa sổ. Box CE-130 trung vị "
                         "0,069x0,061 = 2,2x1,95 Ô LƯỚI, nên k lớn chỉ nội suy song "
                         "tuyến chứ không thêm thông tin (đo được: spread trong cửa sổ "
                         "0,159 ~ đúng bằng std toàn ảnh 0,158 = phẳng).")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--timesteps", type=int, nargs="+",
                    default=[0, 100, 200, 400, 600, 800, 999])
    ap.add_argument("--t-small", type=int, default=200)
    ap.add_argument("--snr-scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    with open(a.config) as f:
        a.snr_scale = yaml.safe_load(f).get("diffusion", {}).get("snr_scale", a.snr_scale)

    data, g = load(a.config, a.split, a.limit)
    alphas = cosine_alphas_cumprod(1000).float()

    print(f"CỬA CHẶN SOFT-ARGMAX CỤC BỘ — CLIP frozen, CHƯA train gì")
    print(f"  split={a.split}  n={len(data)} ảnh  lưới {g}x{g}  k={a.k}  tau={a.tau}")
    print(f"  kappa {a.kappa} (t nhỏ) .. {a.kappa_max} (t lớn)  exemplar={a.exemplar}")
    print(f"  snr_scale={a.snr_scale}\n")

    ok = {"curve": mode_curve, "loop": mode_loop, "modes": mode_modes}[a.mode](data, a, alphas)
    print("\n  Đối chiếu số đã có: C1 oracle_recall 0,1384 | E1 0,2559 | D.1 0,6734")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

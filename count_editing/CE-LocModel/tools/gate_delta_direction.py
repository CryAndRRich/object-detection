#!/usr/bin/env python3
"""CỬA CHẶN cho EXPERIMENT A — `Linear(256->4)` có đoán được HƯỚNG DỊCH không?

VÌ SAO TỒN TẠI
--------------
Cả EXPERIMENT A dựa trên đúng một giả định chưa ai đo:

    delta = box_delta(h)        # nn.Linear(256, 4)
    x     = update_box(x, delta)

Đây ĐÚNG DẠNG câu hỏi mà cửa chặn soft-argmax của vòng 1 đã đo và TRƯỢT: rút toạ độ
từ đặc trưng CLIP frozen (sai số trung vị 1,34 ô lưới, hơn lưới đều chỉ 0,15 ô).

Khác biệt duy nhất, và là lý do đáng thử lại: ở đây mạng không rút toạ độ TỪ ĐẦU mà rút
PHẦN DƯ so với box đã có, sau khi đã đọc ảnh TẠI CHÍNH BOX ĐÓ. Câu hỏi đổi từ
"vật nằm ở đâu trong ảnh?" thành "nhìn vùng này, box đang lệch sang trái hay phải?".

Lập luận "dễ hơn" đó CHƯA ĐƯỢC ĐO. Nếu sai thì mọi thứ xây trên nó vô nghĩa — và đo mất
vài phút, còn train mất ~10 giờ A30 (cạm bẫy 8 trong CLAUDE.md).

CÁCH ĐO
-------
1. Lấy GT box của val (28 lớp rời train, không dính lô annotation rác của test).
2. Dịch mỗi box đi một vector ngẫu nhiên độ dài `d` ô lưới -> `box_lech`.
3. `r = RoIFeatureSampler(patch_raw, box_lech)` với CLIP FROZEN (đọc từ cache).
4. Train `Linear(256->4)` dự đoán delta đưa box về GT (đúng tham số hoá của
   `update_box`: dịch tâm theo đơn vị kích thước box, kích thước nhân theo exp).
5. Đo cosine giữa delta dự đoán và delta thật, trên tập val giữ riêng.

TIÊU CHÍ: cosine > 0,5 trên cột `cos2304`, ở hàng `t = -1, d = 1` ô.
Dưới ngưỡng ⇒ cộng dồn vô nghĩa, DỪNG, thiết kế lại.

HAI LỖI ĐO CỦA BẢN ĐẦU (lần chạy 2026-09-23 cho cosine 0,086 — KHÔNG dùng để phán quyết)
----------------------------------------------------------------------------------------
1. `add_diffusion_noise` TỰ NÓ dịch tâm box 2,12 ô ở `t=249` và 6,64 ô ở `t=999`, trong
   khi `d` cố ý gây ra chỉ 0,5–4 ô. Biến `d` bị nuốt: đo được tương quan giữa delta thật
   và hướng dịch còn **−0,009** ở `t=999` (alpha_bar = 0,00000 ⇒ box là nhiễu thuần), nên
   cả 4 mức `d` cho kết quả TRÙNG KHÍT. Câu hỏi bị đổi thành định vị tuyệt đối — đúng câu
   cửa chặn soft-argmax vòng 1 đã trượt. ⇒ thêm `t = -1` và cột `dịch thật`.
2. `sampler.out` bị đóng băng NGẪU NHIÊN, thành nút thắt 2304→256 mà model thật không có
   (ở đó lớp này ĐƯỢC HỌC). Trên tín hiệu tuyến tính hoàn hảo, phép chiếu ấy kéo cosine
   **0,966 → 0,233** — xấp xỉ đúng con số bảng cũ đo được. ⇒ chấm thêm cột `cos2304`
   TRƯỚC nút thắt, và lấy chính cột đó làm tiêu chí.

ĐÃ LOẠI TRỪ, không phải lỗi: 400 bước AdamW là đủ hội tụ (0,896 so với trần lstsq 0,900
trên dữ liệu tổng hợp) — vẫn in cột `lstsq` mỗi hàng để luôn thấy trần.

KHÔNG train model thật, KHÔNG cần GPU cho CLIP (đọc cache).

CHẠY TRÊN SERVER
----------------
  cd object-detection/count_editing/CE-LocModel
  python tools/run_on_free_gpu.py -- tools/gate_delta_direction.py \
      --cache ../../data/cache_clip --split val \
      --out /mnt/disk1/aiotlab/haitn/output/round2_gate_delta.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection, PatchCache          # noqa: E402
from models.dit_blocks import MIN_WH                               # noqa: E402
from models.roi_sampler import RoIFeatureSampler, box_grid_points  # noqa: E402

GRID = 32                       # ViT-B/16 ở 512px -> lưới 32x32


def fmt(sec):
    """12.3 -> '12s'; 185 -> '3m05s'. In thời gian THỰC TẾ ở mọi giai đoạn để biết
    tiến trình còn sống — đọc cache nguội có thể mất vài phút mà không in gì."""
    sec = int(max(sec, 0))
    m, s = sec // 60, sec % 60
    return f"{m}m{s:02d}s" if m else f"{s}s"


def true_delta(box_lech, box_gt):
    """Delta mà `update_box` cần để đưa `box_lech` về `box_gt`.

    Đảo ngược đúng công thức object-normalized của V-DETR:
        cx' = cx + d_cx * w   ->   d_cx = (cx' - cx) / w
        w'  = w * exp(d_w)    ->   d_w  = log(w' / w)
    Dùng cùng `MIN_WH` như `update_box` để hai bên nhất quán.
    """
    cx, cy, w, h = box_lech.unbind(-1)
    gx, gy, gw, gh = box_gt.unbind(-1)
    w = w.clamp(min=MIN_WH)
    h = h.clamp(min=MIN_WH)
    return torch.stack([
        (gx - cx) / w,
        (gy - cy) / h,
        torch.log(gw.clamp(min=MIN_WH) / w),
        torch.log(gh.clamp(min=MIN_WH) / h),
    ], dim=-1)


def perturb(boxes, d_cells, rng):
    """Dịch tâm box đi đúng `d_cells` ô lưới theo hướng ngẫu nhiên. Kích thước giữ
    nguyên: cửa chặn hỏi về HƯỚNG DỊCH, thêm nhiễu kích thước sẽ trộn hai câu hỏi."""
    ang = torch.from_numpy(rng.uniform(0, 2 * np.pi, len(boxes))).float()
    step = d_cells / GRID
    out = boxes.clone()
    out[:, 0] = (boxes[:, 0] + step * torch.cos(ang)).clamp(0.0, 1.0)
    out[:, 1] = (boxes[:, 1] + step * torch.sin(ang)).clamp(0.0, 1.0)
    return out


def add_diffusion_noise(boxes, t, alphas_cumprod, snr_scale, rng):
    """Đưa box qua đúng đường mà `x_t` đi lúc train: encode -> q_sample -> decode.

    Cần thiết vì `update_box` nhân delta với `w`, mà ở `t` lớn `w` sau decode có đuôi
    chạm 0 (mục 7.1b của kế hoạch). Chỉ dịch `d` ô mà bỏ qua bước này thì cửa chặn sẽ
    đo một chế độ mà mô hình thật không bao giờ gặp.
    """
    from utils.box_ops import decode_diffusion, encode_diffusion
    x0 = encode_diffusion(boxes, snr_scale)
    noise = torch.from_numpy(rng.standard_normal(x0.shape)).float()
    a = alphas_cumprod[t]
    x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise
    return decode_diffusion(x_t, snr_scale)


@torch.no_grad()
def sample_roi(sampler, praw, box):
    """-> (feat_2304 [n,k*k*d_model], r_256 [n,d_model]).

    Trả về CẢ HAI phía của `sampler.out`. Lý do: trong cửa chặn `out` bị đóng băng
    NGẪU NHIÊN, mà đo được phép chiếu ngẫu nhiên 2304->256 làm cosine tụt 0,966 -> 0,233
    ngay cả khi quan hệ tuyến tính là hoàn hảo. Trong model thật lớp đó ĐƯỢC HỌC nên
    không có nút thắt ấy. Chấm cả hai để tách "CLIP có tín hiệu không" khỏi
    "phép chiếu ngẫu nhiên làm mất bao nhiêu".
    """
    B, P, d_in = praw.shape
    g = int(round(P ** 0.5))
    fmap = praw.transpose(1, 2).reshape(B, d_in, g, g)
    pts = box_grid_points(box.unsqueeze(0).to(praw.device), sampler.k)
    samp = torch.nn.functional.grid_sample(fmap, pts * 2.0 - 1.0, mode="bilinear",
                                           padding_mode="border", align_corners=False)
    samp = samp.permute(0, 2, 3, 1)                                  # [1,n,k*k,d_in]
    feat = sampler.proj_point(samp.to(sampler.proj_point.weight.dtype)).flatten(-2)
    return feat[0], sampler.out(feat)[0]


@torch.no_grad()
def collect(ds, cache, sampler, d_cells, t, alphas, snr_scale, seed, dev, max_img):
    """-> (feat [K,2304], r [K,256], delta_thật [K,4], rộng_ô [K], dịch_thật_ô [K])."""
    rng = np.random.default_rng(seed)
    F_, R, D, W, S = [], [], [], [], []
    for i in range(min(len(ds), max_img)):
        s = ds.__getitem__(i, need_image=False)
        gt = torch.as_tensor(s["boxes"], dtype=torch.float32)
        if gt.numel() == 0:
            continue
        patch, _ = cache.get(s["image_id"], s["text"], False)
        praw = torch.from_numpy(patch).unsqueeze(0).to(dev)          # [1,T,768]

        box = perturb(gt, d_cells, rng)
        if t is not None:
            box = add_diffusion_noise(box, t, alphas, snr_scale, rng)
        feat, r = sample_roi(sampler, praw, box)
        F_.append(feat.cpu())
        R.append(r.cpu())
        D.append(true_delta(box, gt))
        W.append(gt[:, 2] * GRID)
        # Độ dịch tâm THỰC TẾ sau cả perturb lẫn nhiễu khuếch tán. Phải đo, vì ở
        # t=249 riêng nhiễu đã dịch ~2,12 ô — lấn át hoàn toàn `d` mà ta cố ý gây ra.
        S.append((box[:, :2] - gt[:, :2]).norm(dim=-1) * GRID)
    return torch.cat(F_), torch.cat(R), torch.cat(D), torch.cat(W), torch.cat(S)


def _cos_stats(pred, d_te, w_te):
    """Cosine trên 2 kênh TÂM (đó là 'hướng dịch'; 2 kênh kích thước là câu hỏi khác),
    kèm tách theo nhóm kích thước."""
    cos = nn.functional.cosine_similarity(pred[:, :2], d_te[:, :2], dim=-1)
    small = w_te < 1.0                      # nhỏ hơn MỘT ô lưới
    mid = (w_te >= 1.0) & (w_te < 3.0)
    big = w_te >= 3.0
    grp = lambda m: float(cos[m].mean()) if m.any() else float("nan")  # noqa: E731
    return {
        "cosine": float(cos.mean()),
        "cosine_median": float(cos.median()),
        "pct_dung_huong": float((cos > 0).float().mean()),
        "cosine_box_nho": grp(small), "n_box_nho": int(small.sum()),
        "cosine_box_vua": grp(mid), "n_box_vua": int(mid.sum()),
        "cosine_box_to": grp(big), "n_box_to": int(big.sum()),
    }


def fit_and_score(r_tr, d_tr, r_te, d_te, w_te, epochs, lr, dev, seed):
    """Train `Linear(d->4)` trên tập train, chấm cosine trên tập test.

    Cosine chứ không phải MSE: cửa chặn hỏi về HƯỚNG. Một mô hình đoán đúng hướng nhưng
    sai độ lớn vẫn dùng được (6 tầng cộng dồn sẽ bù dần); đoán sai hướng thì không.

    Chấm HAI cách. AdamW là cách model thật học; `lstsq` là nghiệm bình phương tối thiểu
    chính xác, tức TRẦN của mọi ánh xạ tuyến tính. Có trần thì mới phân biệt được
    "đặc trưng không mang tín hiệu" với "tối ưu chưa hội tụ" — đã kiểm trên dữ liệu tổng
    hợp: 400 bước đạt 0,896 so với trần 0,900, nên epochs không phải chỗ nghi ngờ.
    """
    torch.manual_seed(seed)
    head = nn.Linear(r_tr.shape[1], 4).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    r_tr_d, d_tr_d = r_tr.to(dev), d_tr.to(dev)
    for _ in range(epochs):
        opt.zero_grad()
        nn.functional.mse_loss(head(r_tr_d), d_tr_d).backward()
        opt.step()
    with torch.no_grad():
        pred = head(r_te.to(dev)).cpu()
    out = _cos_stats(pred, d_te, w_te)

    # TRẦN tuyến tính: thêm cột hằng số để có bias, giải trên float64 cho ổn định.
    try:
        one = torch.ones(len(r_tr), 1, dtype=torch.float64)
        A = torch.cat([r_tr.double(), one], dim=1)
        sol = torch.linalg.lstsq(A, d_tr.double()).solution
        Ate = torch.cat([r_te.double(), torch.ones(len(r_te), 1, dtype=torch.float64)], 1)
        out["cosine_lstsq"] = _cos_stats((Ate @ sol).float(), d_te, w_te)["cosine"]
    except Exception as e:                                   # ma trận suy biến
        out["cosine_lstsq"] = float("nan")
        out["lstsq_error"] = str(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--cache", default="../../data/cache_clip", help="thư mục cache patch token")
    ap.add_argument("--split", default="val",
                    help="val: 28 lớp rời train, không có lô annotation rác của test")
    ap.add_argument("--d-cells", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--timesteps", type=int, nargs="+", default=[-1, 249, 499, 749, 999],
                    help="-1 = KHÔNG thêm nhiễu khuếch tán. PHẢI có -1: nhiễu tự nó "
                         "dịch tâm 2,12 ô ở t=249 và 6,64 ô ở t=999, lấn át biến `d`; "
                         "ở t=999 alpha_bar=0 nên mọi mức `d` cho cùng một kết quả.")
    ap.add_argument("--max-images", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="gate_delta.json")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from utils.diffusion_math import cosine_alphas_cumprod
    alphas = cosine_alphas_cumprod(cfg["diffusion"]["num_timesteps"]).to(torch.float32)
    snr = cfg["diffusion"]["snr_scale"]

    ds = CE130Detection(cfg["data"]["root"], a.split, cfg["data"]["image_size"])

    meta = os.path.join(a.cache, f"{a.split}_meta.json")
    if not os.path.exists(meta):
        raise SystemExit(
            f"\nKHÔNG THẤY CACHE cho split '{a.split}': {meta}\n\n"
            f"Vòng 1 chỉ build cache cho split nào nó train, nên '{a.split}' có thể "
            f"chưa bao giờ được sinh. Chạy:\n\n"
            f"  LOG=/mnt/disk1/aiotlab/haitn/log/cache_{a.split}_$(date +%m%d_%H%M).log\n"
            f"  nohup python tools/run_on_free_gpu.py -- tools/build_cache.py \\\n"
            f"      --config {a.config} --split {a.split} --out {a.cache} \\\n"
            f"      > $LOG 2>&1 &\n"
            f"  echo \"PID $! -> $LOG\"\n")
    cache = PatchCache(a.cache, a.split)
    torch.manual_seed(a.seed)
    sampler = RoIFeatureSampler(768, cfg["model"]["d_model"],
                                cfg["model"]["roi_k"], 0.0).to(dev).eval()
    # `roi.out` vốn zero-init để thí nghiệm bắt đầu từ 0 — ở đây phải khởi tạo thường,
    # nếu không `r` luôn bằng 0 và cửa chặn đo trên một hằng số.
    nn.init.xavier_uniform_(sampler.out.weight)
    nn.init.zeros_(sampler.out.bias)

    n_img = min(len(ds), a.max_images)
    n_row = len(a.d_cells) * len(a.timesteps)
    print(f"split={a.split}  ảnh={n_img}  thiết bị={dev}  |  {n_row} cấu hình "
          f"({len(a.d_cells)} mức d x {len(a.timesteps)} mức t)", flush=True)
    print(f"Mỗi cấu hình: đọc cache {n_img} ảnh -> train 3 x Linear({a.epochs} bước).",
          flush=True)
    print(f"{'d(ô)':>6} {'t':>5} {'dịch thật':>10} | {'cos256':>7} {'lstsq':>7} "
          f"{'cos2304':>8} {'lstsq':>7} | {'nhỏ<1ô':>8} {'vừa':>7} {'to':>7} | "
          f"{'xáo':>6} {'r=0':>6} {'thời gian':>9}", flush=True)
    print(f"{'':>6} {'':>5} {'(ô lưới)':>10} | {'-- sau nút thắt --':^15} "
          f"{'-- TRƯỚC nút thắt --':^16} | {'-- theo cỡ (2304) --':^24} | "
          f"{'đối chứng':^13}", flush=True)

    res = {"config": vars(a), "rows": []}
    t_all = time.time()
    done = 0
    for d_cells in a.d_cells:
        for t in a.timesteps:
            t_row = time.time()
            tt = None if t < 0 else t
            feat, r, d, w, shift = collect(ds, cache, sampler, d_cells, tt, alphas,
                                           snr, a.seed, dev, a.max_images)
            t_collect = time.time() - t_row
            if done == 0:
                # Lần đầu là lần lâu nhất (đọc memmap nguội), in ngay để biết còn sống.
                print(f"  [đọc cache lần đầu: {fmt(t_collect)} cho {len(r)} box]",
                      flush=True)
            n = len(r)
            cut = int(n * 0.7)
            g = torch.Generator().manual_seed(a.seed)
            perm = torch.randperm(n, generator=g)
            tr, te = perm[:cut], perm[cut:]

            fit = lambda X: fit_and_score(X[tr], d[tr], X[te], d[te], w[te],  # noqa: E731
                                          a.epochs, a.lr, dev, a.seed)
            real = fit(r)                      # 256-d, SAU nút thắt ngẫu nhiên
            raw = fit(feat)                    # 2304-d, TRƯỚC nút thắt
            # ĐỐI CHỨNG 1: xáo `r` giữa các mẫu -> phá liên hệ ảnh<->delta.
            shuf = fit_and_score(r[tr][torch.randperm(cut, generator=g)], d[tr],
                                 r[te], d[te], w[te], a.epochs, a.lr, dev, a.seed)
            # ĐỐI CHỨNG 2: không có ảnh. Trần của việc đoán mò theo prior.
            zero = fit_and_score(torch.zeros_like(r[tr]), d[tr],
                                 torch.zeros_like(r[te]), d[te], w[te],
                                 a.epochs, a.lr, dev, a.seed)

            done += 1
            dt = time.time() - t_row
            eta = (time.time() - t_all) / done * (n_row - done)
            sh_med = float(shift.median())
            row = {"d_cells": d_cells, "t": t, "n_box": n, "sec": dt,
                   "dich_that_o_median": sh_med,
                   "real": real, "raw2304": raw, "shuffled": shuf, "zero": zero}
            res["rows"].append(row)
            print(f"{d_cells:6.1f} {t:5d} {sh_med:10.2f} | "
                  f"{real['cosine']:7.3f} {real['cosine_lstsq']:7.3f} "
                  f"{raw['cosine']:8.3f} {raw['cosine_lstsq']:7.3f} | "
                  f"{raw['cosine_box_nho']:8.3f} {raw['cosine_box_vua']:7.3f} "
                  f"{raw['cosine_box_to']:7.3f} | "
                  f"{shuf['cosine']:6.3f} {zero['cosine']:6.3f} {fmt(dt):>9}"
                  f"  [{done}/{n_row}, còn ~{fmt(eta)}]", flush=True)

    res["total_sec"] = time.time() - t_all
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print()
    print("  ĐỌC KẾT QUẢ:")
    print("    - HÀNG PHÁN QUYẾT là t=-1, d=1.0 (không nhiễu khuếch tán). Chỉ hàng đó")
    print("      hỏi đúng câu 'box lệch 1 ô, biết lệch hướng nào không?'. Các hàng t>=0")
    print("      bị nhiễu dịch tâm thêm 2-7 ô, biến câu hỏi thành ĐỊNH VỊ TUYỆT ĐỐI —")
    print("      đúng câu mà cửa chặn soft-argmax vòng 1 ĐÃ TRƯỢT. Xem cột 'dịch thật'.")
    print("    - TIÊU CHÍ: cos2304 > 0,5. Dùng cột 2304 (TRƯỚC nút thắt) chứ không phải")
    print("      256, vì ở đây `sampler.out` bị đóng băng NGẪU NHIÊN còn trong model thật")
    print("      nó ĐƯỢC HỌC. Đo trên tín hiệu tuyến tính hoàn hảo: chiếu ngẫu nhiên")
    print("      2304->256 kéo cosine 0,966 -> 0,233, tức cột 256 phần lớn đo nút thắt.")
    print("    - 'lstsq' là TRẦN tuyến tính chính xác. Nếu lstsq ~ cosine thì AdamW đã")
    print("      hội tụ, loại bỏ nghi ngờ 'train chưa đủ'. Nếu lstsq >> cosine thì tăng")
    print("      --epochs rồi chạy lại.")
    print("    - 'xáo' và 'r=0' phải THẤP HƠN HẲN cột cosine. Nếu xấp xỉ nhau thì")
    print("      Linear chỉ học prior của delta, KHÔNG dùng ảnh -> kết quả vô giá trị.")
    print("    - 'nhỏ<1ô' dự kiến tệ nhất (18-29 % số box, RoI 3x3 thoái hoá thành 1x1).")
    print("      Nếu chỉ nhóm này hỏng thì thiết kế vẫn dùng được, chỉ giới hạn ở box to.")
    print(f"  tổng thời gian {fmt(res['total_sec'])}  ->  {a.out}")


if __name__ == "__main__":
    main()

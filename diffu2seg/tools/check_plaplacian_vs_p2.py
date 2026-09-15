#!/usr/bin/env python3
r"""CỬA CHẶN 1 cho hướng Diffu2Seg — CỬA CHẶN CỐT LÕI. ~40-70 phút, 100 ảnh val.

CÂU HỎI: `p=1,6` có thật sự hơn `p=2,0` không?

Nếu KHÔNG, thì `compute_g` (4 matmul mỗi vòng lặp) là công vô ích. Ở `p=2` thì
`g**0 = 1`, `gamma_ij = 2*A_ij`, và toàn bộ Algorithm 1 sụp xuống thành MỘT DÒNG
khuếch tán tuyến tính:

    f = (lam*f0 + 2*(A f)) / (lam + 2*(A·1))

Implement `p=1,6` khi đó không phải là tối ưu hoá chưa tới — nó là **tin vào một
cơ chế không tồn tại trên dữ liệu này**.

VÌ SAO ĐÁNG ĐO: `p<2` là **toàn bộ đóng góp thuật toán** của Diffuse2Seg (phần
còn lại — trích self-attention SD2 — là của M2N2, CVPR 2025). Paper tuyên bố nó
giữ biên sắc, đo trên COCO/ADE/SA-1B với vật to. CE-130 thì cạnh ngắn trung vị
chỉ **4,65 ô** trên lưới 64, box nhỏ nhất mỗi ảnh trung vị **2,24 ô**, p10 còn
**0,89 ô**. Ở kích thước 2 ô, "giữ biên" có thể không còn ý nghĩa đo được — biên
chiếm trọn vật. Đó là giả thuyết phải đo, không phải phỏng đoán.

THIẾT KẾ — BIẾN DUY NHẤT LÀ `p`: cùng ảnh, cùng `A` (trích SD2 MỘT LẦN rồi dùng
lại cho mọi `p` — vừa đúng phương pháp vừa tiết kiệm), cùng `f0`, `lam`,
`tau_prop`, ngưỡng mask, connected components, dedup. Chỉ `p` đổi.

MỐC SO SÁNH — chốt TRƯỚC KHI CHẠY:
    `p=2,0` TRONG CHÍNH TOOL NÀY   <- mốc có hiệu lực thật
    A  (CE-Loc gốc)   oracle_recall 0,1197   AP50 1,52
    C1 (refine 6 vòng)              0,1384        1,51
    E1 (score đọc RoI)              0,2559        6,36
    D.1 (DiffusionDet)              0,6734        58,13   mean_bestIoU 0,5974

⚠️ A/C1/E1/D.1 đo trên **test**, tool này đo trên **val** (30 vs 21 box/ảnh, và
test có 855 box rác). **KHÔNG so trực tiếp** — chúng chỉ là bối cảnh độ lớn. Mốc
duy nhất so được là `p=2,0` ở đây, vì nó cùng split, cùng ảnh, cùng mọi thứ.
⚠️ D.1 = **trần của DỮ LIỆU**, KHÔNG phải mục tiêu nên nhắm (docs/01 mục 4.3).

NGƯỠNG — chốt TRƯỚC KHI CHẠY:

  Câu hỏi chính — `p<2` có hơn `p=2` không:
    KHÔNG ĐẠT : oracle_recall(1,6) - oracle_recall(2,0) < +0,01
                -> cơ chế p-Laplacian KHÔNG có tác dụng đo được trên CE-130.
                   Giữ nhánh p=2 một dòng, XOÁ compute_g, ghi kết luận âm vào
                   README. Đây là kết quả HỢP LỆ, không phải thất bại.
    ĐẠT       : chênh >= +0,03 VÀ có xu hướng (p=1,6 không phải nhiễu đơn lẻ)
    XÁM       : +0,01..+0,03, hoặc >= +0,03 nhưng không có xu hướng

  Câu hỏi phụ — hướng này có đáng làm tiếp không:
    KHÔNG ĐẠT : max_p oracle_recall < 0,10  (thua cả EXPERIMENT A) -> bỏ
    ĐẠT       : >= 0,26  (vượt E1 0,2559)
    XÁM       : 0,10..0,26

TIÊN QUYẾT (kiểm TRƯỚC, chặn sớm):
    - không NaN/inf trong f ở mọi p -> có thì clamp g hỏng, SỬA CODE, không đọc số
    - n_iter < max_iter trên >= 90 % ảnh -> không thì tau_prop/lam sai, số vô nghĩa
    - n_prompts > 0 trên mọi ảnh

⚠️ ĐỌC ĐÚNG KẾT QUẢ:
 1. `oracle_recall` BỎ QUA SCORE hoàn toàn — đúng thứ cần cho training-free,
    nhưng nó là **TRẦN**, không phải AP. KHÔNG so nó với cột AP50.
 2. Đo VÙNG-của-mask rồi quy ra box, không đo ĐIỂM.
 3. 100 ảnh: sai số chuẩn của oracle_recall quanh 0,2 là **±0,013**. Chênh lệch
    < 0,01 NẰM TRONG NHIỄU — đó là lý do ngưỡng KHÔNG ĐẠT đặt đúng ở 0,01.
 4. Tool này đo GĐ1 (một mức + CC). `p` có thể phát huy khác dưới GĐ2. Nhưng nếu
    `p` không tạo khác biệt ở GĐ1 thì viện "GĐ2 sẽ cứu" là **đúng cạm bẫy #7**
    (đọc tới khi xác nhận giả thuyết rồi dừng) — phải ghi ra, không được viện.

CHẠY (TRÊN SERVER — xem README):
    export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
    python tools/check_plaplacian_vs_p2.py --split val --limit 100 \
        --p-values 2.0 1.8 1.6 1.4 \
        --out /mnt/disk1/aiotlab/haitn/log/d2s_gate1_p.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base import Diffu2SegConfig
from utils.metrics import fmt_time          # noqa: E402
from d2s.pipeline import build_affinity, segment_image  # noqa: E402
from data.ce130_coco import CE130Coco            # noqa: E402
from utils.metrics import quality_one_image, summarise  # noqa: E402

MAIN_FAIL = 0.01
MAIN_PASS = 0.03
SIDE_FAIL = 0.10
SIDE_PASS = 0.26

REFERENCE = {                     # test split, N=300 — bối cảnh, KHÔNG so trực tiếp
    "A": (0.1197, 1.52), "C1": (0.1384, 1.51),
    "E1": (0.2559, 6.36), "D.1": (0.6734, 58.13),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--p-values", type=float, nargs="+", default=[2.0, 1.8, 1.6, 1.4])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.split == "test":
        print("⚠️  test has a corrupted annotation batch (4.2 % of GT). Use val.\n")
    if 2.0 not in args.p_values:
        print("⚠️  p=2.0 missing — it is THE control arm; the gate cannot conclude "
              "without it.\n")

    cfg = Diffu2SegConfig().validate()
    root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
    ds = CE130Coco(os.path.join(root, "ce130_coco", f"ce130_agnostic_{args.split}.json"),
                   os.path.join(root, "all_phase2_V2"), cfg.canvas)
    n = min(args.limit, len(ds))

    print("CỬA CHẶN 1 — p=1,6 có THẬT SỰ hơn p=2,0 trên CE-130 không?")
    print(f"  split={args.split}  n_images={n}  grid_r={cfg.grid_r}  "
          f"stride={cfg.prompt_stride_cells}  quantile={cfg.mask_quantile}")
    print(f"  p sweep: {args.p_values}   (biến DUY NHẤT; A trích một lần, dùng chung)")
    print(f"  NGƯỠNG (chốt trước): ĐẠT >= +{MAIN_PASS} | KHÔNG ĐẠT < +{MAIN_FAIL}\n")

    from d2s.attention import StableDiffusion2AttentionAggregator
    t0 = time.time()
    agg = StableDiffusion2AttentionAggregator(
        timestep=cfg.timesteps[0], attention_resolution=cfg.grid_r,
        weight_down_block_0=cfg.w_down_0, weight_down_block_1=cfg.w_down_1,
        weight_up_block_0=cfg.w_up_0, weight_up_block_1=cfg.w_up_1,
        weight_up_block_2=cfg.w_up_2, hugging_face_model_id=cfg.model_source,
        prompt_text=cfg.prompt_text, device=args.device, torch_dtype=torch.float16)
    print(f"  SD2 loaded in {time.time() - t0:.1f}s\n")

    acc = {p: {"best": [], "hits": 0, "n_gt": 0, "n_pred": 0,
               "n_not_converged": 0, "n_iter": [], "f_max": [],
               "n_in_padding": 0, "n_too_large": 0} for p in args.p_values}
    bad = []
    t_start = time.time()

    for i in range(n):
        s = ds[i]
        A = build_affinity(s["image"], cfg, agg)     # ONE SD2 forward, reused
        if not torch.isfinite(A).all():
            bad.append((s["file_name"], "affinity has NaN/inf"))
            continue

        for p in args.p_values:
            cfg_p = Diffu2SegConfig(**{**cfg.__dict__, "p": p})
            out = segment_image(s["image"], s["valid_h"], cfg_p, A=A)

            if not np.isfinite(out["boxes"]).all():
                bad.append((s["file_name"], f"p={p}: non-finite boxes"))
                continue

            best, hit, n_gt = quality_one_image(out["boxes"], s["gt_cxcywh"],
                                                size=cfg.canvas)
            a = acc[p]
            a["best"].append(best)
            a["hits"] += hit
            a["n_gt"] += n_gt
            a["n_pred"] += out["n_boxes"]
            a["n_iter"].append(out["n_iter"])
            a["f_max"].append(out["f_max"])
            a["n_in_padding"] += out["filter_info"]["n_in_padding"]
            a["n_too_large"] += out["filter_info"]["n_too_large"]
            if not out["converged"]:
                a["n_not_converged"] += 1

        el = time.time() - t_start
        print(f"  [{i + 1:3d}/{n} {100 * (i + 1) / n:5.1f}%] "
              f"{el / (i + 1):5.1f}s/ảnh | elapsed {fmt_time(el)} | "
              f"ETA {fmt_time(el / (i + 1) * (n - i - 1))}", flush=True)

    # ---------------- TIÊN QUYẾT ----------------
    print("\n" + "=" * 74)
    print("TIÊN QUYẾT")
    prereq_ok = True
    if bad:
        prereq_ok = False
        print(f"  ✗ {len(bad)} lỗi số học (NaN/inf) — clamp g hỏng, SỬA CODE:")
        for fn, why in bad[:5]:
            print(f"      {fn}: {why}")
    for p in args.p_values:
        a = acc[p]
        n_img = len(a["n_iter"])
        conv = 1.0 - a["n_not_converged"] / max(n_img, 1)
        flag = "✓" if conv >= 0.90 else "✗"
        if conv < 0.90:
            prereq_ok = False
        print(f"  {flag} p={p}: hội tụ {100 * conv:5.1f} %  "
              f"(n_iter median {np.median(a['n_iter']):.0f}/{cfg.max_iter}, "
              f"f_max median {np.median(a['f_max']):.2e})")

    if not prereq_ok:
        print("\n  => TIÊN QUYẾT KHÔNG ĐẠT. Số dưới đây mô tả cái CAP hoặc một bug,")
        print("     KHÔNG mô tả cơ chế. Sửa trước, đọc sau.")

    # ---------------- KẾT QUẢ ----------------
    print("\n" + "=" * 74)
    print(f"{'p':>6} | {'oracle_recall':>13} | {'mean_bestIoU':>12} | "
          f"{'box/ảnh':>8} | {'vs p=2.0':>9}")
    print("-" * 74)

    res = {}
    for p in args.p_values:
        a = acc[p]
        res[p] = summarise(a["best"], a["hits"], a["n_gt"],
                           n_pred_total=a["n_pred"], n_images=len(a["n_iter"]))

    base = res.get(2.0, {}).get("oracle_recall")
    for p in sorted(args.p_values, reverse=True):
        r = res[p]
        delta = "" if base is None else f"{r['oracle_recall'] - base:+.4f}"
        print(f"{p:>6.1f} | {r['oracle_recall']:>13.4f} | {r['mean_bestIoU']:>12.4f} | "
              f"{r['n_pred_total'] / max(r['n_images'], 1):>8.1f} | {delta:>9}")

    print("\n  Bối cảnh (test split, N=300 — KHÁC SPLIT, không so trực tiếp):")
    for k, (orc, ap50) in REFERENCE.items():
        print(f"    {k:4s} oracle_recall {orc:.4f}   AP50 {ap50:5.2f}")
    print("    ⚠️ D.1 là trần của DỮ LIỆU, không phải mục tiêu nên nhắm.")

    # ---------------- PHÁN QUYẾT ----------------
    print("\n" + "=" * 74)
    best_p = max(res, key=lambda p: res[p]["oracle_recall"])
    best_val = res[best_p]["oracle_recall"]

    if base is None:
        main_verdict = "KHÔNG KẾT LUẬN ĐƯỢC — thiếu nhánh đối chứng p=2.0"
    else:
        d16 = res.get(1.6, {}).get("oracle_recall", float("nan")) - base
        ordered = [res[p]["oracle_recall"] for p in sorted(args.p_values, reverse=True)]
        has_trend = len(ordered) >= 3 and ordered[-1] != max(ordered)
        if np.isnan(d16) or d16 < MAIN_FAIL:
            main_verdict = (f"KHÔNG ĐẠT (Δ={d16:+.4f}) — p-Laplacian không có tác "
                            f"dụng đo được. Giữ p=2 một dòng, XOÁ compute_g, ghi "
                            f"kết luận âm. Đây là kết quả HỢP LỆ.")
        elif d16 >= MAIN_PASS and has_trend:
            main_verdict = f"ĐẠT (Δ={d16:+.4f}, có xu hướng theo p)"
        else:
            main_verdict = (f"XÁM (Δ={d16:+.4f}) — mang số về bàn, KHÔNG tự quyết")

    if best_val < SIDE_FAIL:
        side = f"KHÔNG ĐẠT (max {best_val:.4f} < {SIDE_FAIL}, thua cả EXPERIMENT A)"
    elif best_val >= SIDE_PASS:
        side = f"ĐẠT (max {best_val:.4f} >= {SIDE_PASS}, vượt E1)"
    else:
        side = f"XÁM (max {best_val:.4f} trong {SIDE_FAIL}..{SIDE_PASS})"

    print(f"  CÂU HỎI CHÍNH (p<2 hơn p=2?)      : {main_verdict}")
    print(f"  CÂU HỎI PHỤ  (đáng làm tiếp?)     : {side}")
    print("\n  ⚠️ oracle_recall là TRẦN (bỏ qua score), KHÔNG phải AP50.")
    print("  ⚠️ 100 ảnh -> sai số chuẩn ±0,013; chênh < 0,01 nằm trong nhiễu.")
    print("  ⚠️ Nếu p không tạo khác biệt ở GĐ1, KHÔNG được viện 'GĐ2 sẽ cứu'.")
    print("=" * 74)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"gate": "plaplacian_vs_p2", "split": args.split, "n_images": n,
                       "config": cfg.to_dict(), "p_values": args.p_values,
                       "results": {str(p): res[p] for p in res},
                       "diagnostics": {str(p): {
                           "n_not_converged": acc[p]["n_not_converged"],
                           "median_n_iter": float(np.median(acc[p]["n_iter"])),
                           "median_f_max": float(np.median(acc[p]["f_max"])),
                           "n_in_padding": acc[p]["n_in_padding"],
                           "n_too_large": acc[p]["n_too_large"]} for p in res},
                       "prereq_ok": prereq_ok,
                       "verdict_main": main_verdict, "verdict_side": side}, f, indent=2)
        print(f"  -> {args.out}")

    return 0 if prereq_ok else 1


if __name__ == "__main__":
    sys.exit(main())

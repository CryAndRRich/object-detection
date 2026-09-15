#!/usr/bin/env python3
r"""CỬA CHẶN 0 cho hướng Diffu2Seg — chạy TRƯỚC HẾT, ~5 phút, 30 ảnh val.

CÂU HỎI: self-attention của SD2 **frozen** (t=150, trộn up_block_0/1, tau_att
0,55) có phân biệt được ô lưới **trong cùng một vật** với ô **sang vật khác hoặc
sang nền** trên CE-130 không?

Nếu KHÔNG, thì mọi thứ phía sau — p-Laplacian, cluster, NMS — chỉ là làm mượt
nhiễu, và hướng này chết ngay tại đây với giá 5 phút thay vì hai ngày.

VÌ SAO ĐÁNG ĐO: đây đúng câu mà cửa chặn EXEMPLAR đã hỏi cho CLIP frozen
(CE-LocModel/tools/check_exemplar_signal.py). Ở đó, đường box<->patch của
EXPERIMENT A đo được **trực giao** lúc khởi tạo (cosine +0,0005), và score_AUC
sau khi train xong vẫn chỉ 0,4965–0,4988 — đúng mức tung đồng xu. SD2 khác hai
điểm: lưới 64x64 thay vì 32x32, và nó là feature **sinh ảnh**, không phải feature
phân loại. Nếu SD2 cũng ra ~0,5 thì không có gì để khai thác.

MỐC SO SÁNH — chốt TRƯỚC KHI CHẠY:
    tung đồng xu                     0,5000
    score_AUC của A/B/C1 sau train   0,4965–0,4988
    AUC vùng của CLIP frozen         0,782   (exemplar, n=30)
    ⚠️ LƯỚI ĐỀU MÙ ẢNH               tool TỰ ĐO — xem dưới

⚠️ MỐC "LƯỚI ĐỀU MÙ ẢNH" LÀ QUAN TRỌNG NHẤT, và là bài học trực tiếp từ
check_keypoint_head.py. CE-130 có trung vị 21 vật/ảnh phủ dày, nên một ma trận
affinity **chỉ phụ thuộc khoảng cách** (A_ij = exp(-d²/sigma²), không nhìn ảnh
một chút nào) cũng tự động cho AUC cao: ô gần nhau thì thường cùng vật, đơn giản
vì vật nhỏ và dày. Không có mốc này thì "AUC 0,80" **không phân biệt được** "SD2
đọc được ảnh" với "CE-130 có vật nhỏ và dày".

NGƯỠNG — chốt TRƯỚC KHI CHẠY:
    TIÊN QUYẾT : A không NaN/inf, mỗi hàng tổng trong [0,99; 1,01].
                 Sai -> "KHÔNG ĐỌC ĐƯỢC", dừng, sửa code, KHÔNG diễn giải số.
    KHÔNG ĐẠT  : mean_AUC < 0,65  HOẶC  mean_AUC - AUC_lưới_đều < 0,05
                 -> dừng, KHÔNG viết tiếp GĐ1.
    ĐẠT        : mean_AUC >= 0,78  VÀ  mean_AUC - AUC_lưới_đều >= 0,10
    XÁM        : còn lại — mang số về bàn, KHÔNG tự quyết.

⚠️ ĐỌC ĐÚNG KẾT QUẢ: phép đo này đo **VÙNG**, không đo **ĐIỂM**. AUC cao chứng
minh "ô cùng vật gần nhau hơn ô khác vật"; nó **KHÔNG** chứng minh "tách được hai
vật cùng class nằm cạnh nhau" — mà tách instance mới là thứ CE-130 cần (trung vị
21 vật CÙNG MỘT class mỗi ảnh). Cửa chặn 1 mới trả lời câu đó. Đây đúng là cạm
bẫy #4 của CLAUDE.md: đừng lấy số đo phân biệt VÙNG làm bằng chứng cho cơ chế
định vị ĐIỂM.

Đo trên **val**, không phải test: test có 30 box/ảnh (vs val 21) và một lô
annotation hỏng chiếm 4,2 % GT.

CHẠY (TRÊN SERVER — xem README):
    export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
    python tools/check_attention_separates.py --split val --limit 30 \
        --out /mnt/disk1/aiotlab/haitn/output/d2s_gate0.json
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
from d2s.affinity import to_affinity             # noqa: E402
from data.ce130_coco import CE130Coco            # noqa: E402
from utils.box_ops_np import cxcywh_to_xyxy      # noqa: E402

PASS_AUC = 0.78
PASS_MARGIN = 0.10
FAIL_AUC = 0.65
FAIL_MARGIN = 0.05


def roc_auc(labels, scores):
    """Mann-Whitney rank-sum AUC with average ranks for ties.

    Same identity CE-LocModel uses (measure_box_quality.py) -- no sklearn.
    Returns nan when one class is missing, and the caller averages over images
    ignoring nan.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1

    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def gt_cell_masks(gt_cxcywh, grid_r, canvas):
    """(G, r, r) bool: which latent cells each GT box covers."""
    if len(gt_cxcywh) == 0:
        return np.zeros((0, grid_r, grid_r), dtype=bool)

    cell_px = canvas / float(grid_r)
    xyxy = cxcywh_to_xyxy(gt_cxcywh) * canvas
    out = np.zeros((len(xyxy), grid_r, grid_r), dtype=bool)
    for g, (x1, y1, x2, y2) in enumerate(xyxy):
        c0 = max(int(np.floor(x1 / cell_px)), 0)
        c1 = min(int(np.ceil(x2 / cell_px)), grid_r)
        r0 = max(int(np.floor(y1 / cell_px)), 0)
        r1 = min(int(np.ceil(y2 / cell_px)), grid_r)
        if c1 > c0 and r1 > r0:
            out[g, r0:r1, c0:c1] = True
    return out


def distance_affinity(grid_r, sigma_cells, device="cpu"):
    """THE BLIND BASELINE: A_ij = exp(-d_ij^2 / sigma^2), image never consulted.

    CE-130 objects are small and densely packed, so "nearby cells belong to the
    same object" is true often enough on its own to score well. Any AUC SD2
    produces has to be read against this number, not against 0.5.
    """
    idx = np.arange(grid_r * grid_r)
    rr, cc = idx // grid_r, idx % grid_r
    d2 = (rr[:, None] - rr[None, :]) ** 2 + (cc[:, None] - cc[None, :]) ** 2
    A = np.exp(-d2 / (2.0 * sigma_cells ** 2))
    A /= A.sum(axis=1, keepdims=True)
    return torch.tensor(A, dtype=torch.float32, device=device)


def auc_for_image(A_np, gt_cells, grid_r, n_seeds, rng):
    """AUC over seeds: positives = same GT box, negatives = outside every box."""
    if len(gt_cells) == 0:
        return float("nan"), 0

    any_gt = gt_cells.any(axis=0).reshape(-1)
    aucs = []
    for _ in range(n_seeds):
        g = int(rng.integers(len(gt_cells)))
        inside = gt_cells[g].reshape(-1)
        if inside.sum() < 2:
            continue
        seed = int(rng.choice(np.flatnonzero(inside)))

        labels = inside.copy()
        labels[seed] = False                    # a token's self-attention is trivial
        usable = labels | (~any_gt)
        usable[seed] = False
        if labels.sum() == 0 or (usable & ~labels).sum() == 0:
            continue

        a = roc_auc(labels[usable], A_np[seed][usable])
        if not np.isnan(a):
            aucs.append(a)

    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"],
                    help="val by default; test has a broken annotation batch")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seeds-per-image", type=int, default=20)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sigma-cells", type=float, default=2.5,
                    help="width of the blind distance baseline, in cells")
    ap.add_argument("--timesteps", type=int, nargs="+", default=None,
                    help="quét nhiều timestep trong MỘT lần chạy. Mặc định chỉ "
                         "dùng cfg.timesteps[0]. t=150 là giá trị Diffuse2Seg "
                         "tinh chỉnh CHO SD2; ta đang chạy SD1.5 nên thang có "
                         "thể khác -> nên quét trước khi chốt.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.split == "test":
        print("⚠️  test has a corrupted annotation batch (4.2 % of GT). "
              "Gates should run on val.\n")

    cfg = Diffu2SegConfig().validate()
    root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
    ds = CE130Coco(os.path.join(root, "ce130_coco", f"ce130_agnostic_{args.split}.json"),
                   os.path.join(root, "all_phase2_V2"), cfg.canvas)

    n = min(args.limit, len(ds))
    timesteps = args.timesteps or [cfg.timesteps[0]]

    print(f"CỬA CHẶN 0 — self-attention có tách được vùng trên CE-130?")
    print(f"  model   : {cfg.model_source}")
    print(f"  split={args.split}  n_images={n}  grid_r={cfg.grid_r}  "
          f"tau_att={cfg.tau_att}")
    print(f"  timestep: {timesteps}"
          f"{'   (quét — t=150 vốn tinh chỉnh cho SD2)' if len(timesteps) > 1 else ''}")
    print(f"  device={args.device}  prompt_text={cfg.prompt_text!r}")
    print(f"  NGƯỠNG (chốt trước): ĐẠT >= {PASS_AUC} và margin >= {PASS_MARGIN} | "
          f"KHÔNG ĐẠT < {FAIL_AUC} hoặc margin < {FAIL_MARGIN}\n")

    from d2s.attention import StableDiffusion2AttentionAggregator
    t0 = time.time()
    agg = StableDiffusion2AttentionAggregator(
        timestep=timesteps[0],
        attention_resolution=cfg.grid_r,
        weight_down_block_0=cfg.w_down_0, weight_down_block_1=cfg.w_down_1,
        weight_up_block_0=cfg.w_up_0, weight_up_block_1=cfg.w_up_1,
        weight_up_block_2=cfg.w_up_2,
        hugging_face_model_id=cfg.model_source,
        prompt_text=cfg.prompt_text,
        device=args.device,
        torch_dtype=torch.float16,
    )
    print(f"  model loaded in {time.time() - t0:.1f}s\n")

    blind = distance_affinity(cfg.grid_r, args.sigma_cells, device="cpu").numpy()

    # AUC của baseline mù ảnh KHÔNG phụ thuộc timestep -> gom riêng, tính một lần.
    blind_aucs = []
    by_t = {t: [] for t in timesteps}
    per_image = []
    prereq_failures = []

    t_loop = time.time()

    for i in range(n):
        s = ds[i]
        cells = gt_cell_masks(s["gt_cxcywh"], cfg.grid_r, cfg.canvas)

        a_bl, _ = auc_for_image(blind, cells, cfg.grid_r, args.seeds_per_image,
                                np.random.default_rng(cfg.seed + i))
        if not np.isnan(a_bl):
            blind_aucs.append(a_bl)

        # Một forward cho mỗi timestep; ảnh và seed giữ nguyên nên t là biến DUY NHẤT.
        attns = agg.extract_attention(s["image"], timesteps=timesteps)
        rec = {"file_name": s["file_name"], "n_gt": int(len(s["gt_cxcywh"])),
               "auc_blind": a_bl}

        for t, attn in zip(timesteps, attns):
            A = to_affinity(attn, tau_att=cfg.tau_att, dtype=torch.float32)

            # TIÊN QUYẾT: A hỏng thì mọi số dưới đây vô nghĩa.
            row = A.sum(dim=1)
            if not torch.isfinite(A).all():
                prereq_failures.append((s["file_name"], f"t={t}: A có NaN/inf"))
            elif not bool(((row > 0.99) & (row < 1.01)).all()):
                prereq_failures.append(
                    (s["file_name"],
                     f"t={t}: tổng hàng trong [{row.min():.4f}, {row.max():.4f}]"))

            a_sd, k = auc_for_image(A.float().cpu().numpy(), cells, cfg.grid_r,
                                    args.seeds_per_image,
                                    np.random.default_rng(cfg.seed + i))
            if not np.isnan(a_sd):
                by_t[t].append(a_sd)
            rec[f"auc_t{t}"] = a_sd
            rec["n_seeds_used"] = k

        per_image.append(rec)
        cols = "  ".join(f"t{t}={rec[f'auc_t{t}']:.4f}" for t in timesteps)
        el = time.time() - t_loop
        print(f"  [{i + 1:3d}/{n} {100 * (i + 1) / n:5.1f}%] {s['file_name']:36s} "
              f"n_gt={len(s['gt_cxcywh']):4d}  {cols}  mù={a_bl:.4f} | "
              f"{el / (i + 1):4.1f}s/ảnh | elapsed {fmt_time(el)} | "
              f"ETA {fmt_time(el / (i + 1) * (n - i - 1))}", flush=True)

    # Timestep tốt nhất là cái được đem ra phán quyết.
    best_t = max(timesteps, key=lambda t: np.mean(by_t[t]) if by_t[t] else -1)
    sd_aucs = by_t[best_t]

    if prereq_failures:
        print("\n" + "=" * 70)
        print("TIÊN QUYẾT KHÔNG ĐẠT — A không dùng được. KHÔNG diễn giải số nào.")
        for fn, why in prereq_failures[:10]:
            print(f"  {fn}: {why}")
        return 1

    mean_sd = float(np.mean(sd_aucs)) if sd_aucs else float("nan")
    mean_bl = float(np.mean(blind_aucs)) if blind_aucs else float("nan")
    margin = mean_sd - mean_bl

    print("\n" + "=" * 70)
    if len(timesteps) > 1:
        print("  AUC theo timestep (biến DUY NHẤT là t):")
        for t in timesteps:
            m = float(np.mean(by_t[t])) if by_t[t] else float("nan")
            star = "  <- tốt nhất" if t == best_t else ""
            print(f"    t={t:4d}: {m:.4f}   margin {m - mean_bl:+.4f}{star}")
        print("  ⚠️ t=150 là giá trị Diffuse2Seg tinh chỉnh cho SD2; ta chạy SD1.5")
        print("     nên nếu t khác thắng rõ thì cập nhật config trước khi sang cửa chặn 1.")
        print()
    print(f"  SD2 attention   mean AUC = {mean_sd:.4f}   (n={len(sd_aucs)}, "
          f"median {np.median(sd_aucs):.4f})")
    print(f"  Lưới đều mù ảnh mean AUC = {mean_bl:.4f}   (sigma={args.sigma_cells} ô)")
    print(f"  MARGIN                   = {margin:+.4f}")
    print("\n  Đối chiếu số đã có của dự án:")
    print(f"    tung đồng xu                    0,5000")
    print(f"    score_AUC A/B/C1 sau train      0,4965–0,4988")
    print(f"    AUC vùng CLIP frozen (exemplar) 0,7820")

    # EXIT CODE 0 CHO MỌI PHÁN QUYẾT. "KHÔNG ĐẠT" là một KẾT QUẢ, không phải
    # lỗi chạy. Trả 1 ở đây từng làm run_on_free_gpu.py tưởng job hỏng và chạy
    # lại 3 lần một cửa chặn đã xong (2026-09-15). Chỉ khối tiên quyết ở trên
    # mới trả 1, vì khi đó A hỏng thật và không có số nào để đọc.
    if np.isnan(mean_sd) or mean_sd < FAIL_AUC or margin < FAIL_MARGIN:
        verdict = "KHÔNG ĐẠT"
    elif mean_sd >= PASS_AUC and margin >= PASS_MARGIN:
        verdict = "ĐẠT"
    else:
        verdict = "XÁM — mang số về bàn, KHÔNG tự quyết"

    print(f"\n  => {verdict}")
    print("\n  ⚠️ Đo VÙNG, không đo ĐIỂM. AUC cao KHÔNG chứng minh tách được hai")
    print("     vật cùng class cạnh nhau — cửa chặn 1 mới trả lời câu đó.")
    print("=" * 70)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"gate": "attention_separates", "split": args.split,
                       "n_images": n, "config": cfg.to_dict(),
                       "sigma_cells": args.sigma_cells,
                       "timesteps": timesteps, "best_timestep": best_t,
                       "mean_auc_by_timestep": {
                           str(t): (float(np.mean(by_t[t])) if by_t[t] else None)
                           for t in timesteps},
                       "mean_auc_sd2": mean_sd, "mean_auc_blind": mean_bl,
                       "margin": margin, "verdict": verdict,
                       "per_image": per_image}, f, indent=2)
        print(f"  -> {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""CE-130: segment map từ prompt = TÂM VẬT (density), để làm đầu vào detector.

⚠️ ĐÂY KHÔNG PHẢI reproduce Diffuse2Seg. Hai chỗ cố ý khác paper:

  1. PROMPT là tâm vật (density CountGD) thay cho lưới đều.
  2. KHÔNG tách connected components, KHÔNG NMS, KHÔNG sinh instance mask.

Lý do (2): Alg.2 dòng 11 `ConnectedComponents` là bước paper gọi là "the step
that makes them INSTANCE masks rather than semantic regions". Ở đây ta MUỐN
semantic region — một vùng phủ MỌI vật giống nhau — nên bỏ đúng bước đó.

Tính chất làm việc này chạy được: M2N2 §3.3 ghi "self-attentions DO NOT TAKE
OBJECT INSTANCES INTO ACCOUNT". M2N2 coi đó là vấn đề (họ thêm flood fill để
chặn); ở đây nó là thứ ta cần — một prompt lan sang mọi vật cùng loại, nên
density thiếu vật (blob ≈ 0,9 × số vật, 1,4 % ảnh trống) vẫn không chặn việc
phủ hết, miễn còn một vật cùng loại được chạm.

                          ĐẦU RA (mỗi ảnh một .npz)

  soft    (H, W) float16  trung bình các soft map đã chuẩn hoá, chia cho max
                          -> [0,1]. ĐÂY LÀ KÊNH DÙNG ĐƯỢC NGAY cho detector
                          (liên tục, giống cách ALPHA3 đưa density làm kênh 4).
  labels  (L, H, W) uint8 nhãn cụm ở từng mức granularity (0 = không cụm nào).
                          Rời rạc, conv KHÔNG đọc trực tiếp được — để dành cho
                          ai muốn tự nhị phân hoá / one-hot.
  heights (L,) float32    độ cao cắt dendrogram tương ứng
  points  (M, 2) int16    ô lưới thực sự dùng làm prompt (sau khi loại/gộp)

Ảnh không có điểm nào (14/908 ở val) vẫn ghi file, `soft` toàn 0 và
`n_prompts = 0`, để phía đọc không phải xử lý file thiếu.

VÍ DỤ
  python tools/run_ce130_points.py --split val --limit 20 \
      --out-dir /mnt/disk1/aiotlab/haitn/output/d2s_ce130_points
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from diffuse2seg.config.base import Diffu2SegConfig          # noqa: E402
from diffuse2seg.data.ce130_points import CE130Points        # noqa: E402
from diffuse2seg.d2s.merging import (_upsample_nearest, cluster_at_heights,  # noqa: E402
                                     normalise_maps, symmetric_kl_matrix)
from diffuse2seg.d2s.pipeline import build_affinity          # noqa: E402
from diffuse2seg.d2s.plaplacian import plaplacian_propagate  # noqa: E402
from diffuse2seg.d2s.prompts import f0_onehot, points_to_cells  # noqa: E402
from diffuse2seg.utils.metrics import fmt_time               # noqa: E402


def semantic_maps(f, cfg, H, W, valid_w, valid_h):
    """(K, N) soft map -> (soft [H,W], labels [L,H,W], heights [L]).

    Giống Alg.2 tới bước argmax, rồi DỪNG: không connected components, không
    lọc diện tích, không NMS.
    """
    p, ok = normalise_maps(f)
    p = p[ok]
    L = cfg.n_levels
    if len(p) == 0:
        return (np.zeros((H, W), np.float16),
                np.zeros((L, H, W), np.uint8),
                np.geomspace(cfg.kl_h_min, cfg.kl_h_max, L).astype(np.float32))

    # soft: trung bình mọi prompt rồi chia max. Không phụ thuộc cụm, nên nó
    # sống sót cả khi clustering vô nghĩa (1 prompt, hoặc mọi prompt giống nhau).
    acc = _upsample_nearest(p.mean(axis=0, keepdims=True), cfg.grid_r, H, W,
                            valid_w, valid_h)[0]
    mx = float(acc.max())
    soft = (acc / mx if mx > 0 else acc).astype(np.float16)

    heights = np.geomspace(cfg.kl_h_min, cfg.kl_h_max, L)
    d = symmetric_kl_matrix(p, eps=cfg.kl_eps)
    labels = np.zeros((L, H, W), np.uint8)
    for li, lab in enumerate(cluster_at_heights(d, heights)):
        n_cl = int(lab.max()) + 1 if len(lab) else 0
        if n_cl == 0:
            continue
        bar = np.zeros((n_cl, p.shape[1]), dtype=np.float64)
        cnt = np.bincount(lab, minlength=n_cl).astype(np.float64)
        np.add.at(bar, lab, p)
        bar /= np.maximum(cnt, 1.0)[:, None]
        up = _upsample_nearest(bar, cfg.grid_r, H, W, valid_w, valid_h)
        seg = up.argmax(axis=0).astype(np.int64)
        # Ô không cụm nào thắng (mọi giá trị 0) -> nhãn 0 = "không có".
        seg = np.where(up.max(axis=0) > 0.0, seg + 1, 0)
        # uint8: >255 cụm thì gộp phần đuôi về 255 thay vì tràn âm thầm.
        labels[li] = np.minimum(seg, 255).astype(np.uint8)
    return soft, labels, heights.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="0 = cả split")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--canvas", type=int, default=512,
                    help="512 = hệ chuẩn dự án. 1120 của paper sẽ PHÓNG TO ảnh "
                         "CE-130 (cao 384 cố định) 1,75-2,75 lần.")
    ap.add_argument("--points", default=None,
                    help="mặc định ../data/density_points.json")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--save-labels", action="store_true",
                    help="ghi thêm mảng nhãn 6 mức (nặng hơn ~6x)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = args.data_root or os.path.join(here, "..", "data", "all_phase2_V2")
    pts_json = args.points or os.path.join(here, "..", "data", "density_points.json")

    cfg = Diffu2SegConfig(canvas=args.canvas).validate()
    ds = CE130Points(root, args.split, pts_json, canvas=cfg.canvas)
    end = len(ds) if args.limit <= 0 else min(args.start + args.limit, len(ds))
    n = end - args.start
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 74)
    print(f"CE-130 segment map, prompt = TÂM VẬT (density)")
    print(f"  split {args.split}: {len(ds)} ảnh, chạy {n} (từ {args.start})")
    print(f"  canvas {cfg.canvas}  r {cfg.grid_r}  t {cfg.timesteps[0]}  "
          f"p {cfg.p}  mức {cfg.n_levels}")
    print(f"  điểm: {pts_json} (params {ds.points_params})")
    print(f"  ⚠️ {ds.n_no_points}/{len(ds)} ảnh KHÔNG có điểm density -> soft toàn 0")
    print(f"  ⚠️ KHÔNG connected-components / NMS: đầu ra là VÙNG NGỮ NGHĨA, "
          f"mỗi vùng gộp mọi vật giống nhau")
    print("=" * 74 + "\n", flush=True)

    from diffuse2seg.d2s.attention import StableDiffusion2AttentionAggregator
    t0 = time.time()
    agg = StableDiffusion2AttentionAggregator(
        timestep=cfg.timesteps[0], attention_resolution=cfg.grid_r,
        weight_down_block_0=cfg.w_down_0, weight_down_block_1=cfg.w_down_1,
        weight_up_block_0=cfg.w_up_0, weight_up_block_1=cfg.w_up_1,
        weight_up_block_2=cfg.w_up_2, hugging_face_model_id=cfg.model_source,
        prompt_text=cfg.prompt_text, device=args.device, torch_dtype=torch.float16)
    print(f"model nạp xong trong {time.time() - t0:.1f}s\n", flush=True)

    t_start = time.time()
    n_pr, n_empty = [], 0
    for k, i in enumerate(range(args.start, end)):
        s = ds[i]
        cells = points_to_cells(s["points_xy"], s["W"], s["H"], cfg.grid_r,
                                cfg.canvas, s["valid_w"], s["valid_h"])
        if len(cells) == 0:
            n_empty += 1
            soft = np.zeros((s["H"], s["W"]), np.float16)
            labels = np.zeros((cfg.n_levels, s["H"], s["W"]), np.uint8)
            heights = np.geomspace(cfg.kl_h_min, cfg.kl_h_max,
                                   cfg.n_levels).astype(np.float32)
            n_iter = 0
        else:
            A = build_affinity(s["image"], cfg, agg)
            f0 = f0_onehot(cells, cfg.grid_r, device=A.device, dtype=A.dtype)
            f, n_iter, _ = plaplacian_propagate(
                A, f0, p=cfg.p, lam=cfg.lam, tau_prop=cfg.tau_prop,
                max_iter=cfg.max_iter, g_eps=cfg.g_eps)
            soft, labels, heights = semantic_maps(
                f.detach().cpu().numpy(), cfg, s["H"], s["W"],
                s["valid_w"], s["valid_h"])
            del A, f, f0
            if torch.cuda.is_available() and args.device.startswith("cuda"):
                agg.current_merged_tensor = None
                torch.cuda.empty_cache()

        n_pr.append(len(cells))
        out = {"soft": soft, "heights": heights,
               "points": cells.astype(np.int16),
               "image_id": np.array(s["image_id"]),
               "n_prompts": np.array(len(cells)), "n_iter": np.array(n_iter)}
        if args.save_labels:
            out["labels"] = labels
        np.savez_compressed(os.path.join(args.out_dir, f"{s['image_id']}.npz"), **out)

        el = time.time() - t_start
        done = k + 1
        peak = (torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available() and args.device.startswith("cuda")
                else float("nan"))
        print(f"  [{done:4d}/{n} {100*done/n:5.1f}%] id={s['image_id']:>6s} "
              f"{s['W']:4d}x{s['H']:<4d} prompt={len(cells):3d} "
              f"gt={len(s['gt_boxes_xyxy']):3d} n_iter={n_iter:4d} "
              f"| đỉnh {peak:.2f} GB | {el/done:5.2f}s/ảnh "
              f"| elapsed {fmt_time(el)} | ETA {fmt_time(el/done*(n-done))}",
              flush=True)

    el = time.time() - t_start
    meta = {"tool": "run_ce130_points", "split": args.split,
            "n_images": n, "start": args.start, "canvas": cfg.canvas,
            "grid_r": cfg.grid_r, "elapsed_sec": el,
            "points_json": os.path.abspath(pts_json),
            "points_params": ds.points_params,
            "prompts_per_image_median": float(np.median(n_pr)) if n_pr else 0.0,
            "n_images_without_points": n_empty,
            "config": cfg.to_dict(),
            "note": "semantic regions: KHONG connected-components, KHONG NMS"}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print(f"\n{n} ảnh -> {args.out_dir}")
    print(f"  prompt/ảnh trung vị {np.median(n_pr):.0f} | "
          f"{n_empty} ảnh không có điểm | {fmt_time(el)} ({el/max(n,1):.2f}s/ảnh)")
    print(f"  -> {args.out_dir}/meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

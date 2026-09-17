#!/usr/bin/env python3
r"""Vẽ mask GĐ2 lên N ảnh, chọn theo kết quả của một lần chạy trước.

NHÌN ẢNH TRƯỚC KHI TIN BẤT KỲ CON SỐ NÀO — bài học §5 của docs/02-du-lieu-ce130.md:
visualize bắt được 2 lỗi lớn mà toàn bộ test và 3 vòng rà soát code bỏ sót.

CHỌN ẢNH NÀO: đọc `per_image` trong JSON của `run_paper.py`, xếp theo
`recall@50 = hits_at_50 / n_gt`, rồi lấy:

    --pick best    N ảnh recall cao nhất   -> cơ chế chạy ĐÚNG trông thế nào
    --pick worst   N ảnh thấp nhất         -> hỏng ở đâu
    --pick spread  đều khắp dải            <- MẶC ĐỊNH, và là thứ nên xem trước

⚠️ `--pick best` một mình LÀ CHỌN LỌC THIÊN VỊ. 30 ảnh tốt nhất trong 50 trông
sẽ luôn thuyết phục, kể cả khi trung vị recall chỉ 0,27 (đo được ở lần chạy
2026-09-15). Nên mặc định là `spread`, và mỗi ảnh in kèm recall thật của nó
ngay trên tiêu đề để không ai đọc nhầm một mẫu chọn lọc thành kết quả chung.

MÀU: xanh lá = GT | cam = pred khớp GT ở IoU>=0.5 | đỏ = pred không khớp
     (chỉ vẽ tối đa `--max-draw` mask pred, xếp theo diện tích giảm dần —
      trung bình có 509 mask/ảnh, vẽ hết thì không nhìn được gì)

CHẠY (TRÊN SERVER):
    export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
    python tools/run_on_free_gpu.py -- tools/visualize_best.py \
        --from-json /mnt/disk1/aiotlab/haitn/output/d2s_paper_paco.json \
        --limit 30 --pick spread \
        --out-dir /mnt/disk1/aiotlab/haitn/output/d2s_viz
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt, median_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base import Diffu2SegConfig                  # noqa: E402
from d2s.pipeline import build_affinity, segment_image   # noqa: E402
from utils.mask_ops import mask_iou_matrix               # noqa: E402
from utils.metrics import fmt_time                       # noqa: E402


def build_dataset(name, root, canvas, split="val"):
    if name == "paco":
        from data.paco_val import PacoVal
        return PacoVal(os.path.join(root, "paco", "paco_lvis_v1_val.json"),
                       os.path.join(root, "paco", "images"), canvas=canvas)
    if name == "ce130":
        # ⚠️ CE-130 CHỈ CÓ BOX GT, không có mask -> không tính được AR/recall.
        # Chỉ dùng để XEM mask trông thế nào trên dữ liệu của dự án.
        # file_name trong json là đường dẫn tương đối so với all_phase2_V2/.
        from data.ce130_coco import CE130Coco
        js = os.path.join(root, "ce130_coco", f"ce130_agnostic_{split}.json")
        imgs = os.path.join(root, "all_phase2_V2")
        # Báo lỗi CÓ NỘI DUNG thay vì FileNotFoundError trần. Trên server dùng
        # chung, thiếu dữ liệu là nguyên nhân thường gặp hơn lỗi code, và
        # run_on_free_gpu sẽ retry 3 lần vô ích nếu không nói rõ.
        missing = [p for p in (js, imgs) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "thiếu dữ liệu CE-130:\n  "
                + "\n  ".join(os.path.normpath(m) for m in missing)
                + f"\n\nĐã tìm dưới data-root: {os.path.normpath(root)}\n"
                "CE-130 cần HAI thứ, và cả hai lên server qua zip người dùng "
                "tự upload (không scp/rsync):\n"
                "  data/ce130_coco/       (~40 MB, 5 file json)\n"
                "  data/all_phase2_V2/    (~14 GB, thư mục ảnh)\n"
                "Kiểm nhanh:  ls "
                + os.path.normpath(os.path.join(root, "ce130_coco")) + "\n"
                "Nếu data/ nằm chỗ khác, truyền --data-root <đường dẫn>.")
        return CE130Coco(js, imgs, canvas=canvas)
    from data.coco_val import CocoVal
    return CocoVal(os.path.join(root, "coco", "annotations",
                                "instances_val2017.json"),
                   os.path.join(root, "coco", "val2017"), canvas=canvas)


def pick_indices(per_image, n, mode):
    """-> list các (chỉ số trong dataset, recall) theo tiêu chí `mode`."""
    scored = []
    for k, p in enumerate(per_image):
        rec = p["hits_at_50"] / max(p["n_gt"], 1)
        scored.append((k, rec, p))
    scored.sort(key=lambda x: -x[1])

    if mode == "best":
        chosen = scored[:n]
    elif mode == "worst":
        chosen = scored[-n:]
    else:                                   # spread
        if n >= len(scored):
            chosen = scored
        else:
            idx = np.linspace(0, len(scored) - 1, n).round().astype(int)
            chosen = [scored[i] for i in idx]
    return chosen


def _out_name(file_name, rec, image_id):
    """Tên file an toàn cho MỌI bộ dữ liệu.

    ⚠️ CE-130 dùng file_name kiểu 'val/1386_b2/ground_truth.jpg' — basename của
    CẢ 908 ảnh đều là 'ground_truth', nên đặt tên theo basename sẽ khiến 50 ảnh
    ghi đè lên nhau còn đúng 1 file. Dùng cả đường dẫn tương đối, thay '/'
    bằng '_'.

    `rec` là NaN khi bộ không có mask GT (không có gì để xếp hạng) -> bỏ tiền
    tố điểm và dùng image_id cho thứ tự ổn định.
    """
    stem = os.path.splitext(file_name)[0].strip("./").replace("/", "_")
    if rec != rec:                      # NaN
        return f"{image_id}_{stem}.png"
    return f"{rec:.2f}_{stem}.png"


def _segmap(masks, H, W, seed=0, cell_px=None):
    """Phân hoạch -> ảnh RGB màu đặc, lấp lỗ, làm mượt biên. CHỈ ĐỂ VẼ.

    Ba việc, theo thứ tự, và vì sao cần:

    1. LẤP LỖ. `masks_from_clusters` bỏ component nhỏ hơn `min_area_px` (100px),
       nên phân hoạch có lỗ — đo được 3,63 % pixel trên ảnh mẫu. Nếu để trống,
       nền lộ ra thành đốm ĐEN lỗ chỗ. Lấp bằng láng giềng gần nhất
       (`distance_transform_edt` trả chỉ số ô gần nhất) chứ không tô một màu
       nền, để lỗ nhỏ tan vào vùng bao quanh nó.

    2. LÀM MƯỢT BIÊN. Mask sinh ở lưới r x r rồi upsample NEAREST, nên biên là
       bậc thang cao đúng MỘT Ô. Lọc trung vị trên ẢNH NHÃN làm tròn bậc thang
       mà KHÔNG trộn hai nhãn thành nhãn thứ ba (lọc trung bình sẽ).
       ⚠️ Kernel phải LỚN HƠN bậc thang. Đo: bậc 8 px thì size=5 chỉ đổi 1,0 %
       pixel (vô tác dụng), size=13 đổi 4,4 %. Nên kernel tính theo `cell_px`
       = kích thước một ô quy về pixel ảnh gốc, không phải hằng số.

    3. TÔ MÀU ĐẶC. Không phủ lên ảnh gốc: đã có panel ảnh gốc riêng bên cạnh.

    ⚠️ Cả ba CHỈ đổi thứ hiển thị. Mask dùng để tính AR không đi qua hàm này.
    """
    lab = np.zeros((H, W), dtype=np.int32)          # 0 = chưa gán
    for i, m in enumerate(masks):
        lab[m] = i + 1

    if (lab == 0).any() and len(masks):
        # chỉ số của pixel khác 0 gần nhất
        _, (iy, ix) = distance_transform_edt(
            lab == 0, return_distances=True, return_indices=True)
        lab = lab[iy, ix]

    if cell_px and len(masks):
        k = int(max(3, round(cell_px * 1.5)))
        k += 1 - k % 2                               # kernel lẻ
        sm = median_filter(lab, size=k, mode="nearest")
        # ⚠️ Median XOÁ HẲN vùng nhỏ hơn kernel: đo được vật 8x8 px (đúng 1 ô)
        # biến mất sạch với k=13. Mà vật nhỏ chính là nhóm ta yếu nhất
        # (AR_S 4,83) — làm mượt để rồi xoá chúng là tự bóp méo hình minh hoạ.
        # Nên chỉ nhận kết quả mượt ở nơi nhãn đó CÒN TỒN TẠI sau lọc.
        survived = set(np.unique(sm).tolist())
        lost = [v for v in np.unique(lab) if v not in survived]
        if lost:
            keep_mask = np.isin(lab, lost)
            sm[keep_mask] = lab[keep_mask]           # trả lại vùng bị xoá
        lab = sm

    rng = np.random.default_rng(seed)
    palette = rng.random((len(masks) + 1, 3))
    palette[0] = 0.5                                # không nên còn dùng tới
    return palette[np.clip(lab, 0, len(masks))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-json", default=None,
                    help="JSON của run_paper.py — quyết định ảnh nào được vẽ. "
                         "Bỏ qua thì phải truyền --dataset và ảnh được chọn "
                         "NGẪU NHIÊN (dùng cho bộ không có mask GT, vd CE-130).")
    ap.add_argument("--dataset", default=None,
                    choices=["paco", "coco", "ce130"],
                    help="chỉ khi KHÔNG có --from-json: bộ dữ liệu cần vẽ.")
    ap.add_argument("--split", default="val",
                    help="ce130: train/val/test (mặc định val)")
    ap.add_argument("--seed", type=int, default=0,
                    help="hạt giống chọn ảnh ngẫu nhiên khi không có --from-json")
    ap.add_argument("--canvas", type=int, default=None,
                    help="ghi đè canvas. CHỈ dùng khi KHÔNG có --from-json — "
                         "đổi canvas thì mask khác hẳn, không còn là ảnh của "
                         "con số trong JSON. CE-130: 512 là hệ chuẩn dự án, "
                         "1120 của paper sẽ PHÓNG TO ảnh 1,75-2,75 lần.")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--pick", default="spread",
                    choices=["spread", "best", "worst", "random"])
    ap.add_argument("--mode", default="debug", choices=["debug", "paper"],
                    help="debug: 3 panel (ảnh gốc | mask trộn 6 mức | box vs GT), "
                         "để chẩn đoán. paper: mỗi mức granularity một ảnh "
                         "riêng, không box, crop bỏ pad — như Hình 1 của paper.")
    ap.add_argument("--levels", default=None,
                    help="--mode paper: các mức muốn vẽ, vd '1,3,6'. "
                         "Mặc định vẽ cả 6. Panel ảnh gốc luôn có.")
    ap.add_argument("--max-draw", type=int, default=60,
                    help="số mask pred vẽ tối đa mỗi ảnh (theo diện tích giảm dần)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if not args.from_json and not args.dataset:
        ap.error("cần --from-json, hoặc --dataset khi bộ dữ liệu không có "
                 "mask GT để xếp hạng (vd: --dataset ce130)")

    if args.from_json:
        with open(args.from_json) as f:
            blob = json.load(f)
        saved_cfg = blob["config"]
        per_image = blob["per_image"]
        dataset = blob.get("dataset", "paco")
    else:
        # KHÔNG có JSON -> không có điểm để xếp hạng -> chọn NGẪU NHIÊN.
        # Cấu hình lấy từ config/paper.py, cộng các cờ ghi đè trên dòng lệnh.
        blob, per_image, dataset = None, None, args.dataset
        from config.paper import cfg as paper_cfg
        saved_cfg = paper_cfg.to_dict()

    # Dựng lại ĐÚNG cấu hình đã chạy, không dùng mặc định -- nếu không thì ảnh
    # vẽ ra không phải ảnh của con số trong JSON.
    fields = {k: v for k, v in saved_cfg.items()
              if k in Diffu2SegConfig.__dataclass_fields__}
    for k in ("timesteps", "timestep_weights", "kl_thresholds"):
        if k in fields and isinstance(fields[k], list):
            fields[k] = tuple(fields[k])
    # keep_level_masks CHỈ đổi thứ được TRẢ VỀ, không đổi phép tính nào -- mask
    # và mọi con số vẫn y hệt cấu hình đã lưu trong JSON.
    fields["keep_level_masks"] = (args.mode == "paper")
    if args.canvas is not None:
        if args.from_json:
            ap.error("--canvas không dùng chung với --from-json: đổi canvas là "
                     "đổi mask, ảnh vẽ ra không còn khớp con số trong JSON.")
        fields["canvas"] = args.canvas
    cfg = Diffu2SegConfig(**fields).validate()

    root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
    ds = build_dataset(dataset, root, cfg.canvas, split=args.split)
    start = blob.get("start", 0) if blob else 0

    print("=" * 74)
    if per_image is not None:
        chosen = pick_indices(per_image, args.limit, args.pick)
    else:
        # Không có điểm -> chọn NGẪU NHIÊN, có seed để lặp lại được.
        n_av = len(ds)
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(n_av, size=min(args.limit, n_av), replace=False)
        chosen = [(int(k), float("nan"), None) for k in sorted(idx)]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"VISUALIZE — {len(chosen)} ảnh, chọn kiểu "
          f"'{args.pick if per_image is not None else 'random'}'")
    print(f"  nguồn       : {args.from_json or (dataset + '/' + args.split)}")
    print(f"  cấu hình    : t={cfg.timesteps[0]} w1={cfg.w_up_0} canvas={cfg.canvas} "
          f"r={cfg.grid_r} p={cfg.p}")
    if per_image is not None:
        recs = np.array([r for _, r, _ in chosen])
        all_rec = np.array([p["hits_at_50"] / max(p["n_gt"], 1) for p in per_image])
        print(f"  AR_1000 lần chạy đó : {100 * blob['results']['AR_1000']:.2f}")
        print(f"  recall@50 CẢ {len(per_image)} ảnh  : trung vị "
              f"{np.median(all_rec):.2f}  [{all_rec.min():.2f}, {all_rec.max():.2f}]")
        print(f"  recall@50 {len(chosen)} ảnh được chọn: trung vị "
              f"{np.median(recs):.2f}  [{recs.min():.2f}, {recs.max():.2f}]")
        if args.pick == "best":
            print("  ⚠️ 'best' LÀ MẪU CHỌN LỌC — đừng đọc nó như kết quả chung.")
    else:
        print(f"  tổng số ảnh : {len(ds)}  -> lấy ngẫu nhiên {len(chosen)} "
              f"(seed {args.seed})")
        print("  ⚠️ BỘ NÀY KHÔNG CÓ MASK GT nên KHÔNG có recall/AR — mọi con số")
        print("     recall in ra sẽ là 0.00 vì không có gì để so, KHÔNG phải vì")
        print("     model kém. Đây thuần tuý là xem mask trông thế nào.")
    print("=" * 74 + "\n")

    from d2s.attention import StableDiffusion2AttentionAggregator
    agg = StableDiffusion2AttentionAggregator(
        timestep=cfg.timesteps[0], attention_resolution=cfg.grid_r,
        weight_down_block_0=cfg.w_down_0, weight_down_block_1=cfg.w_down_1,
        weight_up_block_0=cfg.w_up_0, weight_up_block_1=cfg.w_up_1,
        weight_up_block_2=cfg.w_up_2, hugging_face_model_id=cfg.model_source,
        prompt_text=cfg.prompt_text, device=args.device, torch_dtype=torch.float16)

    t_start = time.time()
    for j, (k, rec_saved, rec_info) in enumerate(chosen):
        i = start + k
        s = ds[i]
        A = build_affinity(s["image"], cfg, agg)
        out = segment_image(s["image"], s["valid_h"], cfg, A=A,
                            valid_w=s["valid_w"], orig_hw=(s["H"], s["W"]))
        pred = out["masks_full"]

        iou = mask_iou_matrix(pred, s["gt_masks"])
        hit_pred = (iou.max(axis=1) >= 0.5) if iou.size else np.zeros(len(pred), bool)
        hit_gt = (iou.max(axis=0) >= 0.5) if iou.size else np.zeros(len(s["gt_masks"]), bool)
        rec = hit_gt.mean() if len(hit_gt) else 0.0

        areas = pred.reshape(len(pred), -1).sum(1) if len(pred) else np.zeros(0)
        order = np.argsort(-areas)[:args.max_draw]

        if args.mode == "paper":
            # MỘT file cho mỗi ảnh: ảnh gốc BÊN TRÁI, các mức granularity bên
            # cạnh — đúng bố cục Hình 1 của paper. Segment map là ảnh MÀU ĐẶC
            # (không phủ lên ảnh gốc), vì đã có panel ảnh gốc riêng rồi.
            lv = out["merge_info"].get("level_masks")
            if not lv:
                raise RuntimeError(
                    "không có level_masks — cần cfg.keep_level_masks=True. "
                    "Chạy --mode paper thì tool tự bật, nên lỗi này nghĩa là "
                    "config bị ghi đè ở đâu đó.")
            # ⚠️ mask của Alg.2 ở kích thước ẢNH GỐC (H, W), còn s["image"] là
            # canvas vuông đã pad. Cắt vùng hợp lệ rồi resize về (W, H).
            vh = int(s["valid_h"] * cfg.canvas)
            vw = int(s["valid_w"] * cfg.canvas)
            bg = np.asarray(Image.fromarray(
                s["image"][:vh, :vw]).resize((s["W"], s["H"]), Image.BILINEAR))

            # Một ô lưới phủ canvas/r px trên canvas; quy về pixel ảnh gốc.
            cell_px = (cfg.canvas / cfg.grid_r) * s["W"] / max(vw, 1)
            heights = out["merge_info"]["heights"]
            want = [int(x) for x in args.levels.split(",")] if args.levels else \
                list(range(1, len(heights) + 1))
            want = [i for i in want if 1 <= i <= len(heights)]

            panels = [("Input image", bg)]
            for li in want:
                masks, h = lv[li - 1], heights[li - 1]
                panels.append((f"Granularity {li}/{len(heights)}  "
                               f"h={h:.3f}  {len(masks)} masks",
                               _segmap(masks, s["H"], s["W"], seed=li,
                                       cell_px=cell_px)))

            # Bố cục 2 hàng, hàng dưới CĂN GIỮA. Với 7 panel: 4 trên, 3 dưới.
            # Dùng lưới 2*ncol_top cột và cho mỗi panel rộng 2 cột; hàng thiếu
            # panel được đẩy vào giữa bằng offset lẻ. Với 4+3: hàng trên chiếm
            # cột 0..7, hàng dưới cột 1..6 -> lệch đúng 1 cột mỗi bên.
            n = len(panels)
            n_top = (n + 1) // 2
            n_bot = n - n_top
            ncols = 2 * n_top
            aspect = s["H"] / max(s["W"], 1)
            nrows = 1 if n_bot == 0 else 2
            fig1 = plt.figure(figsize=(5.5 * n_top, 5.5 * aspect * nrows + 0.9))
            gs = fig1.add_gridspec(nrows, ncols)

            for idx, (title, img) in enumerate(panels):
                if idx < n_top:
                    r_, c0 = 0, idx * 2
                else:
                    r_ = 1
                    c0 = (ncols - 2 * n_bot) // 2 + (idx - n_top) * 2
                ax1 = fig1.add_subplot(gs[r_, c0:c0 + 2])
                ax1.imshow(img)
                ax1.set_title(title, fontsize=11)
                ax1.axis("off")

            fig1.suptitle(
                f"{s['file_name']}  |  t={cfg.timesteps[0]} w1={cfg.w_up_0} "
                f"r={cfg.grid_r}  |  {len(s['gt_masks'])} GT  "
                f"recall@50={rec:.2f}", fontsize=13)
            fig1.tight_layout()
            fig1.savefig(
                os.path.join(args.out_dir,
                             _out_name(s["file_name"], rec, s["image_id"])),
                dpi=110, bbox_inches="tight")
            plt.close(fig1)
            n_saved_this = 1

            n_pred = len(pred)
            del A, out, pred
            agg.current_merged_tensor = None
            if torch.cuda.is_available() and args.device.startswith("cuda"):
                torch.cuda.empty_cache()
                peak = torch.cuda.max_memory_allocated() / 1e9
            else:
                peak = float("nan")
            el = time.time() - t_start
            print(f"  [{j + 1:3d}/{len(chosen)}] {os.path.basename(s['file_name']):28s} "
                  f"-> {n_saved_this} mức | n_gt={len(s['gt_masks']):3d} "
                  f"n_pred={n_pred:4d} recall={rec:.2f} | đỉnh GPU {peak:.2f} GB | "
                  f"elapsed {fmt_time(el)} | "
                  f"ETA {fmt_time(el / (j + 1) * (len(chosen) - j - 1))}", flush=True)
            continue

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        raw = s["image"][:int(s["valid_h"] * cfg.canvas),
                         :int(s["valid_w"] * cfg.canvas)]

        axes[0].imshow(raw)
        axes[0].set_title(f"Input image {s['W']}x{s['H']}")

        # giữa: mask pred, mỗi instance một màu
        axes[1].imshow(s["image"])
        if len(order):
            rng = np.random.default_rng(0)
            overlay = np.zeros((s["H"], s["W"], 4))
            for oi in order:
                c = rng.random(3)
                overlay[pred[oi]] = [c[0], c[1], c[2], 0.55]
            axes[1].imshow(overlay, extent=[0, cfg.canvas * s["valid_w"],
                                            cfg.canvas * s["valid_h"], 0])
        axes[1].set_title(f"{len(pred)} masks (drawing {len(order)} largest) | "
                          f"{out['merge_info']['nms']['n_in']} before NMS")

        # phải: box từ mask, so GT
        axes[2].imshow(s["image"])
        sx = cfg.canvas * s["valid_w"] / s["W"]
        sy = cfg.canvas * s["valid_h"] / s["H"]
        for gi, gm in enumerate(s["gt_masks"]):
            ys, xs = np.where(gm)
            if not len(ys):
                continue
            axes[2].add_patch(Rectangle(
                (xs.min() * sx, ys.min() * sy),
                (xs.max() - xs.min() + 1) * sx, (ys.max() - ys.min() + 1) * sy,
                fill=False, edgecolor="lime" if hit_gt[gi] else "darkgreen",
                lw=2.0 if hit_gt[gi] else 1.0,
                linestyle="-" if hit_gt[gi] else ":"))
        for oi in order:
            ys, xs = np.where(pred[oi])
            if not len(ys):
                continue
            axes[2].add_patch(Rectangle(
                (xs.min() * sx, ys.min() * sy),
                (xs.max() - xs.min() + 1) * sx, (ys.max() - ys.min() + 1) * sy,
                fill=False, edgecolor="orange" if hit_pred[oi] else "red", lw=1.0))
        n_hit = int(hit_gt.sum())
        axes[2].set_title(f"{len(s['gt_masks'])} GT (bright green = {n_hit} found) | "
                          f"recall@50 = {rec:.2f}")

        for ax in (axes[1], axes[2]):
            if s["valid_h"] < 0.999:
                ax.axhline(s["valid_h"] * cfg.canvas, color="yellow", ls="--", lw=1.2)
            if s["valid_w"] < 0.999:
                ax.axvline(s["valid_w"] * cfg.canvas, color="yellow", ls="--", lw=1.2)
            ax.set_xlim(0, cfg.canvas)
            ax.set_ylim(cfg.canvas, 0)
        for ax in axes:
            ax.axis("off")

        n_part = int(s["is_part"].sum()) if "is_part" in s else -1
        fig.suptitle(
            f"{s['file_name']}  |  t={cfg.timesteps[0]} w1={cfg.w_up_0} r={cfg.grid_r}"
            f"  |  {len(s['gt_masks'])} GT"
            + (f" ({n_part} PART)" if n_part >= 0 else "")
            + f"  recall@50={rec:.2f}", fontsize=13)
        fig.tight_layout()

        path = os.path.join(args.out_dir,
                            _out_name(s["file_name"], rec, s["image_id"]))
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        # ⚠️ Giải phóng GIỮA các ảnh. A là (h,w,h,w) fp32 = 1,54 GB ở r=140 và
        # current_merged_tensor là một bản nữa. Không xoá thì cả hai sống qua
        # vòng lặp sau, đúng lúc SD đang dựng tensor 6,15 GB cho ảnh kế tiếp.
        # empty_cache() trả phần đã giải phóng về driver — cần trên GPU dùng
        # chung, nơi phần trống thật sự có thể chỉ còn ~18 GB.
        #
        # Lấy n_pred TRƯỚC khi del: dòng log bên dưới cần nó, và `del pred`
        # rồi đọc lại là UnboundLocalError (đã vỡ đúng thế 2026-09-16).
        n_pred = len(pred)
        del A, out, pred
        agg.current_merged_tensor = None
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            peak = torch.cuda.max_memory_allocated() / 1e9
        else:
            peak = float("nan")

        el = time.time() - t_start
        print(f"  [{j + 1:3d}/{len(chosen)}] {os.path.basename(path):40s} "
              f"n_gt={len(s['gt_masks']):3d} n_pred={n_pred:4d} recall={rec:.2f} | "
              f"đỉnh GPU {peak:.2f} GB | "
              f"elapsed {fmt_time(el)} | "
              f"ETA {fmt_time(el / (j + 1) * (len(chosen) - j - 1))}", flush=True)

    print(f"\n{len(chosen)} ảnh -> {args.out_dir}")
    print("  (tên file bắt đầu bằng recall@50, nên `ls` đã là bảng xếp hạng)")
    print("\n⚠️ NHÌN KỸ: mask có bám vật hay tràn nền? hai vật cùng loại cạnh nhau")
    print("   có bị gộp? các mức granularity có ra mask lồng nhau như mong đợi?")
    print("   vùng dưới/phải vạch vàng là PAD — không được sinh mask ở đó.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

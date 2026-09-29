#!/usr/bin/env python3
"""Soi SpatialSoftmax của CE-Loc GỐC (checkpoint bài add) với 4 density: đầy đủ nhất -> bớt 1
vật -> bớt 2 vật -> trống, trên một trong ba loại ảnh (`--image`):
  original     ground_truth.jpg, chưa xoá vật nào
  inpainted_1  ảnh lượt 1 của nhánh (đã xoá T1 — đúng đầu vào bài add, T1 = box đáp án);
               khớp pixel với density Full
  inpainted_2  ảnh lượt 2 (đã xoá T1 và R1); khớp pixel với density -1
Hình vẽ box các vật đã xoá (lỗ inpaint) nét đứt đỏ. Cùng --seed thì ba chế độ chạy trên cùng bộ ảnh.

Density trong samples/ KHÔNG vẽ từ một danh sách GT cố định: hai density lượt 1 của hai nhánh
cùng ảnh lệch nhau ở nhiều blob, không chỉ ở vật bị xoá (đếm lại trên từng ảnh inpaint?) ->
không ghép được một map "đủ mọi vật". Nên:
  Full        density lượt 1 của nhánh (>= 3 lượt) có diện tích blob LỚN NHẤT
  -1 object   density lượt 2 của cùng nhánh (mất thêm blob vật xoá ở lượt 2, R1)
  -2 objects  density lượt 3 (mất thêm R2)
  Empty       toàn (0,0,127) — map trống, có trong dữ liệu (1-3 % mẫu)
Mỗi ảnh gốc một ca.

Hình: mỗi ca 4 hàng (density) x 3 cột (Original | Density Map | SpatialSoftmax Output).
Checkpoint KHÔNG density (conv1 3 kênh, `legacy/train.py --no-density`): model không đọc density ->
1 hàng x 2 cột (ảnh | SpatialSoftmax Output), bỏ mọi số đo theo density (lift_blobs_full, shift_px,
cos_emb). Density vẫn dùng ở bước CHỌN nhánh để ra đúng bộ ảnh như bản có density (so được từng ảnh).
Cột 3 = 512 chấm, mỗi chấm là toạ độ kỳ vọng của MỘT kênh; màu = số ô hiệu dụng của softmax
kênh đó (vàng = nhọn, chấm có nghĩa là một vị trí; tím = trải đều, chấm bị kéo về giữa khung).

Số đo (log + metrics.json; trung bình trên 512 kênh, trọng số w_c = ||W_proj[:, 2c:2c+2]||):
  lift_<nhóm>     khối lượng attention trên ô phủ nhóm box / tỉ lệ diện tích nhóm (1 = ngẫu nhiên)
                  objects = mọi vật trong ảnh gốc; T1 = vật xoá ở lượt 1 (vắng ở mọi density);
                  R1, R2 = vật mất blob ở hàng -1 / -2
  frac_localized  tỉ lệ (trọng số) kênh có < 16 ô hiệu dụng
  shift_px        chấm dịch bao nhiêu pixel (ảnh gốc) so với Full;  cos_emb: vis_emb so với Full

  python tools/inspect_spatial_softmax.py --image original --n 100 --out ../../output/spatial_softmax/density_paper/original
  python tools/inspect_spatial_softmax.py --image inpainted_1 --n 100 --out ../../output/spatial_softmax/density_paper/inpainted_1
"""

import argparse
import warnings
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.legacy.celoc_vision import TARGET, grid_to_canvas, load_vision_encoder, to_input  # noqa: E402

BG = np.array([0, 0, 127])            # nền jet của density = "không có vật"
SETTINGS = ["full", "minus1", "minus2", "empty"]
NO_DENSITY = ["none"]                 # checkpoint 3 kênh: một lần chạy, không density
ROW_LABELS = {"full": "Full", "minus1": "−1 object", "minus2": "−2 objects", "empty": "Empty"}
LOCALIZED = 16                        # kênh "tập trung" nếu < 16 ô hiệu dụng
FONT = {"family": "DejaVu Sans", "size": 12, "weight": "normal"}   # MỌI chữ trên hình
IMAGE_TITLE = {"original": "Original", "inpainted_1": "Inpainted (1 removed)",
               "inpainted_2": "Inpainted (2 removed)"}


# ----------------------------------------------------------------------------- dữ liệu

def xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], float)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def blob_mask(density_img):
    """Pixel density khác nền jet (0,0,127) -> có blob."""
    a = np.asarray(density_img.convert("RGB"), dtype=int)
    return np.abs(a - BG).sum(-1) > 40


def blob_frac(mask, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    reg = mask[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]
    return float(reg.mean()) if reg.size else 0.0


def load_targets(samples_dir):
    by_iid = defaultdict(dict)
    for f in os.listdir(os.path.join(samples_dir, "annotation")):
        stem = f[:-5]
        with open(os.path.join(samples_dir, "annotation", f)) as fh:
            by_iid[stem.rsplit("_", 1)[0]][stem] = np.array(json.load(fh)["target_bbox"], float)
    return by_iid


def image_path(samples_dir, stem):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(samples_dir, "images", stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(stem)


def sample_for_turn(samples_dir, stems, branch_dir, box_xyxy, turn):
    """Sample có target = box xoá ở `turn` VÀ ảnh trùng pixel với inpainted_turn_{turn}.png
    (hai nhánh có thể xoá cùng một vật nên box thôi chưa đủ)."""
    want = xyxy_to_cxcywh(box_xyxy)
    cands = [s for s, t in stems.items() if np.abs(t - want).max() < 0.6]
    ref = np.asarray(Image.open(os.path.join(branch_dir, f"inpainted_turn_{turn}.png")).convert("RGB"), dtype=np.int16)
    for s in cands:
        a = np.asarray(Image.open(image_path(samples_dir, s)).convert("RGB"), dtype=np.int16)
        if a.shape == ref.shape and np.abs(a - ref).mean() < 0.5:
            return s
    return None


def build_case(samples_dir, stems, iid, branch_dirs):
    """Một ca cho một ảnh gốc hoặc (None, lý do). Chọn nhánh >= 3 lượt có density lượt 1 dày nhất."""
    best = None
    for d in branch_dirs:
        ann = json.load(open(os.path.join(d, "annotation.json")))
        inp = [np.array(b, float) for b in ann["inpainted_bboxes"]]
        if len(inp) < 3:
            continue
        turns = [sample_for_turn(samples_dir, stems, d, inp[t - 1], t) for t in (1, 2, 3)]
        if None in turns:
            continue
        dens = [Image.open(os.path.join(samples_dir, "density", s + ".png")) for s in turns]
        area = float(blob_mask(dens[0]).mean())
        if best is None or area > best[0]:
            best = (area, d, ann, inp, turns, dens)
    if best is None:
        return None, "không có nhánh >= 3 lượt khớp được sample"
    area, d, ann, inp, turns, dens = best
    if area == 0:
        return None, "density lượt 1 trống"

    original = Image.open(os.path.join(d, "ground_truth.jpg")).convert("RGB")
    if original.size != dens[0].size:
        return None, "ground_truth.jpg khác kích thước density"
    m = [blob_mask(x) for x in dens]
    r1, r2 = inp[1], inp[2]
    blob = {"R1_full": blob_frac(m[0], r1), "R1_minus1": blob_frac(m[1], r1),
            "R2_full": blob_frac(m[0], r2), "R2_minus2": blob_frac(m[2], r2),
            "T1_full": blob_frac(m[0], inp[0]), "area_full": area}
    if blob["R1_full"] < 0.05 or blob["R2_full"] < 0.05:
        return None, "R1/R2 không có blob ở density Full (bớt blob vô nghĩa)"
    with open(os.path.join(samples_dir, "annotation", turns[0] + ".json")) as fh:
        cls = json.load(fh)["class"]
    return dict(name=os.path.basename(d), iid=iid, cls=cls, stems=turns, original=original,
                inpainted_1=Image.open(image_path(samples_dir, turns[0])).convert("RGB"),
                inpainted_2=Image.open(image_path(samples_dir, turns[1])).convert("RGB"), dens=dens,
                objects=[np.array(b, float) for b in ann["all_bboxes"]], t1=inp[0], r1=r1, r2=r2,
                blob=blob), None


def select_cases(args):
    by_iid = load_targets(args.samples)
    branches = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(args.ce130, "*", "*_b*"))):
        branches[os.path.basename(d).rsplit("_", 1)[0]].append(d)
    iids = args.ids if args.ids else sorted(i for i in branches if i in by_iid)
    if not args.ids:
        np.random.default_rng(args.seed).shuffle(iids)
    cases, skipped = [], defaultdict(int)
    for iid in iids:
        if len(cases) >= args.n:
            break
        if iid not in by_iid or iid not in branches:
            skipped["không có trong samples / all_phase2_V2"] += 1
            continue
        c, why = build_case(args.samples, by_iid[iid], iid, branches[iid])
        if c is None:
            skipped[why] += 1
            if args.ids:
                print(f"  bỏ {iid}: {why}")
            continue
        cases.append(c)
    return cases, dict(skipped)


# ----------------------------------------------------------------------------- đo

def coverage(boxes, scale, n_cell):
    """Tỉ lệ diện tích mỗi ô lưới bị phủ bởi HỢP các box (toạ độ ảnh gốc)."""
    m = np.zeros((TARGET, TARGET), np.float32)
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v * scale)) for v in b]
        m[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 1
    c = TARGET // n_cell
    return m.reshape(n_cell, c, n_cell, c).mean(axis=(1, 3))


def mask_coverage(mask, scale, n_cell):
    """Mask pixel (ảnh gốc) -> tỉ lệ phủ mỗi ô, theo đúng resize NEAREST + độn của dataset."""
    h, w = mask.shape
    nw, nh = int(w * scale), int(h * scale)
    m = np.zeros((TARGET, TARGET), np.float32)
    m[:nh, :nw] = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((nw, nh), Image.NEAREST)) > 0
    c = TARGET // n_cell
    return m.reshape(n_cell, c, n_cell, c).mean(axis=(1, 3))


def lift(att, w, cov):
    """att [C,h,w], w [C], cov [h,w] -> (lift có trọng số, lift không trọng số)."""
    chance = cov.mean()
    if chance == 0:
        return float("nan"), float("nan")
    mass = (att * cov).sum(axis=(1, 2))
    return float((w * mass).sum() / w.sum() / chance), float(mass.mean() / chance)


def holes(case, image):
    """Box các vật đã bị inpaint khỏi ảnh đầu vào."""
    return {"original": [], "inpainted_1": [case["t1"]], "inpainted_2": [case["t1"], case["r1"]]}[image]


def settings_for(enc):
    return SETTINGS if enc.in_channels == 4 else NO_DENSITY


@torch.no_grad()
def run_case(enc, case, w, image):
    use_density = enc.in_channels == 4
    settings = settings_for(enc)
    if use_density:
        dens = {"full": case["dens"][0], "minus1": case["dens"][1], "minus2": case["dens"][2],
                "empty": Image.new("RGB", case["original"].size, tuple(BG))}
        rgbs, ds, inputs = [], [], {}
        for name in settings:
            rgb, den, scale = to_input(case[image], dens[name])
            rgbs.append(rgb)
            ds.append(den)
            inputs[name] = (rgb[0].permute(1, 2, 0).numpy(), den[0, 0].numpy())
        emb, kp, att, _ = enc(torch.cat(rgbs), torch.cat(ds))
    else:
        rgb, _, scale = to_input(case[image])
        inputs = {"none": (rgb[0].permute(1, 2, 0).numpy(), None)}
        emb, kp, att, _ = enc(rgb)
    emb, kp, att = emb.numpy(), kp.numpy(), att.numpy()
    n_cell = att.shape[-1]
    kp_canvas = grid_to_canvas(kp, n_cell)                                # (hàng, cột) pixel canvas
    eff = np.exp(-(att * np.log(att + 1e-12)).sum(axis=(2, 3)))          # [S, C]

    covs = {"objects": coverage(case["objects"], scale, n_cell), "T1": coverage([case["t1"]], scale, n_cell),
            "R1": coverage([case["r1"]], scale, n_cell), "R2": coverage([case["r2"]], scale, n_cell)}
    if use_density:
        covs["blobs_full"] = mask_coverage(blob_mask(case["dens"][0]), scale, n_cell)
    if holes(case, image):
        covs["holes"] = coverage(holes(case, image), scale, n_cell)
    rows = {}
    for i, name in enumerate(settings):
        r = {f"lift_{g}": lift(att[i], w, cv)[0] for g, cv in covs.items()}
        r["eff_cells"] = float((w * eff[i]).sum() / w.sum())
        r["frac_localized"] = float((w * (eff[i] < LOCALIZED)).sum() / w.sum())
        if use_density:
            r["shift_px"] = float((w * np.linalg.norm(kp_canvas[i] - kp_canvas[0], axis=-1)).sum() / w.sum() / scale)
            r["cos_emb"] = float(emb[i] @ emb[0] / (np.linalg.norm(emb[i]) * np.linalg.norm(emb[0]) + 1e-12))
        rows[name] = r
    return dict(rows=rows, kp=kp_canvas, eff=eff, inputs=inputs, scale=scale)


# ----------------------------------------------------------------------------- vẽ

def plot_case(case, res, path, image):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.family": FONT["family"], "font.size": FONT["size"],
                         "font.weight": FONT["weight"], "axes.titleweight": FONT["weight"],
                         "axes.labelweight": FONT["weight"], "figure.titleweight": FONT["weight"],
                         "axes.titlesize": FONT["size"], "axes.labelsize": FONT["size"],
                         "figure.titlesize": FONT["size"], "xtick.labelsize": FONT["size"],
                         "ytick.labelsize": FONT["size"]})
    ext = [0, TARGET, TARGET, 0]
    settings = list(res["inputs"])
    use_density = settings != NO_DENSITY
    n_col = 3 if use_density else 2
    fig, axes = plt.subplots(len(settings), n_col, figsize=(n_col * 3.2, len(settings) * 3.2), squeeze=False)
    for i, name in enumerate(settings):
        rgb, den = res["inputs"][name]
        if use_density:
            a0, a1, a2 = axes[i]
            a1.imshow(den, cmap="gray", vmin=0, vmax=1, extent=ext)
        else:
            a0, a2 = axes[i]
        a0.imshow(rgb, extent=ext)
        a2.imshow(rgb * 0.45, extent=ext)
        order = np.argsort(-res["eff"][i])                                # chấm nhọn vẽ sau (nằm trên)
        sc = a2.scatter(res["kp"][i, order, 1], res["kp"][i, order, 0], s=9, c=res["eff"][i, order],
                        cmap="viridis_r", norm=LogNorm(vmin=1, vmax=256), edgecolors="white", linewidths=0.15)
        if use_density:
            a0.set_ylabel(ROW_LABELS[name])
        for b in holes(case, image):                                      # chỗ vật đã bị xoá
            x1, y1, x2, y2 = b * res["scale"]
            for a in (a0, a2):
                a.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.2, ls="--"))
        for a in axes[i]:
            a.set_xlim(0, TARGET)
            a.set_ylim(TARGET, 0)
            a.set_xticks([])
            a.set_yticks([])
    titles = [IMAGE_TITLE[image], "Density Map", "SpatialSoftmax Output"] if use_density else \
        [IMAGE_TITLE[image], "SpatialSoftmax Output"]
    for a, t in zip(axes[0], titles):
        a.set_title(t)
    right = 0.88 if use_density else 0.84
    fig.subplots_adjust(top=1 - (0.9 if use_density else 0.6) / fig.get_figheight(), right=right,
                        wspace=0.05, hspace=0.08)
    cb = fig.colorbar(sc, cax=fig.add_axes([right + 0.02, 0.2 if not use_density else 0.3,
                                            0.015 if use_density else 0.025, 0.6 if not use_density else 0.4]))
    cb.set_label("Effective cells")
    fig.suptitle(f"{case['iid']} ({case['cls']})")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../weights/celoc/best_paper.pth")
    ap.add_argument("--samples", default="../data/samples/test", help="checkpoint train trên samples/train")
    ap.add_argument("--ce130", default="../data/all_phase2_V2")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ids", nargs="*", help="chỉ định ảnh, vd 1077 568")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image", choices=list(IMAGE_TITLE), required=True)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message="All-NaN slice")
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    enc, info = load_vision_encoder(args.ckpt)
    print(f"[{time.time() - t0:5.1f}s] checkpoint: {info}")
    W = enc.projection.weight.detach().numpy()                       # [D, 1024]
    w = np.linalg.norm(W.reshape(W.shape[0], -1, 2), axis=(0, 2))     # [512]

    cases, skipped = select_cases(args)
    print(f"[{time.time() - t0:5.1f}s] {len(cases)} ca; bỏ: {skipped}")
    if not cases:
        return

    settings = settings_for(enc)
    print(f"[{time.time() - t0:5.1f}s] density {'CÓ' if settings == SETTINGS else 'KHÔNG'} -> setting {settings}")
    all_rows, per_case = [], []
    for k, case in enumerate(cases):
        res = run_case(enc, case, w, args.image)
        all_rows.append(res["rows"])
        per_case.append(dict(iid=case["iid"], branch=case["name"], cls=case["cls"], stems=case["stems"],
                             blob=case["blob"], n_objects=len(case["objects"]), size=case["original"].size,
                             rows=res["rows"]))
        if not args.no_figures:
            plot_case(case, res, os.path.join(args.out, f"{case['iid']}.png"), args.image)
        el = time.time() - t0
        head = (f"[{el:5.1f}s | ETA {el / (k + 1) * (len(cases) - k - 1):4.0f}s] {k + 1}/{len(cases)} "
                f"{case['name']:>9} n_obj {len(case['objects']):3d} | ")
        if settings == SETTINGS:
            rf, re = res["rows"]["full"], res["rows"]["empty"]
            print(head + f"lift_obj full {rf['lift_objects']:.2f} empty {re['lift_objects']:.2f} "
                  f"| loc full {rf['frac_localized']:.2f} | R1 full {rf['lift_R1']:.2f} -1 {res['rows']['minus1']['lift_R1']:.2f} "
                  f"| shift_empty {re['shift_px']:.1f}px cos {re['cos_emb']:.3f}")
        else:
            r = res["rows"]["none"]
            print(head + f"lift_obj {r['lift_objects']:.2f} | lift_holes {r.get('lift_holes', float('nan')):.2f} "
                  f"| loc {r['frac_localized']:.2f} | eff {r['eff_cells']:.1f}")

    cols = ["lift_holes", "lift_objects", "lift_blobs_full", "lift_R1", "lift_R2", "lift_T1", "frac_localized", "eff_cells",
            "shift_px", "cos_emb"]
    if settings != SETTINGS:
        cols = [c for c in cols if c not in ("lift_blobs_full", "shift_px", "cos_emb")]
    print("\nTRUNG VỊ trên các ca (lift: 1 = ngẫu nhiên):")
    print(f"{'setting':>8} " + " ".join(f"{c.replace('lift_', 'L_')[:12]:>12}" for c in cols))
    summary = {}
    for s in settings:
        summary[s] = {c: float(np.nanmedian([r[s].get(c, np.nan) for r in all_rows])) for c in cols}
        print(f"{s:>8} " + " ".join(f"{summary[s][c]:12.3f}" for c in cols))
    out = dict(args=vars(args), checkpoint={k: str(v) for k, v in info.items()}, w_c=w.tolist(),
               skipped=skipped, median=summary, cases=per_case)
    if settings == SETTINGS:
        blob = {k: float(np.median([c["blob"][k] for c in per_case])) for k in per_case[0]["blob"]}
        print(f"\nKiểm density (trung vị tỉ lệ pixel blob trong box): {blob}")
        out["blob_check_median"] = blob
    else:
        for c in per_case:
            c.pop("blob")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[{time.time() - t0:5.1f}s] xong -> {args.out}")


if __name__ == "__main__":
    main()

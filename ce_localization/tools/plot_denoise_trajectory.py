#!/usr/bin/env python3
"""Vẽ MỘT box qua từng bước khử nhiễu của bộ sinh box bài ADD (CE-Loc gốc của bài hoặc checkpoint GAMMA0 / 0.1).

CE-Loc sinh một box mỗi lần: mỗi ca sinh 1 box từ một nhiễu ban đầu (cùng `--seed` ⇒ cùng nhiễu ở mọi ca, nên khác
biệt chỉ do ảnh / text). Mỗi ca một hình, cột = các bước `--snapshots-*` (trái = nhiễu thuần, phải = kết quả):
  hàng 1  box x_t ở bước đó (đỏ đậm), các box của bước TRƯỚC (đỏ nhạt dần), lỗ đích `target_bbox` (xanh lá)
  hàng 2  đường đi của tâm box qua các bước đã vẽ, tới bước đó (× = xuất phát, chấm đỏ = hiện tại), tâm lỗ (xanh)

Ca tự chọn (split của `samples/` — split mà checkpoint của bài đã train):
  train         ảnh samples/train của lớp C (ảnh model ĐÃ TRAIN)
  test_same     ảnh samples/test cùng lớp C (lớp đã thấy, ảnh chưa thấy)
  test_unseen   ảnh samples/test của lớp KHÔNG có trong samples/train
  train_text    ảnh của ca `train` nhưng text = lớp của ca `test_unseen` (đổi object, giữ ảnh)
C = `--class`, mặc định lớp có nhiều mẫu test nhất trong các lớp có cả ở train. `--files rel[:lớp] ...` để tự chọn;
lớp đặc biệt: `<empty>` = chuỗi rỗng "" qua CLIP, `<zero>` = phần text của điều kiện đặt bằng 0 (bỏ hẳn text).
Trường thứ ba tuỳ chọn = density: `sample` (mặc định, density của mẫu) | `empty` (kênh density = 0 toàn bộ — NGOÀI phân phối)
| `blank` (density trống ĐÚNG kiểu dữ liệu: PNG toàn màu nền jet (0, 0, 127) cùng cỡ ảnh, qua cùng bước đọc — với đầu vào
của bài thành 14/255 trong vùng ảnh, 0 ở phần đệm).

Hai vòng lấy mẫu (`--samplers`): `ddpm` (1000 bước, đúng công thức) và `mock` (`x -= eps/100`, vòng repo của bài
dùng thật khi suy luận). `--track x0` vẽ x̂0 (box model đoán là đích ở mỗi bước) thay cho x_t.

`--image original`: thay ảnh inpaint bằng `ground_truth.jpg` (ảnh gốc CHƯA xoá gì) của cùng ảnh gốc trong `--ce130`; density
`sample` khi đó = bản density ĐẦY ĐỦ NHẤT của ảnh gốc đó (`full` của chỉ mục ALPHA3 `--density-index` — không có density của
đúng ảnh gốc); ảnh gốc không có đáp án nên hình chỉ vẽ box đỏ của model (JSON vẫn ghi khoảng cách tới vị trí vật mà mẫu đó sẽ xoá).

Ra (`--out`): `<sampler>_<ca>.png` (chữ trên hình bằng tiếng Anh) + `trajectories.json` (box, tâm, IoU với lỗ ở mọi bước đã vẽ).

~1–3 phút (quét ~26k annotation của samples/ + nạp CLIP), CPU được:
  cd /mnt/disk1/aiotlab/haitn/object-detection/ce_localization
  export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
  python ../tools/run_on_free_gpu.py -- tools/plot_denoise_trajectory.py --ckpt ../weights/add/paper/best_model.pth \\
      --out /mnt/disk1/aiotlab/haitn/output/add/viz
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from ce_localization.data.density import DensityIndex  # noqa: E402
from ce_localization.data.turns import image_inputs  # noqa: E402
from ce_localization.models.box_policy import BoxPolicy, unit_to_boxes  # noqa: E402
from ce_localization.models.detector import build_model  # noqa: E402
from ce_localization.models.text import TextTable, encode_class_names  # noqa: E402
from ce_localization.utils.box_ops_np import box_iou  # noqa: E402
from ce_localization.utils.log import fmt_time  # noqa: E402

SNAP_DDPM = (999, 800, 600, 400, 300, 200, 100, 50, 20, 0)
SNAP_MOCK = (99, 80, 60, 40, 20, 10, 5, 0)


def load_model(path, device, clip_name="openai/clip-vit-base-patch32"):
    """-> (model, style, cần density?, hàm text(names) -> TextTable, mô tả). Nhận checkpoint của bài hoặc của train.py."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ck:                                          # checkpoint CE-Loc gốc của bài
        model, clip, info = BoxPolicy.load_celoc_paper(ck)
        text = lambda names: TextTable(encode_class_names(names, clip_name, device=str(device), state_dict=clip))  # noqa: E731
        return model.to(device).eval(), "paper", info["in_channels"] == 4, text, f"CE-Loc gốc của bài {info}"
    cfg = ck["config"]
    if cfg["model"].get("arch") != "box_policy":
        sys.exit(f"{path}: không phải checkpoint bài add (model.arch {cfg['model'].get('arch')})")
    model = build_model(cfg, pretrained_backbone=False)
    model.load_state_dict(ck["model"])
    text = lambda names: TextTable(encode_class_names(names, cfg["model"]["clip_text"], device=str(device)))  # noqa: E731
    style = cfg["data"].get("input_style", "ours")          # GAMMA "CE-Loc gốc + R-50": paper (box chia canvas)
    if (style == "paper") != (model.box_norm == "canvas"):
        sys.exit(f"{path}: data.input_style {style} lệch model.box_norm {model.box_norm} — tool chưa hỗ trợ tổ hợp này")
    return (model.to(device).eval(), style, cfg["model"].get("in_channels", 3) == 4, text,
            f"{cfg['experiment']} iter {ck.get('iter')}")


def scan_samples(root, log):
    """-> {split: {rel ảnh: lớp}} của samples/train + samples/test."""
    out = {}
    for split in ("train", "test"):
        d = os.path.join(root, split, "annotation")
        files = sorted(os.listdir(d))
        out[split] = {}
        t0 = time.time()
        for i, f in enumerate(files, 1):
            with open(os.path.join(d, f)) as fh:
                out[split][f"{split}/images/{os.path.splitext(f)[0]}.png"] = json.load(fh)["class"]
            if i % 5000 == 0 or i == len(files):
                log(f"  quét annotation {split}: {i}/{len(files)} ({fmt_time(time.time() - t0)})")
    return out


def pick_cases(ann, cls, rng):
    tr_cls = set(ann["train"].values())
    by = lambda split, c: sorted(k for k, v in ann[split].items() if v == c)  # noqa: E731
    if cls is None:
        common = [c for c in set(ann["test"].values()) if c in tr_cls]
        cls = max(common, key=lambda c: (len(by("test", c)), c))
    a = by("train", cls)
    b = by("test", cls)
    if not a or not b:
        sys.exit(f"lớp {cls!r}: {len(a)} ảnh train, {len(b)} ảnh test — cần cả hai")
    cases = [{"name": "train", "file": a[rng.integers(len(a))], "text": cls},
             {"name": "test_same", "file": b[rng.integers(len(b))], "text": cls}]
    unseen = sorted({v for v in ann["test"].values() if v not in tr_cls})
    if unseen:
        u = unseen[rng.integers(len(unseen))]
        c = by("test", u)
        cases.append({"name": "test_unseen", "file": c[rng.integers(len(c))], "text": u})
        cases.append({"name": "train_text", "file": cases[0]["file"], "text": u})
    return cases


def canvas_rgb(x, style):
    img = x[:3].numpy().transpose(1, 2, 0)
    if style == "ours":
        img = img * IMAGENET_STD + IMAGENET_MEAN
    return img.clip(0, 1)


def rect(ax, b, color, lw=1.0, alpha=1.0):
    """Box xyxy; suy biến (w hoặc h < 0) vẽ nét đứt theo trị tuyệt đối."""
    from matplotlib.patches import Rectangle
    x1, y1, x2, y2 = b
    ax.add_patch(Rectangle((min(x1, x2), min(y1, y2)), abs(x2 - x1), abs(y2 - y1), fill=False, ec=color, lw=lw,
                           alpha=alpha, ls="--" if (x2 < x1 or y2 < y1) else "-"))


def iou_one(b, gt):
    b = np.array(b, dtype=np.float64)
    b[2], b[3] = max(b[2], b[0]), max(b[3], b[1])
    return float(box_iou(b[None], np.asarray(gt)[None])[0][0, 0])


def plot_case(path, bg, density, gt, steps, boxes, snaps, T, title):
    """Cột đầu = ĐẦU VÀO model: ảnh RGB (trên) + kênh density đúng như model nhận, thang màu cố định [0, 1] (dưới; None =
    model không có kênh density). Các cột sau = các bước: hàng 1 box (bước trước nhạt dần), hàng 2 đường đi của tâm.
    `gt` None (ảnh gốc chưa xoá: không có đáp án) -> chỉ vẽ box đỏ của model, không khung xanh, không IoU."""
    import matplotlib.pyplot as plt
    idx = [steps.index(t) for t in snaps if t in steps]
    ctr = (boxes[:, :2] + boxes[:, 2:]) / 2
    if gt is not None:
        g = np.asarray(gt)
        gc = ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)
    n = len(idx) + 1
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.4), squeeze=False)
    for ax in axes.ravel():
        ax.set_xlim(0, T)
        ax.set_ylim(T, 0)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axes[0, 0]
    ax.imshow(bg)
    if gt is not None:
        rect(ax, gt, "lime", lw=1.5)
    ax.set_title("input image" + ("\n(green = target hole)" if gt is not None else ""), fontsize=8)
    ax = axes[1, 0]
    if density is None:
        ax.set_facecolor("lightgray")
        ax.text(T / 2, T / 2, "no density\nchannel", ha="center", va="center", fontsize=8)
        ax.set_title("input density", fontsize=8)
    else:
        im = ax.imshow(density, cmap="viridis", vmin=0.0, vmax=1.0)
        if gt is not None:
            rect(ax, gt, "red", lw=1.2)
        ax.set_title(f"input density (0-1)\nmin {density.min():.3f} max {density.max():.3f}", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
    for j, k in enumerate(idx, start=1):
        for row in (0, 1):
            axes[row, j].imshow(bg)
            if gt is not None:
                rect(axes[row, j], gt, "lime", lw=1.5)
        ax = axes[0, j]
        for jj, kk in enumerate(idx[: j - 1]):                             # các box đã qua, nhạt dần về quá khứ
            rect(ax, boxes[kk], "red", lw=0.8, alpha=0.15 + 0.5 * (jj + 1) / max(j - 1, 1))
        rect(ax, boxes[k], "red", lw=2.0)
        ax.set_title(f"t = {steps[k]}" + ("" if gt is None else
                     f"\nIoU {iou_one(boxes[k], gt):.2f} | dist {np.hypot(ctr[k, 0] - gc[0], ctr[k, 1] - gc[1]):.0f} px"),
                     fontsize=8)
        ax = axes[1, j]
        trail = ctr[idx[:j]]                                               # chỉ nối các bước đã vẽ
        ax.plot(trail[:, 0], trail[:, 1], color="red", lw=1.0, alpha=0.8, marker="o", ms=2.5)
        ax.scatter([trail[0, 0]], [trail[0, 1]], marker="x", c="white", s=30, zorder=3)
        ax.scatter([trail[-1, 0]], [trail[-1, 1]], c="red", s=25, zorder=4)
        if gt is not None:
            ax.scatter([gc[0]], [gc[1]], c="lime", s=25, zorder=4)
    axes[0, 1].set_ylabel("box at step t", fontsize=9)
    axes[1, 1].set_ylabel("center path (x = start)", fontsize=9)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint của bài (weights/add/paper/best_model.pth) hoặc GAMMA0 / 0.1")
    ap.add_argument("--samples", default="../data/samples")
    ap.add_argument("--image", default="inpainted", choices=["inpainted", "original"],
                    help="original = ground_truth.jpg (chưa xoá gì) của cùng ảnh gốc")
    ap.add_argument("--ce130", default="../data/all_phase2_V2", help="--image original: lấy ground_truth.jpg")
    ap.add_argument("--density-index", default="../data/density_index.json",
                    help="--image original: chọn density đầy đủ nhất của ảnh gốc")
    ap.add_argument("--class", dest="cls", default=None, help="lớp C của ca train / test_same")
    ap.add_argument("--files", nargs="+", default=None, help="tự chọn ca: <split>/images/<file>.png[:lớp] ...")
    ap.add_argument("--samplers", nargs="+", default=["ddpm", "mock"], choices=["ddpm", "mock"])
    ap.add_argument("--track", default="xt", choices=["xt", "x0"], help="vẽ x_t (box đang khử) hay x̂0 (box đoán là đích)")
    ap.add_argument("--snapshots-ddpm", type=int, nargs="+", default=list(SNAP_DDPM))
    ap.add_argument("--snapshots-mock", type=int, nargs="+", default=list(SNAP_MOCK))
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    t0 = time.time()
    log = lambda *s: print(f"[{fmt_time(time.time() - t0)}]", *s, flush=True)  # noqa: E731
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, style, use_density, text_fn, desc = load_model(a.ckpt, dev)
    log(f"model: {desc} | đầu vào {style} | density {use_density} | T {model.num_timesteps} | {dev}")
    rng = np.random.default_rng(a.seed)
    if a.files:
        cases = []
        for i, f in enumerate(a.files):
            rel, _, rest = f.partition(":")
            cls, _, dens = rest.partition(":")
            if dens not in ("", "sample", "empty", "blank"):
                sys.exit(f"density {dens!r} trong {f!r}: chỉ 'sample', 'empty' hoặc 'blank'")
            if not cls:
                with open(os.path.join(a.samples, rel.replace("/images/", "/annotation/").rsplit(".", 1)[0] + ".json")) as fh:
                    cls = json.load(fh)["class"]
            cases.append({"name": f"case{i}_{cls.replace(' ', '_').strip('<>')}" + (f"_d{dens}" if dens in ("empty", "blank") else ""),
                          "file": rel, "text": cls, "density": dens or "sample"})
    else:
        cases = pick_cases(scan_samples(a.samples, log), a.cls, rng)
    names = sorted({c["text"] for c in cases} - {"<zero>"})
    texts = text_fn([("" if n == "<empty>" else n) for n in names]) if names else None
    if texts is not None:                                                # khoá bảng theo tên hiển thị
        texts.table = {n: texts.table["" if n == "<empty>" else n] for n in names}
    if "<zero>" in {c["text"] for c in cases}:
        texts = texts or TextTable({"<zero>": torch.zeros(512)})
        texts.table["<zero>"] = torch.zeros(next(iter(texts.table.values())).numel())
    os.makedirs(a.out, exist_ok=True)
    T = a.image_size
    key = "x_t" if a.track == "xt" else "x0_hat"
    out = {"ckpt": a.ckpt, "model": desc, "style": style, "seed": a.seed, "track": a.track, "cases": []}
    print(f"\n{'sampler':>7} {'ca':>12} {'text':>14} | IoU cuối | tâm cách lỗ (px) | √(wh) box / lỗ (px)")
    for c in cases:
        stem = os.path.splitext(os.path.basename(c["file"]))[0]
        split = c["file"].split("/")[0]
        with open(os.path.join(a.samples, split, "annotation", stem + ".json")) as fh:
            ann = json.load(fh)
        dmode = c.get("density", "sample") if use_density else "none"
        if a.image == "original":
            iid = stem.rsplit("_", 1)[0]
            br = sorted(glob.glob(os.path.join(a.ce130, "*", f"{iid}_b*")))
            if not br:
                sys.exit(f"không thấy nhánh {iid}_b* trong {a.ce130}")
            img = Image.open(os.path.join(br[0], "ground_truth.jpg")).convert("RGB")
            if dmode == "sample":                                          # không có density của đúng ảnh gốc
                dmode = "full"
            didx = DensityIndex(a.density_index, a.samples) if dmode == "full" else None
            dfull = didx.path(didx.pick(iid, "full")[0]) if didx is not None else None
        else:
            img = Image.open(os.path.join(a.samples, c["file"])).convert("RGB")
            dfull = None
        dpath = (Image.new("RGB", img.size, (0, 0, 127)) if dmode == "blank" else dfull if dmode == "full" else
                 os.path.join(a.samples, split, "density", stem + ".png") if use_density else None)
        x, scale, nw, nh = image_inputs(img, dpath, T, style)
        if dmode == "empty":                                              # density cố tình trống rỗng
            x[3] = 0
        cx, cy, w, h = (v * scale for v in ann["target_bbox"])                   # cxcywh pixel ảnh gốc (cạm bẫy 1)
        gt = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        whwh = torch.tensor([[T, T, T, T]] if style == "paper" else [[nw, nh, nw, nh]], dtype=torch.float32, device=dev)
        bg = canvas_rgb(x, style)
        for s in a.samplers:
            snaps = a.snapshots_ddpm if s == "ddpm" else a.snapshots_mock
            gen = torch.Generator(device=dev.type).manual_seed(a.seed)          # cùng nhiễu ban đầu ở mọi ca
            _, traj = model.sample(x[None].to(dev), texts([c["text"]], dev), torch.tensor([[nh, nw]], device=dev), 1,
                                   generator=gen, sampler=s, record="all", null_text=c["text"] == "<zero>")
            steps = [st["t"] for st in traj]
            boxes = np.stack([unit_to_boxes(st[key][0, 0].to(dev), whwh[0]).cpu().numpy() for st in traj])   # [S,4]
            shown = {"<empty>": '"" (empty string)', "<zero>": "none (text embedding = 0)"}.get(c["text"], f"'{c['text']}'")
            orig = a.image == "original"
            title = (f"{s} | {c['name']}: {c['file']} ({a.image} image) | text {shown} (true class '{ann['class']}') | "
                     f"density {dmode} | tracked: {'x_t' if a.track == 'xt' else 'x0_hat'} | red = generated box"
                     + ("" if orig else ", green = target hole"))
            plot_case(os.path.join(a.out, f"{s}_{c['name']}.png"), bg, x[3].numpy() if x.shape[0] == 4 else None,
                      None if orig else gt, steps, boxes, snaps, T, title)
            ctr = (boxes[:, :2] + boxes[:, 2:]) / 2
            dist = np.hypot(ctr[:, 0] - (gt[0] + gt[2]) / 2, ctr[:, 1] - (gt[1] + gt[3]) / 2)
            idx = [steps.index(t) for t in snaps if t in steps]
            row = {"sampler": s, "case": c["name"], "file": c["file"], "text": c["text"], "gt_class": ann["class"], "density": dmode,
                   "gt_canvas": gt, "steps": [steps[k] for k in idx], "boxes_canvas": boxes[idx].tolist(),
                   "iou": [iou_one(boxes[k], gt) for k in idx], "center_dist_px": dist[idx].tolist()}
            out["cases"].append(row)
            fb = boxes[-1]
            print(f"{s:>7} {c['name']:>12} {c['text'][:14]:>14} | {row['iou'][-1]:.3f}    | {dist[-1]:7.1f}          | "
                  f"{np.sqrt(max(fb[2] - fb[0], 0) * max(fb[3] - fb[1], 0)):5.1f} / {np.sqrt(w * h):5.1f}", flush=True)
    with open(os.path.join(a.out, "trajectories.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log(f"-> {a.out} ({len(os.listdir(a.out))} file)")


if __name__ == "__main__":
    main()

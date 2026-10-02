#!/usr/bin/env python3
"""Soi checkpoint `model.arch: box_refiner` (GAMMA1) — MỘT hình 4 hàng × 6 cột mỗi ca (nhánh, lượt t):

  hàng 1  ảnh inpaint lượt t (đã xoá t vật) + density của CHÍNH ảnh đó (file `samples/` khớp bằng hash pixel)
  hàng 2  ảnh inpaint như hàng 1 + density TRỐNG
  hàng 3  ảnh gốc chưa xoá (`ground_truth.jpg`) + density của ảnh gốc (bản `full` của chỉ mục ALPHA3 — như eval.py)
  hàng 4  ảnh gốc như hàng 3 + density TRỐNG
  (density trống = PNG toàn màu nền jet (0, 0, 127) cùng cỡ ảnh qua cùng bước đọc — như `CE130AddDataset(density="empty")`)

  cột 1     Density Map — kênh density model nhận (thang [0, 1] cố định), lỗ (xanh lá) để so hai bản density
  cột 2–5   quỹ đạo khử nhiễu, mỗi ô = trạng thái SAU một bước: box x_t THẬT (đỏ đậm, không kẹp — có thể ra ngoài canvas) +
            tâm (chấm đỏ); ô cuối = đầu ra (t = 0), vẽ y như các ô khác. Box các trạng thái trước mờ dần (~20 box trên cả
            quỹ đạo); đường đen = tâm qua MỌI trạng thái, từ nhiễu thuần ban đầu (t = 999 với DDIM) tới trạng thái hiện tại.
            DDIM (`--steps` 4, `--eta` 1 như eval / 0 tất định): x_t_next = √ᾱ·x̂0 + nhiễu theo lịch (√ᾱ_t trên tiêu đề).
            `--sampler mock` (vòng của CE-Loc gốc): nhiễu thuần nhưng báo model t = 99..0, `x -= ε̂/100` (ε̂ suy từ x̂0
            tầng 6), 100 bước, vẽ 4 trạng thái cách đều; `--t-start` = DDIM bắt đầu ở t nhỏ hơn 999. x̂0 (box tầng 6 model
            đoán ở bước đó) chỉ vẽ khi `--show-x0`. Lỗ (xanh lá, CHỈ ảnh inpaint): lỗ mới nhất = đích (đậm), lỗ cũ nét đứt.
            Một box mỗi lần sinh; cùng `--seed` ⇒ cùng nhiễu ban đầu ở cả 4 hàng, khác biệt chỉ do ảnh / density.
  cột 6     SpatialSoftmax Output trên C5 (token `vis`): mỗi chấm = một kênh trong 2048, vị trí = toạ độ kỳ vọng (đổi trục
            như `tools/plot_spatial_softmax.py`), màu = số ô hiệu dụng (log), cỡ ∝ ‖W ss_proj‖ của kênh.

`--mode stages`: thay 4 cột quỹ đạo bằng 6 cột = box ra của 6 tầng trong MỘT lượt khử nhiễu ở t = `--stage-t`: trạng
thái 0 = box nhiễu đã kẹp mà tầng 1 nhận (mặc định t = T − 1 = 999), rồi tầng 1..6 (box tầng k là đầu vào RoI của tầng k + 1); hình 4 × 8 cột.

Chọn ca:
  `--files`  file `samples/` (vd test/images/3342_3.png — ảnh đã soi checkpoint của bài) -> dò ngược (split CE-130, nhánh,
             lượt) bằng hash pixel; tên thư mục `samples/` KHÔNG phải split CE-130 (samples/train = train + val).
  `--cases`  `<split>/<nhánh>[:lượt]` (mặc định lượt 1).
  `--quota`  số ca mỗi split train / val / test (mặc định 0 0 4); phần thiếu sau `--files` / `--cases` tự chọn lượt 1 từ
             `--pool` nhánh ngẫu nhiên mỗi split: ưu tiên ca density ảnh gốc khác bản inpaint ở lỗ (`gap` = density TB trong
             lỗ của ảnh gốc − của ảnh inpaint ≥ `--min-gap`), rồi lấy cách đều theo cỡ lỗ √(wh)/canvas ⇒ đủ dải kích cỡ vật.
Text = tên lớp thật. Chữ trên hình tiếng Anh. Ra `--out`: `<split>_<nhánh>_t<lượt>.png` + `refiner_steps.json`.
Chỉ nạp model + CLIP + vài trăm ảnh để chọn, ~3–8 phút, CPU được:
  python tools/plot_refiner_steps.py --ckpt ../weights/add/gamma1/best.pth --quota 5 5 10 \\
      --files test/images/3342_3.png ... --out ../../output/gamma/viz/gamma1/4_step
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.dataset import _read_annotation  # noqa: E402
from ce_localization.data.density import DensityIndex  # noqa: E402
from ce_localization.data.turns import image_inputs, pixel_hash  # noqa: E402
from ce_localization.models.box_policy import _paper_spatial_softmax, unit_to_boxes  # noqa: E402
from ce_localization.models.detector import build_model  # noqa: E402
from ce_localization.models.text import TextTable, encode_class_names  # noqa: E402
from ce_localization.tools.plot_spatial_softmax import SHARP, effective_cells, keypoint_pixels  # noqa: E402
from ce_localization.utils.box_ops_np import box_iou  # noqa: E402

SPLITS = ("train", "val", "test")
ROWS = (("inpainted", "own"), ("inpainted", "blank"), ("original", "own"), ("original", "blank"))


def row_label(kind, dens, turn):
    if kind == "inpainted":
        img = f"Inpainted ({turn} object{'s' if turn > 1 else ''} removed)"
        return f"{img}\ndensity: {'own (inpainted image)' if dens == 'own' else 'blank'}"
    return f"Original (nothing removed)\ndensity: {'own (original image)' if dens == 'own' else 'blank'}"


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if cfg["model"].get("arch") != "box_refiner":
        sys.exit(f"{path}: không phải checkpoint box_refiner (model.arch {cfg['model'].get('arch')})")
    if cfg["data"].get("input_style", "ours") != "paper":
        sys.exit("tool chỉ vẽ đầu vào kiểu bài (`data.input_style: paper`)")
    model = build_model(cfg, pretrained_backbone=False)
    model.load_state_dict(ck["model"])
    if model.ss_kind != "paper":
        sys.exit("tool chỉ vẽ SpatialSoftmax kiểu bài (`model.ss_kind: paper`)")
    return model.to(device).eval(), cfg, ck.get("iter")


def match_sample(samples, image_path, iid):
    """ảnh inpaint của (nhánh, lượt) -> 'train/<stem>' | 'test/<stem>' của file samples/ khớp pixel, None nếu không thấy."""
    h = pixel_hash(image_path)
    for sdir in ("train", "test"):
        for p in sorted(glob.glob(os.path.join(samples, sdir, "images", f"{iid}_*.png"))):
            if pixel_hash(p) == h:
                return f"{sdir}/{os.path.splitext(os.path.basename(p))[0]}"
    return None


def find_turn(a, rel):
    """file samples/ (vd test/images/3342_3.png) -> (split CE-130, nhánh, lượt) có ảnh inpaint trùng pixel, None nếu không."""
    iid = os.path.basename(rel).split("_")[0]
    h = pixel_hash(os.path.join(a.samples, rel))
    for split in SPLITS:
        for bdir in sorted(glob.glob(os.path.join(a.ce130, split, f"{iid}_b*"))):
            for p in sorted(glob.glob(os.path.join(bdir, "inpainted_turn_*.png"))):
                if pixel_hash(p) == h:
                    return split, os.path.basename(bdir), int(os.path.splitext(p)[0].rsplit("_", 1)[1])
    return None


def load_case(a, split, br, turn, dindex, T):
    """(split, nhánh, lượt) -> dict ca (ảnh, đường dẫn density, lỗ, gap, cỡ lỗ) hoặc None nếu thiếu dữ liệu."""
    bdir = os.path.join(a.ce130, split, br)
    iid = br.split("_b")[0]
    ann = _read_annotation(bdir)
    ipath = os.path.join(bdir, f"inpainted_turn_{turn}.png")
    if ann is None or iid not in dindex or not os.path.exists(ipath) or len(ann["inpainted_bboxes"]) < turn:
        return None
    sample = match_sample(a.samples, ipath, iid)
    if sample is None:
        return None
    sdir, stem = sample.split("/")
    imgs = {"inpainted": Image.open(ipath).convert("RGB"),
            "original": Image.open(os.path.join(bdir, "ground_truth.jpg")).convert("RGB")}
    own = {"inpainted": os.path.join(a.samples, sdir, "density", stem + ".png"),
           "original": dindex.path(dindex.pick(iid, "full")[0])}
    den, scale = {}, None
    for kind in imgs:
        x, scale, _, _ = image_inputs(imgs[kind], own[kind], T, "paper")
        den[kind] = x[3].numpy()
    holes = np.asarray(ann["inpainted_bboxes"][:turn], dtype=np.float64).reshape(-1, 4) * scale
    x1, y1, x2, y2 = [int(round(v)) for v in holes[-1]]
    gap = float(den["original"][y1:y2, x1:x2].mean() - den["inpainted"][y1:y2, x1:x2].mean()) if x2 > x1 and y2 > y1 else 0.0
    w, h = holes[-1, 2] - holes[-1, 0], holes[-1, 3] - holes[-1, 1]
    return {"split": split, "branch": br, "turn": turn, "class": ann["class_based_caption"], "sample": sample,
            "imgs": imgs, "own": own, "holes": holes, "gap": gap, "size": float(np.sqrt(max(w * h, 0)) / T)}


def pick_cases(a, dindex, T, rng):
    cases = []
    for rel in a.files or []:
        f = find_turn(a, rel)
        c = None if f is None else load_case(a, *f, dindex, T)
        print(f"--files {rel} -> " + ("KHÔNG khớp nhánh / lượt nào" if c is None else
                                       f"{c['split']}/{c['branch']} lượt {c['turn']}"), flush=True)
        if c is not None:
            c["source"] = rel
            cases.append(c)
    for spec in a.cases or []:
        sb, _, t = spec.partition(":")
        split, br = sb.split("/")
        c = load_case(a, split, br, int(t or 1), dindex, T)
        if c is None:
            print(f"--cases {spec}: thiếu dữ liệu, bỏ", flush=True)
        else:
            cases.append(c)
    for split, quota in zip(SPLITS, a.quota):
        need = quota - sum(c["split"] == split for c in cases)
        if need <= 0:
            continue
        taken = {c["branch"].split("_b")[0] for c in cases}
        names = sorted(os.listdir(os.path.join(a.ce130, split))) if os.path.isdir(os.path.join(a.ce130, split)) else []
        pool = []
        for i in rng.permutation(len(names)):
            if len(pool) >= a.pool:
                break
            if names[i].split("_b")[0] in taken:
                continue
            c = load_case(a, split, names[i], 1, dindex, T)
            if c is not None:
                pool.append(c)
        good = [c for c in pool if c["gap"] >= a.min_gap]
        cand = sorted(good if len(good) >= need else pool, key=lambda c: c["size"])
        idx = sorted(set(np.linspace(0, len(cand) - 1, need).round().astype(int).tolist())) if cand else []
        cases += [cand[i] for i in idx]
        print(f"{split}: pool {len(pool)} (gap >= {a.min_gap}: {len(good)}), chọn {len(idx)} theo cỡ lỗ: "
              + ", ".join(f"{cand[i]['branch']} {cand[i]['size']:.3f}" for i in idx), flush=True)
    return sorted(cases, key=lambda c: (SPLITS.index(c["split"]), c["size"]))


def rect(ax, b, color, lw=1.0, alpha=1.0, ls="-"):
    import matplotlib.patches as mp
    ax.add_patch(mp.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor=color, linewidth=lw,
                              alpha=alpha, linestyle=ls))


def sorted_box(b):
    """xyxy có thể w, h âm (box nhiễu không kẹp) -> xyxy (min, max) để vẽ / chấm."""
    b = np.asarray(b, dtype=np.float64)
    return np.array([min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])])


def center(b):
    b = np.asarray(b)
    return np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2], -1)


@torch.no_grad()
def spatial_softmax(model, x):
    """-> (toạ độ [C,2] (dọc, ngang) trong [−1, 1], attention [C,H,W]) trên C5, đúng phép tính của token `vis`."""
    c5 = model.backbone.forward_c5(x)
    N, C, H, W = c5.shape
    att = F.softmax(c5.reshape(N, C, -1), dim=-1).reshape(N, C, H, W)
    return _paper_spatial_softmax(c5).reshape(N, C, 2)[0].cpu(), att[0].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint box_refiner (vd ../weights/add/gamma1/best.pth)")
    ap.add_argument("--ce130", default="../data/all_phase2_V2")
    ap.add_argument("--samples", default="../data/samples")
    ap.add_argument("--density-index", default="../data/density_index.json")
    ap.add_argument("--files", nargs="*", default=None, help="file samples/ phải có trong các ca, vd test/images/3342_3.png")
    ap.add_argument("--cases", nargs="*", default=None, help="<split>/<nhánh>[:lượt], vd test/1050_b1:2")
    ap.add_argument("--quota", type=int, nargs=3, default=[0, 0, 4], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--pool", type=int, default=40, help="số nhánh ngẫu nhiên xét mỗi split khi tự chọn")
    ap.add_argument("--min-gap", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eta", type=float, default=1.0, help="DDIM eta: 1 = như eval (thêm nhiễu MỚI mỗi bước), 0 = tất định")
    ap.add_argument("--sampler", default="ddim", choices=["ddim", "mock"],
                    help="mock = vòng của CE-Loc gốc: nhiễu thuần nhưng t = mock_steps−1..0, x -= ε̂/mock_steps")
    ap.add_argument("--t-start", type=int, default=None, help="DDIM bắt đầu ở t này (mặc định T − 1 = 999)")
    ap.add_argument("--mock-steps", type=int, default=100, help="số bước vòng mock (bài: 100); vẽ 4 trạng thái cách đều")
    ap.add_argument("--show-x0", action="store_true", help="vẽ thêm x̂0 (box tầng 6 model đoán ở bước đó), xanh chấm chấm")
    ap.add_argument("--mode", default="steps", choices=["steps", "stages"],
                    help="steps = quỹ đạo khử nhiễu (4 trạng thái); stages = MỘT lượt ở t = --stage-t, cột = box ra của 6 tầng")
    ap.add_argument("--stage-t", type=int, default=None, help="--mode stages: mức nhiễu của lượt (mặc định T − 1, cao nhất)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FuncFormatter, NullFormatter
    from matplotlib.lines import Line2D

    dev = torch.device(a.device)
    model, cfg, it = load_model(a.ckpt, dev)
    T = cfg["data"]["image_size"]
    ab = model.alphas_cumprod.cpu()
    if a.stage_t is None:
        a.stage_t = model.num_timesteps - 1
    if not 0 <= a.stage_t < model.num_timesteps:
        sys.exit(f"--stage-t {a.stage_t} ngoài [0, {model.num_timesteps - 1}]")
    dindex = DensityIndex(a.density_index, a.samples)
    w = model.memory.ss_proj.weight.detach().reshape(model.memory.ss_proj.out_features, -1, 2).norm(dim=(0, 2)).cpu().numpy()
    size = 4 + 60 * w / w.max()
    os.makedirs(a.out, exist_ok=True)
    cases = pick_cases(a, dindex, T, np.random.default_rng(a.seed))
    name = "mock" if a.sampler == "mock" else "DDIM"
    summary = []
    for cs in cases:
        split, br, turn, cls, holes = cs["split"], cs["branch"], cs["turn"], cs["class"], cs["holes"]
        hole = holes[-1]                                                          # lỗ mới nhất = đích
        hc = center(hole)
        text = TextTable(encode_class_names([cls], cfg["model"]["clip_text"], device=str(dev)))([cls], dev)
        ncol = 8 if a.mode == "stages" else 6
        fig, axes = plt.subplots(4, ncol, figsize=(ncol * 3.3, 4 * 3.5), squeeze=False)
        case = {k: cs[k] for k in ("split", "branch", "turn", "class", "sample", "size")}
        case.update({"source": cs.get("source"), "density_gap_in_hole": cs["gap"], "rows": []})
        for r, (kind, dens) in enumerate(ROWS):
            img = cs["imgs"][kind]
            dsrc = cs["own"][kind] if dens == "own" else Image.new("RGB", img.size, (0, 0, 127))
            x, _, nw, nh = image_inputs(img, dsrc, T, "paper")
            xb, vhw = x[None].to(dev), torch.tensor([[nh, nw]], device=dev)
            g = torch.Generator(device=dev.type).manual_seed(a.seed)
            canvas = torch.full((4,), float(T))
            if a.mode == "stages":                                               # MỘT lượt ở t = --stage-t: 6 tầng
                _, steps = model.sample(xb, text, vhw, 1, generator=g, steps=1, return_stages=True, t_start=a.stage_t)
                st0 = steps[0]
                boxes = [st0["noisy"][0, 0].numpy()] + [b for b in st0["stages"][:, 0, 0].numpy()]
                tlabel = [f"input t={st0['t']}"] + [f"stage {k}" for k in range(1, len(boxes))]
                snaps = list(range(1, len(boxes)))
            else:
                n_run = a.mock_steps if a.sampler == "mock" else a.steps
                u, steps = model.sample(xb, text, vhw, 1, generator=g, steps=n_run, eta=a.eta, return_stages=True,
                                        sampler=a.sampler, t_start=a.t_start)
                # trạng thái: x_t trước mỗi bước (t của bước đó) + đầu ra (t = 0); box THẬT, không kẹp
                boxes = [sorted_box(unit_to_boxes(st["x"][0, 0], canvas).numpy()) for st in steps]
                boxes.append(sorted_box(unit_to_boxes(u[0, 0].cpu(), canvas).numpy()))
                tlabel = [f"t={st['t']}" for st in steps] + ["output"]
                snaps = sorted(set(np.linspace(1, len(boxes) - 1, 4).round().astype(int).tolist()))
            cents = [center(b) for b in boxes]
            trail = max(1, len(boxes) // 20)                                     # box vết: ~20 box trên cả quỹ đạo
            bg = x[:3].numpy().transpose(1, 2, 0)
            den = x[3].numpy()
            inp = kind == "inpainted"
            ax = axes[r, 0]
            im = ax.imshow(den, cmap="viridis", vmin=0, vmax=1)
            rect(ax, hole, "lime", lw=1.2)
            ax.set_title(f"Density Map ({dens})\nmin {den.min():.3f} max {den.max():.3f} | in hole "
                         f"{den[int(hole[1]):int(hole[3]), int(hole[0]):int(hole[2])].mean():.3f}", fontsize=8.5)
            ax.set_ylabel(row_label(kind, dens, turn), fontsize=9.5)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            row = {"image": kind, "density": dens, "states": []}
            iou_h = lambda b: float(box_iou(b[None].clip(min=0), hole[None])[0][0, 0])  # noqa: E731
            for k, si in enumerate(snaps):
                ax = axes[r, 1 + k]
                ax.imshow(bg)
                if inp:
                    for hb in holes[:-1]:
                        rect(ax, hb, "lime", lw=1.0, ls="--")
                    rect(ax, hole, "lime", lw=1.6)
                    ax.plot(*hc, "+", color="lime", ms=10, mew=2)
                prev = list(range(0, si, trail))                                 # box các trạng thái trước: mờ dần
                for n_, j in enumerate(prev):
                    rect(ax, boxes[j], "red", lw=0.8, alpha=0.08 + 0.4 * (n_ + 1) / max(len(prev), 1))
                rect(ax, boxes[si], "red", lw=2.2)
                pc = np.stack(cents[: si + 1])
                ax.plot(pc[:, 0], pc[:, 1], "-", color="black", lw=1.4, alpha=0.85, zorder=5)
                for j in [0] + snaps[:k]:
                    ax.plot(*cents[j], "o", color="0.7", mec="black", mew=0.6, ms=5, zorder=6)
                    ax.annotate(tlabel[j], cents[j], xytext=(4, 4), textcoords="offset points", fontsize=7,
                                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7), zorder=7)
                ax.plot(*cents[si], "o", color="red", mec="black", mew=0.6, ms=7, zorder=8)
                if a.show_x0 and a.mode == "steps":
                    rect(ax, steps[si - 1]["stages"][-1, 0, 0].numpy(), "deepskyblue", lw=1.2, ls=":")
                is_out = si == len(boxes) - 1
                if a.mode == "stages":
                    ttl = f"stage {si}/{len(boxes) - 1} output (t = {steps[0]['t']})"
                    rec = {"stage": si, "t": steps[0]["t"], "box": boxes[si].tolist(), "center": cents[si].tolist()}
                else:
                    ttl = (f"after {name} step {si}/{len(steps)}: "
                           + ("output (t = 0)" if is_out else
                              f"x_t, t = {steps[si]['t']} (sqrt(abar) {float(ab[steps[si]['t']].sqrt()):.2f})"))
                    rec = {"after_step": si, "t": 0 if is_out else steps[si]["t"], "box": boxes[si].tolist(),
                           "center": cents[si].tolist()}
                if inp:
                    rec["iou_hole"] = iou_h(boxes[si])
                    rec["center_dist"] = float(np.linalg.norm(cents[si] - hc))
                    ttl += f"\nIoU(box, hole) {rec['iou_hole']:.2f} | center dist {rec['center_dist']:.0f} px"
                ax.set_title(ttl, fontsize=8.5)
                row["states"].append(rec)
            row["path"] = [[lab, *b.tolist()] for lab, b in zip(tlabel, boxes)]
            xy, att = spatial_softmax(model, xb)
            eff = effective_cells(att).numpy()
            px, py = keypoint_pixels(xy.numpy(), att.shape[1])
            ax = axes[r, ncol - 1]
            ax.imshow(bg)
            order = np.argsort(-eff)
            sc = ax.scatter(px[order], py[order], c=eff[order], s=size[order], cmap="plasma_r",
                            norm=LogNorm(vmin=1, vmax=att.shape[1] * att.shape[2]), edgecolors="black", linewidths=0.3,
                            alpha=0.9)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            # nhãn trục log dạng số thường ("1", "10", "100"), KHÔNG mathtext "$10^{k}$": matplotlib cũ + pyparsing ≥ 3.3
            # (Kaggle) cảnh báo PyparsingDeprecationWarning mỗi lần dựng / parse mathtext
            cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            cb.ax.yaxis.set_minor_formatter(NullFormatter())
            sharp = float((eff < SHARP).mean())
            ax.set_title(f"SpatialSoftmax Output ({att.shape[0]} ch)\nsharp (<{SHARP} cells) {sharp * 100:.1f}% | "
                         f"median cells {np.median(eff):.0f}", fontsize=8.5)
            if inp:
                rect(ax, hole, "lime", lw=1.2)
            row["sharp_frac"], row["median_cells"] = sharp, float(np.median(eff))
            case["rows"].append(row)
            for c in range(ncol):
                axes[r, c].set_xlim(0, T)
                axes[r, c].set_ylim(T, 0)
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
        handles = [Line2D([], [], color="red", lw=2.2, marker="o",
                          label="box + center of this stage" if a.mode == "stages" else
                          "box + center at this state (last = output)"),
                   Line2D([], [], color="red", alpha=0.3, label="earlier stages / input box (fading)" if a.mode == "stages"
                          else "boxes at earlier states (fading)"),
                   Line2D([], [], color="black", marker="o", mfc="0.7", lw=1.4,
                          label="center path: input -> stage 1..k" if a.mode == "stages" else "center path, every step")]
        if a.show_x0:
            handles.append(Line2D([], [], color="deepskyblue", ls=":", label="predicted x0 (stage 6) of that step"))
        handles += [Line2D([], [], color="lime", lw=1.6, marker="+", label="latest removed object (target)"),
                    Line2D([], [], color="lime", lw=1.0, ls="--", label="earlier removed objects")]
        fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
        seen = "train image, seen class" if split == "train" else "unseen class"
        fig.suptitle(f"{os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))}/{os.path.basename(a.ckpt)} (iter {it}) | "
                     f"{split} {br}, turn {turn} ({seen} '{cls}') | text = '{cls}' | one box per generation, seed {a.seed} | "
                     + (f"ONE denoising pass at t={a.stage_t}: noisy box (clipped to canvas, as stage 1 sees it) -> 6 stages"
                        if a.mode == "stages" else
                        f"mock sampler (paper): pure noise labelled t={a.mock_steps - 1}..0, x -= eps/{a.mock_steps}"
                        if a.sampler == "mock" else
                        f"DDIM {a.steps} steps, eta {a.eta:g}, from t={model.num_timesteps - 1 if a.t_start is None else a.t_start}"),
                     fontsize=10)
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        fig.savefig(os.path.join(a.out, f"{split}_{br}_t{turn}.png"), dpi=100)
        plt.close(fig)
        summary.append(case)
        print(f"{split:5s} {br:10s} t{turn} ({cls}, hole size {cs['size']:.3f}, gap {cs['gap']:.3f}) | " + " | ".join(
            f"{r['image'][:4]}/{r['density']}: " + (f"IoU out {r['states'][-1]['iou_hole']:.2f}" if "iou_hole" in r["states"][-1]
                                                     else "no hole") for r in case["rows"]), flush=True)
    with open(os.path.join(a.out, "refiner_steps.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

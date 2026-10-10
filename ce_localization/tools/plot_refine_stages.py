#!/usr/bin/env python3
"""Soi checkpoint `model.arch: propose_refine` (GAMMA2 / 2.1 / 3 / 3.1 — docs/EXPERIMENT_GAMMA.md mục 13) — MỘT hình mỗi ca:

  CẢ BA hàng: CÙNG ảnh inpaint lượt t + density của CHÍNH ảnh đó, cùng seed ⇒ cùng box CE-Loc ở hàng 1–2 (= biến thể eval):
  hàng 1  `_t0`    box CE-Loc (vòng mock 100 bước của bài) vào thẳng refine, t* = 0 (không nhiễu)
  hàng 2  `_t<t*>` box CE-Loc + nhiễu tới t* (`--refine-t`, mặc định 100) -> refine
  hàng 3  `_noise` refine THUẦN từ box nhiễu ở t = 999 (không dùng CE-Loc)
  cột 2   box CE-Loc (đỏ) + box stage 1 nhận (nét đứt cam); hàng 3 chỉ có box nhiễu
  cột 1     ảnh đầu vào (canvas letterbox, khung cắt về vùng ảnh thật; lỗ xanh lá — lỗ mới nhất đậm, lỗ cũ nét đứt)
  cột 3–8   box ra của 6 stage refine trong MỘT lượt (refine 1 bước như eval): stage k đậm, box stage trước mờ, đường tâm
            (đề xuất ->) nhiễu -> stage 1..k; tiêu đề: IoU / khoảng cách tâm tới lỗ mới nhất + attention của query lên [t ; text ; vis]
  cột 9     SpatialSoftmax Output trên C5 ResNet18 (chung CE-Loc + refine, mask phần đệm nếu `ss_mask`): mỗi chấm một kênh,
            màu = số ô hiệu dụng (log), cỡ ∝ ‖W ss_proj‖ của refine (token `vis`)
Mô hình có box vật (GAMMA3 / 3.1): truyền box vật đang có như eval. Tiêu đề ô: IoU với lỗ MỚI NHẤT / với lỗ BẤT KỲ.
`--scan N`: KHÔNG vẽ ngay — chạy nhanh N ca test ngẫu nhiên (lượt ngẫu nhiên), đo IoU stage 6 với lỗ mới nhất của 3 hàng, rồi vẽ
`--pick` ca mỗi nhóm có chênh lớn nhất (giải thích chênh metric giữa biến thể): `noise_needed` (t* ≫ t0: box CE-Loc phải cộng
nhiễu refine mới sửa), `celoc_helps` (t* ≫ noise: thiếu CE-Loc thì refine đi lỗ khác / trượt), `noise_wins` (noise ≫ t*); ghi
`scan.json` (mọi ca đã quét).
Chọn ca như `tools/plot_refiner_steps.py` (`--files` / `--cases` / `--quota`). Chữ trên hình tiếng Anh; `--name` = tên hiển thị.
  python tools/plot_refine_stages.py --ckpt ../weights/add/gamma2/best.pth --name "CE-Loc (frozen) + Refiner" \\
      --cases test/3342_b1:3 --quota 0 0 0 --out ../../output/add/viz/gamma2/stages
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.density import DensityIndex  # noqa: E402
from ce_localization.data.turns import image_inputs  # noqa: E402
from ce_localization.engine.diffusion import diffusion_from_boxes  # noqa: E402
from ce_localization.models.box_policy import unit_to_boxes  # noqa: E402
from ce_localization.models.detector import build_model  # noqa: E402
from ce_localization.models.propose_refine import refine_noisy_boxes  # noqa: E402
from ce_localization.models.text import TextTable, encode_class_names  # noqa: E402
from ce_localization.tools.plot_refiner_steps import center, pick_cases, rect  # noqa: E402
from ce_localization.tools.plot_spatial_softmax import SHARP, effective_cells, keypoint_pixels  # noqa: E402
from ce_localization.utils.box_ops_np import box_iou  # noqa: E402

ROW_LABEL = {"t0": "CE-Loc -> Refiner\n(no noise, t*=0)", "ce": "CE-Loc + noise\n-> Refiner", "noise": "Refiner only\n(from pure noise)"}
ROW_SPECS = ("t0", "ce", "noise", "t0_nogeo", "ce_nogeo", "noise_nogeo")   # `_nogeo`: TẮT box vật (= biến thể `_nogeo` của eval)


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if cfg["model"].get("arch") != "propose_refine":
        sys.exit(f"{path}: không phải checkpoint propose_refine (model.arch {cfg['model'].get('arch')})")
    model = build_model(cfg, pretrained_backbone=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, ck.get("iter")


@torch.no_grad()
def run_one(model, x, vhw, text, refine_t, generator, objects=None, use_objects=True):
    """Đúng `ProposeRefine.sample_variants` với 1 box, refine 1 bước -> (box CE-Loc, box nhiễu stage 1 nhận, box 6 stage [6,4],
    attention [6,3], t, (toạ độ SpatialSoftmax [C,2], attention [C,H,W]))."""
    T = x.shape[-1]
    feats, kp = model.encode(x, vhw)
    whwh = model._whwh(vhw)
    cond = model.proposer.cond_from_keypoints(kp, text)
    u = model.proposer.sample_from_cond(cond, 1, generator, "mock", obj=model.proposer_obj(objects, vhw, T, cond))
    box_ce = unit_to_boxes(u.reshape(-1, 4), torch.full((4,), float(T), device=x.device))
    if refine_t == "noise":
        t = model.num_timesteps - 1
        xt = torch.randn((1, 4), device=x.device, generator=generator)
    else:
        t = int(refine_t)
        ab = model.alphas_cumprod[t]
        xt = ab.sqrt() * diffusion_from_boxes(box_ce, whwh, model.snr) + (1 - ab).sqrt() * torch.randn(
            (1, 4), device=x.device, generator=generator)
    noisy = refine_noisy_boxes(xt, whwh, model.snr)
    tb = torch.full((1,), t, device=x.device, dtype=torch.long)
    preds, attn = model.refine(feats, model.vis_token(kp), text, tb, noisy.view(1, 1, 4), True,
                               geo=model._geo(objects, 1, x.device) if model.geo and use_objects else None,
                               rel=model._rel(objects, 1, whwh, x.device) if model.relation and use_objects else None)
    xy, att = model.proposer.vision.keypoints(x, vhw)
    return (box_ce[0].cpu().numpy(), noisy[0].cpu().numpy(), preds[:, 0].cpu().numpy(), attn[:, 0].float().cpu().numpy(),
            t, (xy[0].cpu(), att[0].cpu()))


def iou_holes(b, holes):
    """-> (IoU với lỗ mới nhất, IoU lớn nhất với lỗ bất kỳ)."""
    iou = box_iou(np.asarray(b)[None].clip(min=0), np.asarray(holes))[0][0]
    return float(iou[-1]), float(iou.max())


def scan_pool(a, dindex, T, rng):
    """`--scan N` ca test ngẫu nhiên (nhánh + lượt ngẫu nhiên), mỗi ảnh gốc một ca."""
    from ce_localization.data.dataset import _read_annotation
    from ce_localization.tools.plot_refiner_steps import load_case
    names = sorted(os.listdir(os.path.join(a.ce130, "test")))
    out, seen = [], set()
    for i in rng.permutation(len(names)):
        if len(out) >= a.scan:
            break
        iid = names[i].split("_b")[0]
        if iid in seen:
            continue
        ann = _read_annotation(os.path.join(a.ce130, "test", names[i]))
        n = len(ann["inpainted_bboxes"]) if ann else 0
        c = load_case(a, "test", names[i], int(rng.integers(1, n + 1)), dindex, T) if n else None
        if c is not None:
            seen.add(iid)
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint propose_refine (vd ../weights/add/gamma2/best.pth)")
    ap.add_argument("--name", default=None, help="tên hiển thị trên tiêu đề (docs/EXPERIMENT_GAMMA.md mục 13.0)")
    ap.add_argument("--ce130", default="../data/all_phase2_V2")
    ap.add_argument("--samples", default="../data/samples")
    ap.add_argument("--density-index", default="../data/density_index.json")
    ap.add_argument("--turn-index", default="../data/turn_index.json", help="box vật đang có (chỉ cần cho GAMMA3 / 3.1)")
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--quota", type=int, nargs=3, default=[0, 0, 4], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--min-gap", type=float, default=0.03)
    ap.add_argument("--refine-t", default="100", help="t* của hàng `ce`")
    ap.add_argument("--rows", nargs="+", default=["t0", "ce", "noise"], choices=ROW_SPECS,
                    help="các hàng: t0 / ce (t* = --refine-t) / noise; hậu tố _nogeo = tắt box vật (GAMMA3 / 3.1). Có/không box vật: "
                         "--rows ce ce_nogeo noise noise_nogeo")
    ap.add_argument("--scan", type=int, default=0, help="quét N ca test rồi chỉ vẽ ca chênh nhất mỗi nhóm (xem docstring)")
    ap.add_argument("--pick", type=int, default=2, help="--scan: số ca vẽ mỗi nhóm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullFormatter

    dev = torch.device(a.device)
    model, cfg, it = load_model(a.ckpt, dev)
    T = cfg["data"]["image_size"]
    dindex = DensityIndex(a.density_index, a.samples)
    w = model.memory.ss_proj.weight.detach().reshape(model.memory.ss_proj.out_features, -1, 2).norm(dim=(0, 2)).cpu().numpy()
    size = 4 + 60 * w / w.max()
    tindex = None
    if model.needs_objects:
        from ce_localization.data.turns import TurnIndex
        tindex = TurnIndex(a.turn_index)
    os.makedirs(a.out, exist_ok=True)
    rt_of = {"t0": "0", "ce": a.refine_t, "noise": "noise"}
    rows = [(spec, rt_of[spec.split("_")[0]], not spec.endswith("_nogeo")) for spec in a.rows]
    if any(not g for *_, g in rows) and not model.needs_objects:
        sys.exit("hàng _nogeo chỉ có nghĩa với model có box vật (model.geo / model.relation)")
    texts = {}

    def compute(cs):
        """-> list kết quả 3 hàng (run_one) + (x, nw, nh) của ảnh inpaint."""
        cls = cs["class"]
        if cls not in texts:
            texts[cls] = TextTable(encode_class_names([cls], cfg["model"]["clip_text"], device=str(dev)))([cls], dev)
        x, scale, nw, nh = image_inputs(cs["imgs"]["inpainted"], cs["own"]["inpainted"], T, "paper")
        xb, vhw = x[None].to(dev), torch.tensor([[nh, nw]], device=dev)
        objects = None
        if tindex is not None:                                                   # vật đang có ở lượt t: như CE130AddDataset
            b = tindex.branches[f"{cs['split']}/{cs['branch']}"]
            objs = np.asarray(b["objects"], dtype=np.float64).reshape(-1, 4)
            rm = np.asarray(b["removed"], dtype=int)
            objects = [torch.from_numpy(objs[(rm == 0) | (rm > cs["turn"])] * scale).float()]
        res = []
        for _, rt, use in rows:
            g = torch.Generator(device=dev.type).manual_seed(a.seed)             # cùng seed ⇒ cùng box CE-Loc mọi hàng
            res.append(run_one(model, xb, vhw, texts[cls], rt, g, objects, use))
        return res, (x, nw, nh, None if objects is None else objects[0].numpy())

    if a.scan:
        pool = scan_pool(a, dindex, T, np.random.default_rng(a.seed))
        scanned = []
        for n_, cs in enumerate(pool, 1):
            res, _ = compute(cs)
            ious = {m: iou_holes(r_[2][-1], cs["holes"]) for (m, *_), r_ in zip(rows, res)}
            scanned.append((cs, ious))
            if n_ % 10 == 0 or n_ == len(pool):
                print(f"  scan {n_}/{len(pool)}", flush=True)
        lat = lambda c, m: c[1][m][0]  # noqa: E731
        have = {m for m, *_ in rows}
        pairs = [("noise_needed", "ce", "t0"), ("celoc_helps", "ce", "noise"), ("noise_wins", "noise", "ce"),
                 ("boxes_help", "ce", "ce_nogeo"), ("boxes_hurt", "ce_nogeo", "ce"),
                 ("boxes_help_noise", "noise", "noise_nogeo"), ("boxes_hurt_noise", "noise_nogeo", "noise")]
        groups = {g_: (lambda c, p=p_, q=q_: lat(c, p) - lat(c, q)) for g_, p_, q_ in pairs if p_ in have and q_ in have}
        cases, used = [], set()
        for gname, key in groups.items():
            for c in sorted(scanned, key=key, reverse=True):
                if sum(cc.get("group") == gname for cc in cases) >= a.pick:
                    break
                if c[0]["branch"] in used or key(c) <= 0.2:
                    continue
                used.add(c[0]["branch"])
                cases.append({**c[0], "group": gname})
                print(f"{gname}: {c[0]['branch']} t{c[0]['turn']} | IoU latest " + " / ".join(
                    f"{m} {lat(c, m):.2f}" for m in have), flush=True)
        with open(os.path.join(a.out, "scan.json"), "w") as f:
            json.dump([{"branch": c["branch"], "turn": c["turn"], "class": c["class"], "size": c["size"],
                        "iou_latest_any": i_} for c, i_ in scanned], f, indent=1)
        m_ = {m: float(np.mean([i_[m][0] for _, i_ in scanned])) for m, *_ in rows}
        print(f"scan {len(scanned)} ca, IoU stage 6 với lỗ mới nhất TB: " + " / ".join(f"{m} {v:.3f}" for m, v in m_.items()),
              flush=True)
    else:
        cases = pick_cases(a, dindex, T, np.random.default_rng(a.seed))
    name = a.name or f"{os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))}/{os.path.basename(a.ckpt)}"
    summary = []
    for cs in cases:
        split, br, turn, cls, holes = cs["split"], cs["branch"], cs["turn"], cs["class"], cs["holes"]
        hole, hc = holes[-1], center(holes[-1])
        res, (x, nw, nh, objs_drawn) = compute(cs)
        fig, axes = plt.subplots(len(rows), 9, figsize=(9 * 3.0, len(rows) * (3.0 * nh / nw + 1.0)), squeeze=False,
                                 layout="constrained")
        case = {k: cs[k] for k in ("split", "branch", "turn", "class", "sample", "size")}
        case.update({"group": cs.get("group"), "refine_t": a.refine_t, "rows": []})
        bg = x[:3].numpy().transpose(1, 2, 0)

        def draw_holes(ax):
            for hb in holes[:-1]:
                rect(ax, hb, "lime", lw=1.0, ls="--")
            rect(ax, hole, "lime", lw=1.6)
            ax.plot(*hc, "+", color="lime", ms=10, mew=2)

        def score(b):
            il, ia = iou_holes(b, holes)
            return f"\nIoU latest {il:.2f} / any {ia:.2f} | center dist {np.linalg.norm(center(b) - hc):.0f} px"

        for r, ((mode, _, use_obj), (box_ce, noisy, stages, attn, t, (xy, att))) in enumerate(zip(rows, res)):
            ce = not mode.startswith("noise")
            boxes = [box_ce, noisy] + list(stages)                               # 0 = CE-Loc, 1 = đầu vào stage 1, 2.. = stage 1..6
            cents = [center(b) for b in boxes]
            first = 0 if ce else 1
            ax = axes[r, 0]
            ax.imshow(bg)
            draw_holes(ax)
            if objs_drawn is not None and use_obj:                               # GAMMA3 / 3.1: box vật model được cho
                for ob in objs_drawn:
                    rect(ax, ob, "deepskyblue", lw=0.7, alpha=0.9)
            ax.set_title(f"Input image (turn {turn}: {turn} removed, density: own)"
                         + ("" if objs_drawn is None else f"\n+ {len(objs_drawn)} existing object boxes (blue, model input)"
                            if use_obj else "\nobject boxes OFF (not given to model)"), fontsize=9)
            ax.set_ylabel(ROW_LABEL[mode.split("_")[0]] + ("" if use_obj else "\n[no object boxes]"), fontsize=10)
            ax = axes[r, 1]
            ax.imshow(bg)
            draw_holes(ax)
            rect(ax, noisy, "orange", lw=1.6, ls="--")
            if ce:
                rect(ax, box_ce, "red", lw=2.2)
                ax.plot(*cents[0], "o", color="red", mec="black", mew=0.6, ms=7, zorder=8)
                ax.set_title("CE-Loc proposal (mock 100 steps)\n" + ("fed directly, t*=0 (orange = stage-1 input)" if t == 0 else
                             f"+ noise to t*={t} (orange, stage-1 input)") + score(box_ce), fontsize=8.5)
            else:
                ax.plot(*cents[1], "o", color="orange", mec="black", mew=0.6, ms=7, zorder=8)
                ax.set_title(f"pure noise box, t={t}\n(orange, stage-1 input; clipped)" + score(noisy), fontsize=8.5)
            row = {"mode": mode, "objects": use_obj, "noisy": noisy.tolist(), "t": t, "stages": []}
            if ce:
                row["ce"], row["ce_iou_latest_any"] = box_ce.tolist(), iou_holes(box_ce, holes)
            for k in range(6):
                ax = axes[r, 2 + k]
                ax.imshow(bg)
                draw_holes(ax)
                for j in range(1, k + 2):                                        # đầu vào + stage trước: mờ dần
                    rect(ax, boxes[j], "red", lw=0.8, alpha=0.12 + 0.4 * j / (k + 2))
                if ce:
                    rect(ax, boxes[0], "red", lw=0.8, ls=":", alpha=0.5)
                rect(ax, boxes[2 + k], "red", lw=2.2)
                pc = np.stack(cents[first: 3 + k])
                ax.plot(pc[:, 0], pc[:, 1], "-", color="black", lw=1.2, alpha=0.85, zorder=5)
                ax.plot(*cents[2 + k], "o", color="red", mec="black", mew=0.6, ms=7, zorder=8)
                at = attn[k]
                ax.set_title(f"refine stage {k + 1}/6 (t={t})\nattn t/text/vis {at[0]:.2f}/{at[1]:.2f}/{at[2]:.2f}"
                             + score(boxes[2 + k]), fontsize=8.5)
                row["stages"].append({"stage": k + 1, "box": boxes[2 + k].tolist(), "attn": at.tolist(),
                                      "iou_latest_any": iou_holes(boxes[2 + k], holes)})
            eff = effective_cells(att).numpy()
            px, py = keypoint_pixels(xy.numpy(), att.shape[1])
            ax = axes[r, 8]
            ax.imshow(bg)
            order = np.argsort(-eff)
            sc = ax.scatter(px[order], py[order], c=eff[order], s=size[order], cmap="plasma_r",
                            norm=LogNorm(vmin=1, vmax=att.shape[1] * att.shape[2]), edgecolors="black", linewidths=0.3,
                            alpha=0.9)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            cb.ax.yaxis.set_minor_formatter(NullFormatter())
            sharp = float((eff < SHARP).mean())
            ax.set_title(f"SpatialSoftmax Output ({att.shape[0]} ch)\nsharp (<{SHARP} cells) {sharp * 100:.1f}% | "
                         f"median cells {np.median(eff):.0f}", fontsize=8.5)
            rect(ax, hole, "lime", lw=1.2)
            row["sharp_frac"] = sharp
            case["rows"].append(row)
            for c in range(9):
                axes[r, c].set_xlim(0, nw)
                axes[r, c].set_ylim(nh, 0)
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
        handles = [Line2D([], [], color="red", lw=2.2, marker="o", label="box + center (this column)"),
                   Line2D([], [], color="orange", lw=1.4, ls="--", label="stage-1 input"),
                   Line2D([], [], color="red", alpha=0.35, label="earlier stages (fading)"),
                   Line2D([], [], color="red", ls=":", alpha=0.6, label="CE-Loc proposal"),
                   Line2D([], [], color="black", lw=1.2, label="center path: (proposal ->) input -> stage 1..k"),
                   Line2D([], [], color="lime", lw=1.6, marker="+", label="latest removed object (target)"),
                   Line2D([], [], color="lime", lw=1.0, ls="--", label="earlier removed objects")]
        if objs_drawn is not None:
            handles.append(Line2D([], [], color="deepskyblue", lw=0.8, label="existing objects (model input)"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
        grp = f" | case type: {cs['group']}" if cs.get("group") else ""
        fig.suptitle(f"{name} (iter {it}) | {split} {br}, turn {turn} ('{cls}') | text = '{cls}' | one box, seed {a.seed} | "
                     f"rows: {' / '.join(a.rows)} (ce: t*={a.refine_t}) | 1 pass = 6 stages{grp}", fontsize=10)
        fig.savefig(os.path.join(a.out, (f"{cs['group']}_" if cs.get("group") else "") + f"{split}_{br}_t{turn}.png"), dpi=100)
        plt.close(fig)
        summary.append(case)
        print(f"{split:5s} {br:10s} t{turn} ({cls}) | IoU latest stage 6: " + " / ".join(
            f"{rw['mode']} {rw['stages'][-1]['iou_latest_any'][0]:.2f}" for rw in case["rows"]), flush=True)
    with open(os.path.join(a.out, "refine_stages.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

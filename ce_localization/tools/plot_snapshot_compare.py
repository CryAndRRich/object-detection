#!/usr/bin/env python3
"""So nhiều run theo snapshot weight — 2 hình, KHÔNG model, KHÔNG GPU (chỉ đọc `history.json` + JSON eval của snapshot):
- `<prefix>_overview.png`: 4 ô loss train theo epoch (tổng / ε / aux / aux÷x_t, làm mượt cùng số epoch cho mọi run kể cả run nhiều
  iter / epoch hơn) + C-NLL TB theo snapshot (CE-130 `test_all`, CE-CoCount `review5`) kèm mốc box GT; chú thích = nhãn + thời lượng
  train (cộng mọi phiên resume).
- `<prefix>_cnll_all_boxes_density.png`: phân bố C-NLL của CẢ 30 box / ảnh (trục log có dấu) ở vài epoch, hàng CE-130 / CE-CoCount.

Run = tên thư mục: `<weights-root>/<run>/history.json`, `<snap-root>/<run>/{ce130_test_all,cocount_review5}/epNNN[_boxes].json`
(eval mock; dump DELTA6 cũ có khoá `_mock` cũng đọc được). Nhãn bắt đầu bằng "+" = thêm vào run đầu (hình density ghép nhãn run đầu).
Chữ trong hình tiếng Anh, không tên thí nghiệm.

  cd object-detection/ce_localization
  python tools/plot_snapshot_compare.py --out-prefix ../../output/add/viz/snapshots/delta7x_vs_delta6 \\
    --run "delta6=CE-Loc + Box Obj + Refiner" \\
    --run "delta7=+ Positive targets (pick random)" \\
    --run "delta7_0_2=+ Positive targets (pick all pairs)" \\
    --run "delta7_1=+ Positive targets (pick random) + LogScale, GIoU weight 0.10"
"""

import argparse
import json
import os

import numpy as np

COLORS = ["#2a6fd6", "#6250d6", "#c2418f", "#e8643a", "#16a085", "#b8860b"]
SETS = (("ce130_test_all", "inpainted", "CE-130"), ("cocount_review5", "cocount", "CoCount"))
LOSSES = (("loss", "Total train loss"), ("loss_eps", "Denoising loss: MSE(eps_pred, eps)"),
          ("loss_aux", "Refiner aux loss: MSE(refined, x0)"), ("aux_ratio", "Refiner aux / MSE(x_t, x0)"))


def result(path, key):
    """Kết quả `key` của một JSON eval / dump (`<key>` hoặc `<key>_mock` khi eval nhiều sampler)."""
    with open(path) as f:
        r = json.load(f)["results"]
    return r[key] if key in r else r[key + "_mock"]


def load_history(path):
    """-> (epoch [N], iter / epoch, {loss: [N]}, giây train cộng mọi phiên, số phiên, config) từ `history.json` của `train.py`."""
    with open(path) as f:
        h = json.load(f)
    rec = {e["iter"]: e for e in h["history"] if "loss" in e}   # resume ghi lại iter trùng -> giữ bản sau
    it = np.array(sorted(rec))
    ipe = max(e["iter"] / e["epoch"] for e in h["history"] if "epoch" in e)
    el = [rec[i]["elapsed_sec"] for i in it]
    ends = [a for a, b in zip(el, el[1:]) if b < a] + [el[-1]]   # elapsed đặt lại 0 mỗi phiên resume
    losses = {k: np.array([rec[i].get(k, np.nan) for i in it], float) for k, _ in LOSSES}
    return it / ipe, ipe, losses, float(sum(ends)), len(ends), h["config"]


def smooth(y, n):
    """Trung bình trượt n điểm, đệm hai đầu bằng giá trị biên (giữ độ dài)."""
    n = max(1, int(round(n)))
    yp = np.concatenate([np.full(n // 2, y[0]), y, np.full(n - n // 2 - 1, y[-1])])
    return np.convolve(yp, np.ones(n) / n, mode="valid")


def signed_log(x):
    """Trục log có dấu: C-NLL âm vẫn vẽ được (-1 / 1 đối xứng quanh 0)."""
    return np.sign(x) * np.log10(1 + np.abs(x))


def hm(sec):
    return f"{int(sec // 3600)}h{int(sec % 3600 // 60):02d}m"


def setup_style(plt):
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e6", "grid.linewidth": 0.8, "axes.edgecolor": "#888", "xtick.color": "#555",
                         "ytick.color": "#555", "axes.titlelocation": "left", "axes.titlesize": 13, "axes.labelcolor": "#333"})


def overview(plt, runs, a, path):
    import matplotlib.ticker as mt
    fig, axs = plt.subplots(2, 3, figsize=(20, 11.2))
    ref_ipe, gts = None, {}
    for r in runs:
        ep, ipe, h, sec, _, cfg = load_history(os.path.join(a.weights_root, r["run"], "history.json"))
        ref_ipe = ref_ipe or ipe
        win = a.smooth_iters / ref_ipe * ipe / max(1, np.median(np.diff(ep * ipe)))   # cùng số epoch như run đầu
        for ax, (k, title) in zip(axs.flat[:4], LOSSES):
            if np.isfinite(h[k]).any():
                ax.plot(ep, smooth(h[k], win), color=r["color"], lw=2)
            ax.set_title(title)
        tm = hm(sec)
        for ax, (sub, key, name) in zip(axs.flat[4:], SETS):
            d = os.path.join(a.snap_root, r["run"], sub)
            pts = [(e, result(os.path.join(d, f"ep{e:03d}.json"), key)) for e in a.epochs
                   if os.path.exists(os.path.join(d, f"ep{e:03d}.json"))]
            if pts:
                ax.plot([e for e, _ in pts], [v["cnll"] for _, v in pts], color=r["color"], lw=2, marker="o", ms=7,
                        label=f"{r['label']} (training time {tm})")
                gts.setdefault(name, pts[-1][1]["cnll_gt"])
            ax.set_title(f"CNLL {name}")
        r["cfg"] = cfg
    for ax, (_, _, name) in zip(axs.flat[4:], SETS):
        if name in gts:
            ax.axhline(gts[name], color="#555", ls="--", lw=1.2)
            ax.text(min(a.epochs) + 13, gts[name], f"Ground truth {gts[name]:.2f}", color="#555", va="center", fontsize=11,
                    bbox=dict(fc="white", ec="none", pad=1))
    for i, ax in enumerate(axs.flat):
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.yaxis.set_major_locator(mt.LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(mt.NullFormatter())
    h, lab = axs.flat[4].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=12.5)
    t = runs[0]["cfg"]["training"]
    lr = f"{t['lr']:.0e}".replace("e-0", "e-")
    fig.text(0.01, 0.975, f"{a.header}batch {t['batch_size']}, {round(ep[-1])} epochs, lr {lr} cosine to 0. "
             f"Loss smoothed over {a.smooth_iters / ref_ipe:.1f} epochs; CNLL (mock sampler) from weight snapshots.", fontsize=13,
             va="top")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95), h_pad=3, w_pad=3)
    fig.savefig(path, dpi=126)
    plt.close(fig)


def density(plt, runs, a, path):
    from scipy.ndimage import gaussian_filter1d
    fig, axs = plt.subplots(2, len(a.density_epochs), figsize=(5 * len(a.density_epochs), 11.3), squeeze=False)
    grid = np.linspace(signed_log(-1.5), signed_log(15000), 600)
    base = runs[0]["label"]
    for row, (sub, key, name) in enumerate(SETS):
        for col, e in enumerate(a.density_epochs):
            ax = axs[row, col]
            for r in runs:
                p = os.path.join(a.snap_root, r["run"], sub, f"ep{e:03d}_boxes.json")
                if not os.path.exists(p):
                    continue
                v = signed_log(np.concatenate([np.asarray(x["cnll"], float) for x in result(p, key) if x.get("cnll")]))
                hist, _ = np.histogram(v, bins=len(grid), range=(grid[0], grid[-1]), density=True)
                lab = f"{base} {r['label']}" if r["label"].startswith("+") else r["label"]
                ax.plot(grid, gaussian_filter1d(hist, 9 if row else 6), color=r["color"], lw=2, label=lab)
            ax.set_title(f"{name}, epoch {e}")
            ax.set_yticks([])
            ax.set_ylim(bottom=0)
            ax.spines["left"].set_visible(False)
            ax.grid(axis="y", visible=False)
            ticks = np.array([-1, 1, 10, 100, 1000, 10000], float)
            ax.set_xticks(signed_log(ticks), [f"{t:g}" for t in ticks])
            if col == 0:
                ax.set_ylabel("share of boxes")
            if row == 1:
                ax.set_xlabel("CNLL of a box (log scale, lower is better)")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=12.5)
    fig.text(0.01, 0.975, "Distribution of CNLL over all 30 boxes per image (mock sampler, weight snapshots). A peak further left means "
             "more good boxes; a fatter right tail means more failed boxes.", fontsize=13, va="top")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95), h_pad=3, w_pad=2)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="RUN=nhãn tiếng Anh (lặp được; run đầu = mốc)")
    ap.add_argument("--out-prefix", required=True, help="ghi <prefix>_overview.png + <prefix>_cnll_all_boxes_density.png")
    ap.add_argument("--weights-root", default="../weights/add")
    ap.add_argument("--snap-root", default="../../output/add/snapshots")
    ap.add_argument("--epochs", type=int, nargs="+", default=list(range(10, 130, 10)), help="snapshot vẽ C-NLL ở overview")
    ap.add_argument("--density-epochs", type=int, nargs="+", default=[10, 40, 80, 120])
    ap.add_argument("--smooth-iters", type=float, default=500, help="cửa sổ làm mượt loss, tính theo iter của run đầu")
    ap.add_argument("--colors", nargs="+", default=None, help="màu theo thứ tự run (mặc định bảng cố định)")
    ap.add_argument("--header", default="Kaggle 2x T4, ")
    ap.add_argument("--only", choices=["overview", "density"], default=None)
    a = ap.parse_args()
    colors = a.colors or COLORS
    if len(a.run) > len(colors):
        raise SystemExit(f"{len(a.run)} run > {len(colors)} màu — thêm --colors")
    runs = [dict(zip(("run", "label"), s.split("=", 1)), color=c) for s, c in zip(a.run, colors)]
    if any("label" not in r for r in runs):
        raise SystemExit("--run phải dạng RUN=nhãn")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_style(plt)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
    for kind, fn in (("overview", overview), ("cnll_all_boxes_density", density)):
        if a.only and not kind.startswith(a.only):
            continue
        p = f"{a.out_prefix}_{kind}.png"
        fn(plt, runs, a, p)
        print(f"-> {p}")


if __name__ == "__main__":
    main()

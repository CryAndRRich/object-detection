#!/usr/bin/env python3
"""Độ ổn định theo epoch từ snapshot weight (`training.snapshot_every_epochs`): đọc eval của từng snapshot (`<dir>/epNNN.json` +
dump `epNNN_boxes.json` của `eval.py`), KHÔNG model, KHÔNG GPU. Mỗi run, mỗi epoch: C-NLL (box thấp nhất / ảnh) trung bình / trung vị /
TB bỏ 1 % ảnh lớn nhất (như `cnll_outliers.py`), mốc box GT, `hit50@K_any`, `mean_iou_any`. Ghi `<out>/snapshots.png` +
`<out>/summary.json`, in bảng.

  cd object-detection/ce_localization
  S=../../output/add/snapshots
  python tools/plot_snapshots.py --run DELTA6=$S/delta6 --run DELTA6.1=$S/delta6_1 --run DELTA6.2=$S/delta6_2 --out ../../output/add/viz/snapshots
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.tools.cnll_outliers import per_sample, summarize  # noqa: E402


def load_run(d, key=None):
    """-> list theo epoch {epoch, n, mean, median, trim1, gt, hit50, mean_iou} của mọi `epNNN_boxes.json` trong `d`."""
    out = []
    for p in sorted(glob.glob(os.path.join(d, "ep*_boxes.json"))):
        ep = int(re.match(r"ep(\d+)_boxes\.json$", os.path.basename(p)).group(1))
        with open(p) as f:
            res = json.load(f)["results"]
        k = key or (next(iter(res)) if len(res) == 1 else None)
        if k is None:
            raise SystemExit(f"{p}: nhiều khoá {list(res)} — chọn bằng --key")
        ps = per_sample(res[k])
        del res
        s = summarize([v[0] for v in ps.values()])
        row = {"epoch": ep, "n": s["n"], "mean": s["mean"], "median": s["median"], "trim1": s["mean_trim1"],
               "gt": float(np.nanmean([v[2] for v in ps.values()]))}
        m = p[: -len("_boxes.json")] + ".json"
        if os.path.exists(m):
            with open(m) as f:
                j = json.load(f)
            r = j["results"][k]
            row.update(hit50=r.get(f"hit50@{j['n_samples']}_any"), mean_iou=r.get("mean_iou_any"))
        out.append(row)
    if not out:
        raise SystemExit(f"{d}: không có ep*_boxes.json")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="TÊN=thư mục chứa epNNN.json + epNNN_boxes.json (lặp được)")
    ap.add_argument("--key", default=None, help="khoá kết quả trong dump (mặc định: khoá duy nhất, vd inpainted)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {}
    for spec in a.run:
        name, d = spec.split("=", 1)
        runs[name] = load_run(d, a.key)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(runs, f, indent=1)

    for name, rows in runs.items():
        print(f"\n{name} ({rows[0]['n']} ảnh có C-NLL)\n  epoch   C-NLL TB  trung vị  bỏ 1 %   box GT   hit50   mean_iou")
        for r in rows:
            f = lambda v, w=8, p=3: f"{v:{w}.{p}f}" if v is not None else " " * (w - 1) + "-"  # noqa: E731
            print(f"  {r['epoch']:5d} {f(r['mean'], 10)} {f(r['median'])} {f(r['trim1'])} {f(r['gt'])} {f(r.get('hit50'), 7, 4)} "
                  f"{f(r.get('mean_iou'), 8, 4)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = (("mean", "C-NLL mean (best of K per image)", True), ("median", "C-NLL median", False),
              ("trim1", "C-NLL mean, drop worst 1% images", False), ("hit50", "hit50@K_any", False))
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
    for ax, (k, title, log) in zip(axes, panels):
        for name, rows in runs.items():
            pts = [(r["epoch"], r[k]) for r in rows if r.get(k) is not None]
            if pts:
                line, = ax.plot(*zip(*pts), marker="o", label=name)
                if k == "mean":
                    ax.axhline(rows[0]["gt"], color=line.get_color(), ls=":", lw=1)
        ax.set(title=title, xlabel="epoch", yscale="log" if log else "linear")
        ax.legend(fontsize=8)
    axes[0].text(0.01, 0.01, "dotted: GT boxes (mean)", transform=axes[0].transAxes, fontsize=8)
    plt.tight_layout()
    path = os.path.join(a.out, "snapshots.png")
    plt.savefig(path, dpi=110)
    print(f"\n-> {path}\n-> {os.path.join(a.out, 'summary.json')}")


if __name__ == "__main__":
    main()

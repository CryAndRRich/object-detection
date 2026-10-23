#!/usr/bin/env python3
"""Ảnh outlier của C-NLL (điểm TRUNG BÌNH "box tốt nhất / ảnh" nhảy loạn giữa các model) — đọc dump `<out>_boxes.json` của `eval.py`
(mỗi mẫu `cnll` từng box, `cnll_gt` từng lỗ GT, `n_obj`), KHÔNG model, KHÔNG GPU.

Mỗi model, mỗi mẫu: `sel` = C-NLL thấp nhất trong K box (như bài: chọn box q cao nhất); rồi:
  1. trung bình / trung vị / trung bình bỏ p % mẫu `sel` lớn nhất (`--trim`), phần tổng do 1 % mẫu lớn nhất góp;
  2. outlier = `--top` mẫu `sel` lớn nhất của từng model: trùng nhau giữa các model bao nhiêu, có trùng ở MỌI model không,
     số vật / C-NLL box GT của chúng so với cả tập;
  3. `sel` của từng model trên hợp các outlier (bảng + heatmap).
Ghi `<out>/summary.json` + 3 hình: `sorted_sel.png` (đuôi phân bố), `trimmed_mean.png` (trung bình theo % bỏ), `outliers_heatmap.png`.

  cd object-detection/ce_localization
  O=../../output/add/ce130_test_all
  python tools/cnll_outliers.py --dump DELTA6=$O/test_delta6_boxes.json DELTA6.2=$O/test_delta6_2_boxes.json … --key inpainted_mock \\
      --out ../../output/add/viz/cnll_outliers/ce130_test_all_mock
"""

import argparse
import json
import os

import numpy as np

__all__ = ["per_sample", "summarize", "outlier_overlap"]

TRIMS = (0.0, 0.5, 1.0, 2.0, 5.0)


def per_sample(rows):
    """rows = `dump["results"][khoá]` -> {mẫu: (sel, trung vị K box, sel box GT | nan, n_obj)} (bỏ mẫu không có C-NLL: < 5 vật)."""
    out = {}
    for x in rows:
        if "cnll" not in x:
            continue
        sid = f"{x['image_id']}|{x.get('t', 0)}"
        c = np.asarray(x["cnll"], dtype=np.float64)
        gt = np.asarray(x.get("cnll_gt", []), dtype=np.float64)
        out[sid] = (float(c.min()), float(np.median(c)), float(gt.min()) if gt.size else float("nan"), int(x["n_obj"]))
    return out


def summarize(sel, trims=TRIMS):
    """sel [N] -> trung bình, trung vị, trung bình bỏ p % mẫu lớn nhất, phần tổng của 1 % lớn nhất, max."""
    v = np.sort(np.asarray(sel, dtype=np.float64))[::-1]
    n = len(v)
    k1 = max(1, n // 100)
    out = {"n": n, "mean": float(v.mean()), "median": float(np.median(v)), "max": float(v[0]),
           "top1pct_share": float(v[:k1].sum() / v.sum()) if v.sum() > 0 else float("nan")}
    for p in trims:
        out[f"mean_trim{p:g}"] = float(v[int(round(n * p / 100)):].mean())
    return out


def outlier_overlap(per_model, top):
    """per_model {model: {mẫu: (sel, …)}} trên CÙNG tập mẫu -> (top mẫu sel lớn nhất của mỗi model, ma trận trùng |A∩B| / top,
    mẫu là outlier ở MỌI model)."""
    tops = {m: [s for s, _ in sorted(d.items(), key=lambda kv: -kv[1][0])[:top]] for m, d in per_model.items()}
    names = list(tops)
    mat = [[len(set(tops[a]) & set(tops[b])) / top for b in names] for a in names]
    common = set.intersection(*(set(t) for t in tops.values())) if tops else set()
    return tops, mat, sorted(common)


def plot_all(per_model, tops, out, key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    names = list(per_model)
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in names:
        v = np.sort([t[0] for t in per_model[m].values()])[::-1]
        ax.plot(np.arange(1, len(v) + 1), np.clip(v, 1e-2, None), label=m, lw=1.3)
    gt = np.sort([t[2] for t in next(iter(per_model.values())).values() if np.isfinite(t[2])])[::-1]
    ax.plot(np.arange(1, len(gt) + 1), np.clip(gt, 1e-2, None), "k--", lw=1.0, label="GT boxes")
    ax.set(xscale="log", yscale="log", xlabel="image rank (worst first)", ylabel="C-NLL of best box per image",
           title=f"{key}: tail of per-image best-box C-NLL")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out, "sorted_sel.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ps = np.linspace(0, 5, 51)
    for m in names:
        v = np.sort([t[0] for t in per_model[m].values()])[::-1]
        ax.plot(ps, [v[int(round(len(v) * p / 100)):].mean() for p in ps], label=m, lw=1.3)
    ax.set(xlabel="% of images dropped (largest C-NLL first)", ylabel="mean C-NLL of best box per image",
           title=f"{key}: mean after dropping outlier images")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out, "trimmed_mean.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    union = []
    for m in names:
        for s in tops[m]:
            if s not in union:
                union.append(s)
    first = next(iter(per_model.values()))
    union.sort(key=lambda s: -np.nanmax([per_model[m][s][0] for m in names]))
    mat = np.array([[per_model[m][s][0] for m in names] + [first[s][2]] for s in union])
    fig, ax = plt.subplots(figsize=(1.0 * len(names) + 4, 0.22 * len(union) + 1.5))
    im = ax.imshow(np.clip(mat, 1e-2, None), aspect="auto", cmap="magma", norm=LogNorm(vmin=0.1, vmax=max(10.0, np.nanmax(mat))))
    ax.set_xticks(range(len(names) + 1), names + ["GT"], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(union)), [f"{s.split('|')[0][:38]} t{s.split('|')[1]} ({first[s][3]} obj)" for s in union], fontsize=6)
    fig.colorbar(im, ax=ax, label="C-NLL of best box (log)")
    ax.set_title(f"{key}: union of top outlier images of every model", fontsize=10)
    fig.savefig(os.path.join(out, "outliers_heatmap.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", nargs="+", required=True, help="NHÃN=đường_dẫn <out>_boxes.json của eval.py")
    ap.add_argument("--key", default="inpainted", help="khoá kết quả trong dump (vd inpainted | inpainted_mock | cocount_mock)")
    ap.add_argument("--top", type=int, default=50, help="số mẫu outlier mỗi model (sel lớn nhất)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    per_model = {}
    for spec in a.dump:
        name, path = spec.split("=", 1)
        with open(path) as f:
            rows = json.load(f)["results"][a.key]
        per_model[name] = per_sample(rows)
        if not per_model[name]:
            raise SystemExit(f"{path}: không có `cnll` trong dump (dump của branch `new` ghi `cnll_F1`)")
    ids = set.intersection(*(set(d) for d in per_model.values()))
    if any(len(d) != len(ids) for d in per_model.values()):
        print(f"⚠️ tập mẫu khác nhau giữa các dump -> chỉ dùng {len(ids)} mẫu chung")
    per_model = {m: {s: d[s] for s in ids} for m, d in per_model.items()}
    first = next(iter(per_model.values()))
    summ = {m: summarize([t[0] for t in d.values()]) for m, d in per_model.items()}
    summ["GT boxes"] = summarize([t[2] for t in first.values() if np.isfinite(t[2])])
    tops, mat, common = outlier_overlap(per_model, a.top)
    nobj = np.array([t[3] for t in first.values()])
    gt = np.array([t[2] for t in first.values()])
    def describe(sids):
        return {"n_obj_median": float(np.median([first[s][3] for s in sids])) if sids else None,
                "frac_n_obj_le8": float(np.mean([first[s][3] <= 8 for s in sids])) if sids else None,
                "gt_sel_median": float(np.nanmedian([first[s][2] for s in sids])) if sids else None}
    union = sorted(set().union(*map(set, tops.values())))
    res = {"key": a.key, "n_samples": len(ids), "top": a.top, "summary": summ, "overlap_names": list(per_model),
           "overlap_matrix": mat, "outliers_in_all_models": common,
           "all_images": {"n_obj_median": float(np.median(nobj)), "frac_n_obj_le8": float(np.mean(nobj <= 8)),
                          "gt_sel_median": float(np.nanmedian(gt))},
           "outliers_union": describe(union), "outliers_common": describe(common),
           "outlier_values": {s: {m: per_model[m][s][0] for m in per_model} | {"GT": first[s][2], "n_obj": first[s][3]} for s in union}}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    w = max(len(m) for m in summ)
    print(f"{a.key}: {len(ids)} mẫu | C-NLL box tốt nhất / ảnh")
    print(f"{'':{w}s}  mean    median  " + "  ".join(f"bỏ{p:g}%" for p in TRIMS[1:]) + "  top1%góp  max")
    for m, s in summ.items():
        print(f"{m:{w}s}  {s['mean']:6.2f}  {s['median']:6.2f}  " + "  ".join(f"{s[f'mean_trim{p:g}']:5.2f}" for p in TRIMS[1:])
              + f"  {s['top1pct_share']:6.0%}  {s['max']:.0f}")
    print(f"outlier = {a.top} mẫu tệ nhất / model; trùng nhau (|A∩B| / {a.top}):")
    for m, row in zip(per_model, mat):
        print(f"  {m:{w}s} " + " ".join(f"{x:4.2f}" for x in row))
    print(f"outlier ở MỌI model: {len(common)} | n_obj trung vị outlier (hợp) {res['outliers_union']['n_obj_median']} vs cả tập "
          f"{res['all_images']['n_obj_median']}; % <= 8 vật {res['outliers_union']['frac_n_obj_le8']:.0%} vs "
          f"{res['all_images']['frac_n_obj_le8']:.0%}; C-NLL box GT trung vị {res['outliers_union']['gt_sel_median']:.2f} vs "
          f"{res['all_images']['gt_sel_median']:.2f}")
    plot_all(per_model, tops, a.out, a.key)
    print(f"-> {a.out}/summary.json, sorted_sel.png, trimmed_mean.png, outliers_heatmap.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Chỉ số user study từ `ratings.jsonl` (docs/EXPERIMENT_GAMMA.md mục 17). Không model; chạy lúc nào cũng được (chấm dở vẫn
là mẫu ngẫu nhiên vì thứ tự màn đã xáo). Màn chấm lại (`repeat_of`) chỉ dùng đo độ nhất quán, không vào chỉ số.

Mỗi model (CI 95 % = bootstrap theo màn, `--boot` lần):
  box_ok        box ổn / box hiện                       any_ok  ảnh có >= 1 box ổn
  top1_ok       box hạng 1 (phiếu cao nhất) ổn           all_ok  cả K box ổn
  mean_ok       số box ổn TB / ảnh                       reasons tỉ lệ box không ổn CÓ từng lý do (một box chọn được nhiều lý do
                                                                ⇒ tổng có thể > 1); n_reasons = số lý do TB / box không ổn
  short         màn hiện < K box (model không đủ K box hợp lệ trong 30 mẫu); chỉ số theo ẢNH tính box thiếu là KHÔNG ổn
                (màn 0 box: top1 / any / all đều hỏng), `box_ok` chỉ tính box đã hiện
  in_hole / out_hole   box_ok của box trùng lỗ GT (IoU >= 0,5 với lỗ bất kỳ) / không trùng lỗ — model có đề xuất được chỗ
                       hợp lý NGOÀI vết inpaint không
  auto_on_object       box_ok của box IoA >= 0,5 với vật có sẵn (= `on_object` của eval) / còn lại
  by_objects           box_ok theo số vật có sẵn (<= 10 / 11–30 / 31–100 / > 100)
  n_distinct           số box khác biệt TB / 30 mẫu (mọi mẫu, không cần chấm)
Khớp người <-> chỉ số tự động (mọi box đã chấm): AUC phân biệt ổn / không ổn của IoU với lỗ, IoA với vật (thấp = tốt),
C-NLL F1 / F2 (thấp = tốt; ảnh >= 5 vật). Cặp model: hiệu box_ok trên CÙNG ảnh đã chấm cả hai (bootstrap cặp).
Nhất quán: tỉ lệ box cùng nhãn ổn / không ổn giữa lần chấm gốc và chấm lại.

  python user_study/score.py --items ../../output/add/user_study/ce130/items.json
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.user_study.app import REASONS, as_reasons  # noqa: E402
from ce_localization.user_study.selection import ioa  # noqa: E402
from ce_localization.utils.box_ops_np import box_iou  # noqa: E402

__all__ = ["OBJ_BINS", "load_ratings", "box_table", "auc", "score"]

HIT = 0.5
OBJ_BINS = ((0, 10, "<=10"), (11, 30, "11-30"), (31, 100, "31-100"), (101, 10 ** 9, ">100"))


def load_ratings(path):
    """-> {mã màn: bản ghi sau cùng}."""
    latest = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                latest[rec["id"]] = rec
    return latest


def _cnll_fn():
    from ce_localization.engine.add_eval import cnll        # nạp torch: chỉ khi cần
    return cnll


def box_table(d, ratings):
    """Mỗi box của màn đã chấm (không tính màn chấm lại; bỏ model không còn trong items) -> list dict: s (mã màn), model, image_id, rank, ok, label, iou_hole, ioa_obj,
    n_obj, cnll_F1, cnll_F2 (`reasons` = list lý do, [] = ổn)."""
    cnll = _cnll_fn()
    rows, cache = [], {}
    for s, rec in sorted(ratings.items()):
        if rec.get("repeat_of") is not None or rec["model"] not in d["models"]:
            continue
        it = d["items"][rec["image_id"]]
        sel = it["models"][rec["model"]]
        b = np.asarray(sel["boxes"], dtype=np.float64).reshape(-1, 4)
        obj = np.asarray(it["objects"], dtype=np.float64).reshape(-1, 4)
        holes = np.asarray(it["holes"], dtype=np.float64).reshape(-1, 4)
        key = (rec["image_id"], rec["model"])
        if key not in cache:
            c1 = cnll(b, obj, it["wh"], "F1") if len(b) else None
            c2 = cnll(b, obj, it["wh"], "F2") if len(b) else None
            cache[key] = (box_iou(b, holes)[0].max(1) if len(holes) and len(b) else np.zeros(len(b)),
                          ioa(b, obj).max(1) if len(obj) and len(b) else np.zeros(len(b)), c1, c2)
        ih, io, c1, c2 = cache[key]
        for j, lab in enumerate(rec["labels"]):
            rs = as_reasons(lab)
            rows.append({"s": s, "model": rec["model"], "image_id": rec["image_id"], "rank": j, "ok": not rs,
                         "reasons": rs, "iou_hole": float(ih[j]), "ioa_obj": float(io[j]), "n_obj": len(obj),
                         "cnll_F1": None if c1 is None else float(c1[j]), "cnll_F2": None if c2 is None else float(c2[j])})
    return rows


def auc(score_, ok):
    """AUC (Mann–Whitney) của `score_` phân biệt ok=True (điểm cao) với ok=False; thiếu một lớp -> nan."""
    s, y = np.asarray(score_, dtype=np.float64), np.asarray(ok, dtype=bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                                    # hạng trung bình cho giá trị bằng nhau
        m = s == v
        ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _ci_ratio(num, den, boot, rng):
    """sum(num) / sum(den), bootstrap theo phần tử (màn)."""
    n, d = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    if not len(n) or d.sum() == 0:
        return [float("nan")] * 3
    if boot <= 0:
        return [float(n.sum() / d.sum()), float("nan"), float("nan")]
    j = rng.integers(0, len(n), size=(boot, len(n)))
    with np.errstate(divide="ignore", invalid="ignore"):
        bs = n[j].sum(1) / d[j].sum(1)
    bs = bs[np.isfinite(bs)]
    return [float(n.sum() / d.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def _ci(vals, boot, rng):
    v = np.asarray(vals, dtype=np.float64)
    if not len(v):
        return [float("nan")] * 3
    if boot <= 0:
        return [float(v.mean()), float("nan"), float("nan")]
    bs = v[rng.integers(0, len(v), size=(boot, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def score(d, ratings, boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    rows = box_table(d, ratings)
    out = {"n_screens_rated": sum(r.get("repeat_of") is None for r in ratings.values()),
           "n_repeat_rated": sum(r.get("repeat_of") is not None for r in ratings.values()), "models": {}, "pairs": {},
           "auto_vs_human": {}}
    K = d["k"]
    per_screen = {(rec["model"], s): [] for s, rec in ratings.items()                    # kể cả màn 0 box
                  if rec.get("repeat_of") is None and rec["model"] in d["models"]}
    for r in rows:
        per_screen[(r["model"], r["s"])].append(r)
    for m in d["models"]:
        sc = [v for (mm, _), v in per_screen.items() if mm == m]
        br = [r for r in rows if r["model"] == m]
        res = {"n_screens": len(sc), "n_boxes": len(br),
               "n_distinct": float(np.mean([it["models"][m]["n_distinct"] for it in d["items"].values()]))}
        if sc:
            res["box_ok"] = _ci_ratio([sum(r["ok"] for r in v) for v in sc], [len(v) for v in sc], boot, rng)
            res["top1_ok"] = _ci([bool(v) and v[0]["ok"] for v in sc], boot, rng)
            res["any_ok"] = _ci([any(r["ok"] for r in v) for v in sc], boot, rng)
            res["all_ok"] = _ci([len(v) == K and all(r["ok"] for r in v) for v in sc], boot, rng)
            res["mean_ok"] = float(np.mean([sum(r["ok"] for r in v) for v in sc]))
            res["short"] = float(np.mean([len(v) < K for v in sc]))
            bad = [r["reasons"] for r in br if not r["ok"]]
            res["reasons"] = {lab: (sum(lab in x for x in bad) / len(bad) if bad else float("nan")) for lab in REASONS}
            res["n_reasons"] = float(np.mean([len(x) for x in bad])) if bad else float("nan")
            for name, f in (("in_hole", lambda r: r["iou_hole"] >= HIT), ("out_hole", lambda r: r["iou_hole"] < HIT),
                            ("auto_on_object", lambda r: r["ioa_obj"] >= HIT), ("auto_off_object", lambda r: r["ioa_obj"] < HIT)):
                sub = [r["ok"] for r in br if f(r)]
                res[name] = {"n": len(sub), "box_ok": float(np.mean(sub)) if sub else float("nan")}
            res["by_objects"] = {}
            for lo, hi, name in OBJ_BINS:
                sub = [r["ok"] for r in br if lo <= r["n_obj"] <= hi]
                res["by_objects"][name] = {"n": len(sub), "box_ok": float(np.mean(sub)) if sub else float("nan")}
        out["models"][m] = res
    by_img = {}
    for (m, s), v in per_screen.items():
        if v:
            by_img.setdefault(ratings[s]["image_id"], {})[m] = np.mean([r["ok"] for r in v])
    for a, b in itertools.combinations(d["models"], 2):
        diff = [x[a] - x[b] for x in by_img.values() if a in x and b in x]
        out["pairs"][f"{a} - {b}"] = {"n_images": len(diff), "box_ok_diff": _ci(diff, boot, rng)}
    for name, key, sign in (("iou_hole", "iou_hole", 1), ("ioa_obj", "ioa_obj", -1), ("cnll_F1", "cnll_F1", -1),
                            ("cnll_F2", "cnll_F2", -1)):
        sub = [(sign * r[key], r["ok"]) for r in rows if r[key] is not None]
        out["auto_vs_human"][name] = {"n": len(sub), "auc": auc(*zip(*sub)) if sub else float("nan")}
    same, n = 0, 0
    for rec in ratings.values():
        o = ratings.get(rec.get("repeat_of")) if rec.get("repeat_of") is not None else None
        if o is not None:
            for x, y in zip(o["labels"], rec["labels"]):
                same += (not as_reasons(x)) == (not as_reasons(y))
                n += 1
    out["consistency"] = {"n_boxes": n, "agree": same / n if n else float("nan")}
    return out


def _fmt(ci):
    return f"{ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]" if not np.isnan(ci[1]) else f"{ci[0]:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--ratings", default=None, help="mặc định ratings.jsonl cạnh items.json")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=None, help="mặc định score.json cạnh items.json")
    a = ap.parse_args()
    with open(a.items) as f:
        d = json.load(f)
    here = os.path.dirname(os.path.abspath(a.items))
    res = score(d, load_ratings(a.ratings or os.path.join(here, "ratings.jsonl")), boot=a.boot)
    print(f"{res['n_screens_rated']} màn đã chấm (+ {res['n_repeat_rated']} chấm lại), K = {d['k']}")
    print(f"{'model':14s} {'màn':>5s}  {'box_ok':24s} {'top1_ok':24s} {'any_ok':24s} {'all_ok':24s} "
          f"{'in_hole':>8s} {'out_hole':>8s} {'distinct':>8s} {'short':>6s}")
    for m, r in res["models"].items():
        if not r["n_screens"]:
            print(f"{m:14s} {0:5d}")
            continue
        print(f"{m:14s} {r['n_screens']:5d}  {_fmt(r['box_ok']):24s} {_fmt(r['top1_ok']):24s} {_fmt(r['any_ok']):24s} "
              f"{_fmt(r['all_ok']):24s} {r['in_hole']['box_ok']:8.3f} {r['out_hole']['box_ok']:8.3f} {r['n_distinct']:8.1f} {r['short']:6.3f}")
    for p, v in res["pairs"].items():
        print(f"  {p:28s} hiệu box_ok {_fmt(v['box_ok_diff'])} ({v['n_images']} ảnh)")
    print("  AUC chỉ số tự động phân biệt ổn / không ổn: " + " | ".join(
        f"{k} {v['auc']:.3f} (n {v['n']})" for k, v in res["auto_vs_human"].items()))
    print(f"  nhất quán chấm lại: {res['consistency']['agree']:.3f} ({res['consistency']['n_boxes']} box)")
    out = a.out or os.path.join(here, "score.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

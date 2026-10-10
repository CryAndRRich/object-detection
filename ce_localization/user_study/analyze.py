#!/usr/bin/env python3
"""Phân tích nhãn user study theo HẠNG box sau khi gộp box trùng (docs/EXPERIMENT_GAMMA.md mục 17.3). Không model, chỉ đọc JSON.

Quy ước chấm: lý do "Other" CHỈ dùng khi box này TRÙNG một box khác của cùng màn (selection.py luôn hiện K = 4 box, nhưng khi 30 mẫu
không đủ 4 vùng tách nhau thì NMS nới ngưỡng tới 1,0 và nhận cả box trùng). Gộp:
  1. mỗi box có "Other" nối với box BẠN của nó = box khác có IoU lớn nhất (IoU = 0 thì lấy box có IoA hai chiều lớn nhất — box nằm lọt
     trong box kia); không chạm box nào (IoU = IoA = 0) ⇒ KHÔNG ghép được, giữ là box riêng, đếm vào `other_unpaired`
  2. các box nối nhau thành nhóm (thành phần liên thông); mỗi nhóm giữ box có SỐ NHỎ NHẤT (hạng cao hơn = nhận trước), dù "Other" tích
     ở box nào
  3. nhãn của box giữ = HỢP các lý do KHÁC "Other" của mọi box trong nhóm (người chấm chấm vị trí ở box không tích Other). Nhóm mà
     MỌI box đều tích Other và không box nào có lý do khác (không box nào được chấm làm đại diện) ⇒ box giữ "không rõ" (`ambiguous`): tính là ổn trong `error`, và
     báo thêm `error_excl_ambiguous` (bỏ các box đó khỏi mẫu số)
Sau gộp: `n_distinct` = số box giữ / màn (1–4); box giữ đánh lại hạng 1..n theo số gốc tăng dần ("box thứ k tách biệt").
Lỗi = có ít nhất một trong on_object / wrong_size / implausible; tỉ lệ từng lý do tính trên mọi box hạng k (một box chọn được nhiều lý do).
Màn chấm lại (`repeat_of`) bỏ.

  cd object-detection/ce_localization
  python user_study/analyze.py --model gamma4 --out ../../output/add/user_study/analysis/summary.json \\
      --set ce130 ../../output/add/user_study/ce130/items.json ../../output/add/user_study/ce130/ratings.jsonl \\
      --set cocount_resize ../../output/add/user_study/cocount_resize/items_objsize.json \\
            ../../output/add/user_study/cocount_resize/ratings_objsize_rater3.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.user_study.app import as_reasons  # noqa: E402
from ce_localization.user_study.score import load_ratings  # noqa: E402
from ce_localization.user_study.selection import ioa  # noqa: E402
from ce_localization.utils.box_ops_np import box_iou  # noqa: E402

__all__ = ["ERRORS", "dedupe", "analyze"]

ERRORS = ("on_object", "wrong_size", "implausible")


def dedupe(boxes, labels):
    """boxes [K,4], labels list K list lý do -> (kept: list (số gốc 0-based, set lý do lỗi, không rõ), n_unpaired). Xem docstring
    module."""
    K = len(labels)
    labs = [set(as_reasons(x)) for x in labels]
    parent = list(range(K))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    unpaired = 0
    if K > 1:
        b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        iou = box_iou(b, b)[0]
        a = ioa(b, b)
        cont = np.maximum(a, a.T)
        for m in (iou, cont):
            np.fill_diagonal(m, -1.0)
        for i in range(K):
            if "other" not in labs[i]:
                continue
            j = int(iou[i].argmax()) if iou[i].max() > 0 else int(cont[i].argmax())
            if max(iou[i, j], cont[i, j]) <= 0:
                unpaired += 1
                continue
            ri, rj = find(i), find(j)
            parent[max(ri, rj)] = min(ri, rj)
    elif K == 1 and "other" in labs[0]:
        unpaired = 1
    groups = {}
    for i in range(K):
        groups.setdefault(find(i), []).append(i)
    kept = []
    for g in sorted(groups.values(), key=min):
        reasons = set().union(*(labs[i] for i in g)) - {"other"}
        if len(g) == 1 and "other" in labs[g[0]]:
            reasons |= {"other"}                         # Other không ghép được: giữ nguyên nhãn để báo riêng
        kept.append((min(g), reasons, len(g) > 1 and not reasons and all("other" in labs[i] for i in g)))
    return kept, unpaired


def _rate(x):
    return float(np.mean(x)) if len(x) else float("nan")


def analyze(d, ratings, model, k=4):
    """d = items.json, ratings = {id: bản ghi} -> dict: phân bố n_distinct, lỗi theo hạng tách biệt k (và theo số gốc)."""
    recs = [r for r in ratings.values() if r["model"] == model and r.get("repeat_of") is None]
    dist = np.zeros(k + 1, dtype=int)
    by_rank = {i: [] for i in range(1, k + 1)}           # hạng tách biệt -> list set lý do
    by_orig = {i: [] for i in range(1, k + 1)}           # số gốc của box giữ -> list set lý do
    by_n = {i: [] for i in range(1, k + 1)}              # n_distinct của màn -> list (mọi box giữ) set lý do
    unpaired = n_other = 0
    for r in recs:
        sel = d["items"][r["image_id"]]["models"][model]
        n_other += sum("other" in as_reasons(x) for x in r["labels"])
        kept, u = dedupe(sel["boxes"], r["labels"])
        unpaired += u
        dist[len(kept)] += 1
        for rank, (orig, reasons, amb) in enumerate(kept, 1):
            by_rank[rank].append((reasons, amb))
            by_orig[orig + 1].append((reasons, amb))
            by_n[len(kept)].append((reasons, amb))

    def table(groups):
        out = {}
        for key, ra in groups.items():
            rs = [x for x, _ in ra]
            err = [bool(x & set(ERRORS)) for x in rs]
            out[str(key)] = {"n": len(rs), "n_ambiguous": int(sum(a_ for _, a_ in ra)),
                             "error_excl_ambiguous": _rate([bool(x & set(ERRORS)) for x, a_ in ra if not a_]), "error": _rate(err), **{e: _rate([e in x for x in rs]) for e in ERRORS},
                        "error_only": {e: _rate([x & set(ERRORS) == {e} for x in rs]) for e in ERRORS},
                        "multi": _rate([len(x & set(ERRORS)) > 1 for x in rs]),
                        "other_unpaired": int(sum("other" in x for x in rs))}
        return out

    n = len(recs)
    return {"model": model, "n_screens": n, "n_other_boxes": int(n_other), "other_unpaired": int(unpaired),
            "n_distinct": {str(i): {"n": int(dist[i]), "rate": float(dist[i] / n) if n else float("nan")} for i in range(1, k + 1)},
            "mean_distinct": float((dist * np.arange(k + 1)).sum() / n) if n else float("nan"),
            "by_rank": table(by_rank), "by_orig_index": table(by_orig), "by_n_distinct": table(by_n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs=3, action="append", required=True, metavar=("NAME", "ITEMS", "RATINGS"))
    ap.add_argument("--model", default="gamma4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    for name, items, rat in a.set:
        with open(items) as f:
            d = json.load(f)
        r = res[name] = analyze(d, load_ratings(rat), a.model, d["k"])
        print(f"== {name}: {r['n_screens']} màn, {r['n_other_boxes']} box Other ({r['other_unpaired']} không ghép được)")
        print("   số box tách biệt / ảnh: " + " | ".join(f"{i}: {v['rate']:.3f} ({v['n']})" for i, v in r["n_distinct"].items())
              + f" | TB {r['mean_distinct']:.2f}")
        for rank, v in r["by_rank"].items():
            print(f"   box tách biệt {rank} (n {v['n']:5d}): lỗi {v['error']:.3f} | " + " · ".join(f"{e} {v[e]:.3f}" for e in ERRORS))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

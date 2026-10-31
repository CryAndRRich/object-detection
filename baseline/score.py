#!/usr/bin/env python3
"""Chấm box của baseline (jsonl {"image_id", "boxes": [[x1, y1, x2, y2] × K] pixel ẢNH GỐC}) bằng đúng code chấm của CE-Loc — CPU,
vài phút / tập. Ghi `--out X.json` (khoá như eval.py: `inpainted` | `cocount[_objsize[_exemplar]]`, kèm `prior`) + `X_boxes.json`.
`--first 1`: chỉ box đầu mỗi ảnh (box chính của baseline, như bản tác giả bài 1 box / ảnh).

  cd object-detection/baseline
  python score.py --dataset ce130 --boxes X_raw.jsonl --out X.json
  python score.py --dataset cocount --boxes Y_raw.jsonl --out Y.json --obj-size mean exemplar
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.data import DATASETS, geometry, load_samples  # noqa: E402
from baseline.scoring import read_boxes, records_from_boxes, score_records, write_outputs  # noqa: E402
from ce_localization.data.cocount import load_review  # noqa: E402

SHOW = ("cnll", "cnll_median", "cnll_mean_trim1", "cnll_gt", "best_iou@{K}_any", "hit50@{K}_any", "best_iou@{K}_latest", "on_object",
        "in_image", "degenerate")


def add_data_args(ap):
    """Đường dẫn dữ liệu test dùng chung với `predict.py` (mặc định: chạy trong `object-detection/baseline/`)."""
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--split", default="test_all", choices=["test", "test_all", "val"])
    ap.add_argument("--turn-index", default="../data/turn_index.json", help="CE-130; cũng là nguồn mốc `prior` (lỗ train)")
    ap.add_argument("--samples-root", default="../data/samples")
    ap.add_argument("--cocount-root", default="../data/cocount")
    ap.add_argument("--cocount-review", default="../../output/add/box_review/cocount.jsonl")
    ap.add_argument("--cocount-min-obj", type=int, default=5)
    ap.add_argument("--obj-size", nargs="+", default=["mean", "exemplar"], choices=["mean", "exemplar"], help="CoCount: chấm thêm sau đổi cỡ")
    ap.add_argument("--seed", type=int, default=0, help="seed (mốc prior: 0 như eval CE-Loc)")
    ap.add_argument("--limit", type=int, default=None)


def load_data(a):
    """-> (mẫu, TurnIndex | None, geo_fn)."""
    review = load_review(a.cocount_review) if a.dataset == "cocount" else None
    samples, index = load_samples(a.dataset, turn_index=a.turn_index if os.path.exists(a.turn_index) else None,
                                  samples_root=a.samples_root, split=a.split, cocount_root=a.cocount_root, review=review,
                                  min_obj=a.cocount_min_obj, limit=a.limit)
    return samples, index, lambda s: geometry(s, index=index, samples_root=a.samples_root, cocount_root=a.cocount_root, review=review)


def run_score(a, samples, index, geo, boxes, meta, log=print):
    t0 = time.time()
    recs, K = records_from_boxes(samples, boxes, geo)
    results, prior, dump = score_records(recs, a.dataset, K, index=index, obj_size=a.obj_size if a.dataset == "cocount" else None,
                                         seed=a.seed)
    write_outputs(a.out, {"dataset": a.dataset, "split": a.split if a.dataset == "ce130" else None, "n_samples": K, "seed": a.seed,
                          **({"cocount_review": a.cocount_review, "cocount_min_obj": a.cocount_min_obj} if a.dataset == "cocount" else {}),
                          **meta}, results, prior, dump)
    for key, res in results.items():
        pri = prior.get(key, {})
        log(f"\n  === {key} ({res['n']} mẫu; C-NLL trên {res['n_cnll']} mẫu >= 5 vật; K = {K}) — baseline | prior ({time.time() - t0:.0f} s)")
        for k in (k.format(K=K) for k in SHOW):
            if k in res:
                log(f"  {k:20s} {res[k]:10.4f} {pri.get(k, float('nan')):10.4f}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_data_args(ap)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--meta", nargs="*", default=[], help="k=v ghi vào JSON")
    a = ap.parse_args()
    samples, index, geo = load_data(a)
    print(f"[score] {a.dataset}: {len(samples)} mẫu", flush=True)
    run_score(a, samples, index, geo, read_boxes(a.boxes, a.first), {"source": a.boxes, "first": a.first, **dict(kv.split("=", 1) for kv in a.meta)})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Đọc kết quả eval trong OUTPUT_DIR và in bảng so sánh với baseline đã công bố.

Đọc:
- ``<output>/metrics.json``  — JSONL do detectron2 ghi; lấy bản ghi cuối có "bbox/AP"
- ``<output>/inference/crowdhuman_metrics.json`` — mMR/Recall nếu là CrowdHuman
- ``baselines/baselines.yaml``

In kèm số iteration và batch size thật, vì baseline dùng schedule dài hơn ta nhiều nên
không ghi rõ hai số đó thì bảng so sánh sẽ gây hiểu sai.

Dùng:
    python tools/summarize.py output/minitrain_res50 --dataset coco_minitrain
    python tools/summarize.py output/voc0712_res50   --dataset voc0712
    python tools/summarize.py output/crowdhuman_fbox_res50 --dataset crowdhuman
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_baselines():
    path = os.path.join(REPO, "baselines/baselines.yaml")
    try:
        import yaml
    except ImportError:
        sys.exit("cần PyYAML: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def read_metrics(output_dir):
    """Lấy bản ghi metric cuối cùng có kết quả bbox."""
    path = os.path.join(output_dir, "metrics.json")
    if not os.path.isfile(path):
        return {}
    best = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if any(k.startswith("bbox/") for k in rec):
                best = rec
    return best


def read_crowdhuman(output_dir):
    path = os.path.join(output_dir, "inference/crowdhuman_metrics.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def fmt(v):
    return "-" if v is None else f"{v:.1f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output_dir")
    p.add_argument("--dataset", required=True,
                   choices=["coco_minitrain", "voc0712", "crowdhuman", "crowdhuman_zeroshot"])
    p.add_argument("--label", default="DiffusionDet (tự train)")
    args = p.parse_args()

    bl = load_baselines()[args.dataset]
    metrics = read_metrics(args.output_dir)
    ch = read_crowdhuman(args.output_dir)

    print("=" * 78)
    print(f"{args.dataset}  —  eval trên {bl['eval_on']}")
    print(f"metric: {bl['metric']}")
    print("=" * 78)

    iters = metrics.get("iteration")
    if iters is not None:
        print(f"Kết quả của ta ở iteration {int(iters)}")
    if not metrics and not ch:
        print(f"!! Chưa thấy kết quả nào trong {args.output_dir}")
        print("   (cần metrics.json của detectron2, hoặc inference/crowdhuman_metrics.json)")

    if args.dataset == "crowdhuman":
        header = f"{'Method':<28}{'AP50':>8}{'mMR↓':>8}{'Recall':>8}"
        print("\n" + header)
        print("-" * len(header))
        for m in bl["methods"]:
            print(f"{m['name']:<28}{fmt(m.get('AP50')):>8}{fmt(m.get('mMR')):>8}"
                  f"{fmt(m.get('Recall')):>8}")
        print("-" * len(header))
        if ch:
            print(f"{args.label:<28}{fmt(ch.get('AP50')):>8}{fmt(ch.get('mMR')):>8}"
                  f"{fmt(ch.get('Recall')):>8}")
            if metrics.get("bbox/AP") is not None:
                print(f"  (COCO-style trên cùng model: AP={metrics['bbox/AP']:.1f} "
                      f"AP50={metrics.get('bbox/AP50', float('nan')):.1f})")
    else:
        header = f"{'Method':<34}{'Backbone':<12}{'AP':>7}{'AP50':>7}{'AP75':>7}"
        print("\n" + header)
        print("-" * len(header))
        for m in bl["methods"]:
            print(f"{m['name']:<34}{m.get('backbone', '-'):<12}"
                  f"{fmt(m.get('AP')):>7}{fmt(m.get('AP50')):>7}{fmt(m.get('AP75')):>7}")
        print("-" * len(header))
        if metrics:
            print(f"{args.label:<34}{'R50-FPN':<12}"
                  f"{fmt(metrics.get('bbox/AP')):>7}{fmt(metrics.get('bbox/AP50')):>7}"
                  f"{fmt(metrics.get('bbox/AP75')):>7}")

    if "reference_upper_bound" in bl:
        rb = bl["reference_upper_bound"]
        print(f"\nTham chiếu trần trên — {rb['note']}:")
        for m in rb["methods"]:
            print(f"  {m['name']:<32}{m['backbone']:<8}AP={m['AP']}")

    print(f"\nMức so sánh: {bl['comparable']}")
    print("LƯU Ý: " + " ".join(bl["note"].split()))
    print("Baseline dùng schedule đầy đủ trên 8 GPU; ta chạy 2x T4 với compute nhỏ hơn "
          "nhiều.\nKhi báo cáo phải ghi rõ iteration + batch size, đừng so trực tiếp như "
          "thể cùng điều kiện.")


if __name__ == "__main__":
    main()

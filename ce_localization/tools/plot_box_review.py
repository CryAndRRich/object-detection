#!/usr/bin/env python3
"""Minh hoạ lọc box vật CE-CoCount bằng tay (`box_review/`): mỗi file một hình 2 hàng × 1 cột — trên: mọi box SAM gốc (`exam_bbox`),
dưới: box còn lại sau rà (bỏ `bad` ∪ `auto`). Chỉ đọc ảnh + annotation + bản rà, KHÔNG model.

  cd object-detection/ce_localization
  python tools/plot_box_review.py --names INTER_HOU_BBT0_HCP0_00089_00070_0_240_negative \\
      --review ../../output/add/box_review/cocount.jsonl --out ../../output/add/viz/box_review
"""

import argparse
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.cocount import load_review, read_cocount  # noqa: E402


def plot(root, name, review, out, color="#00e5ff", prefix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    img = Image.open(os.path.join(root, "Image", name + ".jpg")).convert("RGB")
    w, h = img.size
    panels = ((read_cocount(root, name)["objects"], "Before: {} boxes"),
              (read_cocount(root, name, review=review)["objects"], "After: {} boxes left"))
    fig, axes = plt.subplots(2, 1, figsize=(6, 2 * 6 * h / w + 0.9))
    for ax, (boxes, title) in zip(axes, panels):
        ax.imshow(img)
        for x1, y1, x2, y2 in boxes:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.8))
        ax.set_title(title.format(len(boxes)), fontsize=12)
        ax.set_axis_off()
    fig.tight_layout()
    path = os.path.join(out, prefix + name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="+", required=True, help="tên file CE-CoCount (không đuôi)")
    ap.add_argument("--root", default="../data/cocount")
    ap.add_argument("--review", default="../../output/add/box_review/cocount.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--numbered", action="store_true", help="tên hình thêm số thứ tự 01_, 02_ … theo --names")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    review = load_review(a.review)
    for i, n in enumerate(a.names, 1):
        print("->", plot(a.root, n, review, a.out, prefix=f"{i:02d}_" if a.numbered else ""))


if __name__ == "__main__":
    main()

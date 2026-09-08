#!/usr/bin/env python3
"""Vẽ box từ JSON COCO (sinh bởi convert_ce130.py) lên ảnh — CỬA CHẶN bắt buộc trước
khi train EXPERIMENT D.

Vì sao bắt buộc: bài học ``docs/bai-hoc-ce-loc-detection.md`` §5 — visualize bắt được 2
lỗi hình học lớn ở vòng 1 CE-Loc mà toàn bộ unit test và 3 vòng rà soát code đều bỏ sót.
Test ``tests/test_convert_ce130.py`` chỉ kiểm logic (dedupe, xyxy->xywh, degenerate) trên
dữ liệu GIẢ — không thay được việc nhìn bằng mắt trên ảnh THẬT.

Không cần detectron2 — chỉ đọc thẳng json + PIL, độc lập hoàn toàn với pipeline train,
để không có chuyện lỗi ở loader che giấu lỗi ở converter (hai đường đọc khác nhau, cùng
ra một kết quả mới đáng tin).

Chạy (từ ``object-detection/diffusiondet/``):

    python tools/visualize_ce130_coco.py \\
        --json ../data/ce130_coco/ce130_agnostic_train.json \\
        --image-root ../data/all_phase2_V2 \\
        --out /tmp/ce130_viz --n 12
"""

import argparse
import json
import os
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="file COCO json do convert_ce130.py sinh")
    p.add_argument("--image-root", required=True,
                   help="thư mục chứa file_name TƯƠNG ĐỐI ghi trong json (all_phase2_V2/)")
    p.add_argument("--out", required=True, help="thư mục ghi ảnh đã vẽ box")
    p.add_argument("--n", type=int, default=12, help="số ảnh lấy ngẫu nhiên để vẽ")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suspect-only", action="store_true",
                   help="chỉ vẽ ảnh NGHI ANNOTATION HỎNG (>=5 box chiếm >50%% diện tích "
                        "ảnh). Split test có 16 ảnh như vậy (855 box, 4,2%% GT của test) "
                        "trong khi train/val có 0 — random 12 ảnh gần như không bao giờ "
                        "trúng, nên phải xem riêng.")
    args = p.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    with open(args.json) as f:
        coco = json.load(f)

    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    ann_by_image = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    images = coco["images"]
    if args.suspect_only:
        area_of = {im["id"]: im["width"] * im["height"] for im in images}
        huge = {}
        for a in coco["annotations"]:
            if a["area"] > 0.5 * area_of[a["image_id"]]:
                huge[a["image_id"]] = huge.get(a["image_id"], 0) + 1
        suspect = {iid for iid, n in huge.items() if n >= 5}
        images = [im for im in images if im["id"] in suspect]
        print(f"--suspect-only: {len(images)} ảnh nghi annotation hỏng "
              f"(>=5 box chiếm >50% diện tích ảnh)")
        if not images:
            print("  (không có ảnh nào — split này sạch theo tiêu chí đó)")
            return

    rng = random.Random(args.seed)
    sample = rng.sample(images, min(args.n, len(images)))

    os.makedirs(args.out, exist_ok=True)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    n_no_box = 0
    n_total_box_drawn = 0
    for im_info in sample:
        img_path = os.path.join(args.image_root, im_info["file_name"])
        if not os.path.exists(img_path):
            print(f"[MISSING] {img_path}")
            continue
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        anns = ann_by_image.get(im_info["id"], [])
        if not anns:
            n_no_box += 1
        for ann in anns:
            x, y, w, h = ann["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)
            label = cat_name.get(ann["category_id"], "?")
            draw.text((x + 2, max(0, y - 12)), label, fill=(255, 0, 0), font=font)
        n_total_box_drawn += len(anns)

        ce130_id = im_info.get("ce130_image_id", im_info["id"])
        out_path = os.path.join(args.out, f"{ce130_id}.jpg")
        img.save(out_path, quality=90)
        print(f"  {out_path}  ({len(anns)} box, {im_info['width']}x{im_info['height']})")

    print(f"\n{len(sample)} ảnh, {n_total_box_drawn} box vẽ được, "
          f"{n_no_box} ảnh KHÔNG có box nào (kiểm tra lại nếu > 0 và dataset không rỗng)")
    print(f"-> Mở {args.out} và NHÌN BẰNG MẮT trước khi train: box phải bao đúng vật thể "
          f"thật trong ảnh, không lệch trục / không đổi vị trí.")


if __name__ == "__main__":
    main()

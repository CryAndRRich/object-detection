#!/usr/bin/env python3
"""Chuyển annotation CrowdHuman (.odgt) sang COCO json để detectron2 đọc được.

Vì sao cần: repo DiffusionDet gốc chỉ có config cho COCO và LVIS, không có CrowdHuman.

**Chọn loại box.** Mỗi instance trong odgt có 3 box:

- ``fbox`` — full body (có thể vượt ra ngoài phần thấy được)
- ``vbox`` — visible region
- ``hbox`` — head

Baseline trong Table 7 của paper DiffusionDet (Faster R-CNN 85,0 AP50 / 50,4 mMR /
90,2 Recall) khớp chính xác baseline FPN **full-body** của paper CrowdHuman gốc
(84,95 / 50,42 / 90,24), nên để so sánh với Table 7 phải dùng ``fbox`` — đây là mặc định.
Table 1 (zero-shot COCO -> CrowdHuman) thì paper ghi rõ dùng visible box, tức ``vbox``.

**Vùng ignore.** ``tag == "mask"`` (không phải người: tượng, ảnh in, phản chiếu...) và
``extra.ignore == 1`` được ghi thành annotation với ``iscrowd=1``. Cách này khớp với cả
hai phía của pipeline detectron2:

- ``DiffusionDetDatasetMapper`` bỏ mọi annotation ``iscrowd != 0`` khi train, nên vùng
  ignore không bị học thành positive;
- ``COCOEvaluator`` coi ``iscrowd=1`` là vùng ignore, detection trúng vào đó không bị
  tính false positive.

Chạy:

    python tools/convert_crowdhuman.py --data-root data --box-type fbox
    python tools/convert_crowdhuman.py --data-root data --box-type vbox
"""

import argparse
import json
import os


def clip_box(box, width, height):
    """odgt có box vượt biên ảnh (nhất là fbox) -> cắt về trong ảnh.

    Trả về ``[x, y, w, h]`` hoặc None nếu box rỗng sau khi cắt.
    """
    x, y, w, h = box
    x1, y1 = max(0.0, float(x)), max(0.0, float(y))
    x2, y2 = min(float(width), float(x) + float(w)), min(float(height), float(y) + float(h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def convert(odgt_path, image_dir, out_path, box_type):
    images, annotations = [], []
    ann_id = 1
    n_ignore = n_pos = n_dropped = 0

    with open(odgt_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    # đọc kích thước ảnh: odgt không chứa width/height nên phải mở ảnh
    from PIL import Image

    for img_id, rec in enumerate(records, 1):
        file_name = rec["ID"] + ".jpg"
        path = os.path.join(image_dir, file_name)
        with Image.open(path) as im:
            width, height = im.size

        images.append({"id": img_id, "file_name": file_name, "width": width, "height": height})

        for gt in rec["gtboxes"]:
            is_ignore = gt["tag"] != "person" or gt.get("extra", {}).get("ignore", 0) == 1
            box = clip_box(gt[box_type], width, height)
            if box is None:
                n_dropped += 1
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": box,
                "area": box[2] * box[3],
                "iscrowd": 1 if is_ignore else 0,
            })
            ann_id += 1
            n_ignore += is_ignore
            n_pos += not is_ignore

    coco = {
        "info": {
            "description": f"CrowdHuman ({box_type}) chuyển sang COCO format",
            "box_type": box_type,
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coco, f)

    print(f"  {os.path.basename(out_path)}: {len(images)} ảnh, "
          f"{n_pos} person + {n_ignore} ignore(iscrowd=1) = {len(annotations)} ann"
          + (f", bỏ {n_dropped} box rỗng sau khi cắt biên" if n_dropped else ""))
    return len(images), n_pos, n_ignore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.environ.get("OBJDET_DATA_ROOT", "data"))
    p.add_argument("--box-type", default="fbox", choices=["fbox", "vbox", "hbox"],
                   help="fbox = full body (baseline Table 7); vbox = visible (Table 1)")
    p.add_argument("--out-dir", default=None,
                   help="nơi ghi json; mặc định <data-root>/crowdhuman/annotations, hoặc "
                        "biến môi trường OBJDET_CROWDHUMAN_ANN_DIR. Cần đổi khi dữ liệu "
                        "nằm ở /kaggle/input (read-only).")
    args = p.parse_args()

    ch = os.path.join(args.data_root, "crowdhuman")
    out_dir = (args.out_dir
               or os.environ.get("OBJDET_CROWDHUMAN_ANN_DIR")
               or os.path.join(ch, "annotations"))
    print(f"CrowdHuman -> COCO json, box_type={args.box_type}, out_dir={out_dir}")
    for split, img_dir in (("train", "images_train"), ("val", "images_val")):
        convert(
            odgt_path=os.path.join(ch, f"annotation_{split}.odgt"),
            image_dir=os.path.join(ch, img_dir),
            out_path=os.path.join(out_dir, f"crowdhuman_{args.box_type}_{split}.json"),
            box_type=args.box_type,
        )


if __name__ == "__main__":
    main()

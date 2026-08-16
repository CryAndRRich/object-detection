"""Convert a COCO ``instances_*.json`` into the ODVG jsonl the trainer reads.

Works for any COCO-format detection json -- COCO-minitrain, and the CrowdHuman /
VOC json produced by ``object-detection/tools/`` -- because the category ids are
read from the file rather than assumed.

    python tools/coco2odvg.py \
        --input ../../data/coco_minitrain/annotations/instances_minitrain2017.json \
        --output-jsonl ../../data/coco_minitrain/annotations/minitrain_odvg.jsonl \
        --output-label-map ../../data/coco_minitrain/annotations/label_map.json

Two files come out:

  * the jsonl, one image per line, boxes as absolute ``xyxy``;
  * a label map ``{"0": "person", ...}`` keyed by *contiguous* index.

The contiguous remap matters: COCO numbers 80 classes inside a 91-slot id space, so
using raw ids would leave 11 holes in the prompt vocabulary. Training uses these
contiguous labels; evaluation goes through the original COCO json and its raw ids,
which is why ``PostProcess`` reads its categories from the val file.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def coco_to_xyxy(bbox):
    """COCO ``[x, y, w, h]`` -> ``[x1, y1, x2, y2]``, rounded to 2 decimals."""
    x, y, width, height = bbox
    return [round(x, 2), round(y, 2), round(x + width, 2), round(y + height, 2)]


def convert(
    input_json: str,
    output_jsonl: str,
    output_label_map: str = None,
    keep_crowd: bool = False,
    min_box_size: float = 1.0,
):
    with open(input_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = sorted(coco["categories"], key=lambda c: c["id"])
    id_to_index = {c["id"]: i for i, c in enumerate(categories)}
    label_map = {str(i): c["name"] for i, c in enumerate(categories)}
    print(f"{len(categories)} categories, ids {categories[0]['id']}..{categories[-1]['id']}")

    images = {img["id"]: img for img in coco["images"]}
    by_image = defaultdict(list)
    dropped_crowd = dropped_small = 0

    for ann in coco["annotations"]:
        if not keep_crowd and ann.get("iscrowd", 0):
            dropped_crowd += 1
            continue
        _, _, w, h = ann["bbox"]
        if w < min_box_size or h < min_box_size:
            dropped_small += 1
            continue
        by_image[ann["image_id"]].append(
            {"bbox": coco_to_xyxy(ann["bbox"]), "label": id_to_index[ann["category_id"]]}
        )

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    written = empty = 0
    with open(output_jsonl, "w", encoding="utf-8") as out:
        for image_id, image in images.items():
            instances = by_image.get(image_id, [])
            if not instances:
                # Images with no annotation are kept: they are the only negative
                # evidence the model gets that a prompted category can be absent.
                empty += 1
            out.write(
                json.dumps(
                    {
                        "filename": image["file_name"],
                        "height": image["height"],
                        "width": image["width"],
                        "detection": {"instances": instances},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"wrote {written} lines to {output_jsonl} ({empty} without annotations)")
    if dropped_crowd or dropped_small:
        print(f"dropped {dropped_crowd} crowd and {dropped_small} degenerate boxes")

    if output_label_map:
        Path(output_label_map).parent.mkdir(parents=True, exist_ok=True)
        with open(output_label_map, "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
        print(f"wrote label map to {output_label_map}")
    return label_map


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="COCO instances json")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-label-map", default=None)
    parser.add_argument("--keep-crowd", action="store_true", help="keep iscrowd=1 annotations")
    parser.add_argument("--min-box-size", type=float, default=1.0, help="drop boxes thinner than this, in pixels")
    args = parser.parse_args()
    convert(args.input, args.output_jsonl, args.output_label_map, args.keep_crowd, args.min_box_size)


if __name__ == "__main__":
    main()

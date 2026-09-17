"""CE-130 loader, reading the COCO json that already exists on disk.

Reads data/ce130_coco/ce130_agnostic_{split}.json rather than walking
all_phase2_V2/ directly. That json was produced by
diffusiondet/tools/convert_ce130.py and already solved the two problems this
loader would otherwise have to re-solve, and re-solve identically:

  * BRANCH DEDUPE. all_phase2_V2/{split}/{id}_b{N}/ has one directory per
    branch, but every branch of one id shares the same ground_truth.jpg (md5
    identical, verified across the whole set). 8829 branches are only
    1911/908/779 images. Counting branches would duplicate data.
  * ANNOTATION CHOICE. fixed_annotation.json exists only for val/test and
    disagrees between branches of the same image on 86.5 % of val / 79.7 % of
    test; annotation.json is perfectly consistent. The converter picks
    deterministically.

Verified against the files: val has 908 images / 38289 annotations, one
category ("object"), file_name relative to all_phase2_V2/.

                          THE BOX FORMAT TRAP

CE-130 ships THREE conventions and mixing them fails silently -- the box stays
inside the image and nothing asserts:

    data/ce130_coco/*.json            COCO xywh   (x,y = TOP-LEFT)   <- here
    all_phase2_V2/.../annotation.json all_bboxes  xyxy
    samples/.../annotation/*.json     target_bbox cxcywh

This loader converts xywh -> xyxy before anything else touches the numbers.

                        GEOMETRY, AND WHY PADDING IS KNOWN

Every CE-130 image is exactly 384 px tall and 384-1918 wide, so an
aspect-preserving resize onto a square canvas ALWAYS pads at the bottom, never
on the right. `valid_h` marks the boundary. Padding is filled with the CLIP mean
rather than black, matching CE-LocModel: a black block is -1.79 sigma after
normalisation and hands the encoder a hard fake edge.
"""

import json
import os

import numpy as np
from PIL import Image

from utils.box_ops_np import scale_to_canvas, xywh_to_xyxy

__all__ = ["CE130Coco", "resize_and_pad"]

# CLIP's channel means, the same constant CE-LocModel pads with.
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float64)


def resize_and_pad(img, target=512):
    """Aspect-preserving resize, top-left anchored, padded with the CLIP mean.

    Returns (uint8 RGB canvas, valid_h). Padding always lands at the BOTTOM
    because CE-130 images are never taller than they are wide.
    """
    W, H = img.size
    s = min(target / float(W), target / float(H))
    nw, nh = int(W * s), int(H * s)

    canvas = np.empty((target, target, 3), dtype=np.uint8)
    canvas[:] = (CLIP_MEAN * 255).round().astype(np.uint8)
    canvas[:nh, :nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    return canvas, nh / float(target)


class CE130Coco:
    """Indexable view over one CE-130 split. Returns numpy, no torch."""

    def __init__(self, coco_json, image_root, canvas=512):
        self.coco_json = coco_json
        self.image_root = image_root
        self.canvas = canvas

        with open(coco_json, "r") as f:
            blob = json.load(f)

        self.images = sorted(blob["images"], key=lambda im: im["id"])
        by_image = {im["id"]: [] for im in self.images}
        for ann in blob["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            by_image[ann["image_id"]].append(ann["bbox"])
        self.boxes_xywh = by_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        path = os.path.join(self.image_root, im["file_name"])

        with Image.open(path) as raw:
            raw = raw.convert("RGB")
            W, H = raw.size
            canvas, valid_h = resize_and_pad(raw, self.canvas)

        assert (W, H) == (im["width"], im["height"]), \
            f"json says {im['width']}x{im['height']}, file is {W}x{H}: {path}"

        xywh = np.asarray(self.boxes_xywh[im["id"]], dtype=np.float64).reshape(-1, 4)
        if len(xywh):
            xyxy = xywh_to_xyxy(xywh)
            # Degenerate boxes exist in the source (85 of 71852 in train) and
            # would make IoU undefined.
            keep = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
            gt, _, valid_h_box = scale_to_canvas(xyxy[keep], W, H, self.canvas)
            assert abs(valid_h_box - valid_h) < 1e-9, "image and box padding disagree"
        else:
            gt = np.zeros((0, 4), dtype=np.float64)

        return {
            "image": canvas,
            "valid_h": valid_h,
            # CE-130 ảnh luôn rộng >= cao (H = 384 cố định, W = 384..1918) nên
            # resize theo min() luôn lấp kín chiều ngang -> valid_w = 1.0.
            # Trả ra tường minh để dùng chung đường chạy với PACO, vốn có ảnh
            # dọc và phải pad hai phía.
            "valid_w": 1.0,
            "gt_cxcywh": gt,
            # ⚠️ CE-130 CHỈ CÓ BOX, KHÔNG CÓ MASK. Trả mảng rỗng đúng shape để
            # mọi thứ hạ nguồn (mask_iou_matrix, AR) chạy được mà không phải
            # phân nhánh — chúng sẽ báo 0 GT, và đó là sự thật: không có mask
            # nào để so. Đừng đọc "recall 0" trên CE-130 là model kém.
            "gt_masks": np.zeros((0, H, W), dtype=bool),
            "gt_areas": np.zeros(0, dtype=np.float64),
            "image_id": im["id"],
            "ce130_image_id": im.get("ce130_image_id"),
            "file_name": im["file_name"],
            "W": W,
            "H": H,
        }

    def resolution_ceiling(self, i, grid_r):
        """Fraction of this image's GT whose short side reaches one latent cell.

        An upper bound on recall that r imposes before any mechanism runs. On a
        1918x384 image the scale factor is 0.267, so a median 39x32 px object
        shrinks to ~1.3 x 1.07 cells; below one cell it cannot be represented at
        all. Reported alongside results so a low oracle_recall on wide images is
        read as a resolution limit, not a failure of p.
        """
        gt = self[i]["gt_cxcywh"]
        if len(gt) == 0:
            return float("nan")
        short_cells = np.minimum(gt[:, 2], gt[:, 3]) * grid_r
        return float((short_cells >= 1.0).mean())

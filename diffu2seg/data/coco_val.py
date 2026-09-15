"""COCO val2017 loader — for reproducing Diffuse2Seg in the regime it was
tuned for.

WHY A SECOND LOADER INSTEAD OF EXTENDING ce130_coco.py: that file is allowed to
assume padding is always at the BOTTOM, because every CE-130 image is exactly
384 px tall and at least that wide, so W >= H holds for all 3598 of them. COCO
has both orientations (640x427 and 427x640 are both common), so padding can land
at the bottom OR on the right. Folding both cases into one class would mean
weakening an invariant that is genuinely true for CE-130 and load-bearing there.

                    THE TWO NUMBERS THAT MAKE COCO DIFFERENT

Measured on instances_val2017.json (5000 images, 36781 annotations):

                         CE-130 val        COCO val2017
    objects per image    ~21 (median)      4 (median), 7.4 (mean)
    median box           39 x 32 px        53.7 x 62.2 px
    classes per image    1                 many
    segmentation         absent            polygon, present

Objects are ~5x sparser and larger. This is the regime Diffuse2Seg reports on,
which is the whole point of running it here.

                           WHAT IS AND IS NOT GT

`segmentation` is present and is a list of polygons for every non-crowd
annotation. This loader returns BOXES only. Decoding polygons to masks needs
pycocotools and is a separate step; the box path reuses everything already
written and tested.

`iscrowd=1` annotations are dropped. They mark unsegmented crowds with a RLE
region rather than an instance, so counting them as instances would penalise a
method that correctly returns individual objects.
"""

import json
import os

import numpy as np
from PIL import Image

from utils.box_ops_np import xywh_to_xyxy, xyxy_to_cxcywh

__all__ = ["CocoVal", "resize_and_pad_any"]

# Same padding constant as the CE-130 loader: the CLIP channel means. Black
# padding is -1.79 sigma after normalisation and hands the encoder a hard fake
# edge that the attention will happily latch onto.
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float64)


def resize_and_pad_any(img, target):
    """Aspect-preserving resize onto a square canvas, anchored top-left.

    Unlike the CE-130 version this returns BOTH (valid_w, valid_h), because a
    portrait COCO image pads on the right instead of the bottom.

    Returns (uint8 RGB canvas, valid_w, valid_h), each valid_* in (0, 1].
    """
    W, H = img.size
    s = min(target / float(W), target / float(H))
    nw, nh = max(1, int(W * s)), max(1, int(H * s))

    canvas = np.empty((target, target, 3), dtype=np.uint8)
    canvas[:] = (CLIP_MEAN * 255).round().astype(np.uint8)
    canvas[:nh, :nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    return canvas, nw / float(target), nh / float(target)


class CocoVal:
    """Indexable view over COCO val2017. Returns numpy, no torch.

    Mirrors CE130Coco's interface so the same tools drive both, with one extra
    key: `valid_w`. Tools that only read `valid_h` keep working on landscape
    images (valid_w == 1.0 there), which is the majority of COCO.
    """

    def __init__(self, coco_json, image_root, canvas=1120, min_boxes=1):
        self.coco_json = coco_json
        self.image_root = image_root
        self.canvas = canvas

        with open(coco_json, "r") as f:
            blob = json.load(f)

        by_image = {}
        for ann in blob["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            by_image.setdefault(ann["image_id"], []).append(ann["bbox"])

        # Images with no usable annotation are dropped rather than scored as
        # zero: 48 of the 5000 have only crowd regions or nothing at all, and
        # counting them would quietly pull every average down for a reason that
        # has nothing to do with the method.
        self.images = sorted(
            (im for im in blob["images"]
             if len(by_image.get(im["id"], ())) >= min_boxes),
            key=lambda im: im["id"])
        self.boxes_xywh = by_image

        self.categories = {c["id"]: c["name"] for c in blob["categories"]}
        self.n_dropped = len(blob["images"]) - len(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        path = os.path.join(self.image_root, im["file_name"])

        with Image.open(path) as raw:
            raw = raw.convert("RGB")
            W, H = raw.size
            canvas, valid_w, valid_h = resize_and_pad_any(raw, self.canvas)

        assert (W, H) == (im["width"], im["height"]), \
            f"json says {im['width']}x{im['height']}, file is {W}x{H}: {path}"

        xywh = np.asarray(self.boxes_xywh[im["id"]], dtype=np.float64).reshape(-1, 4)
        xyxy = xywh_to_xyxy(xywh)
        # Degenerate boxes would make IoU undefined.
        keep = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])

        # Scale the boxes by the SAME integer pixel extent the image actually
        # occupies, not by the raw float ratio. resize() lands on int(W*s)
        # pixels, so a box scaled by s alone can exceed valid_w by up to one
        # canvas pixel -- 0.0004 in normalised units on a 1120 canvas. Small,
        # but it puts GT outside the region the padding filter calls real, and
        # every such box would be judged against predictions that were
        # correctly clipped.
        gt = xyxy_to_cxcywh(xyxy[keep] * np.array(
            [valid_w, valid_h, valid_w, valid_h]) / np.array([W, H, W, H]))

        return {
            "image": canvas,
            "valid_h": valid_h,
            "valid_w": valid_w,
            "gt_cxcywh": gt,
            "image_id": im["id"],
            "file_name": im["file_name"],
            "W": W,
            "H": H,
        }

    def resolution_ceiling(self, i, grid_r):
        """Fraction of this image's GT whose short side reaches one latent cell.

        An upper bound on recall imposed by the grid alone, before any mechanism
        runs. At the paper's canvas=1120 / grid_r=140 one cell is 8 canvas px,
        and a median COCO object (53.7 x 62.2 px on a 640-wide image, scale
        1.75) is ~11.7 cells across -- far above the limit. Reported anyway, so
        that a low recall is never mistaken for a resolution problem when it is
        not one.
        """
        gt = self[i]["gt_cxcywh"]
        if len(gt) == 0:
            return float("nan")
        short_cells = np.minimum(gt[:, 2], gt[:, 3]) * grid_r
        return float((short_cells >= 1.0).mean())

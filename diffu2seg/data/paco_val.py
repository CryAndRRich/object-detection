"""PACO-LVIS val loader — the one dataset from Diffuse2Seg's training-free
tables that this project can actually obtain.

Of the paper's five (SA-1B, ADE20K, EntitySeg, UVO, PACO):
  SA-1B     only downloadable in whole 10 GB shards, and their 1000-image val
            sample is not published, so the split cannot be reproduced anyway
  ADE20K    the instance-level release needs an approved MIT CSAIL account;
            the freely downloadable ADEChallengeData2016 is SEMANTIC only
            (pixel -> one of 150 class ids, no instances) and cannot produce
            instance AR at all
  EntitySeg even the low-resolution images are 11.4 GB in three shards that do
            not separate val
  UVO       video from Kinetics-400, needs approval plus frame extraction
  PACO      annotations 128 MB, images are 2410 ordinary COCO train2017 files

                     WHAT THIS DATASET IS, AND THE CATCH

PACO is PART segmentation. Measured on paco_lvis_v1_val.json:

    annotations        31919 over 2410 images (13.2/image, median 7, max 279)
    OBJECT             10974  (34.4 %)
    PART               20945  (65.6 %)   e.g. "chair:apron", "car:antenna"
    size bands         S 68.2 %   M 24.2 %   L 7.6 %
    median area        362 px^2, i.e. a ~19 px square
    images             median 640 x 480

⚠️ TWO THIRDS OF THE TARGETS ARE PARTS. That is precisely what the paper's six
granularity levels exist for: one chair is simultaneously one OBJECT mask and
about eight PART masks, and a method must return both. Stage 1 emits a SINGLE
level. A low number here therefore cannot distinguish "the implementation is
wrong" from "stage 2 is not written yet" -- which must be said whenever the
result is quoted. The paper's AR_1000 on PACO is 13.6 (M2N2 9.6, DiffSeg 9.8,
UnSAM 9.3), all WITH multi-granularity.

RESOLUTION CEILING at the paper's canvas=1120 / grid_r=140: 90.7 % of GT have a
short side of at least one latent cell (median 4.23 cells, p10 1.09). So even a
perfect mask tops out near 0.907, and `resolution_ceiling` reports it per image.

                  TWO MASK ENCODINGS, SPLIT EXACTLY ALONG OBJECT/PART

Measured, not assumed:

    OBJECT  10974 annotations -> segmentation is a LIST OF POLYGONS
    PART    20945 annotations -> segmentation is a COMPRESSED RLE DICT

Every part is RLE and every object is polygon, and all of them carry
`iscrowd = 0`, so the usual "crowds are RLE" test does not separate them. A
loader that assumes polygons decodes 34 % of the data and throws `could not
convert string to float: 'c'` on the rest -- the 'c' is the first character of
the key `counts`.

Polygons are rasterised with PIL. RLE needs `pycocotools.mask.decode`, whose
`counts` string is LEB128-style delta-compressed; hand-rolling that decoder is
possible and is exactly the kind of thing that fails silently on the odd mask.
So pycocotools IS a dependency here -- but only for this loader, and the import
is deferred so that CE-130 and the rest of the test suite never touch it.
"""

import json
import os

import numpy as np
from PIL import Image, ImageDraw

__all__ = ["PacoVal", "resize_and_pad_any", "polygons_to_mask",
           "segmentation_to_mask"]

# Same padding constant as the other loaders: CLIP channel means. Black padding
# is -1.79 sigma after normalisation and gives the encoder a hard fake edge.
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float64)


def resize_and_pad_any(img, target):
    """Aspect-preserving resize onto a square canvas, anchored top-left.

    Returns (uint8 RGB canvas, valid_w, valid_h). PACO images come in both
    orientations, so padding can land at the bottom OR on the right.
    """
    W, H = img.size
    s = min(target / float(W), target / float(H))
    nw, nh = max(1, int(W * s)), max(1, int(H * s))

    canvas = np.empty((target, target, 3), dtype=np.uint8)
    canvas[:] = (CLIP_MEAN * 255).round().astype(np.uint8)
    canvas[:nh, :nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    return canvas, nw / float(target), nh / float(target)


def polygons_to_mask(polys, W, H):
    """COCO polygon list -> (H, W) bool mask.

    `polys` is a list of flat [x0,y0,x1,y1,...] lists; an annotation split by
    occlusion has several, and their union is the mask.
    """
    m = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(m)
    for p in polys:
        if len(p) < 6:                       # fewer than 3 points is not a polygon
            continue
        # A FLAT list of floats, not a list of (x, y) tuples: older Pillow
        # rejects the tuple form with "incorrect coordinate type". The flat form
        # is accepted by every version and is what COCO stores anyway.
        d.polygon([float(v) for v in p], fill=1)
    return np.array(m, dtype=bool)


def segmentation_to_mask(seg, W, H):
    """Either encoding -> (H, W) bool mask.

    PACO mixes them: objects are polygons, parts are compressed RLE (see the
    module docstring). pycocotools is imported lazily so that nothing else in
    this project gains the dependency.
    """
    if isinstance(seg, list):
        return polygons_to_mask(seg, W, H)

    if isinstance(seg, dict) and "counts" in seg:
        try:
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise ImportError(
                "PACO parts (20945 of 31919 annotations) are stored as "
                "compressed RLE, which needs pycocotools:\n"
                "    pip install pycocotools\n"
                "Only this loader needs it; CE-130 does not."
            ) from exc
        rle = dict(seg)
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("utf-8")
        m = mask_utils.decode(rle)
        if m.ndim == 3:                      # a list of RLEs merges to one mask
            m = m.any(axis=2)
        return m.astype(bool)

    raise TypeError(f"unrecognised segmentation encoding: {type(seg).__name__}")


class PacoVal:
    """Indexable view over PACO-LVIS val. Returns numpy, no torch.

    Interface matches the other loaders (`image`, `valid_h`, `valid_w`,
    `gt_cxcywh`) and adds what AR needs: `gt_masks` at ORIGINAL image
    resolution, and `gt_areas` in ORIGINAL image pixels for the COCO size bands.

    Masks are decoded lazily per image rather than up front: 31919 masks at
    640x480 bool would be ~9.8 GB resident.
    """

    def __init__(self, ann_json, image_root, canvas=1120, include=("OBJECT", "PART")):
        self.ann_json = ann_json
        self.image_root = image_root
        self.canvas = canvas
        self.include = tuple(include)

        with open(ann_json, "r") as f:
            blob = json.load(f)

        self.cat_supercat = {c["id"]: c["supercategory"] for c in blob["categories"]}
        self.cat_name = {c["id"]: c["name"] for c in blob["categories"]}

        # NOTE: iscrowd is 0 on every PACO val annotation, including the 20945
        # RLE-encoded parts, so it says nothing about the encoding. The filter
        # is kept because the field is part of the COCO contract, not because
        # it separates anything here.
        by_image = {}
        self.n_rle = self.n_poly = 0
        for ann in blob["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            if self.cat_supercat[ann["category_id"]] not in self.include:
                continue
            if isinstance(ann["segmentation"], dict):
                self.n_rle += 1
            else:
                self.n_poly += 1
            by_image.setdefault(ann["image_id"], []).append(ann)

        self.anns = by_image
        self.images = sorted((im for im in blob["images"] if by_image.get(im["id"])),
                             key=lambda im: im["id"])
        self.n_dropped = len(blob["images"]) - len(self.images)

    def __len__(self):
        return len(self.images)

    def _path(self, im):
        # file_name is "train2017/000000149115.jpg"; images were downloaded flat.
        return os.path.join(self.image_root, os.path.basename(im["file_name"]))

    def __getitem__(self, i):
        im = self.images[i]
        path = self._path(im)

        with Image.open(path) as raw:
            raw = raw.convert("RGB")
            W, H = raw.size
            canvas, valid_w, valid_h = resize_and_pad_any(raw, self.canvas)

        assert (W, H) == (im["width"], im["height"]), \
            f"json says {im['width']}x{im['height']}, file is {W}x{H}: {path}"

        anns = self.anns[im["id"]]
        boxes, areas, masks, is_part = [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            areas.append(a["area"])
            masks.append(segmentation_to_mask(a["segmentation"], W, H))
            is_part.append(self.cat_supercat[a["category_id"]] == "PART")

        xyxy = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        # Scale by the integer pixel extent the image actually occupies, not by
        # the raw float ratio: resize() lands on int(W*s) pixels, so a box
        # scaled by s alone can sit just outside valid_w/valid_h.
        scale = np.array([valid_w, valid_h, valid_w, valid_h]) / np.array([W, H, W, H])
        b = xyxy * scale
        gt = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                       b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=-1) \
            if len(b) else np.zeros((0, 4))

        return {
            "image": canvas,
            "valid_h": valid_h,
            "valid_w": valid_w,
            "gt_cxcywh": gt,
            "gt_masks": np.asarray(masks, dtype=bool) if masks
                        else np.zeros((0, H, W), dtype=bool),
            "gt_areas": np.asarray(areas, dtype=np.float64),
            "is_part": np.asarray(is_part, dtype=bool),
            "image_id": im["id"],
            "file_name": os.path.basename(im["file_name"]),
            "W": W,
            "H": H,
        }

    def resolution_ceiling(self, i, grid_r):
        """Fraction of this image's GT whose short side reaches one latent cell.

        Measured over the whole val split at grid_r=140: 90.7 %, median 4.23
        cells, p10 1.09. An upper bound on recall that the grid imposes before
        any mechanism runs -- reported so a low AR is not misread as a broken
        mechanism when it is a resolution limit.
        """
        im = self.images[i]
        W, H = im["width"], im["height"]
        s = self.canvas / float(max(W, H))
        cell_px = self.canvas / float(grid_r)
        short = np.array([min(a["bbox"][2], a["bbox"][3]) * s / cell_px
                          for a in self.anns[im["id"]]
                          if a["bbox"][2] > 0 and a["bbox"][3] > 0])
        if not len(short):
            return float("nan")
        return float((short >= 1.0).mean())

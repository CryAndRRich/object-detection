"""COCO-minitrain for CATEGORY-CONDITIONED DETECTION — the EXPERIMENT A.1 control.

Same interface as `CE130Detection`, so NOTHING in models/ or train.py changes.
That is the whole point: A.1 varies the DATA and nothing else, so a difference in
the result can only come from the data.

WHY THIS EXISTS
---------------
EXPERIMENT A/B scored AP50 ~0.015 on CE-130, and four causes were stacked on top
of each other with no way to tell them apart:

  (1) the code/design is wrong          <- what A.1 isolates
  (2) zero-shot: train 72 classes / test 28, overlap ZERO
  (3) only 1,911 training images
  (4) tiny, very dense objects (37.6/image, 0.41 % of the area each)

COCO-minitrain removes (2) and (3) at once: 80 classes shared between train and
eval, and 73,531 samples. So:

    AP50 < 0.05     the stack itself is broken -> fix it before any CE-Loc work
    AP50 0.05-0.15  the stack works; the architecture is simply small
    AP50 > 0.15     the stack is fine -> CE-130's low score is the TASK, not a bug

Write the thresholds down BEFORE reading the number, or any result can be
rationalised after the fact.

ONE IMAGE, ONE CLASS -- HOW
---------------------------
The model takes ONE text per forward pass and has a 1-D score head ("does this
box match the text I was given?"), which fits CE-130 exactly (100 % of its images
contain a single class). COCO does not: 2.94 classes/image on average, and only
20.7 % of images hold exactly one.

So one image becomes several samples, one per class present:

    image_42 + "person" -> the 4 person boxes    (cars and dogs ignored)
    image_42 + "car"    -> the 2 car boxes
    image_42 + "dog"    -> the 1 dog box

The model is only ever asked about one class, and the GT holds only that class.
This keeps the 1-D score head correct: the class identity lives in the INPUT
text, not in the width of the output head. A 80-way head would let the model skip
the text entirely -- fine on a fixed 80-class benchmark, useless for CE-Loc,
whose test set is 28 unseen classes.

READ THIS BEFORE COMPARING A.1 TO A/B
-------------------------------------
DENSITY DIFFERS BY 15x, and it is the one variable A.1 does NOT control:

    boxes per (image, class):  COCO 2.47  (median 1; 58.7 % of pairs hold ONE box)
                              CE-130 37.6

With N=100 proposals the structural precision ceiling min(M,N)/N is ~0.01 here
versus ~0.376 on CE-130. AP is unaffected, RAW PRECISION IS NOT -- so read AP50
and ignore raw precision, or the ceiling gets mistaken for a model failure.
N is deliberately left at 100 anyway: changing it would add a second variable to
an experiment whose entire value is that only one thing moved.

FOUR COCO-SPECIFIC TRAPS (all verified against the real annotation file)
-----------------------------------------------------------------------
1. COCO `bbox` is [x, y, w, h] TOP-LEFT, not xyxy and not cxcywh. Feeding it
   straight into `scale_to_canvas` (which wants xyxy) silently produces boxes
   that are still inside the image and trip no assertion -- exactly the failure
   mode that has already cost this project twice.
2. `iscrowd=1` (2,071 annotations) marks uncountable crowds with sloppy boxes;
   they are dropped, leaving 181,475.
3. Category ids run 1..90 with gaps, NOT 1..80. Never index a list with them.
4. 23.3 % of COCO images are PORTRAIT, so padding is not always at the bottom --
   unlike CE-130, where every image is 384px tall. `valid_h` therefore means
   "fraction of canvas height that is real image", which is 1.0 for portrait
   images (they pad on the RIGHT instead). It is only used to keep placeholder
   boxes out of the padding, so a conservative 1.0 costs nothing; `valid_w` is
   returned alongside for callers that want the tighter bound.

Measured on the real file: 0 degenerate boxes and 0 boxes outside the image, so
unlike CE-130 no filtering is needed -- the guard stays in anyway, cheap and it
documents the check actually ran.
"""

import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from data.ce130_dataset import resize_and_pad
from utils.box_ops_np import filter_degenerate, flip_horizontal, scale_to_canvas

__all__ = ["COCODetection"]


class COCODetection:
    """Returns numpy dicts, identical in shape to `CE130Detection.__getitem__`.

    One item == one (image, class) pair.
    """

    def __init__(self, ann_file, img_dir, target=512, flip_prob=0.0, seed=None,
                 min_boxes=1, per_class=True):
        """`min_boxes` filters pairs by GT count. The default 1 keeps everything;
        raising it to 3 moves COCO towards CE-130's density, which is a SEPARATE
        experiment -- do not turn it on in the same run that changes the dataset.

        `per_class=False` is EXPERIMENT A.2: one sample per IMAGE carrying every
        box of every class, plus a `labels` array of class indices. The model then
        has no text to condition on and must name the class through an 80-way head.

        The two modes describe the SAME annotations, only regrouped -- 25,000
        samples with 7.26 boxes each instead of 73,531 with 2.47. Say so when
        comparing: A.2 resolves a whole image in one forward pass while A.1 needs
        ~2.94, so A.2's per-image numbers are not per-pair numbers.
        """
        self.ann_file = ann_file
        self.img_dir = img_dir
        self.target = target
        self.flip_prob = flip_prob
        self.min_boxes = min_boxes
        self.per_class = per_class
        self.rng = np.random.default_rng(seed)
        # Contiguous 0..79 indices. COCO ids run 1..90 WITH GAPS, so they can never
        # index a head; this mapping is the only place the conversion happens.
        self.cat_ids, self.cat_names = [], []
        self.items = self._scan()

    # ------------------------------------------------------------------ index

    def _scan(self):
        with open(self.ann_file, "r") as f:
            d = json.load(f)

        imgs = {i["id"]: i for i in d["images"]}
        # ids are 1..90 with gaps -> always go through this dict, never an index
        cats = {c["id"]: c["name"] for c in d["categories"]}

        # Sorted COCO ids -> contiguous 0..79. Sorting makes the mapping identical
        # for minitrain and val2017, so a checkpoint trained on one evaluates
        # correctly on the other; deriving it from file order would not.
        self.cat_ids = sorted(cats)
        self.cat_names = [cats[c] for c in self.cat_ids]
        cat_index = {c: i for i, c in enumerate(self.cat_ids)}

        by_pair = defaultdict(list)
        for a in d["annotations"]:
            if a.get("iscrowd", 0):          # sloppy crowd boxes, not countable objects
                continue
            x, y, w, h = a["bbox"]           # TOP-LEFT xywh -> xyxy right here,
            by_pair[(a["image_id"], a["category_id"])].append(
                [x, y, x + w, y + h])        # so nothing downstream sees COCO format

        pairs = []
        for (iid, cid), boxes in sorted(by_pair.items()):
            b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
            b, _ = filter_degenerate(b)
            if len(b) < self.min_boxes:
                continue
            pairs.append((iid, cid, b))

        if self.per_class:
            return [{
                "image_id": f"{iid}_{cid}",          # UNIQUE PER PAIR: the cache and
                                                     # every per-image log key off this,
                                                     # and one image yields ~3 samples
                "img_path": os.path.join(self.img_dir, imgs[iid]["file_name"]),
                "boxes_xyxy_px": b,
                "labels": np.full(len(b), cat_index[cid], dtype=np.int64),
                "text": cats[cid],
                "orig_wh": (imgs[iid]["width"], imgs[iid]["height"]),
            } for iid, cid, b in pairs]

        # A.2: regroup the SAME pairs by image, concatenating classes.
        by_img = defaultdict(list)
        for iid, cid, b in pairs:
            by_img[iid].append((cid, b))
        items = []
        for iid, groups in sorted(by_img.items()):
            b = np.concatenate([g[1] for g in groups])
            lab = np.concatenate([np.full(len(g[1]), cat_index[g[0]], dtype=np.int64)
                                  for g in groups])
            items.append({
                "image_id": str(iid),
                "img_path": os.path.join(self.img_dir, imgs[iid]["file_name"]),
                "boxes_xyxy_px": b,
                "labels": lab,
                # No single class describes the image. Empty rather than a guess:
                # A.2 must never read a class from its input, and an empty string
                # would still be a token the encoder could learn from -- so the
                # encoder is built without a text tower at all (use_text=False).
                "text": "",
                "orig_wh": (imgs[iid]["width"], imgs[iid]["height"]),
            })
        return items

    # ----------------------------------------------------------------- access

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx, need_image=True):
        """`need_image=False` skips the JPEG decode (only the geometry is needed
        when patch tokens come from a cache). Mirrors CE130Detection exactly."""
        it = self.items[idx]
        img = Image.open(it["img_path"])
        W, H = img.size                        # header only, no pixel decode

        if need_image:
            canvas, valid_h = resize_and_pad(img.convert("RGB"), self.target)
        else:
            canvas = None
            s = min(self.target / W, self.target / H)
            valid_h = int(H * s) / float(self.target)
        valid_w = int(W * min(self.target / W, self.target / H)) / float(self.target)

        boxes, _, _ = scale_to_canvas(it["boxes_xyxy_px"], W, H, self.target)

        if self.flip_prob > 0 and self.rng.random() < self.flip_prob:
            if canvas is not None:
                canvas = canvas[:, ::-1].copy()
            boxes = flip_horizontal(boxes)
            did_flip = True
        else:
            did_flip = False

        return {
            "image": canvas,                   # uint8 [T,T,3], CLIP-mean padded
            "boxes": boxes,                    # cxcywh [0,1] — CANONICAL
            # Row i of `labels` describes row i of `boxes`. flip_horizontal only
            # rewrites cx, never reorders, so the correspondence survives it.
            "labels": it["labels"],            # int64 [M], contiguous 0..79
            "text": it["text"],                # class name; 15/80 are multi-word
                                               # ("" when per_class=False)
            "valid_h": valid_h,
            "valid_w": valid_w,                # extra vs CE-130: COCO pads on the
                                               # right for the 23.3 % portrait images
            "image_id": it["image_id"],
            "orig_size": (W, H),
            "flipped": did_flip,
        }

    # ------------------------------------------------------------ statistics

    def stats(self):
        n = [len(it["boxes_xyxy_px"]) for it in self.items]
        return {
            "n_images": len(self.items),       # == n PAIRS; named n_images so the
                                               # train/eval log format is unchanged
            "n_boxes_total": int(np.sum(n)),
            "boxes_per_image_median": float(np.median(n)) if n else 0.0,
            "boxes_per_image_mean": float(np.mean(n)) if n else 0.0,
            "boxes_per_image_max": int(np.max(n)) if n else 0,
            "n_classes": len({int(l) for it in self.items for l in it["labels"]}),
            "n_source_images": len({it["image_id"].split("_")[0] for it in self.items}),
            "per_class": self.per_class,
        }

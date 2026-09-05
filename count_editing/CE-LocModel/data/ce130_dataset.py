"""CE-130 for CATEGORY-CONDITIONED DETECTION.

Input:  (ground_truth.jpg, class name)   Output: boxes of that class only.

The logic (parsing / dedupe / coordinates) is PURE NUMPY, so it is testable
locally without torch. `to_tensor` only happens in collate.

FOUR THINGS THAT MUST BE RIGHT — each one is a bug that was actually measured:

1. TWO SOURCES, TWO DIFFERENT BOX FORMATS:
     all_phase2_V2/*/annotation.json : all_bboxes, inpainted_bboxes -> **xyxy**
     samples/*/annotation/*.json     : target_bbox                  -> **cxcywh**
   (measured: 10,368/10,376 consistently xyxy; 19,998/19,998 cxcywh, and
   cross-checking the two sources gives IoU = 1.0000)

2. KEEP `all_bboxes` AS IS; do NOT subtract `inpainted_bboxes`.
   `ground_truth.jpg` is the ORIGINAL image with nothing removed. Pixel evidence:
   the diff between it and `inpainted_turn_1.png` over the inpainted_bboxes[0]
   region is 51.96/255 versus 1.41/255 for the whole image -> the object IS in
   the image and is only removed on a later turn.
   Round 1 subtracted them -> threw away 7-8 % of REAL objects.

3. DEDUPE BY IMAGE. Branches _b1/_b2/_b3 share one ground_truth.jpg (md5 matches
   186/186) and identical all_bboxes (1,571/1,571) -> 1,911 / 908 / 779 images.

4. PAD WITH THE CLIP MEAN, not black. Black gives -1.79 sigma after normalisation
   (a dark block, creating a fake edge for the ViT); the CLIP mean gives exactly
   0.000.

Also: `fixed_annotation.json` only exists for val/test -> fall back to
annotation.json. Filter 14/37,110 degenerate boxes. Every image is exactly 384px
tall so W>=H always -> padding is ALWAYS at the bottom, and `valid_h` is a single
scalar threshold.
"""

import glob
import json
import os

import numpy as np
from PIL import Image

from utils.box_ops_np import filter_degenerate, flip_horizontal, scale_to_canvas

__all__ = ["CLIP_MEAN", "CLIP_STD", "CE130Detection", "resize_and_pad"]

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def resize_and_pad(img, target=512):
    """Aspect-preserving resize, top-left anchored, padded with the CLIP mean.
    Returns (uint8 RGB image, valid_h).

    Every CE-130 image is 384px tall and >= 384 wide -> new_w is always target,
    and padding goes at the BOTTOM.
    """
    W, H = img.size
    s = min(target / W, target / H)
    nw, nh = int(W * s), int(H * s)

    canvas = np.empty((target, target, 3), dtype=np.uint8)
    canvas[:] = (CLIP_MEAN * 255).round().astype(np.uint8)
    canvas[:nh, :nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    return canvas, nh / float(target)


class CE130Detection:
    """Returns numpy dicts. A torch Dataset wraps this above (keeping this file torch-free)."""

    def __init__(self, root, split, target=512, flip_prob=0.0, seed=None):
        self.root = root
        self.split = split
        self.target = target
        self.flip_prob = flip_prob
        self.rng = np.random.default_rng(seed)
        self.items = self._scan()

    # ------------------------------------------------------------------ index

    def _scan(self):
        """Dedupe by source image: exactly one branch per image-id."""
        by_image = {}
        for br in sorted(glob.glob(os.path.join(self.root, self.split, "*"))):
            ann = self._read_annotation(br)
            if ann is None:
                continue
            iid = os.path.basename(br).split("_b")[0]
            if iid not in by_image:                       # every branch is equivalent
                by_image[iid] = (br, ann)

        items = []
        for iid, (br, ann) in sorted(by_image.items()):
            img_path = os.path.join(br, "ground_truth.jpg")
            if not os.path.exists(img_path):
                continue
            boxes = np.asarray(ann.get("all_bboxes", []), dtype=np.float64).reshape(-1, 4)
            boxes, _ = filter_degenerate(boxes)           # drop 14/37,110 boxes with w/h <= 0
            items.append({
                "image_id": iid,
                "img_path": img_path,
                "boxes_xyxy_px": boxes,                   # NOT minus inpainted_bboxes
                "text": ann.get("class_based_caption", ""),
            })
        return items

    @staticmethod
    def _read_annotation(branch_dir):
        """fixed_annotation.json only exists for val/test -> fall back to annotation.json."""
        for name in ("fixed_annotation.json", "annotation.json"):
            p = os.path.join(branch_dir, name)
            if os.path.exists(p):
                with open(p, "r") as f:
                    return json.load(f)
        return None

    # ----------------------------------------------------------------- access

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx, need_image=True):
        """`need_image=False` skips the JPEG decode + resize entirely.

        Measured 16.9 ms/image (135 ms per batch of 8) purely to build a canvas the
        caller throws away when patch tokens come from the cache. The box geometry
        still needs (W,H), but PIL reads those from the header without decoding
        pixels, which is ~1000x cheaper.
        """
        it = self.items[idx]
        img = Image.open(it["img_path"])
        W, H = img.size                       # header only, no decode yet

        if need_image:
            canvas, valid_h = resize_and_pad(img.convert("RGB"), self.target)
        else:
            canvas = None
            s = min(self.target / W, self.target / H)
            valid_h = int(H * s) / float(self.target)
        boxes, _, _ = scale_to_canvas(it["boxes_xyxy_px"], W, H, self.target)

        if self.flip_prob > 0 and self.rng.random() < self.flip_prob:
            if canvas is not None:
                canvas = canvas[:, ::-1].copy()
            boxes = flip_horizontal(boxes)
            did_flip = True
        else:
            did_flip = False

        return {
            "image": canvas,                 # uint8 [512,512,3], CLIP-mean padded
            "boxes": boxes,                  # cxcywh [0,1] — CANONICAL
            "text": it["text"],              # single-word category name
            "valid_h": valid_h,              # real-image boundary (bounds placeholders)
            "image_id": it["image_id"],
            "orig_size": (W, H),
            "flipped": did_flip,
        }

    # ------------------------------------------------------------ statistics

    def stats(self):
        n = [len(it["boxes_xyxy_px"]) for it in self.items]
        return {
            "n_images": len(self.items),
            "n_boxes_total": int(np.sum(n)),
            "boxes_per_image_median": float(np.median(n)) if n else 0.0,
            "boxes_per_image_mean": float(np.mean(n)) if n else 0.0,
            "boxes_per_image_max": int(np.max(n)) if n else 0,
            "n_classes": len({it["text"] for it in self.items}),
        }


def normalize_for_clip(canvas_uint8):
    """uint8 HWC -> CLIP-normalised float32 CHW. Padded regions become exactly 0.000."""
    x = np.asarray(canvas_uint8, dtype=np.float32) / 255.0
    return ((x - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)


class PatchCache:
    """Reads cached CLIP patch tokens (fp16 memmap) + per-class text embeddings.

    CLIP is frozen, so its features are fixed and cacheable. Measured on an A30:
    CLIP takes 76.8 % of per-batch time, so caching gives ~4.3x speedup
    (328ms -> 76ms/batch).

    TWO VERSIONS are cached (original + horizontally flipped) because you CANNOT
    flip already-cached tokens: the ViT mixes global information across 12 layers,
    so token (i,j) is no longer "the feature of cell (i,j) alone".
    """

    def __init__(self, cache_dir, split):
        import json
        with open(os.path.join(cache_dir, f"{split}_meta.json")) as f:
            self.meta = json.load(f)
        n, v, t, d = self.meta["shape"]
        self.patch = np.memmap(os.path.join(cache_dir, f"{split}_patch.f16"),
                               dtype=np.float16, mode="r", shape=(n, v, t, d))
        self.text = np.load(os.path.join(cache_dir, f"{split}_text.npy"))
        self.idx_image = {k: i for i, k in enumerate(self.meta["image_ids"])}
        self.idx_class = {k: i for i, k in enumerate(self.meta["classes"])}
        self.n_ver = v

    def get(self, image_id, text, flipped=False):
        """-> (patch [T,D] fp32, text [1,D] fp32)."""
        v = 1 if (flipped and self.n_ver > 1) else 0
        return (np.asarray(self.patch[self.idx_image[image_id], v], dtype=np.float32),
                np.asarray(self.text[self.idx_class[text]], dtype=np.float32))

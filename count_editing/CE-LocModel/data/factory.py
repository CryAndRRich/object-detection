"""ONE place that turns a config into a dataset.

Without this, every entry point (train, eval, visualize, preflight, cache) grows
its own `if dataset == "coco"` branch, they drift, and an experiment silently
evaluates on a different split than it trained on. A single factory means adding
EXPERIMENT A.1 touches exactly one function.

Both datasets return the same numpy dict, so nothing downstream cares which one
it got:

    image [T,T,3] uint8 | boxes cxcywh[0,1] | text | valid_h | image_id | flipped

`config["data"]["dataset"]` selects: "ce130" (default, so every existing config
keeps working untouched) or "coco".

SPLIT NAMES DIFFER, deliberately mapped here rather than in the configs:
  ce130  train / val / test  -- three real splits with DISJOINT classes
  coco   train -> minitrain 25K, val|test -> val2017
         COCO has no third split; "test" resolves to val2017 as well so that
         `--split test` works everywhere. Both therefore report the SAME numbers
         on coco -- that is intended, not a bug, and eval prints the resolved
         file so it cannot be mistaken for a held-out test set.
"""

import os

__all__ = ["build_dataset", "dataset_kind"]


def dataset_kind(cfg):
    return cfg.get("data", {}).get("dataset", "ce130")


def build_dataset(cfg, split, flip_prob=0.0, seed=None):
    """Config + split -> dataset. `flip_prob` is passed by the caller (train uses
    the config value; eval must always use 0.0), so augmentation can never leak
    into evaluation just because a config had it enabled."""
    kind = dataset_kind(cfg)
    size = cfg["data"]["image_size"]

    if kind == "ce130":
        from data.ce130_dataset import CE130Detection
        return CE130Detection(cfg["data"]["root"], split, size, flip_prob, seed)

    if kind == "coco":
        from data.coco_dataset import COCODetection
        root = cfg["data"]["root"]
        if split == "train":
            ann = os.path.join(root, "coco_minitrain/annotations/instances_minitrain2017.json")
            img = os.path.join(root, "coco_minitrain/images/train2017")
        elif split in ("val", "test"):
            ann = os.path.join(root, "coco/annotations/instances_val2017.json")
            img = os.path.join(root, "coco/val2017")
        else:
            raise ValueError(f"unknown coco split: {split!r}")
        for p in (ann, img):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} not found — data/README.md documents the layout; "
                    f"data.root should point at object-detection/data/")
        return COCODetection(ann, img, size, flip_prob, seed,
                             min_boxes=cfg["data"].get("min_boxes", 1),
                             # A.2 groups by image instead of by (image, class)
                             per_class=cfg["data"].get("per_class", True))

    raise ValueError(f"unknown data.dataset: {kind!r} (expected 'ce130' or 'coco')")

"""ODVG dataset: one jsonl line per image, for OD or phrase-grounding data.

**OD mode** (what this project uses). Each line carries boxes with integer category
labels; a ``label_map`` json turns those into names. The prompt for an image is
built from the categories actually present *plus randomly sampled absent ones* --
the negatives are what teach the model to answer "no" for a category that is in
the prompt but not in the image. Without them the model learns to fire on every
prompted category.

**VG mode.** Each region carries its own phrase, so the prompt is the set of
distinct phrases and no label map is needed.

Line format (OD)::

    {"filename": "000001.jpg", "height": 480, "width": 640,
     "detection": {"instances": [{"bbox": [x1, y1, x2, y2], "label": 3}, ...]}}

Boxes are absolute ``xyxy``; ``transforms`` normalizes them at the end.
"""

import json
import os
import random
from typing import Callable, List, Optional

import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset

from util.vl_utils import build_caption

from . import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ODVGDataset(VisionDataset):
    """Args:
        root: image directory that ``filename`` is relative to.
        anno: path to the ``.jsonl`` annotation file.
        label_map_anno: path to ``{"0": "person", ...}``. Its presence selects OD mode.
        max_labels: categories per prompt, positives plus sampled negatives.
    """

    def __init__(
        self,
        root: str,
        anno: str,
        label_map_anno: str = None,
        max_labels: int = 80,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.dataset_mode = "OD" if label_map_anno else "VG"
        self.max_labels = max_labels

        if self.dataset_mode == "OD":
            with open(label_map_anno, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)
            self.label_index = set(self.label_map.keys())

        with open(anno, "r", encoding="utf-8") as f:
            self.metas = [json.loads(line) for line in f if line.strip()]

        print(f"  == {self.dataset_mode} dataset: {len(self)} images", end="")
        if self.dataset_mode == "OD":
            print(f", {len(self.label_map)} labels")
        else:
            print()

    def __len__(self) -> int:
        return len(self.metas)

    def _sample_categories(self, present: List[str]) -> List[str]:
        """Positive categories plus negatives, shuffled.

        Shuffling matters: the model must not learn that the first category in the
        prompt is the likely answer.
        """
        categories = list(present)
        absent = sorted(self.label_index.difference(present))
        num_to_add = min(len(absent), self.max_labels - len(categories))
        if num_to_add > 0:
            categories.extend(random.sample(absent, num_to_add))
        random.shuffle(categories)
        return categories

    def _od_target(self, meta):
        instances = meta["detection"]["instances"]
        boxes = [obj["bbox"] for obj in instances]
        present = {str(obj["label"]) for obj in instances}

        prompt_labels = self._sample_categories(sorted(present))
        caption_list = [self.label_map[label] for label in prompt_labels]
        # Index within this image's prompt, which is what the model predicts against.
        name_to_index = {name: i for i, name in enumerate(caption_list)}
        classes = [name_to_index[self.label_map[str(obj["label"])]] for obj in instances]
        return boxes, classes, caption_list

    def _vg_target(self, meta):
        regions = meta["grounding"]["regions"]
        boxes = [obj["bbox"] for obj in regions]
        phrases = [obj["phrase"] for obj in regions]

        paired = list(zip(boxes, phrases))
        random.shuffle(paired)
        boxes, phrases = (list(x) for x in zip(*paired)) if paired else ([], [])

        unique = list(dict.fromkeys(phrases))  # order-stable dedup
        name_to_index = {phrase: i for i, phrase in enumerate(unique)}
        classes = [name_to_index[p] for p in phrases]
        return boxes, classes, unique

    def __getitem__(self, index: int):
        meta = self.metas[index]
        abs_path = os.path.join(self.root, meta["filename"])
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"{abs_path} not found (root={self.root})")

        image = Image.open(abs_path).convert("RGB")
        width, height = image.size

        if self.dataset_mode == "OD":
            boxes, classes, caption_list = self._od_target(meta)
        else:
            boxes, classes, caption_list = self._vg_target(meta)

        target = {
            "size": torch.as_tensor([int(height), int(width)]),
            "cap_list": caption_list,
            "caption": build_caption(caption_list),
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(classes, dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def make_coco_transforms(image_set: str, fix_size: bool = False, strong_aug: bool = False, args=None):
    """The DETR augmentation recipe, driven by the ``data_aug_*`` config fields."""
    normalize = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    scales = getattr(args, "data_aug_scales", [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800])
    max_size = getattr(args, "data_aug_max_size", 1333)
    scales2_resize = getattr(args, "data_aug_scales2_resize", [400, 500, 600])
    scales2_crop = getattr(args, "data_aug_scales2_crop", [384, 600])

    overlap = getattr(args, "data_aug_scale_overlap", None)
    if overlap:
        overlap = float(overlap)
        scales = [int(s * overlap) for s in scales]
        max_size = int(max_size * overlap)
        scales2_resize = [int(s * overlap) for s in scales2_resize]
        scales2_crop = [int(s * overlap) for s in scales2_crop]

    if image_set == "train":
        if fix_size:
            return T.Compose([T.RandomHorizontalFlip(), T.RandomResize([(max_size, max(scales))]), normalize])

        # Either a plain multi-scale resize, or resize-crop-resize, which is what
        # exposes the model to objects at scales the raw images do not contain.
        scale_aug = T.RandomSelect(
            T.RandomResize(scales, max_size=max_size),
            T.Compose(
                [
                    T.RandomResize(scales2_resize),
                    T.RandomSizeCrop(*scales2_crop),
                    T.RandomResize(scales, max_size=max_size),
                ]
            ),
        )
        if strong_aug:
            return T.Compose([T.RandomHorizontalFlip(), scale_aug, T.RandomErasing(p=0.3), normalize])
        return T.Compose([T.RandomHorizontalFlip(), scale_aug, normalize])

    if image_set in ("val", "eval_debug", "train_reg", "test"):
        if os.environ.get("GFLOPS_DEBUG") == "INFO":
            print("fixed-size eval transform (FLOP counting mode)")
            return T.Compose([T.ResizeDebug((1280, 800)), normalize])
        return T.Compose([T.RandomResize([max(scales)], max_size=max_size), normalize])

    raise ValueError(f"unknown image_set {image_set!r}")


def build_odvg(image_set: str, args, datasetinfo: dict) -> ODVGDataset:
    return ODVGDataset(
        datasetinfo["root"],
        datasetinfo["anno"],
        datasetinfo.get("label_map"),
        max_labels=args.max_labels,
        transforms=make_coco_transforms(
            image_set,
            fix_size=getattr(args, "fix_size", False),
            strong_aug=getattr(args, "strong_aug", False),
            args=args,
        ),
    )


__all__ = ["ODVGDataset", "build_odvg", "make_coco_transforms"]

"""COCO-format dataset, used for evaluation.

Training runs on ODVG jsonl; evaluation runs on a COCO ``instances_*.json`` so that
``pycocotools`` can compute AP. Labels are kept as raw ``category_id`` (not
remapped to contiguous indices) because that is what the evaluator compares
against, and what ``PostProcess`` emits.
"""

import os
from typing import Optional

import torch
import torchvision

from .odvg import make_coco_transforms


class CocoDetection(torchvision.datasets.CocoDetection):
    """``(image, target)`` with target fields the eval loop needs.

    Target keys: ``boxes`` (normalized cxcywh after transforms), ``labels`` (raw
    category ids), ``image_id``, ``area``, ``iscrowd``, ``orig_size``, ``size``.
    """

    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoAnnotations()

    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        image, target = self.prepare(image, {"image_id": image_id, "annotations": target})
        if self._transforms is not None:
            image, target = self._transforms(image, target)
        return image, target


class ConvertCocoAnnotations:
    """COCO annotation dicts -> tensors, dropping crowd boxes and empty ones."""

    def __call__(self, image, target):
        w, h = image.size
        image_id = target["image_id"]
        anno = [obj for obj in target["annotations"] if obj.get("iscrowd", 0) == 0]

        boxes = torch.as_tensor([obj["bbox"] for obj in anno], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        boxes[:, 0::2] = boxes[:, 0::2].clamp(min=0, max=w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(min=0, max=h)

        classes = torch.as_tensor([obj["category_id"] for obj in anno], dtype=torch.int64)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        target = {
            "boxes": boxes[keep],
            "labels": classes[keep],
            "image_id": torch.as_tensor([image_id]),
            "area": torch.as_tensor([obj["area"] for obj in anno], dtype=torch.float32)[keep],
            "iscrowd": torch.as_tensor([obj.get("iscrowd", 0) for obj in anno], dtype=torch.int64)[keep],
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
        }
        return image, target


def build_coco(image_set: str, args, datasetinfo: dict) -> CocoDetection:
    root = datasetinfo["root"]
    ann_file = datasetinfo["anno"]
    assert os.path.exists(ann_file), f"annotation file not found: {ann_file}"
    return CocoDetection(
        root,
        ann_file,
        transforms=make_coco_transforms(
            image_set, fix_size=getattr(args, "fix_size", False), strong_aug=False, args=args
        ),
    )


__all__ = ["CocoDetection", "ConvertCocoAnnotations", "build_coco"]

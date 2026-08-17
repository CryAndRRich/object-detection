"""Dataset registry.

A run is described by a datasets json::

    {"train": [{"root": ".../images", "anno": ".../train_odvg.jsonl",
                "label_map": ".../label_map.json", "dataset_mode": "odvg"}],
     "val":   [{"root": ".../val2017", "anno": ".../instances_val2017.json",
                "dataset_mode": "coco"}]}

Training is ODVG (text prompts); validation is COCO format so ``pycocotools`` can
score it. Several train entries are concatenated.
"""

from .coco import build_coco
from .coco_eval import CocoEvaluator, get_coco_api_from_dataset
from .odvg import build_odvg


def build_dataset(image_set: str, args, datasetinfo: dict):
    mode = datasetinfo.get("dataset_mode", "odvg")
    if mode == "coco":
        return build_coco(image_set, args, datasetinfo)
    if mode == "odvg":
        return build_odvg(image_set, args, datasetinfo)
    raise ValueError(f"unknown dataset_mode {mode!r}; expected 'odvg' or 'coco'")


__all__ = ["CocoEvaluator", "build_coco", "build_dataset", "build_odvg", "get_coco_api_from_dataset"]

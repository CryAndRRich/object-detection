"""COCO AP evaluation, gathered across distributed ranks.

Thin driver around ``pycocotools``: accumulate per-image detections, all-gather
them at the end so every rank scores the same full set, then run ``COCOeval``.

``useCats=False`` runs a class-agnostic evaluation, which answers "did we find the
objects at all?" separately from "did we name them right?" -- useful when
diagnosing whether a localization change (this project's whole subject) helped.
"""

import contextlib
import copy
import io
from typing import Dict, List, Sequence

import numpy as np
import torch

from util.misc import all_gather


class CocoEvaluator:
    """Args:
        coco_gt: a loaded ``pycocotools.coco.COCO``.
        iou_types: subset of ``("bbox",)`` -- segmentation is not supported here.
        useCats: pass ``False`` for class-agnostic AP.
    """

    def __init__(self, coco_gt, iou_types: Sequence[str] = ("bbox",), useCats: bool = True):
        from pycocotools.cocoeval import COCOeval

        assert isinstance(iou_types, (list, tuple))
        assert all(t == "bbox" for t in iou_types), f"only bbox evaluation is supported, got {iou_types}"

        self.coco_gt = copy.deepcopy(coco_gt)
        self.iou_types = list(iou_types)
        self.useCats = useCats

        self.coco_eval = {}
        for iou_type in self.iou_types:
            evaluator = COCOeval(self.coco_gt, iouType=iou_type)
            evaluator.useCats = useCats
            self.coco_eval[iou_type] = evaluator

        self.img_ids: List[int] = []
        self.eval_imgs = {iou_type: [] for iou_type in self.iou_types}

    def update(self, predictions: Dict[int, dict]):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            # pycocotools chatters on every loadRes; keep the training log readable.
            with contextlib.redirect_stdout(io.StringIO()):
                coco_dt = self.coco_gt.loadRes(results) if results else type(self.coco_gt)()

            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            coco_eval.params.useCats = self.useCats
            eval_imgs = self._evaluate(coco_eval)
            self.eval_imgs[iou_type].append(eval_imgs)

    @staticmethod
    def _evaluate(coco_eval):
        """Run ``COCOeval.evaluate`` quietly and return its per-image results."""
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()
        return np.asarray(coco_eval.evalImgs).reshape(
            len(coco_eval.params.catIds) if coco_eval.params.useCats else 1,
            len(coco_eval.params.areaRng),
            len(coco_eval.params.imgIds),
        )

    def prepare(self, predictions: Dict[int, dict], iou_type: str) -> List[dict]:
        results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0 or len(prediction["boxes"]) == 0:
                continue

            boxes = _xyxy_to_xywh(prediction["boxes"]).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            results.extend(
                {"image_id": original_id, "category_id": labels[k], "bbox": box, "score": scores[k]}
                for k, box in enumerate(boxes)
            )
        return results

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            img_ids, eval_imgs = _merge(self.img_ids, np.concatenate(self.eval_imgs[iou_type], 2))
            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = list(img_ids)
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval.evalImgs = list(eval_imgs.flatten())
            self.eval_imgs[iou_type] = eval_imgs

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()


def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def _merge(img_ids, eval_imgs):
    """All-gather per-image results and drop duplicates from padded samplers."""
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_ids = [i for per_rank in all_img_ids for i in per_rank]
    merged_eval_imgs = np.concatenate([p for p in all_eval_imgs], 2)

    merged_ids = np.array(merged_ids)
    unique_ids, index = np.unique(merged_ids, return_index=True)
    return unique_ids, merged_eval_imgs[..., index]


def get_coco_api_from_dataset(dataset):
    """Pull the ``COCO`` object out of a (possibly wrapped) dataset."""
    import torch.utils.data
    import torchvision

    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        else:
            break
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    raise TypeError(
        f"evaluation needs a COCO-format dataset, got {type(dataset).__name__}; "
        'set dataset_mode="coco" for the val entry'
    )


__all__ = ["CocoEvaluator", "get_coco_api_from_dataset"]

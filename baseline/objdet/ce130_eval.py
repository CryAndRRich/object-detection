"""Evaluator detectron2 cho eval định kỳ trên val CE-130: `oracle_recall` / `mean_bestIoU` /
`score_AUC` bằng đúng `ce_localization.utils.metrics_np` (định nghĩa của ALPHA).

Dùng để CHỌN checkpoint (`hooks.BestCheckpointer` theo "ce130/oracle_recall", cạm bẫy 3), không phải
số báo cáo: số báo cáo lấy từ `predict.py` + bộ chấm chung. Mỗi ảnh lấy `budget` box điểm cao nhất
trong đầu ra của model (sau hậu xử lý chuẩn của chính model, như lúc dump).
"""

import itertools

import numpy as np
from detectron2.data import DatasetCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm

from ce_localization.utils.box_ops_np import xyxy_to_cxcywh
from ce_localization.utils.metrics_np import quality, summarise

__all__ = ["CE130BoxQualityEvaluator", "gt_xyxy_of_record"]


def gt_xyxy_of_record(record):
    """GT xyxy của một bản ghi DatasetCatalog (bbox COCO XYWH_ABS), bỏ vùng iscrowd."""
    boxes = [[x, y, x + w, y + h] for a in record.get("annotations", []) if not a.get("iscrowd", 0)
             for x, y, w, h in [a["bbox"]]]
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


class CE130BoxQualityEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, budget=200, iou_thr=0.5, distributed=True):
        self._gt = {d["image_id"]: gt_xyxy_of_record(d) for d in DatasetCatalog.get(dataset_name)}
        self._budget, self._iou_thr, self._distributed = budget, iou_thr, distributed
        self._acc = []

    def reset(self):
        self._acc = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            inst = out["instances"].to("cpu")
            boxes = inst.pred_boxes.tensor.numpy().astype(np.float64) if len(inst) else np.zeros((0, 4))
            scores = inst.scores.numpy().astype(np.float64) if len(inst) else np.zeros(0)
            keep = np.argsort(-scores, kind="stable")[:self._budget]
            gt = self._gt[inp["image_id"]]
            best, hit, n_gt, auc = quality(xyxy_to_cxcywh(boxes[keep]), scores[keep], xyxy_to_cxcywh(gt),
                                           size=1, iou_thr=self._iou_thr)
            self._acc.append((best, hit, n_gt, auc))

    def evaluate(self):
        acc = self._acc
        if self._distributed:
            comm.synchronize()
            acc = list(itertools.chain(*comm.gather(acc, dst=0)))
            if not comm.is_main_process():
                return {}
        res = summarise([a[0] for a in acc], sum(a[1] for a in acc), sum(a[2] for a in acc), [a[3] for a in acc])
        return {"ce130": {k: float(res[k]) for k in ("oracle_recall", "mean_bestIoU", "score_AUC")}}

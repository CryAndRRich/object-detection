"""Evaluator CrowdHuman cho detectron2: AP50 / mMR / Recall.

Vì sao cần thêm: ``COCOEvaluator`` cho AP/AP50 kiểu COCO, nhưng Table 7 của paper
DiffusionDet báo cáo **AP50 / mMR / Recall** — mMR là metric riêng của benchmark người đi
bộ mà detectron2 không có. Phần toán nằm ở ``objdet/mmr.py`` (numpy thuần, có self-test).

Khả năng so sánh: toolkit chính thức của CrowdHuman còn vài chi tiết riêng, nên con số ở
đây dùng để đối chiếu tương đối với Table 7. Nếu cần số chính thức tuyệt đối thì chạy lại
bằng toolkit gốc.
"""

import itertools
import json
import logging
import os
from collections import OrderedDict, defaultdict

import numpy as np

from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm

from .mmr import compute_mmr_and_recall

logger = logging.getLogger(__name__)


class CrowdHumanEvaluator(DatasetEvaluator):
    """Tính mMR / Recall / AP50 theo giao thức CrowdHuman.

    Dùng song song với ``COCOEvaluator`` (bọc trong ``DatasetEvaluators``) để có cả AP
    COCO-style lẫn 3 số của Table 7.
    """

    def __init__(self, dataset_name, output_dir=None, score_thresh=0.05):
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._score_thresh = score_thresh
        self._json_file = MetadataCatalog.get(dataset_name).json_file
        self._gt = None
        self._predictions = []

    def _load_gt(self):
        """Nạp GT từ chính json đã đăng ký; ``iscrowd=1`` là vùng ignore."""
        with open(self._json_file) as f:
            coco = json.load(f)
        by_img = defaultdict(lambda: ([], []))     # image_id -> (person, ignore)
        for ann in coco["annotations"]:
            x, y, w, h = ann["bbox"]
            by_img[ann["image_id"]][1 if ann.get("iscrowd", 0) else 0].append([x, y, x + w, y + h])
        self._gt = {
            img["id"]: (
                np.array(by_img[img["id"]][0], dtype=np.float64).reshape(-1, 4),
                np.array(by_img[img["id"]][1], dtype=np.float64).reshape(-1, 4),
            )
            # đi theo danh sách images để ảnh không có annotation nào vẫn được đếm
            for img in coco["images"]
        }

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            inst = out["instances"].to("cpu")
            inst = inst[inst.scores > self._score_thresh]
            boxes = inst.pred_boxes.tensor.numpy()
            scores = inst.scores.numpy()
            dets = (np.concatenate([boxes, scores[:, None]], axis=1)
                    if len(boxes) else np.zeros((0, 5)))
            self._predictions.append({"image_id": inp["image_id"], "dets": dets})

    def evaluate(self):
        preds = list(itertools.chain(*comm.gather(self._predictions, dst=0)))
        if not comm.is_main_process():
            return {}

        if self._gt is None:
            self._load_gt()

        empty = (np.zeros((0, 4)), np.zeros((0, 4)))
        per_image = {img_id: (np.zeros((0, 5)), gts, igs) for img_id, (gts, igs) in self._gt.items()}
        for p in preds:
            gts, igs = self._gt.get(p["image_id"], empty)
            per_image[p["image_id"]] = (p["dets"], gts, igs)

        res = compute_mmr_and_recall(per_image)
        logger.info(f"CrowdHuman {self._dataset_name}: AP50={res['AP50']:.2f} "
                    f"mMR={res['mMR']:.2f} Recall={res['Recall']:.2f}")

        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            with open(os.path.join(self._output_dir, "crowdhuman_metrics.json"), "w") as f:
                json.dump(res, f, indent=2)

        return OrderedDict({"crowdhuman": res})

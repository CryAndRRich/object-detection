"""Self-test cho objdet/box_quality_metrics.py (dùng bởi tools/measure_box_quality_ce130.py)
— không cần detectron2/torch (đối chiếu với các case đã biết trong
measure_box_quality.py gốc của CE-LocModel, để chắc công thức viết lại độc lập không
lệch nghĩa).

Chạy: python tests/test_measure_box_quality_ce130.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from objdet.box_quality_metrics import (  # noqa: E402
    box_iou_xyxy, quality_one_image, roc_auc, summarise,
)


def check(name, cond):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")
    assert cond, name


def main():
    print("test_measure_box_quality_ce130:")

    # ---------------- box_iou_xyxy ----------------
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [100, 100, 110, 110]])
    iou = box_iou_xyxy(a, b)
    check("IoU box trùng nhau tuyệt đối = 1.0", abs(iou[0, 0] - 1.0) < 1e-9)
    # overlap [5,10]x[5,10] = 25, union = 100+100-25=175 -> 25/175
    check("IoU box chồng lấn một phần đúng công thức", abs(iou[0, 1] - 25 / 175) < 1e-9)
    check("IoU box không chạm nhau = 0", iou[0, 2] == 0.0)
    check("box_iou_xyxy rỗng không crash", box_iou_xyxy(np.zeros((0, 4)), b).shape == (0, 3))

    # ---------------- roc_auc ----------------
    check("AUC hoàn hảo (positive toàn điểm cao) = 1.0",
          abs(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8]) - 1.0) < 1e-9)
    check("AUC ngược hoàn toàn = 0.0",
          abs(roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.9, 0.8]) - 0.0) < 1e-9)
    check("AUC ngẫu nhiên/tie hoàn toàn ~ 0.5",
          abs(roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-9)
    check("AUC thiếu 1 class -> nan", np.isnan(roc_auc([0, 0, 0], [0.1, 0.2, 0.3])))

    # ---------------- quality_one_image ----------------
    gt = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
    pred_perfect = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
    scores_perfect = np.array([0.9, 0.9])
    best, hit, ngt, auc, scored_hit = quality_one_image(pred_perfect, scores_perfect, gt)
    check("perfect: hit cả 2 GT", hit == 2 and ngt == 2)
    check("perfect: mean best IoU = 1.0", abs(best.mean() - 1.0) < 1e-9)
    check("perfect: scored_hit == oracle hit khi score đều tốt", scored_hit == 2)

    # box tốt nhưng KHÔNG có GT nào -> n_gt=0, không crash
    best0, hit0, ngt0, auc0, sh0 = quality_one_image(pred_perfect, scores_perfect, np.zeros((0, 4)))
    check("không có GT -> (rỗng, 0, 0, nan, 0)",
          len(best0) == 0 and hit0 == 0 and ngt0 == 0 and np.isnan(auc0) and sh0 == 0)

    # không có prediction nào -> mọi GT best_iou = 0, hit = 0
    best1, hit1, ngt1, auc1, sh1 = quality_one_image(np.zeros((0, 4)), np.zeros(0), gt)
    check("không có prediction -> best toàn 0, hit=0, ngt=2",
          (best1 == 0).all() and hit1 == 0 and ngt1 == 2)

    # box TỐT nhưng SCORE ngược (đúng kịch bản "score head hỏng" mà tool này tồn tại để bắt):
    # 1 box đúng GT[0] điểm THẤP, 1 box rác điểm CAO -> oracle_recall vẫn cao nhưng
    # recall_scored (theo greedy score-order) thấp hơn hẳn.
    pred_mixed = np.array([[0, 0, 10, 10], [50, 50, 60, 60]])   # box 2 không trúng GT nào
    scores_mixed = np.array([0.1, 0.9])                          # box đúng điểm thấp hơn box rác
    best2, hit2, ngt2, auc2, sh2 = quality_one_image(pred_mixed, scores_mixed, gt)
    check("score hỏng: oracle vẫn thấy 1/2 GT trúng (box đúng có mặt, bất kể score)",
          hit2 == 1)
    check("score hỏng: AUC thấp vì box khớp bị xếp điểm thấp hơn box không khớp",
          auc2 < 0.5)

    # ---------------- summarise ----------------
    res = summarise([best], hit, ngt, [auc] if not np.isnan(auc) else [], scored_hit)
    check("summarise: oracle_recall == hit/n_gt", abs(res["oracle_recall"] - hit / ngt) < 1e-9)
    check("summarise: score_head_cost = oracle - scored",
          abs(res["score_head_cost"] - (res["oracle_recall"] - res["recall_scored"])) < 1e-9)
    res_empty = summarise([], 0, 0, [], 0)
    check("summarise: n_gt=0 không chia-cho-0 (oracle_recall=0.0, không NaN/crash)",
          res_empty["oracle_recall"] == 0.0 and res_empty["mean_bestIoU"] == 0.0)

    print("ALL OK")


def test_box_quality_metrics():
    """Wrapper cho pytest — xem ghi chú trong test_convert_ce130.py."""
    main()


if __name__ == "__main__":
    main()

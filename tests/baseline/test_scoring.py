"""Bộ chấm chung của baseline (`baseline/scoring.py`, `baseline/tools/score_predictions.py`).

Điểm then chốt: chấm dump trên toạ độ ẢNH GỐC phải cho ĐÚNG số như chấm kiểu ALPHA (box trên canvas
letterbox 1024, chuẩn hoá theo vùng ảnh thật) — nếu không, bảng docs/SCORE.md so hai thước khác nhau.
"""

import json
import os

import numpy as np
import pytest

from baseline.scoring import (budget_tag, iid_of_path, load_gt, read_dump, records_from_pred, score_pred,
                              score_row, top_budget, write_dump)
from baseline.tools.score_predictions import parse_budgets, score_dump_file
from ce_localization.data.dataset import scale_boxes, scan_ce130
from ce_localization.engine.evaluate import score
from ce_localization.utils.box_ops_np import xyxy_to_cxcywh
from tests.ce_localization.helpers import _fake_ce130

# (W, H) sao cho letterbox 1024 cho nw, nh NGUYÊN (int() không cắt) -> hai đường phải trùng đến sai số máy
SIZES = [(512, 384), (640, 320), (1024, 384), (256, 256)]
KEYS = ("AP50", "AP75", "AP_coco", "recall", "oracle_recall", "mean_bestIoU", "score_AUC", "kept_per_image",
        "AP_pt", "oracle_recall_pt", "score_AUC_pt")


def _case(seed=0, n_pred=60):
    rng = np.random.default_rng(seed)
    gt_table, pred = {}, {}
    for i, (w, h) in enumerate(SIZES):
        n = int(rng.integers(5, 40))
        x1, y1 = rng.uniform(0, w * 0.9, n), rng.uniform(0, h * 0.9, n)
        gt = np.stack([x1, y1, x1 + rng.uniform(4, w * 0.2, n), y1 + rng.uniform(4, h * 0.2, n)], 1)
        gt = scale_boxes(gt, 1.0, w, h)
        near = gt[rng.integers(0, len(gt), n_pred // 2)] + rng.normal(0, 3, (n_pred // 2, 4))
        x1, y1 = rng.uniform(0, w, n_pred - len(near)), rng.uniform(0, h, n_pred - len(near))
        far = np.stack([x1, y1, x1 + rng.uniform(5, 60, len(x1)), y1 + rng.uniform(5, 60, len(x1))], 1)
        boxes = np.concatenate([near, far])
        iid = str(100 + i)
        gt_table[iid] = {"wh": (w, h), "gt": gt, "text": "x", "img_path": f"/x/test/{iid}_b1/ground_truth.jpg"}
        pred[iid] = {"boxes_xyxy": boxes, "scores": rng.uniform(0, 1, len(boxes))}
    return gt_table, pred


def _alpha_records(gt_table, pred, T=1024):
    """Bản ghi y hệt `ce_localization.engine.evaluate.predict` (canvas letterbox T, chuẩn hoá (nw, nh))."""
    recs = []
    for iid in sorted(gt_table):
        w, h = gt_table[iid]["wh"]
        sc = min(T / w, T / h)
        nw, nh = int(w * sc), int(h * sc)
        assert nw == w * sc and nh == h * sc
        whwh = np.array([nw, nh, nw, nh], dtype=np.float64)
        gt_c = scale_boxes(gt_table[iid]["gt"], sc, nw, nh)
        b = np.asarray(pred[iid]["boxes_xyxy"]) * sc
        recs.append({"image_id": iid, "boxes": xyxy_to_cxcywh(b / whwh), "scores": np.asarray(pred[iid]["scores"]),
                     "keep": np.zeros(0, dtype=int), "gt": xyxy_to_cxcywh(gt_c / whwh),
                     "gt_size_px": 512.0 / T * np.sqrt((gt_c[:, 2] - gt_c[:, 0]) * (gt_c[:, 3] - gt_c[:, 1]))})
    return recs


@pytest.mark.parametrize("order", ["topk_first", "nms_first"])
def test_same_numbers_as_alpha_scoring(order):
    gt_table, pred = _case()
    ours = score(records_from_pred(pred, gt_table, budget=None), [], 100, 0.5, oracle=True, order=order)
    alpha = score(_alpha_records(gt_table, pred), [], 100, 0.5, oracle=True, order=order)
    for k in KEYS:
        assert ours[k] == pytest.approx(alpha[k], abs=1e-9), k
    assert ours["oracle_score"]["AP50"] == pytest.approx(alpha["oracle_score"]["AP50"], abs=1e-9)
    for grp in ("density_recall", "size_recall"):
        for name, v in alpha[grp].items():
            assert ours[grp][name]["oracle_recall"] == pytest.approx(v["oracle_recall"], abs=1e-9), (grp, name)
            assert ours[grp][name]["n_gt"] == v["n_gt"]


def test_perfect_prediction_scores_one():
    gt_table, _ = _case(1)
    pred = {k: {"boxes_xyxy": v["gt"], "scores": np.linspace(1, 0.5, len(v["gt"]))} for k, v in gt_table.items()}
    res = score_pred(pred, gt_table)
    for order in ("topk_first", "nms_first"):
        r = res[order]
        assert r["oracle_recall"] == pytest.approx(1.0)
        assert r["AP_pt"] == pytest.approx(1.0)
    # < 100 GT mỗi ảnh -> không bị top-k chặn: AP50 = 1 ở cả hai thứ tự
    assert all(len(v["gt"]) <= 100 for v in gt_table.values())
    assert res["nms_first"]["AP50"] == pytest.approx(1.0)


def test_budget_keeps_highest_scores_only():
    b, s = top_budget(np.arange(12).reshape(3, 4), [0.1, 0.9, 0.5], 2)
    assert s.tolist() == [0.9, 0.5] and b[0].tolist() == [4, 5, 6, 7]
    b, s = top_budget(np.zeros((3, 4)), [0.1, 0.9, 0.5], None)
    assert len(s) == 3
    with pytest.raises(ValueError):
        top_budget(np.zeros((2, 4)), [0.1], 5)
    # box đúng duy nhất có score thấp nhất: ngân sách 200 bỏ mất nó, 'all' thì còn
    gt_table = {"1": {"wh": (100, 100), "gt": np.array([[10.0, 10, 30, 30]]), "text": "", "img_path": ""}}
    boxes = np.concatenate([np.tile([[60.0, 60, 90, 90]], (299, 1)), [[10.0, 10, 30, 30]]])
    pred = {"1": {"boxes_xyxy": boxes, "scores": np.r_[np.full(299, 0.9), 0.1]}}
    assert score_pred(pred, gt_table, budget=200)["nms_first"]["oracle_recall"] == 0.0
    assert score_pred(pred, gt_table, budget=None)["nms_first"]["oracle_recall"] == 1.0
    assert budget_tag(200) == "B200" and budget_tag(None) == "Ball" and parse_budgets(["200", "all"]) == [200, None]


def test_missing_or_extra_image_is_an_error():
    gt_table, pred = _case()
    with pytest.raises(KeyError):
        records_from_pred({k: v for k, v in list(pred.items())[1:]}, gt_table)
    with pytest.raises(KeyError):
        records_from_pred({**pred, "999": pred["100"]}, gt_table)


def test_score_row_matches_score_md_columns():
    gt_table, pred = _case()
    res = score_pred(pred, gt_table)
    row = score_row({"run": "BASELINE0", "config": "baseline/configs/x.yaml", "steps": 1, "iter": 12000,
                     "batch": 2, "where": "A30 server", "train_time": "1h", "date": "2026-10-01"}, res)
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    assert len(cells) == 25                       # đúng số cột của docs/SCORE.md
    assert cells[0] == "BASELINE0" and cells[3] == "1" and cells[-1] == "2026-10-01"
    assert "," in cells[4] and "." not in cells[4]  # số kiểu Việt: 0,530
    row4 = score_row({"run": "B", "steps": 4, "budget": None}, res)
    assert "4 (gộp" in row4


def test_iid_of_path_matches_scan_rule():
    assert iid_of_path("/d/all_phase2_V2/test/4499_b2/ground_truth.jpg") == "4499"


def test_full_flow_dump_then_score(tmp_path):
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root)
    gt = load_gt(root, "test")
    items = {it["image_id"]: it for it in scan_ce130(root, "test")}
    assert set(gt) == set(items) and all(v["wh"] == (200, 150) for v in gt.values())
    pred = {iid: {"boxes_xyxy": v["gt"], "scores": np.linspace(0.9, 0.5, len(v["gt"]))} for iid, v in gt.items()}
    path = str(tmp_path / "out" / "BASELINEX_test.json")
    write_dump(path, {"run": "BASELINEX", "config": "c.yaml", "split": "test", "steps": "—"}, pred)
    meta, back = read_dump(path)
    assert meta["run"] == "BASELINEX" and set(back) == set(pred)
    out = score_dump_file(path, root, [200, None], log=lambda *a, **k: None)
    assert list(out["results"]) == ["Ball"]        # dump < 200 box/ảnh: B200 trùng Ball -> chỉ chấm một lần
    assert out["results"]["Ball"]["nms_first"]["AP50"] == pytest.approx(1.0)
    with open(os.path.splitext(path)[0] + "_metrics.json", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["score_rows"]["Ball"].startswith("| BASELINEX |")

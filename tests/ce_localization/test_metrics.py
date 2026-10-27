"""Chấm bài add (IoU, on_object, C-NLL như bài Add One Take One), prior / đổi cỡ, lịch lr + thứ tự batch, config kế thừa."""

import os

import numpy as np
import pytest
import yaml


def _rec(boxes, holes, objects, wh=(100.0, 100.0), t=2, **kw):
    return {"boxes": np.asarray(boxes, dtype=np.float64), "holes": np.asarray(holes, dtype=np.float64).reshape(-1, 4),
            "objects": np.asarray(objects, dtype=np.float64).reshape(-1, 4), "wh": np.asarray(wh), "t": t, "image_id": "x", **kw}


def _boxes(rng, n, lo=0, hi=80, size=(4, 15)):
    xy = rng.uniform(lo, hi, (n, 2))
    return np.concatenate([xy, xy + rng.uniform(*size, (n, 2))], 1)


def test_add_metrics_hand_computed():
    from ce_localization.engine.add_eval import add_metrics
    holes = [[0, 0, 10, 10], [50, 50, 60, 60]]                       # lỗ cũ, lỗ MỚI NHẤT
    boxes = [[0, 0, 10, 10],                                          # trùng lỗ cũ: IoU any 1, latest 0
             [50, 50, 60, 65],                                        # IoU latest 100/150
             [80, 80, 70, 90],                                        # suy biến (w < 0) -> IoU 0
             [95, 95, 105, 105]]                                      # tràn ra ngoài ảnh
    objects = [[20, 20, 40, 40], [94, 94, 104, 104]]
    r = add_metrics([_rec(boxes, holes, objects)])
    assert r["best_iou@4_any"] == 1.0 and np.isclose(r["best_iou@4_latest"], 100 / 150) and r["hit50@4_any"] == r["hit50@4_latest"] == 1.0
    assert np.isclose(r["mean_iou_any"], (1 + 100 / 150) / 4) and np.isclose(r["mean_iou_latest"], (100 / 150) / 4)
    assert r["box_hit50_any"] == 0.5 and r["box_hit50_latest"] == 0.25 and r["hole_cover"] == 1.0 and r["degenerate"] == 0.25
    assert r["in_image"] == 0.5 and r["on_object"] == 0.25                # box 3: giao 9×9 / 100 >= 0,5
    assert r["n_cnll"] == 0 and np.isnan(r["cnll"])                       # < 5 vật -> bỏ C-NLL
    r2 = add_metrics([_rec(boxes, holes, objects)], with_holes=False)
    assert "mean_iou_any" not in r2 and "cnll_gt" not in r2 and r2["on_object"] == 0.25
    assert not any("latest" in k for k in add_metrics([_rec(boxes, holes, objects)], latest=False))


def test_cnll_definition_f1all_and_paper_aggregate():
    """C-NLL = −log N(x) − min_i(−log N(z_i)), z = [w, h, IoU lớn nhất, TB khoảng cách 3 tâm gần nhất] của vật cùng lớp (bỏ chính nó);
    chiều quan hệ tính với MỌI vật (lớp kia = `objects_all` [cùng lớp ; lớp kia]); `cnll` = TB trên ảnh của box THẤP nhất trong K,
    `cnll_gt` cùng cách trên lỗ GT; < 5 vật cùng lớp -> bỏ ảnh; `record_cnll` (dump) khớp."""
    from scipy.stats import multivariate_normal
    from ce_localization.engine.add_eval import CNLL_RIDGE, add_metrics, cnll, other_class, record_cnll
    objs = np.array([[10.0 + 15 * k, 10.0, 20.0 + 15 * k, 20.0 + k] for k in range(6)])        # một hàng vật cùng lớp
    others = np.array([[10.0 + 15 * k, 60.0, 20.0 + 15 * k, 70.0] for k in range(4)])           # lớp kia
    wh = (120.0, 100.0)
    box = np.array([[10.0, 60.0, 20.0, 70.0], [55.0, 35.0, 65.0, 45.0]])                        # đè vật lớp kia | chỗ trống
    c = np.stack([(objs[:, 0] + objs[:, 2]) / 2, (objs[:, 1] + objs[:, 3]) / 2], 1)
    d = np.linalg.norm(c[:, None] - c[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    z = np.stack([(objs[:, 2] - objs[:, 0]) / 120, (objs[:, 3] - objs[:, 1]) / 100, np.zeros(6),
                  np.sort(d, 1)[:, :3].mean(1) / np.sqrt(120 * 100)], 1)                      # vật không chồng nhau: IoU 0
    mvn = multivariate_normal(z.mean(0), np.cov(z.T, bias=True) + CNLL_RIDGE * np.eye(4))
    bc = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], 1)
    x = np.stack([np.full(2, 10 / 120), np.full(2, 10 / 100), np.zeros(2),
                  np.sort(np.linalg.norm(bc[:, None] - c[None], axis=-1), 1)[:, :3].mean(1) / np.sqrt(120 * 100)], 1)
    f1 = cnll(box, objs, wh)
    assert np.allclose(f1, -mvn.logpdf(x) - (-mvn.logpdf(z)).min())
    fa = cnll(box, objs, wh, others=others)
    assert fa[0] > f1[0] + 1.0 and np.allclose(cnll(box, objs, wh, others=np.zeros((0, 4))), f1)
    assert cnll(box, objs[:4], wh, others=others) is None
    with pytest.raises(ValueError, match="objects_all"):
        other_class(objs, np.concatenate([others, objs]))
    rng = np.random.default_rng(5)
    recs = []
    for i in range(5):
        o = _boxes(rng, 6 + i)
        recs.append(_rec(_boxes(rng, 10), _boxes(rng, 4), o, objects_all=np.concatenate([o, _boxes(rng, 3)])))
    recs.append(_rec(_boxes(rng, 10), _boxes(rng, 4), _boxes(rng, 3)))                       # < 5 vật: bỏ
    m = add_metrics(recs, latest=False)
    rc = [record_cnll(r) for r in recs[:5]]
    assert record_cnll(recs[5]) is None and m["n_cnll"] == 5
    assert np.isclose(m["cnll"], np.mean([c_.min() for c_, _ in rc])) and np.isclose(m["cnll_gt"], np.mean([g.min() for _, g in rc]))
    r0 = recs[0]
    assert np.allclose(rc[0][0], cnll(r0["boxes"], r0["objects"], (100.0, 100.0), others=other_class(r0["objects"], r0["objects_all"])))


def test_select_nms_and_fit_cache():
    """Minh hoạ: k box C-NLL thấp nhất sau NMS (box trùng IoU > ngưỡng bị bỏ, không fit lại). Gaussian nhớ theo NỘI DUNG (sửa box vật tại
    chỗ -> fit lại, kết quả trùng từng bit với bộ nhớ rỗng)."""
    from ce_localization.engine import add_eval as ae
    rng = np.random.default_rng(3)
    objs, boxes = _boxes(rng, 7, size=(6, 12)), _boxes(rng, 12, hi=90, size=(3, 25))
    b = np.array([[10, 10, 30, 30], [11, 10, 31, 30], [60, 60, 80, 80], [40, 0, 50, 10]], dtype=np.float64)
    c = np.array([1.0, 2.0, 3.0, 0.5])                                  # box 1 trùng box 0 (IoU 0,9)
    assert ae.select_nms(b, c, 3) == [3, 0, 2] and ae.select_nms(b, c, 3, iou=0.95) == [3, 0, 1] and ae.select_nms(b, c, 9) == [3, 0, 2]
    ae._FIT.clear()
    cold = ae.cnll(boxes, objs, (100.0, 80.0))
    assert np.array_equal(ae.cnll(boxes, objs, (100.0, 80.0)), cold) and len(ae._FIT) == 1
    objs[0] += 3.0
    moved = ae.cnll(boxes, objs, (100.0, 80.0))
    ae._FIT.clear()
    assert np.array_equal(ae.cnll(boxes, objs, (100.0, 80.0)), moved) and not np.array_equal(moved, cold)


def test_prior_and_resize_to_objects():
    from ce_localization.engine.add_eval import prior_records, resize_to_objects
    rec = _rec([[0, 0, 1, 1], [10, 10, 30, 20]], [[0, 0, 1, 1]], [[0, 0, 4, 6], [10, 10, 16, 20]], wh=(200.0, 100.0),
               exemplars=np.array([[0, 0, 2, 2]], dtype=np.float64))
    pri = prior_records([rec], np.array([[0.5, 0.5, 0.1, 0.2]]), n_samples=3)
    assert np.allclose(pri[0]["boxes"], [[90, 40, 110, 60]] * 3)
    assert np.allclose(resize_to_objects([rec], "mean")[0]["boxes"], [[-2, -3.5, 3, 4.5], [17.5, 11, 22.5, 19]])   # (w, h) TB = (5, 8)
    assert np.allclose(resize_to_objects([rec], "exemplar")[0]["boxes"][1], [19, 14, 21, 16])
    assert resize_to_objects([{**rec, "objects": np.zeros((0, 4))}])[0] is not None


def test_lr_schedule_and_epoch_batches():
    """Cosine theo iter (+ warmup tuyến tính, vắng = như cũ), về 0 ở t_max; `EpochBatches` (worker bền) == `epoch_batches` cả khi bắt đầu
    giữa epoch; batch hai rank rời nhau."""
    import math
    from ce_localization.engine.train_utils import EpochBatches, epoch_batches, lr_factor
    tr = {"schedule": "cosine", "cosine_t_max": 1000}
    cos = lambda it: 0.5 * (1 + math.cos(math.pi * min(it, 1000) / 1000))  # noqa: E731
    assert all(lr_factor(it, tr) == cos(it) for it in (0, 500, 1000, 2000))
    tw = {**tr, "cosine_warmup_iters": 100}
    assert lr_factor(0, tw) == cos(0) / 100 and lr_factor(49, tw) == cos(49) * 0.5 and lr_factor(500, tw) == cos(500)
    s = EpochBatches(23, 2, 1, 2, 7)
    for ep, start in ((0, 0), (3, 2)):
        s.set(ep, start)
        assert list(s) == epoch_batches(23, 2, 1, 2, 7, ep)[start:] and len(s) == 5 - start
    a, b = epoch_batches(23, 2, 0, 2, 7, 1), epoch_batches(23, 2, 1, 2, 7, 1)
    assert not {i for x in a for i in x} & {i for x in b for i in x}
    with pytest.raises(ValueError):
        lr_factor(0, {"schedule": "multistep"})


def test_configs_inherit_only_intended_keys(tmp_path):
    """Config kế thừa (`base`, gộp sâu): delta5 = delta4 + box log + GIoU 0 + val ddim; delta5_k chỉ đổi `aux_giou_weight`; delta6 /
    6_1 / 6_2 = delta4 / 5_1 / 5_4 + giao thức CE-Loc gốc (không val, best theo loss train, 120 epoch = 37.440 iter, lr 1e-4 cosine về 0,
    train lấy ngẫu nhiên 30 vật); delta7 = delta6 + đích train = box positive ngẫu nhiên;
    7_0_1 / 7_0_2 = delta7 + mọi box / ảnh mỗi iter | mỗi cặp (ảnh, box) một mẫu (120 epoch theo cặp); 7_1 = 6_1 + đích của 7."""
    from ce_localization.utils.config import load_config, merge
    from tests.ce_localization.helpers import CFG_DIR
    L = lambda r: load_config(os.path.join(CFG_DIR, f"{r}.yaml"))  # noqa: E731
    d4, d5 = L("delta4"), L("delta5")
    assert d5 == merge(d4, {"experiment": "delta5", "description": d5["description"],
                            "model": {"box_coder": {"type": "log", "min_size_ratio": 1 / 128}, "refiner": {"aux_giou_weight": 0.0}},
                            "eval": {"sample_kw": {"sampler": "ddim", "ddim_steps": 100, "eta": 1.0}}})
    for run, w in (("delta5_1", 0.1), ("delta5_2", 0.5), ("delta5_3", 0.25), ("delta5_4", 0.75), ("delta5_5", 1.0)):
        assert L(run) == merge(d5, {"experiment": run, "description": L(run)["description"], "model": {"refiner": {"aux_giou_weight": w}}})
    proto = yaml.safe_load(open(os.path.join(CFG_DIR, "protocol_celoc.yaml")))
    assert proto["training"]["max_iter"] == proto["training"]["cosine_t_max"] == 120 * (19998 // d4["training"]["batch_size"])
    assert (d4["model"]["obj_max"], proto["model"]) == (300, {"obj_max": 30})
    for run, src in (("delta6", "delta4"), ("delta6_1", "delta5_1"), ("delta6_2", "delta5_4")):
        assert L(run) == merge(merge(L(src), proto), {"experiment": run, "description": L(run)["description"]})
    assert L("delta7") == merge(L("delta6"), {"experiment": "delta7", "description": L("delta7")["description"], "data": {
        "train_target": "positive", "positive_file": "../../output/add/box_review/ce130_train.jsonl", "positive_hole_iou": 0.3,
        "positive_sampling": "random"}})
    assert L("delta7_0_1") == merge(L("delta7"), {"experiment": "delta7_0_1", "description": L("delta7_0_1")["description"],
                                                  "data": {"positive_sampling": "all"}})
    pos_keys = ("train_target", "positive_file", "positive_hole_iou", "positive_sampling")
    assert L("delta7_1") == merge(L("delta6_1"), {"experiment": "delta7_1", "description": L("delta7_1")["description"],
                                                "data": {k: L("delta7")["data"][k] for k in pos_keys}})
    ipe = 70567 // 64                                               # cặp (mẫu, box) của bản positive 17.454 mẫu, IoU 0,3
    assert L("delta7_0_2") == merge(L("delta7"), {"experiment": "delta7_0_2", "description": L("delta7_0_2")["description"],
                                                  "data": {"positive_sampling": "pair"}, "training": {
                                                      "cosine_warmup_iters": ipe, "cosine_t_max": 120 * ipe, "max_iter": 120 * ipe}})
    assert L("delta8") == merge(L("delta7_0_2"), {"experiment": "delta8", "description": L("delta8")["description"], "training": {
        "max_iter": 10 * ipe, "snapshot_every_epochs": 1, "snapshot_initial": True}})
    assert L("delta8")["training"]["cosine_t_max"] == 120 * ipe and L("delta7_0_2")["training"]["snapshot_initial"] is False
    (tmp_path / "a.yaml").write_text("x: {y: 1, z: [1]}\nk: 1\n")
    (tmp_path / "b.yaml").write_text("base: a.yaml\nx: {z: [2]}\n")
    assert load_config(str(tmp_path / "b.yaml")) == {"x": {"y": 1, "z": [2]}, "k": 1}

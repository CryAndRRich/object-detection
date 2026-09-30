"""EXPERIMENT BETA — cửa G1 (docs/EXPERIMENT_BETA.md mục 7): nhãn điểm density, cỡ giả kNN, matcher +
loss chế độ `point`, chỉ số điểm, TRỌN LUỒNG G0 (tool) -> train -> dừng -> --resume -> eval trên
CE-130 giả. Dùng lại dữ liệu giả / text giả / `_run_train` của test_alpha.py (cùng quy ước: không
tải gì, ép `--device cpu`).
"""

import json
import os
import sys

import numpy as np
import pytest
import torch
import yaml

from ce_localization.alpha import criterion as C
from ce_localization.alpha.data import AlphaCE130, collate, scale_points
from ce_localization.alpha.density import JET, decode_jet
from ce_localization.alpha.evaluate import point_metrics
from ce_localization.alpha.head import apply_deltas
from ce_localization.alpha.points import (PointTable, find_peaks, knn_distance, match_points_to_boxes,
                                          pseudo_boxes, pseudo_sizes)
from tests.ce_localization.test_alpha import (CFG0, CFG_DIR, _fake_ce130, _fake_density,
                                              _fake_text_table, _run_train, _targets)

CFG_B0 = os.path.join(CFG_DIR, "beta0.yaml")
PSEUDO = {"knn": 3, "beta": 1.0, "min_frac": 0.02, "max_frac": 0.30}


def _gauss(h, w, centers, sigma=2.5, peak=1.0):
    yy, xx = np.mgrid[0:h, 0:w] + 0.5
    d = sum(np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)) for cx, cy in centers)
    return np.round(d / d.max() * 255 * peak).astype(np.uint8)


# ----------------------------------------------------------------------------- tách đỉnh

def test_find_peaks_separate_blobs_background_and_merged():
    c = [(10.5, 12.5), (40.5, 12.5), (25.5, 30.5)]
    lv = _gauss(48, 56, c)
    p = find_peaks(lv, tau=8, radius=2)
    assert len(p) == 3
    for cx, cy in c:
        assert np.min(np.hypot(p[:, 0] - cx, p[:, 1] - cy)) < 0.51           # đúng tâm pixel
    assert len(find_peaks(np.zeros((20, 20), np.uint8), 8, 2)) == 0          # nền -> 0 điểm
    assert len(find_peaks(_gauss(40, 40, [(15.5, 20.5), (18.5, 20.5)]), 8, 2)) == 1   # 2 vật sát -> gộp 1
    lo = _gauss(48, 56, c, peak=0.1)                                          # đỉnh thấp hơn tau -> bỏ
    assert len(find_peaks(lo, tau=64, radius=2)) == 0


def test_find_peaks_plateau_is_one_point_at_centroid_and_survives_jet():
    lv = np.zeros((20, 30), np.uint8)
    lv[5:8, 10:14] = 200                                                       # đỉnh phẳng 3×4
    lv[4, 9:15] = lv[8, 9:15] = 100
    p = find_peaks(lv, 8, 1)
    assert p.tolist() == [[12.0, 6.5]]                     # trọng tâm cột 10..13, hàng 5..7 (+0,5: pixel liên tục)
    dec, _ = decode_jet(JET[_gauss(48, 56, [(10.5, 12.5), (40.5, 30.5)])])     # qua mã màu jet thật
    assert len(find_peaks(dec, 8, 2)) == 2


# ----------------------------------------------------------------------------- cỡ giả

def test_knn_and_pseudo_sizes_by_hand():
    p = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [10.0, 10.0]])
    d = knn_distance(p, k=3)
    assert d[0] == pytest.approx((3 + 4 + np.hypot(10, 10)) / 3)
    assert d[1] == pytest.approx((3 + 5 + np.hypot(7, 10)) / 3)
    assert knn_distance(p[:2], k=3).tolist() == [3.0, 3.0]                     # < k láng giềng
    assert np.isnan(knn_distance(p[:1])).all()
    s = pseudo_sizes(p, 3, beta=2.0, s_min=5.0, s_max=12.0)
    assert np.allclose(s, np.clip(2.0 * d, 5.0, 12.0))
    assert pseudo_sizes(p[:1], 3, 1.0, 5.0, 12.0).tolist() == [12.0]           # 1 điểm -> s_max
    b = pseudo_boxes(p, s)
    assert np.allclose((b[:, :2] + b[:, 2:]) / 2, p) and np.allclose(b[:, 2] - b[:, 0], s)
    assert (b[:, 0] < 0).any()                                                 # KHÔNG kẹp: giữ tâm


def test_scale_points_drops_outside_valid_region():
    p = scale_points([[10, 10], [150, 20], [20, 149]], 0.5, 70, 60)
    assert p.tolist() == [[5.0, 5.0]]


def test_match_points_to_boxes_one_to_one_inside_only():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [50, 50, 60, 60]], float)
    pts = np.array([[4.0, 5.0], [9.0, 5.0], [30.0, 30.0]])
    m = match_points_to_boxes(pts, boxes)
    assert sorted(m["pairs"]) == [(0, 0), (1, 1)]                              # gần tâm box nhất, 1-1
    assert m["inside"][2].sum() == 0 and m["inside"][:, 2].sum() == 0
    assert match_points_to_boxes(np.zeros((0, 2)), boxes)["pairs"] == []


# ----------------------------------------------------------------------------- dữ liệu

def _write_points(path, table):
    with open(path, "w") as f:
        json.dump({"params": {"tau": 8, "radius": 2}, "points": table}, f)
    return PointTable(path)


def test_dataset_point_targets_are_pseudo_boxes_gt_only_for_scoring(tmp_path):
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root, n_train=2, n_val=1, n_test=1)
    ref = AlphaCE130(root, "train", 128)
    ids = [it["image_id"] for it in ref.items]
    pts = {ids[0]: [[20.0, 30.0], [100.0, 50.0], [60.0, 120.0], [199.0, 10.0]], ids[1]: []}
    pt = _write_points(str(tmp_path / "p.json"), pts)
    ds = AlphaCE130(root, "train", 128, targets="point", points=pt, pseudo=PSEUDO)
    s, s_ref = ds[0], ref[0]
    nh, nw = s["valid_hw"]
    exp = scale_points(pts[ids[0]], 0.64, nw, nh)
    b = s["boxes"].double().numpy()
    assert np.allclose((b[:, :2] + b[:, 2:]) / 2, exp, atol=1e-4)             # tâm box giả = điểm × scale
    w = b[:, 2] - b[:, 0]
    assert np.all(w >= PSEUDO["min_frac"] * nh - 1e-4) and np.all(w <= PSEUDO["max_frac"] * nh + 1e-4)
    assert torch.equal(s["gt_boxes"], s_ref["boxes"])                          # box GT giữ nguyên để chấm
    assert torch.equal(s["image"], s_ref["image"])
    assert len(ds[1]["boxes"]) == 0 and len(ds[1]["gt_boxes"]) > 0             # ảnh density trống
    assert torch.equal(ref[0]["gt_boxes"], ref[0]["boxes"])                    # ALPHA: gt == đích
    bt = collate([s, ds[1]])
    assert len(bt["gt_boxes"]) == 2 and len(bt["boxes"][1]) == 0
    with pytest.raises(ValueError):                                            # thiếu điểm của một ảnh
        AlphaCE130(root, "train", 128, targets="point", points=_write_points(
            str(tmp_path / "q.json"), {ids[0]: []}), pseudo=PSEUDO)


# ----------------------------------------------------------------------------- matcher + loss

CFG_L = {"alpha": 0.25, "gamma": 2.0, "class_weight": 2.0, "l1_weight": 5.0, "giou_weight": 2.0,
         "center_weight": 5.0, "size_weight": 1.0}
CFG_M = {"ota_k": 5, "center_radius": 2.5}


def test_point_matcher_prefers_box_centred_on_point_and_ignores_far_boxes():
    gt = torch.tensor([[10.0, 10.0, 30.0, 30.0], [60.0, 60.0, 90.0, 90.0]])   # box giả, tâm (20,20) (75,75)
    pred = torch.tensor([[[12.0, 12.0, 28.0, 28.0], [0.0, 0.0, 5.0, 5.0], [65.0, 60.0, 85.0, 90.0],
                          [40.0, 40.0, 50.0, 50.0]]])
    (sel, gi), = C.match(torch.zeros(1, 4, 1), pred, _targets(gt), mode="point")
    m = dict(zip(torch.nonzero(sel).flatten().tolist(), gi.tolist()))
    assert m.get(0) == 0 and m.get(2) == 1 and 1 not in m and 3 not in m


def test_point_loss_by_hand_and_gradient_split_center_vs_size():
    wh = (100.0, 80.0)
    gt = torch.tensor([[10.0, 10.0, 30.0, 30.0]])                             # điểm (20,20), ŝ = 20
    base = torch.tensor([[12.0, 13.0, 36.0, 29.0]])                           # tâm (24,21), w 24, h 16
    delta = torch.zeros(1, 4, requires_grad=True)
    boxes = apply_deltas(delta, base)[None]
    tg = _targets(gt, wh)
    crit = C.AlphaCriterion(CFG_L, CFG_M, mode="point")
    loss, st = crit.loss_one(torch.full((1, 1, 1), 2.0), boxes, tg)
    assert st["n_matched"] == 1
    assert float(st["loss_center"]) == pytest.approx(4 / 100 + 1 / 80, rel=1e-5)
    assert float(st["loss_size"]) == pytest.approx(abs(np.log(24 / 20)) + abs(np.log(16 / 20)), rel=1e-5)
    assert float(st["center_px"]) == pytest.approx(np.hypot(4, 1), rel=1e-5)
    for w_c, w_s, zero in ((1.0, 0.0, [2, 3]), (0.0, 1.0, [0, 1])):          # tâm chỉ vào dx,dy; cỡ chỉ dw,dh
        c = C.AlphaCriterion({**CFG_L, "center_weight": w_c, "size_weight": w_s}, CFG_M, mode="point")
        d = torch.zeros(1, 4, requires_grad=True)
        c.loss_one(torch.full((1, 1, 1), 2.0), apply_deltas(d, base)[None], tg)[0].backward()
        assert torch.all(d.grad[0, zero] == 0) and torch.any(d.grad[0, [i for i in range(4) if i not in zero]] != 0)
    assert crit.log_keys == ("loss_ce", "loss_center", "loss_size", "center_px")
    tot, st6 = crit(torch.full((6, 1, 1, 1), 2.0), torch.stack([boxes] * 6), tg)
    assert torch.allclose(tot, 6 * loss) and "loss_center_final" in st6


def test_point_loss_empty_targets_and_box_mode_keys_unchanged():
    crit = C.AlphaCriterion(CFG_L, CFG_M, mode="point")
    tg = C.build_targets([torch.zeros(0, 4)], torch.tensor([[100.0] * 4]))
    boxes = torch.rand(1, 5, 4, requires_grad=True)
    loss, st = crit.loss_one(torch.zeros(1, 5, 1), boxes * 10 + torch.tensor([0, 0, 20, 20.0]), tg)
    assert st["n_matched"] == 0 and torch.isfinite(loss)
    _, sb = C.AlphaCriterion(CFG_L, CFG_M)(torch.zeros(6, 1, 3, 1), torch.rand(6, 1, 3, 4) * 10 + 5,
                                           _targets(torch.tensor([[5.0, 5.0, 15.0, 15.0]])))
    assert {k for k in sb if k.endswith("_final")} == {"loss_ce_final", "loss_bbox_final", "loss_giou_final",
                                                        "n_matched_final", "iou_matched_final"}
    with pytest.raises(ValueError):
        C.AlphaCriterion(CFG_L, CFG_M, mode="points")


# ----------------------------------------------------------------------------- chỉ số điểm

def test_point_metrics_by_hand():
    gt = np.array([[0.2, 0.2, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]])              # cxcywh
    boxes = np.array([[0.21, 0.2, 0.9, 0.9],       # tâm trong GT0 (box to: điểm không chấm cỡ)
                      [0.19, 0.22, 0.1, 0.1],      # bản trùng GT0
                      [0.72, 0.68, 0.1, 0.1],      # trong GT1
                      [0.45, 0.45, 0.1, 0.1]])     # ngoài mọi GT
    sc = np.array([0.9, 0.8, 0.7, 0.6])
    r = point_metrics([{"boxes": boxes, "scores": sc, "keep": np.arange(4), "gt": gt}])
    assert r["oracle_recall_pt"] == 1.0 and r["recall_pt"] == 1.0 and r["precision_pt"] == 0.5
    # PR theo score: TP, FP, TP, FP -> rec (.5,.5,1,1), prec (1,.5,.67,.5) -> AP = .5·1 + .5·(2/3)
    assert r["AP_pt"] == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert r["score_AUC_pt"] == pytest.approx(1.0)                            # 3 dương (kể cả bản trùng), âm thấp nhất
    assert r["count_MAE@0.5"] == 2 and r["count_MAE@0.3"] == 2
    assert r["by_density"]["<=30 vật"]["kept_recall_pt"] == 1.0
    r2 = point_metrics([{"boxes": boxes, "scores": sc, "keep": np.array([3]), "gt": gt}])
    assert r2["oracle_recall_pt"] == 1.0 and r2["recall_pt"] == 0.0 and r2["AP_pt"] == 0.0


def test_point_metrics_vs_box_metrics_through_score():
    """IoU >= 0,5 => tâm box nằm trong GT (tính cả biên) => oracle_recall_pt >= oracle_recall.
    Box trùng hệt GT => AP_pt = AP50 = 1. Đi qua `score()` như eval thật (có mặt ở kết quả)."""
    from ce_localization.alpha.evaluate import score
    rng = np.random.default_rng(0)
    rec, exact = [], []
    for _ in range(4):
        gt = np.c_[rng.uniform(0.2, 0.8, (6, 2)), rng.uniform(0.05, 0.15, (6, 2))]
        boxes = np.r_[gt[:4], np.c_[rng.uniform(0.1, 0.9, (6, 2)), rng.uniform(0.05, 0.2, (6, 2))]]
        rec.append({"boxes": boxes, "scores": rng.uniform(size=10), "keep": np.arange(10), "gt": gt,
                    "gt_size_px": np.full(6, 40.0)})
        g2 = np.c_[np.linspace(0.1, 0.9, 5)[:, None].repeat(2, 1), np.full((5, 2), 0.08)]
        exact.append({"boxes": g2.copy(), "scores": rng.uniform(size=5), "keep": np.arange(5), "gt": g2,
                      "gt_size_px": np.full(5, 40.0)})
    res = score(rec, [0.0] * 6, top_k=100, nms_thr=0.5, oracle=False)
    assert res["oracle_recall_pt"] >= res["oracle_recall"] and "point" in res
    ex = score(exact, [0.0] * 6, top_k=100, nms_thr=0.5, oracle=False)
    assert ex["AP_pt"] == pytest.approx(1.0) and ex["AP50"] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- config

def test_beta0_config_only_differs_from_alpha0_by_beta_keys():
    with open(CFG0) as f:
        c0 = yaml.safe_load(f)
    with open(CFG_B0) as f:
        c = yaml.safe_load(f)
    assert c["data"].pop("targets") == "point" and c["data"].pop("points").endswith("density_points.json")
    assert set(c["data"].pop("pseudo_size")) == {"knn", "beta", "min_frac", "max_frac"}
    assert c["loss"].pop("center_weight") == c0["loss"]["l1_weight"] and "size_weight" in c["loss"]
    c["loss"].pop("size_weight")
    assert c["eval"].pop("select_metric") == "oracle_recall_pt"
    c0["eval"].pop("select_metric")
    for k in ("experiment", "description"):
        c.pop(k), c0.pop(k)
    assert c == c0


# ----------------------------------------------------------------------------- trọn luồng

def _beta_cfg(tmp_path, root, points):
    with open(CFG_B0) as f:
        cfg = yaml.safe_load(f)
    cfg["data"].update(root=root, image_size=128, num_workers=0, points=points)
    cfg["model"]["pretrained_backbone"] = False
    cfg["diffusion"]["num_proposals"] = 20
    cfg["training"].update(max_iter=4, steps=[3], warmup_iters=2, log_every=1, ckpt_every=2, eval_every=2)
    cfg["eval"]["batch_size"] = 2
    p = str(tmp_path / "cfg_beta0.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f)
    return p


def _run_g0(monkeypatch, base, out, report):
    import ce_localization.tools.build_density_points as g0
    monkeypatch.setattr(sys, "argv", ["build_density_points.py", "--ce130", os.path.join(base, "all_phase2_V2"),
                                      "--samples", os.path.join(base, "samples"),
                                      "--density-index", os.path.join(base, "density_index.json"),
                                      "--out", out, "--report", report, "--workers", "0"])
    g0.main()


def test_g0_tool_writes_points_and_report(tmp_path, monkeypatch, capsys):
    base = str(tmp_path)
    _fake_ce130(os.path.join(base, "all_phase2_V2"))
    _fake_density(base)
    out, rep = str(tmp_path / "density_points.json"), str(tmp_path / "rep.json")
    _run_g0(monkeypatch, base, out, rep)
    log = capsys.readouterr().out
    assert "yaml: pseudo_size:" in log
    pt = PointTable(out)
    ds = AlphaCE130(os.path.join(base, "all_phase2_V2"), "train", 128)
    assert all(it["image_id"] in pt for it in ds.items)
    with open(rep) as f:
        r = json.load(f)
    tr = r["chosen"]["train"]["all"]
    # blob giả vẽ tại tâm MỌI box (bản full) -> nhãn gần như đủ và đúng
    assert tr["precision"] > 0.8 and tr["recall"] > 0.6 and set(r["chosen"]) == {"train", "val", "test"}
    assert {"beta", "min_frac", "max_frac"} <= set(r["pseudo_size"])


def test_full_flow_beta0_train_resume_eval_and_gt_never_in_loss(tmp_path, monkeypatch):
    """G0 -> train 2 iter -> --resume tới 4 == train liền 4 ; eval ghi chỉ số điểm. Đổi box GT của
    ảnh train (giữ điểm) -> weight sau train Y HỆT: box GT không vào loss."""
    base = str(tmp_path)
    root = os.path.join(base, "all_phase2_V2")
    _fake_ce130(root)
    _fake_density(base)
    pts = str(tmp_path / "density_points.json")
    _run_g0(monkeypatch, base, pts, str(tmp_path / "rep.json"))
    cfg_path = _beta_cfg(tmp_path, root, pts)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", a, "--max-iter", "2"])
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", a, "--resume"])
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", b])
    ka = torch.load(os.path.join(a, "last.pth"), weights_only=False)
    kb = torch.load(os.path.join(b, "last.pth"), weights_only=False)
    assert ka["iter"] == kb["iter"] == 4
    for k in kb["model"]:
        assert torch.allclose(ka["model"][k].float(), kb["model"][k].float(), atol=1e-5), k
    assert "oracle_recall_pt" in kb["best"] and kb["config"]["data"]["targets"] == "point"

    for br in os.listdir(os.path.join(root, "train")):                        # box GT train khác hẳn
        p = os.path.join(root, "train", br, "annotation.json")
        with open(p) as f:
            ann = json.load(f)
        ann["all_bboxes"] = [[1.0, 1.0, 9.0, 9.0]]
        with open(p, "w") as f:
            json.dump(ann, f)
    c = str(tmp_path / "c")
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", c])
    kc = torch.load(os.path.join(c, "last.pth"), weights_only=False)
    for k in kb["model"]:
        assert torch.equal(kb["model"][k], kc["model"][k]), k

    import ce_localization.eval_alpha as ea
    import ce_localization.train_alpha as ta
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    out = str(tmp_path / "res.json")
    monkeypatch.setattr(sys, "argv", ["eval_alpha.py", "--ckpt", os.path.join(a, "best.pth"), "--split", "test",
                                      "--nms", "--steps", "1", "--batch-size", "2", "--num-workers", "0",
                                      "--out", out, "--device", "cpu"])
    ea.main()
    with open(out) as f:
        res = json.load(f)
    for key in ("steps1", "steps1_nmsfirst"):
        p = res["results"][key]["point"]
        assert 0 <= p["oracle_recall_pt"] <= 1 and 0 <= p["AP_pt"] <= 1
        assert set(p["by_density"]) == {"<=30 vật", "31-100 vật", ">100 vật"}

"""Engine (`engine/`): ghép box khuếch tán + DDIM, SimOTA + loss (chế độ box / điểm), hậu xử lý
và chỉ số (top-k / NMS trước, trần oracle, recall theo độ dày, chỉ số điểm), lịch lr, chia batch.
"""

import numpy as np
import pytest
import torch

from ce_localization.engine import criterion as C
from ce_localization.engine.diffusion import cosine_alphas_cumprod, ddim_sample, prepare_train_boxes
from ce_localization.engine.evaluate import point_metrics
from ce_localization.engine.train_utils import epoch_batches, warmup_multistep
from ce_localization.models.head import apply_deltas
from tests.ce_localization.helpers import CFG_L, CFG_M, _targets


def test_prepare_train_boxes_placeholders_subsample_and_t():
    ac = cosine_alphas_cumprod(1000)
    g = torch.Generator().manual_seed(0)
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0]] * 3)
    gts = [torch.tensor([[10.0, 10.0, 40.0, 30.0]]), torch.rand(50, 2).repeat(1, 2) * 50 + torch.tensor([0, 0, 5, 5.0]),
           torch.zeros(0, 4)]
    boxes, t = prepare_train_boxes(gts, whwh, 20, ac, 2.0, g)
    assert boxes.shape == (3, 20, 4) and t.shape == (3,) and len(set(t.tolist())) == 3   # mỗi ảnh một t
    # DiffusionDet kẹp CXCYWH về [0,1] (không kẹp xyxy): tâm và cỡ nằm trong vùng thật
    cx, cy = (boxes[..., 0] + boxes[..., 2]) / 2, (boxes[..., 1] + boxes[..., 3]) / 2
    w, h = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]
    assert (cx >= 0).all() and (cx <= 128).all() and (w >= 0).all() and (w <= 128 + 1e-4).all()
    assert (cy >= 0).all() and (cy <= 96).all() and (h >= 0).all() and (h <= 96 + 1e-4).all()
    b0, _ = prepare_train_boxes(gts[:1], whwh[:1], 20, ac, 2.0, torch.Generator().manual_seed(0), t=0)
    assert torch.allclose(b0[0, 0], gts[0][0], atol=1.0)               # t=0: nhiễu < 1 px
    # placeholder: randn/6 + 0.5 -> tâm trung bình ~0.5 vùng thật
    bb, _ = prepare_train_boxes([torch.zeros(0, 4)], whwh[:1], 4000, ac, 2.0,
                                torch.Generator().manual_seed(1), t=0)
    cx = (bb[0, :, 0] + bb[0, :, 2]) / 2 / 128
    assert abs(cx[1:].mean().item() - 0.5) < 0.02 and abs(cx[1:].std().item() - 1 / 6) < 0.02


def test_ddim_sample_shapes_and_renewal():
    ac = cosine_alphas_cumprod(1000)
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0]])

    def head_fn(boxes, t):
        S, B, N = 3, boxes.shape[0], boxes.shape[1]
        return torch.zeros(S, B, N, 1), boxes[None].repeat(S, 1, 1, 1)

    o1 = ddim_sample(head_fn, 1, 16, whwh, ac, steps=1)
    assert o1["boxes"].shape == (1, 16, 4) and o1["stage_boxes"].shape == (3, 1, 16, 4)
    o4 = ddim_sample(head_fn, 1, 16, whwh, ac, steps=4)                # score 0.5 -> renewal bỏ hết
    assert o4["boxes"].shape == (1, 64, 4) and o4["scores"].shape == (1, 64)   # 4 bước đều vào ensemble
    with pytest.raises(ValueError):
        ddim_sample(head_fn, 2, 16, whwh.repeat(2, 1), ac, steps=4)


def test_simota_matches_obvious_queries_and_dedupes():
    gt = torch.tensor([[10.0, 10.0, 30.0, 30.0], [60.0, 60.0, 90.0, 90.0]])
    pred = torch.tensor([[[10.0, 10.0, 30.0, 30.0], [11.0, 11.0, 31.0, 31.0], [60.0, 60.0, 90.0, 90.0],
                          [0.0, 0.0, 5.0, 5.0], [40.0, 40.0, 50.0, 50.0], [59.0, 61.0, 89.0, 91.0]]])
    logits = torch.zeros(1, 6, 1)
    (sel, gi), = C.match(logits, pred, _targets(gt))
    matched = dict(zip(torch.nonzero(sel).flatten().tolist(), gi.tolist()))
    assert matched[0] == 0 and matched[2] == 1                       # box trùng GT được ghép đúng GT
    assert 3 not in matched and 4 not in matched                     # ngoài vùng tâm -> không ghép
    assert len(gi) == int(sel.sum())                                 # mỗi query tối đa 1 GT


def test_simota_terminates_when_more_gt_than_queries():
    """Ảnh CE-130 có tới 501 box, N=200: vòng cứu phải DỪNG (hành vi gốc)."""
    g = torch.Generator().manual_seed(0)
    xy = torch.rand(12, 2, generator=g) * 80
    gt = torch.cat([xy, xy + 10], 1)
    pred = gt[:5].clone()[None]
    (sel, gi), = C.match(torch.zeros(1, 5, 1), pred, _targets(gt))
    assert int(sel.sum()) == len(gi) <= 5


def test_simota_cap_does_not_change_result_when_enough_queries(monkeypatch):
    """Số GT <= số query: vòng cứu tự dừng trước trần -> kết quả Y HỆT khi gần như không có trần
    (= bản gốc). Lưu ý: GT có thể vắng trong `gi` — hai GT cùng chọn một query không thuộc
    `anchor_matching_gt` cũ thì query giữ cả hai, `max(1)` chỉ trả một (hành vi gốc)."""
    orig = C.dynamic_k_matching
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        xy = torch.rand(15, 2, generator=g) * 80
        gt = torch.cat([xy, xy + 5 + torch.rand(15, 2, generator=g) * 15], 1)
        q = torch.rand(40, 2, generator=g) * 90
        pred = torch.cat([q, q + 3 + torch.rand(40, 2, generator=g) * 20], 1)[None]
        logits = torch.randn(1, 40, 1, generator=g)
        (s1, g1), = C.match(logits, pred, _targets(gt))
        monkeypatch.setattr(C, "dynamic_k_matching", lambda c, i, n, k=5: orig(c, i, n, k, 10 ** 5))
        (s2, g2), = C.match(logits, pred, _targets(gt))
        monkeypatch.setattr(C, "dynamic_k_matching", orig)
        assert torch.equal(s1, s2) and torch.equal(g1, g2), seed


def test_simota_empty_gt():
    (sel, gi), = C.match(torch.zeros(1, 4, 1), torch.rand(1, 4, 4) * 10,
                         C.build_targets([torch.zeros(0, 4)], torch.tensor([[100.0] * 4])))
    assert sel.sum() == 0 and len(gi) == 0


def test_criterion_normalises_by_matched_queries_and_sums_stages():
    cfg_l = {"alpha": 0.25, "gamma": 2.0, "class_weight": 2.0, "l1_weight": 5.0, "giou_weight": 2.0}
    crit = C.Criterion(cfg_l, {"ota_k": 5, "center_radius": 2.5})
    gt = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
    pred = torch.tensor([[[10.0, 10.0, 30.0, 30.0], [0.0, 0.0, 4.0, 4.0], [80, 80, 99, 99.0]]])
    logits = torch.tensor([[[0.3], [-1.0], [0.5]]])
    tg = _targets(gt)
    loss, st = crit.loss_one(logits, pred, tg)
    (sel, gi), = C.match(logits, pred, tg)
    n = int(sel.sum())
    tcls = sel.float()[None, :, None]
    from ce_localization.engine.criterion import sigmoid_focal_loss
    ce = sigmoid_focal_loss(logits.flatten(0, 1), tcls.flatten(0, 1)).sum() / n
    assert torch.allclose(st["loss_ce"], ce) and st["n_matched"] == n
    assert abs(float(st["loss_bbox"])) < 1e-6 and abs(float(st["loss_giou"])) < 1e-6
    tot, st6 = crit(torch.stack([logits] * 6), torch.stack([pred] * 6), tg)
    assert torch.allclose(tot, 6 * loss) and len(st6["loss_per_stage"]) == 6


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
    crit = C.Criterion(CFG_L, CFG_M, mode="point")
    loss, st = crit.loss_one(torch.full((1, 1, 1), 2.0), boxes, tg)
    assert st["n_matched"] == 1
    assert float(st["loss_center"]) == pytest.approx(4 / 100 + 1 / 80, rel=1e-5)
    assert float(st["loss_size"]) == pytest.approx(abs(np.log(24 / 20)) + abs(np.log(16 / 20)), rel=1e-5)
    assert float(st["center_px"]) == pytest.approx(np.hypot(4, 1), rel=1e-5)
    for w_c, w_s, zero in ((1.0, 0.0, [2, 3]), (0.0, 1.0, [0, 1])):          # tâm chỉ vào dx,dy; cỡ chỉ dw,dh
        c = C.Criterion({**CFG_L, "center_weight": w_c, "size_weight": w_s}, CFG_M, mode="point")
        d = torch.zeros(1, 4, requires_grad=True)
        c.loss_one(torch.full((1, 1, 1), 2.0), apply_deltas(d, base)[None], tg)[0].backward()
        assert torch.all(d.grad[0, zero] == 0) and torch.any(d.grad[0, [i for i in range(4) if i not in zero]] != 0)
    assert crit.log_keys == ("loss_ce", "loss_center", "loss_size", "center_px")
    tot, st6 = crit(torch.full((6, 1, 1, 1), 2.0), torch.stack([boxes] * 6), tg)
    assert torch.allclose(tot, 6 * loss) and "loss_center_final" in st6


def test_point_loss_empty_targets_and_box_mode_keys_unchanged():
    crit = C.Criterion(CFG_L, CFG_M, mode="point")
    tg = C.build_targets([torch.zeros(0, 4)], torch.tensor([[100.0] * 4]))
    boxes = torch.rand(1, 5, 4, requires_grad=True)
    loss, st = crit.loss_one(torch.zeros(1, 5, 1), boxes * 10 + torch.tensor([0, 0, 20, 20.0]), tg)
    assert st["n_matched"] == 0 and torch.isfinite(loss)
    _, sb = C.Criterion(CFG_L, CFG_M)(torch.zeros(6, 1, 3, 1), torch.rand(6, 1, 3, 4) * 10 + 5,
                                           _targets(torch.tensor([[5.0, 5.0, 15.0, 15.0]])))
    assert {k for k in sb if k.endswith("_final")} == {"loss_ce_final", "loss_bbox_final", "loss_giou_final",
                                                        "n_matched_final", "iou_matched_final"}
    with pytest.raises(ValueError):
        C.Criterion(CFG_L, CFG_M, mode="points")


def test_nms_first_keeps_distinct_boxes_topk_first_keeps_duplicates():
    """Ca của matcher một-nhiều: 3 bản trùng điểm cao + 2 vật khác điểm thấp hơn, top-k 2.
    top-k trước rồi NMS -> 1 box ; NMS trước rồi top-k -> 2 vật khác nhau (như DiffusionDet)."""
    from ce_localization.engine.evaluate import POSTPROCESS
    b = np.array([[0.2, 0.2, 0.1, 0.1], [0.201, 0.2, 0.1, 0.1], [0.2, 0.201, 0.1, 0.1],
                  [0.7, 0.7, 0.1, 0.1], [0.5, 0.2, 0.1, 0.1]])
    sc = np.array([0.9, 0.89, 0.88, 0.5, 0.4])
    assert len(POSTPROCESS["topk_first"](b, sc, 2, 0.5)) == 1
    assert POSTPROCESS["nms_first"](b, sc, 2, 0.5).tolist() == [0, 3]


def test_oracle_records_score_is_best_iou_and_boxes_unchanged():
    """Trần oracle: score = IoU lớn nhất với GT tính tay, box giữ nguyên từng bit, keep = hậu xử lý
    cùng thứ tự trên score mới; ảnh không GT -> score 0."""
    from ce_localization.engine.evaluate import POSTPROCESS, oracle_score_records
    rng = np.random.default_rng(0)
    rec = [{"boxes": np.c_[rng.uniform(0.1, 0.9, (30, 2)), rng.uniform(0.02, 0.2, (30, 2))],
            "scores": rng.uniform(size=30), "keep": np.arange(5),
            "gt": np.c_[rng.uniform(0.1, 0.9, (8, 2)), rng.uniform(0.02, 0.2, (8, 2))]} for _ in range(3)]
    rec.append({**rec[0], "gt": np.zeros((0, 4))})

    def iou_1(a, g):                                      # cxcywh, một cặp, tính tay
        ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
        gx1, gy1, gx2, gy2 = g[0] - g[2] / 2, g[1] - g[3] / 2, g[0] + g[2] / 2, g[1] + g[3] / 2
        iw, ih = max(0, min(ax2, gx2) - max(ax1, gx1)), max(0, min(ay2, gy2) - max(ay1, gy1))
        inter = iw * ih
        return inter / (a[2] * a[3] + g[2] * g[3] - inter)
    for order in ("topk_first", "nms_first"):
        for r, o in zip(rec, oracle_score_records(rec, 10, 0.5, order)):
            want = np.array([max((iou_1(b, g) for g in r["gt"]), default=0.0) for b in r["boxes"]])
            assert np.allclose(o["scores"], want) and np.array_equal(o["boxes"], r["boxes"])
            assert np.array_equal(o["keep"], POSTPROCESS[order](r["boxes"], o["scores"], 10, 0.5))


def test_density_recall_bins():
    from ce_localization.engine.evaluate import density_recall
    gt = lambda n: np.tile([[0.5, 0.5, 0.1, 0.1]], (n, 1))              # noqa: E731
    rec = [{"boxes": gt(1), "scores": np.ones(1), "keep": np.array([0]), "gt": gt(5)},
           {"boxes": gt(1), "scores": np.ones(1), "keep": np.array([], dtype=int), "gt": gt(40)}]
    d = density_recall(rec)
    assert d["<=30 vật"] == {"oracle_recall": 1.0, "kept_recall": 1.0, "n_gt": 5, "n_img": 1}
    assert d["31-100 vật"]["oracle_recall"] == 1.0 and d["31-100 vật"]["kept_recall"] == 0.0
    assert d[">100 vật"]["n_img"] == 0


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
    from ce_localization.engine.evaluate import score
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


def test_warmup_multistep_matches_detectron2():
    assert warmup_multistep(0, [9000, 11000]) == pytest.approx(0.01)
    assert warmup_multistep(500, [9000, 11000]) == pytest.approx(0.01 * 0.5 + 0.5)
    assert warmup_multistep(1000, [9000, 11000]) == 1.0
    assert warmup_multistep(9000, [9000, 11000]) == pytest.approx(0.1)
    assert warmup_multistep(11999, [9000, 11000]) == pytest.approx(0.01)


def test_epoch_batches_disjoint_across_ranks_and_reproducible():
    a = epoch_batches(11, 1, 0, 2, 0, 3)
    b = epoch_batches(11, 1, 1, 2, 0, 3)
    assert len(a) == len(b) == 5
    assert not set(sum(a, [])) & set(sum(b, []))
    assert a == epoch_batches(11, 1, 0, 2, 0, 3) and a != epoch_batches(11, 1, 0, 2, 0, 4)


# ----------------------------------------------------------------------------- GAMMA (bài add)

def test_linear_schedule_matches_celoc_original():
    from ce_localization.utils.diffusion_math import linear_alphas_cumprod
    ref = torch.cumprod(1.0 - torch.linspace(0.0001, 0.02, 1000), dim=0)      # diffusion_module.py của bài
    assert torch.equal(linear_alphas_cumprod(1000), ref)


def test_ddpm_sample_recovers_x0_with_oracle_eps():
    from ce_localization.models.box_policy import ddpm_sample
    from ce_localization.utils.diffusion_math import linear_alphas_cumprod
    ac = linear_alphas_cumprod(50, 1e-3, 0.2)
    x0 = torch.tensor([[0.3, -0.2, 0.5, -0.7]]).repeat(3, 1)
    eps_fn = lambda x, t: (x - ac[t][:, None].sqrt() * x0) / (1 - ac[t][:, None]).sqrt()  # noqa: E731
    out = ddpm_sample(eps_fn, 3, ac, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(out, x0, atol=1e-4)


def _rec(boxes, holes, objects, t=None, wh=(100.0, 100.0)):
    return {"image_id": "x", "t": t or len(holes), "wh": np.asarray(wh), "boxes": np.asarray(boxes, float),
            "holes": np.asarray(holes, float).reshape(-1, 4), "objects": np.asarray(objects, float).reshape(-1, 4)}


def test_add_metrics_hand_computed():
    from ce_localization.engine.add_eval import add_metrics
    holes = [[0, 0, 10, 10], [50, 50, 60, 60]]                       # lỗ cũ, lỗ MỚI NHẤT
    boxes = [[0, 0, 10, 10],                                          # trùng lỗ cũ: IoU any 1, latest 0
             [50, 50, 60, 65],                                        # IoU latest 100/150
             [80, 80, 70, 90],                                        # suy biến (w < 0) -> IoU 0
             [95, 95, 105, 105]]                                      # tràn ra ngoài ảnh
    objects = [[20, 20, 40, 40], [94, 94, 104, 104]]
    r = add_metrics([_rec(boxes, holes, objects)])
    assert r["best_iou@4_any"] == 1.0 and np.isclose(r["best_iou@4_latest"], 100 / 150)
    assert r["hit50@4_any"] == r["hit50@4_latest"] == 1.0
    assert np.isclose(r["mean_iou_any"], (1 + 100 / 150) / 4) and np.isclose(r["mean_iou_latest"], (100 / 150) / 4)
    assert r["box_hit50_any"] == 0.5 and r["box_hit50_latest"] == 0.25
    assert r["hole_cover"] == 1.0 and r["n_multi_hole"] == 1 and r["degenerate"] == 0.25
    assert r["in_image"] == 0.5                                       # box 0, 1 nằm trọn; 2 suy biến; 3 tràn
    assert r["on_object"] == 0.25                                     # box 3: giao 9×9 / 100 >= 0,5
    assert r["n_cnll"] == 0 and np.isnan(r["cnll_F1_n1_mean"])        # < 5 vật -> bỏ C-NLL
    assert r["by_turn"][2]["n"] == 1 and r["by_turn"][2]["best_any"] == 1.0
    r2 = add_metrics([_rec(boxes, holes, objects)], with_holes=False)
    assert "mean_iou_any" not in r2 and r2["on_object"] == 0.25


def test_cnll_matches_gaussian_definition():
    """C-NLL F2 = −log N(x | μ, Σ + ridge) − min_i(−log N(z_i)) với z = [cx, cy, w, h] / (nw, nh) của vật có sẵn;
    F1 tính tay cho một box."""
    from scipy.stats import multivariate_normal
    from ce_localization.engine.add_eval import CNLL_RIDGE, cnll
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 80, (8, 2))
    wh = rng.uniform(5, 15, (8, 2))
    objs = np.concatenate([xy, xy + wh], 1)
    box = np.array([[30.0, 40.0, 42.0, 50.0]])
    W, H = 100.0, 80.0
    z = np.stack([(objs[:, 0] + objs[:, 2]) / 2 / W, (objs[:, 1] + objs[:, 3]) / 2 / H,
                  (objs[:, 2] - objs[:, 0]) / W, (objs[:, 3] - objs[:, 1]) / H], 1)
    mvn = multivariate_normal(z.mean(0), np.cov(z.T, bias=True) + CNLL_RIDGE * np.eye(4))
    x = np.array([[36 / W, 45 / H, 12 / W, 10 / H]])
    exp = -mvn.logpdf(x) - (-mvn.logpdf(z)).min()
    assert np.allclose(cnll(box, objs, (W, H), "F2"), exp)
    c1 = cnll(np.concatenate([box, objs[:1]]), objs, (W, H), "F1")
    assert c1.shape == (2,) and np.isfinite(c1).all() and c1[1] >= 0     # vật có sẵn: >= vật "điển hình nhất"
    assert cnll(box, objs[:4], (W, H)) is None


def test_prior_records_map_unit_boxes_to_valid_region():
    from ce_localization.engine.add_eval import prior_records
    rec = _rec([[0, 0, 1, 1]] * 2, [[0, 0, 1, 1]], [], wh=(200.0, 100.0))
    pri = prior_records([rec], np.array([[0.5, 0.5, 0.1, 0.2]]), n_samples=3)
    assert pri[0]["boxes"].shape == (3, 4) and np.allclose(pri[0]["boxes"], [[90, 40, 110, 60]] * 3)

"""Hình học box, toán khuếch tán, matcher — bản numpy tham chiếu, mỗi test có đáp án
giải tích hoặc một ĐỐI CHỨNG ÂM (cài công thức sai để chứng minh bộ test bắt được lỗi)."""

import os
import sys

import numpy as np
import pytest

from ce_localization.utils.box_ops_np import (
    box_iou, canvas_to_pixel, compute_scale, cxcywh_to_xyxy, decode_diffusion,
    encode_diffusion, filter_degenerate, flip_horizontal, generalized_box_iou,
    scale_to_canvas, xyxy_to_cxcywh,
)
from ce_localization.utils.box_ops_np import decode_diffusion, encode_diffusion
from ce_localization.utils.diffusion_np import (
    cosine_alphas_cumprod, ddim_step, ddim_time_pairs, linear_alphas_cumprod,
    make_placeholders, prepare_diffusion_concat, predict_noise_from_start, q_sample,
)
from ce_localization.utils.matcher_np import (
    _center_prior_mask, build_cost, hungarian_match, match, simota_match,
)

PROJECT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ce_localization")

# ============================================================================
# từ test_box_ops.py
# ============================================================================

def test_iou_with_itself_is_one():
    """The most basic invariant. Round 1 produced GIoU 5e5 from inverted boxes."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        x1, y1 = rng.uniform(0, 400, 2)
        w, h = rng.uniform(1, 100, 2)
        b = np.array([[x1, y1, x1 + w, y1 + h]])
        assert box_iou(b, b)[0][0, 0] == pytest.approx(1.0, abs=1e-12)
        assert generalized_box_iou(b, b)[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_iou_against_known_analytic_value():
    """A 5% box fully inside a 25% box -> IoU = (0.05/0.25)^2 = 0.0400."""
    big = np.array([[0.0, 0.0, 0.25 * 512, 0.25 * 512]])
    small = np.array([[0.0, 0.0, 0.05 * 512, 0.05 * 512]])
    assert box_iou(big, small)[0][0, 0] == pytest.approx(0.04, abs=1e-12)


def test_giou_in_range_and_has_gradient_when_disjoint():
    """GIoU in [-1,1]; for DISJOINT boxes IoU=0 but GIoU still has a gradient.

    Important for CE-130: the median object is only 0.41 % of image area, so most
    prediction-GT pairs early in training are disjoint -> L1 alone is nearly blind.
    """
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    far = np.array([[100.0, 100.0, 110.0, 110.0]])
    assert box_iou(a, far)[0][0, 0] == 0.0
    g0 = generalized_box_iou(a, far)[0, 0]
    assert -1.0 <= g0 < 0.0
    # move closer -> GIoU must INCREASE (i.e. it has a gradient w.r.t. distance)
    near = np.array([[50.0, 50.0, 60.0, 60.0]])
    assert generalized_box_iou(a, near)[0, 0] > g0


def test_six_step_round_trip_on_real_boxes():
    """Round-trip through the WHOLE chain: xyxy px -> canvas -> encode -> decode -> px."""
    rng = np.random.default_rng(0)
    for W, H in [(582, 384), (1918, 384), (384, 384), (469, 384)]:
        px = []
        for _ in range(300):
            x1, y1 = rng.uniform(0, W - 60), rng.uniform(0, H - 60)
            w, h = rng.uniform(5, 50), rng.uniform(5, 50)
            px.append([x1, y1, x1 + w, y1 + h])
        px = np.array(px)
        n, _, _ = scale_to_canvas(px, W, H)
        back = decode_diffusion(encode_diffusion(n, 2.0), 2.0)
        out = canvas_to_pixel(back, W, H)
        assert np.abs(out - px).max() < 1e-9, f"({W},{H}) error too large"


def test_NEGATIVE_CONTROL_decoding_divided_twice():
    """[NEGATIVE CONTROL] Round-1 bug: extent = (norm+1)/4 instead of (norm+1)/2.

    Boxes end up half-sized -> IoU between right and wrong is only 0.25. This test
    proves the suite catches the exact bug that killed round 1.
    """
    b_px = np.array([[100.0, 100.0, 160.0, 150.0]])
    n, _, _ = scale_to_canvas(b_px, 512, 512)
    x = encode_diffusion(n, 2.0)

    correct = decode_diffusion(x, 2.0)
    wrong = correct.copy()
    wrong[:, 2:] /= 2.0  # divide one extra time

    iou = box_iou(cxcywh_to_xyxy(correct) * 512, cxcywh_to_xyxy(wrong) * 512)[0][0, 0]
    assert iou == pytest.approx(0.25, abs=1e-9), f"IoU={iou}, expected 0.25"


def test_encode_decode_are_inverse_for_every_snr():
    """decode(encode(x)) == x, and snr_scale does NOT leak outside."""
    rng = np.random.default_rng(1)
    n = np.column_stack([
        rng.uniform(0.05, 0.95, 500), rng.uniform(0.05, 0.95, 500),
        rng.uniform(0.01, 0.40, 500), rng.uniform(0.01, 0.40, 500),
    ])
    for snr in [1.0, 2.0, 3.0]:
        assert np.abs(decode_diffusion(encode_diffusion(n, snr), snr) - n).max() < 1e-12


def test_negative_norm_w_is_normal():
    """Small objects give NEGATIVE w after encoding — as in DiffusionDet, NOT a bug.

    Measured on real data: 99.8 % of CE-130 boxes have norm_w < 0. Correcting
    round 1: this is not a CE-Loc-specific anomaly.
    """
    w_typical = 0.0686  # CE-130 median
    assert encode_diffusion(np.array([[0.5, 0.5, w_typical, w_typical]]), 2.0)[0, 2] < 0


def test_flip_is_an_involution():
    rng = np.random.default_rng(2)
    n = np.column_stack([
        rng.uniform(0.1, 0.9, 200), rng.uniform(0.1, 0.9, 200),
        rng.uniform(0.02, 0.3, 200), rng.uniform(0.02, 0.3, 200),
    ])
    assert np.abs(flip_horizontal(flip_horizontal(n)) - n).max() < 1e-15
    f = flip_horizontal(n)
    assert np.abs(f[:, 1:] - n[:, 1:]).max() == 0.0  # only cx changes


def test_filter_degenerate_boxes():
    b = np.array([
        [10.0, 10.0, 50.0, 50.0],   # good
        [10.0, 10.0, 10.0, 50.0],   # w = 0
        [10.0, 10.0, 50.0, 5.0],    # h < 0
    ])
    clean, keep = filter_degenerate(b)
    assert clean.shape[0] == 1 and keep.tolist() == [True, False, False]


def test_padding_is_always_at_the_bottom():
    """Every CE-130 image is exactly 384px tall -> W >= H -> new_w == 512, padding
    at the BOTTOM.

    If this invariant fails, bounding placeholder cy with a SINGLE scalar threshold
    collapses (a 2D mask would be required).
    """
    for W in [384, 469, 582, 1918]:
        H = 384
        s = compute_scale(W, H, 512)
        assert int(W * s) == 512, f"W={W}: new_w != 512"
        _, _, valid_h = scale_to_canvas(np.array([[0.0, 0.0, 10.0, 10.0]]), W, H)
        assert 0.0 < valid_h <= 1.0


def test_xyxy_cxcywh_are_inverse():
    rng = np.random.default_rng(3)
    b = rng.uniform(0, 400, (500, 4))
    b[:, 2:] += b[:, :2] + 1.0
    assert np.abs(cxcywh_to_xyxy(xyxy_to_cxcywh(b)) - b).max() < 1e-10



# ============================================================================
# từ test_diffusion.py
# ============================================================================

AB = cosine_alphas_cumprod(1000)


SNR = 2.0


def test_cosine_matches_measured_values():
    """The sqrt(alpha_bar) table must match the measured numbers — catches an
    accidental switch to the linear schedule."""
    got = np.sqrt(AB[[249, 499, 749]])
    assert got[0] == pytest.approx(0.92, abs=0.01)
    assert got[1] == pytest.approx(0.70, abs=0.01)
    assert got[2] == pytest.approx(0.38, abs=0.01)


def test_NEGATIVE_CONTROL_linear_retains_far_less_signal():
    """[NEGATIVE CONTROL] linear leaves only 5.8 % signal at t=749 (cosine: 38 %).

    This is why cosine gives 3.70x the AP: at large t, linear makes boxes almost
    pure noise, so the matcher assigns arbitrarily and gradients are 4.6x noisier.
    """
    lin = np.sqrt(linear_alphas_cumprod(1000)[[249, 499, 749]])
    assert lin[2] == pytest.approx(0.058, abs=0.005)
    assert np.sqrt(AB[749]) > 6 * lin[2]


def test_amplification_factor_at_large_t():
    """1/sqrt(alpha_bar) is enormous at the end of the range -> clamping x_start
    is MANDATORY."""
    amp = 1.0 / np.sqrt(AB)
    assert amp[999] > 1000
    assert (amp > 10).mean() > 0.05


def test_q_sample_at_t0_preserves_x_start():
    x0 = encode_diffusion(np.array([[0.3, 0.4, 0.08, 0.07]]), SNR)
    noise = np.random.default_rng(0).standard_normal((1, 4))
    assert np.abs(q_sample(x0, 0, noise, AB) - x0).max() < 0.05


def test_predict_noise_is_the_inverse_of_q_sample():
    rng = np.random.default_rng(0)
    x0 = encode_diffusion(np.array([[0.3, 0.4, 0.08, 0.07]]), SNR)
    noise = rng.standard_normal((1, 4))
    for t in [10, 250, 500, 900]:
        xt = q_sample(x0, t, noise, AB)
        assert np.abs(predict_noise_from_start(xt, t, x0, AB) - noise).max() < 1e-9


def test_x_T_has_std_one_not_snr():
    """[NEGATIVE CONTROL for bug 3] x_T ~ N(0,I), NOT scaled by snr_scale.

    Round 1 initialised x_T with std 2.0 — wrong, because at t=T-1 alpha_bar ~ 0,
    so x_T must be pure standard noise.
    """
    x_T = np.random.default_rng(0).standard_normal((5000, 4))
    assert x_T.std() == pytest.approx(1.0, abs=0.05)
    assert (x_T * SNR).std() == pytest.approx(2.0, abs=0.1)  # the wrong version


def test_ddim_converges_under_an_oracle():
    """Given an oracle returning the true x_start, DDIM must land on that x_start.

    NOTE: this test ALONE is not enough (round 1 had it and was still wrong) — it
    must be paired with the negative control below.
    """
    target = encode_diffusion(np.array([[0.3, 0.4, 0.08, 0.07]]), SNR)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 4))
    for t, tn in ddim_time_pairs(1000, 4):
        x, _ = ddim_step(x, target, t, tn, AB, SNR, eta=0.0)
    assert np.abs(x - target).max() < 1e-9


def test_NEGATIVE_CONTROL_no_clamp_and_no_recomputed_pred_noise():
    """[NEGATIVE CONTROL for bugs 1+2] — a test with real DETECTION POWER.

    An untrained model often returns x_start outside the valid range. Without
    clamping and then RECOMPUTING pred_noise from the clamped version, pred_noise
    is off by hundreds of units and the next DDIM step explodes.

    (The "converges under an oracle" test above does NOT catch this, because the
    oracle always returns valid values — exactly the trap round 1 fell into.)
    """
    x_bad = np.array([[8.0, -6.0, 300.0, 0.5]])       # out of range
    x_clamped = encode_diffusion(decode_diffusion(x_bad, SNR), SNR)
    assert np.abs(x_clamped).max() <= SNR + 1e-12

    xt = np.array([[0.1, 0.2, -0.3, 0.4]])
    pn_wrong = predict_noise_from_start(xt, 500, x_bad, AB)
    pn_right = predict_noise_from_start(xt, 500, x_clamped, AB)
    assert np.abs(pn_wrong - pn_right).max() > 100, "the negative control does not discriminate"

    # ddim_step (the correct version) must give finite values, not explode
    x_next, x_start_used = ddim_step(xt, x_bad, 500, 490, AB, SNR, eta=0.0)
    assert np.abs(x_start_used).max() <= SNR + 1e-12
    assert np.abs(x_next).max() < 10


def test_ddim_time_pairs_end_at_minus_one():
    pairs = ddim_time_pairs(1000, 4)
    assert len(pairs) == 4
    assert pairs[0][0] == 999 and pairs[-1][1] == -1


def test_placeholders_never_land_in_the_padding():
    """[FIX 1b] Measured 13.7 % of placeholders landing in padding; after bounding
    it must be 0 %."""
    rng = np.random.default_rng(0)
    for valid_h in [1.0, 0.701, 0.265]:      # padding 0 %, 29.9 %, 73.5 %
        ph = make_placeholders(50000, (0.07, 0.06), valid_h, rng)
        assert (ph[:, 1] > valid_h).sum() == 0, f"valid_h={valid_h} still has boxes in padding"

    # the ORIGINAL (unbounded) version does land in padding — proving the test
    # discriminates
    original = rng.standard_normal(50000) / 6.0 + 0.5
    assert (original > 0.701).mean() > 0.05


def test_placeholder_size_follows_the_data():
    """[FIX 1] The original gives w ~ 0.5 (7.3x larger than CE-130's median 0.0686)."""
    rng = np.random.default_rng(0)
    original = make_placeholders(20000, None, 1.0, rng)
    assert np.median(original[:, 2]) == pytest.approx(0.5, abs=0.02)
    assert np.median(original[:, 2]) / 0.0686 > 7.0

    fixed = make_placeholders(20000, (0.0686, 0.0609), 1.0, rng)
    ratio = np.median(fixed[:, 2]) / 0.0686
    assert 0.7 < ratio < 1.3, f"ratio {ratio} — should be around 1x the real objects"


def test_LIMIT_of_the_placeholder_fix_at_large_t():
    """Records finding (a): the placeholder fix ONLY helps at small t.

    The model sees x_t = q_sample(x_start), not x_start. Clamping pulls everything
    toward a median of 0.5 at large t — both GT and placeholders become "half-image
    boxes". This test locks that fact in so nobody over-expects from Fix 1.
    """
    rng = np.random.default_rng(0)
    x0 = encode_diffusion(np.full((20000, 4), 0.0686), SNR)
    med = {}
    for t in [0, 300, 700, 999]:
        xt = q_sample(x0, t, rng.standard_normal(x0.shape), AB)
        med[t] = np.median(decode_diffusion(xt, SNR)[:, 2])
    assert med[0] == pytest.approx(0.0686, abs=0.01)
    assert med[300] < 0.2
    assert med[999] == pytest.approx(0.5, abs=0.05), "at large t it must approach 0.5"
    assert med[0] < med[300] < med[700] < med[999]


def test_prepare_diffusion_concat_pad_crop_and_mask():
    rng = np.random.default_rng(0)
    N = 100
    gt = np.column_stack([
        rng.uniform(0.1, 0.9, 30), rng.uniform(0.1, 0.6, 30),
        rng.uniform(0.03, 0.12, 30), rng.uniform(0.03, 0.12, 30),
    ])
    # M < N -> pad
    x_t, noise, is_gt = prepare_diffusion_concat(gt, N, 500, AB, SNR, valid_h=0.7, rng=rng)
    assert x_t.shape == (N, 4) and noise.shape == (N, 4)
    assert is_gt.sum() == 30 and is_gt[:30].all()
    assert np.abs(x_t).max() <= SNR + 1e-12       # clamped

    # M > N -> crop, every slot is real GT
    gt_big = np.repeat(gt, 10, axis=0)            # 300 boxes
    _, _, is_gt2 = prepare_diffusion_concat(gt_big, N, 500, AB, SNR, rng=rng)
    assert is_gt2.all()

    # M == 0 -> no GT at all
    _, _, is_gt3 = prepare_diffusion_concat(np.zeros((0, 4)), N, 500, AB, SNR, rng=rng)
    assert not is_gt3.any()


def test_t_is_one_value_for_the_whole_image():
    """The N boxes are ONE sample from a distribution over box sets -> one shared t.

    With a per-box t the process could not be reproduced at inference (all boxes
    travel from T down to 0 together). DiffusionDet: torch.randint(..., (1,)).
    """
    rng = np.random.default_rng(0)
    gt = np.array([[0.5, 0.5, 0.1, 0.1]])
    x_a, _, _ = prepare_diffusion_concat(gt, 8, 500, AB, SNR, rng=np.random.default_rng(7))
    x_b, _, _ = prepare_diffusion_concat(gt, 8, 500, AB, SNR, rng=np.random.default_rng(7))
    assert np.abs(x_a - x_b).max() == 0.0        # same seed + same t -> identical



# ============================================================================
# từ test_matcher.py
# ============================================================================

def _data(n_pred=100, n_gt=30, seed=0):
    rng = np.random.default_rng(seed)
    gt = np.column_stack([
        rng.uniform(0.1, 0.9, n_gt), rng.uniform(0.1, 0.6, n_gt),
        rng.uniform(0.03, 0.12, n_gt), rng.uniform(0.03, 0.12, n_gt),
    ])
    pred = np.column_stack([
        rng.uniform(0.0, 1.0, n_pred), rng.uniform(0.0, 1.0, n_pred),
        rng.uniform(0.02, 0.30, n_pred), rng.uniform(0.02, 0.30, n_pred),
    ])
    return pred, gt, rng.uniform(0, 1, n_pred)


@pytest.mark.parametrize("method", ["hungarian", "simota"])
def test_one_proposal_matches_at_most_one_GT(method):
    """'1-to-k' means one GT receives k proposals, NOT the reverse."""
    pred, gt, sc = _data()
    pi, _ = match(pred, gt, sc, method=method)
    _, counts = np.unique(pi, return_counts=True)
    assert (counts > 1).sum() == 0


@pytest.mark.parametrize("method", ["hungarian", "simota"])
def test_every_GT_gets_matched(method):
    """No GT is left behind — SimOTA has the rescue loop (loss.py:428-438)."""
    pred, gt, sc = _data()
    _, gi = match(pred, gt, sc, method=method)
    assert len(set(gi.tolist())) == gt.shape[0]


@pytest.mark.parametrize("method", ["hungarian", "simota"])
def test_empty_gt_does_not_crash(method):
    pred, _, sc = _data()
    pi, gi = match(pred, np.zeros((0, 4)), sc, method=method)
    assert len(pi) == 0 and len(gi) == 0


def test_hungarian_on_a_hand_worked_example():
    """3 predictions exactly matching 3 GT but permuted -> must recover the permutation."""
    gt = np.array([[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]])
    pred = gt[[2, 0, 1]]
    pi, gi = hungarian_match(pred, gt)
    order = gi[np.argsort(pi)]
    assert order.tolist() == [2, 0, 1]


def test_dynamic_k_grows_with_model_quality():
    """k is proportional to IoU -> the worse the model, the smaller k. dynamic-k is
    NOT a bootstrap.

    This is the evidence that SimOTA degenerates into Hungarian early in training.
    """
    gt = np.array([[0.5, 0.5, 0.2, 0.2]])
    ks = {}
    for target_iou, name in [(0.10, "untrained"), (0.60, "average"), (1.00, "good")]:
        # build predictions with approximately the target IoU by scaling around GT
        scale = target_iou ** 0.5
        pred = np.tile(np.array([[0.5, 0.5, 0.2 * scale, 0.2 * scale]]), (100, 1))
        pi, _ = simota_match(pred, gt)
        ks[name] = len(pi)
    assert ks["untrained"] <= ks["average"] <= ks["good"]
    assert ks["untrained"] <= 2, "an untrained model must give k ~1 (near Hungarian)"


def test_simota_nearly_matches_hungarian_for_an_untrained_model():
    """When every IoU is low, SimOTA gives k=1, so the pair count approaches Hungarian."""
    rng = np.random.default_rng(5)
    gt = np.column_stack([
        rng.uniform(0.1, 0.9, 20), rng.uniform(0.1, 0.6, 20),
        np.full(20, 0.05), np.full(20, 0.05),
    ])
    pred = np.column_stack([
        rng.uniform(0, 1, 200), rng.uniform(0, 1, 200),
        np.full(200, 0.05), np.full(200, 0.05),
    ])
    n_h = len(hungarian_match(pred, gt)[0])
    n_s = len(simota_match(pred, gt)[0])
    assert n_h == 20
    assert abs(n_s - n_h) <= 5, f"SimOTA {n_s} vs Hungarian {n_h} — should be close"


def test_NEGATIVE_CONTROL_center_prior_adds_cost_rather_than_assigning_labels():
    """[NEGATIVE CONTROL] Round 1 got two things wrong: OR instead of AND, and
    assigning positive labels directly.

    The correct approach only NARROWS the search space; the number of positive
    labels is still decided by the matcher and does not balloon (round 1's bug gave
    ~140 labels for 48 GT).
    """
    pred, gt, sc = _data()
    pi, _ = simota_match(pred, gt, sc, use_center_prior=True)
    _, counts = np.unique(pi, return_counts=True)
    assert (counts > 1).sum() == 0
    assert len(pi) < 4 * gt.shape[0], "the positive-label count ballooned"

    # AND is stricter than OR: the AND mask must be a subset of the OR mask
    m_and = _center_prior_mask(pred, gt)
    p, g = pred, gt
    cx, cy = p[:, 0][:, None], p[:, 1][:, None]
    gx, gy, gw, gh = g[:, 0][None], g[:, 1][None], g[:, 2][None], g[:, 3][None]
    in_boxes = (cx > gx - gw / 2) & (cx < gx + gw / 2) & (cy > gy - gh / 2) & (cy < gy + gh / 2)
    r = 2.5 * np.sqrt(gw * gh)
    in_centers = (cx > gx - r) & (cx < gx + r) & (cy > gy - r) & (cy < gy + r)
    assert m_and.sum() <= (in_boxes | in_centers).sum()
    assert np.array_equal(m_and, in_boxes & in_centers)


def test_cost_uses_the_loss_weights():
    """Matcher and loss must be ON THE SAME SCALE, otherwise the matcher picks
    pairs the loss dislikes."""
    from ce_localization.utils.matcher_np import COST_CLASS, COST_GIOU, COST_L1
    assert (COST_L1, COST_GIOU, COST_CLASS) == (5.0, 2.0, 2.0)


def test_cost_does_not_explode_on_inverted_predicted_boxes():
    """An untrained model can return w<0 -> cxcywh_to_xyxy yields inverted boxes.

    build_cost must sort the corners, otherwise GIoU returns nonsense (round 1: 5e5).
    """
    pred = np.array([[0.5, 0.5, -0.2, -0.3], [0.3, 0.3, 0.1, 0.1]])
    gt = np.array([[0.5, 0.5, 0.1, 0.1]])
    cost, iou = build_cost(pred, gt)
    assert np.isfinite(cost).all() and np.abs(cost).max() < 1e3
    assert (iou >= 0).all() and (iou <= 1).all()


def test_label_stability_baseline():
    """A quantitative reference point for tracking metric #1 during training.

    Add 0.03 noise to the coordinates and count the % of pairs preserved. Round 1
    measured ~55 %; having this number is what makes it possible to tell, during
    training, whether 55 % is "normal" or "broken".
    """
    pred, gt, sc = _data(seed=11)
    rng = np.random.default_rng(0)
    pi0, gi0 = hungarian_match(pred, gt, sc)
    original = dict(zip(pi0.tolist(), gi0.tolist()))

    kept = total = 0
    for _ in range(20):
        noisy = pred + rng.standard_normal(pred.shape) * 0.03
        pi1, gi1 = hungarian_match(noisy, gt, sc)
        for p, g in zip(pi1.tolist(), gi1.tolist()):
            total += 1
            kept += int(original.get(p, -1) == g)
    ratio = kept / total
    assert 0.2 < ratio < 0.9, f"preserved ratio {ratio:.2f} is outside the plausible range"


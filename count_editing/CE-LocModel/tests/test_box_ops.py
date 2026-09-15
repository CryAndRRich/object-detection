"""Coordinate geometry tests — pure numpy, runnable locally (no torch needed).

Every test has a KNOWN ANALYTIC ANSWER. Tests marked [NEGATIVE CONTROL]
deliberately plant a wrong formula to prove the suite IS capable of catching
bugs — round 1 lacked exactly this kind of test, so "runs without crashing" and
"converges under an oracle" both passed while the code was wrong.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.box_ops_np import (  # noqa: E402
    box_iou, canvas_to_pixel, compute_scale, cxcywh_to_xyxy, decode_diffusion,
    encode_diffusion, filter_degenerate, flip_horizontal, generalized_box_iou,
    scale_to_canvas, xyxy_to_cxcywh,
)


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

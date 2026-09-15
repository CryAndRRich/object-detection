"""Matcher tests — pure numpy, runnable locally."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.matcher_np import (  # noqa: E402
    _center_prior_mask, build_cost, hungarian_match, match, simota_match,
)


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
    from utils.matcher_np import COST_CLASS, COST_GIOU, COST_L1
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

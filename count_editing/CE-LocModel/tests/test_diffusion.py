"""Diffusion math tests — pure numpy, runnable locally.

Checks round 1's 4 numerical bugs, each with its own NEGATIVE CONTROL test.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.box_ops_np import decode_diffusion, encode_diffusion  # noqa: E402
from utils.diffusion_np import (  # noqa: E402
    cosine_alphas_cumprod, ddim_step, ddim_time_pairs, linear_alphas_cumprod,
    make_placeholders, prepare_diffusion_concat, predict_noise_from_start, q_sample,
)

AB = cosine_alphas_cumprod(1000)
SNR = 2.0


# --------------------------------------------------------------------- schedule

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


# ------------------------------------------------------------------- q_sample

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


# ----------------------------------------------------------------------- DDIM

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


# --------------------------------------------------------- prepare_diffusion

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

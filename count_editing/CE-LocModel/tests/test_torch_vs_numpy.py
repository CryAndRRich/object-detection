"""Compare the TORCH version against the NUMPY version — the most important gate.

The numpy version is thoroughly checked by 49 tests with analytic answers. These
tests ensure the torch version does NOT drift from it, and they run WITHOUT
training and WITHOUT a GPU. They catch the exact class of bug that killed round 1:
dividing twice, forgetting the clamp, forgetting snr_scale.
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import box_ops, box_ops_np, diffusion_math, diffusion_np, matcher, matcher_np  # noqa: E402

TOL = 1e-5


def _t(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float64)


def _boxes(n=64, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(0.05, 0.95, n), rng.uniform(0.05, 0.95, n),
        rng.uniform(0.01, 0.35, n), rng.uniform(0.01, 0.35, n),
    ])


# ------------------------------------------------------------------- box_ops

def test_format_conversions_match():
    b = _boxes(128)
    assert np.abs(box_ops.cxcywh_to_xyxy(_t(b)).numpy() - box_ops_np.cxcywh_to_xyxy(b)).max() < TOL
    x = box_ops_np.cxcywh_to_xyxy(b)
    assert np.abs(box_ops.xyxy_to_cxcywh(_t(x)).numpy() - box_ops_np.xyxy_to_cxcywh(x)).max() < TOL


@pytest.mark.parametrize("snr", [1.0, 2.0, 3.0])
def test_encode_decode_match(snr):
    """Catches 'divided twice' and 'forgot the clamp' if the torch version drifts."""
    b = _boxes(128)
    e_np = box_ops_np.encode_diffusion(b, snr)
    e_t = box_ops.encode_diffusion(_t(b), snr).numpy()
    assert np.abs(e_t - e_np).max() < TOL

    raw = np.concatenate([e_np, e_np * 5.0])          # contains OUT-OF-RANGE values -> clamp
    d_np = box_ops_np.decode_diffusion(raw, snr)
    d_t = box_ops.decode_diffusion(_t(raw), snr).numpy()
    assert np.abs(d_t - d_np).max() < TOL
    assert d_t.min() >= 0.0 and d_t.max() <= 1.0


def test_iou_and_giou_match():
    a = box_ops_np.cxcywh_to_xyxy(_boxes(40, 1))
    b = box_ops_np.cxcywh_to_xyxy(_boxes(25, 2))
    assert np.abs(box_ops.box_iou(_t(a), _t(b))[0].numpy() - box_ops_np.box_iou(a, b)[0]).max() < TOL
    assert np.abs(box_ops.generalized_box_iou(_t(a), _t(b)).numpy()
                  - box_ops_np.generalized_box_iou(a, b)).max() < TOL


def test_sanitize_inverted_boxes():
    """An untrained model returns w<0 -> corners must be sorted, otherwise GIoU
    gives 5e5 as in round 1."""
    bad = _t(np.array([[0.6, 0.7, 0.2, 0.3], [0.1, 0.1, 0.5, 0.5]]))
    s = box_ops.sanitize_boxes(bad)
    assert (s[:, 2] >= s[:, 0]).all() and (s[:, 3] >= s[:, 1]).all()
    g = box_ops.generalized_box_iou(s, s)
    assert torch.isfinite(g).all() and g.abs().max() <= 1.0 + 1e-9


# ----------------------------------------------------------------- diffusion

def test_schedule_matches():
    ab_np = diffusion_np.cosine_alphas_cumprod(1000)
    ab_t = diffusion_math.cosine_alphas_cumprod(1000).numpy()
    assert np.abs(ab_t - ab_np).max() < 1e-12
    # and matches the measured values
    assert np.sqrt(ab_t[[249, 499, 749]]) == pytest.approx([0.92, 0.70, 0.38], abs=0.01)


def test_q_sample_and_predict_noise_match():
    ab_np = diffusion_np.cosine_alphas_cumprod(1000)
    ab_t = diffusion_math.cosine_alphas_cumprod(1000)
    rng = np.random.default_rng(0)
    x0 = box_ops_np.encode_diffusion(_boxes(32), 2.0)
    nz = rng.standard_normal(x0.shape)

    for t in [0, 100, 500, 900, 999]:
        a = diffusion_np.q_sample(x0, t, nz, ab_np)
        b = diffusion_math.q_sample(_t(x0), t, _t(nz), ab_t).numpy()
        assert np.abs(a - b).max() < TOL, f"q_sample diverges at t={t}"

        pa = diffusion_np.predict_noise_from_start(a, t, x0, ab_np)
        pb = diffusion_math.predict_noise_from_start(_t(a), t, _t(x0), ab_t).numpy()
        assert np.abs(pa - pb).max() < 1e-4, f"predict_noise diverges at t={t}"


def test_ddim_time_pairs_match():
    assert diffusion_math.ddim_time_pairs(1000, 4) == diffusion_np.ddim_time_pairs(1000, 4)


def test_placeholders_share_the_same_statistics():
    """Not an element-wise comparison (the RNGs differ) but a STATISTICAL one plus
    two invariants."""
    g = torch.Generator().manual_seed(0)
    med = (0.0686, 0.0609)
    ph_t = diffusion_math.make_placeholders(20000, med, 0.7, generator=g).numpy()
    ph_n = diffusion_np.make_placeholders(20000, med, 0.7, np.random.default_rng(0))

    assert (ph_t[:, 1] <= 0.7).all(), "torch: a placeholder landed in the padding"
    assert (ph_n[:, 1] <= 0.7).all(), "numpy: a placeholder landed in the padding"
    assert abs(np.median(ph_t[:, 2]) - np.median(ph_n[:, 2])) < 0.01
    assert abs(np.median(ph_t[:, 0]) - np.median(ph_n[:, 0])) < 0.02


def test_prepare_diffusion_concat_matches_shape_and_range():
    ab_t = diffusion_math.cosine_alphas_cumprod(1000)
    gt = torch.as_tensor(_boxes(30), dtype=torch.float32)
    g = torch.Generator().manual_seed(0)

    x_t, nz, is_gt = diffusion_math.prepare_diffusion_concat(
        gt, 100, 500, ab_t, 2.0, valid_h=0.7, generator=g)
    assert x_t.shape == (100, 4) and nz.shape == (100, 4)
    assert int(is_gt.sum()) == 30 and bool(is_gt[:30].all())
    assert x_t.abs().max() <= 2.0 + 1e-6                      # clamped

    # M > N -> crop, every slot is real GT
    _, _, is2 = diffusion_math.prepare_diffusion_concat(
        gt.repeat(10, 1), 100, 500, ab_t, 2.0, generator=g)
    assert bool(is2.all())

    # M == 0
    _, _, is3 = diffusion_math.prepare_diffusion_concat(
        torch.zeros(0, 4), 100, 500, ab_t, 2.0, generator=g)
    assert not bool(is3.any())


# ------------------------------------------------------------------- matcher

def test_cost_matrix_matches():
    p, gtb = _boxes(80, 3), _boxes(20, 4)
    sc = np.random.default_rng(0).uniform(-3, 3, 80)
    # the numpy version takes PROBABILITIES, the torch one takes LOGITS -> feed
    # equivalent inputs
    c_np, i_np = matcher_np.build_cost(p, gtb, 1 / (1 + np.exp(-sc)))
    c_t, i_t = matcher.build_cost(_t(p), _t(gtb), _t(sc))
    assert np.abs(c_t.numpy() - c_np).max() < 1e-4
    assert np.abs(i_t.numpy() - i_np).max() < TOL


@pytest.mark.parametrize("method", ["hungarian", "simota"])
def test_matchers_give_the_same_result(method):
    p, gtb = _boxes(100, 5), _boxes(25, 6)
    sc = np.random.default_rng(1).uniform(-3, 3, 100)
    pi_n, gi_n = matcher_np.match(p, gtb, 1 / (1 + np.exp(-sc)), method=method)
    pi_t, gi_t = matcher.match(_t(p), _t(gtb), _t(sc), method=method)

    pairs_n = sorted(zip(pi_n.tolist(), gi_n.tolist()))
    pairs_t = sorted(zip(pi_t.tolist(), gi_t.tolist()))
    assert pairs_t == pairs_n, f"{method}: the matchings differ"


@pytest.mark.parametrize("method", ["hungarian", "simota"])
def test_torch_matcher_invariants(method):
    p, gtb = _boxes(100, 7), _boxes(30, 8)
    pi, gi = matcher.match(_t(p), _t(gtb), method=method)
    _, counts = np.unique(pi.numpy(), return_counts=True)
    assert (counts > 1).sum() == 0, "a proposal matched >1 GT"
    assert len(set(gi.tolist())) == 30, "a GT is still unmatched"


def test_cost_weights_are_identical_in_both_versions():
    assert (matcher.COST_L1, matcher.COST_GIOU, matcher.COST_CLASS) == \
           (matcher_np.COST_L1, matcher_np.COST_GIOU, matcher_np.COST_CLASS) == (5.0, 2.0, 2.0)


# ------------------------------------- the generator must be on the same device

def test_generator_on_wrong_device_is_caught_early():
    """`torch.randn(device='cuda', generator=<cpu gen>)` raises a cryptic
    RuntimeError. This actually happened in the val loop (train.py:162) — training
    passed because it does not pass a generator, only validation does. The guard in
    the detector must catch it first, with a message that names the fix.

    Still checkable on a machine without CUDA: the reverse setup (cuda generator,
    cpu tensor) cannot be built, so the guard's device-comparison logic is tested
    directly.
    """
    from models.detector import _check_generator

    _check_generator(None, "cuda")                      # None is always valid
    _check_generator(torch.Generator(), "cpu")          # matching
    _check_generator(torch.Generator(device="cpu"), torch.device("cpu"))

    class _FakeCuda:                                    # a fake generator "on cuda"
        device = torch.device("cuda:0")

    with pytest.raises(AssertionError, match="device='cpu'"):
        _check_generator(_FakeCuda(), "cpu")

    if torch.cuda.is_available():
        with pytest.raises(AssertionError, match="device='cuda'"):
            _check_generator(torch.Generator(device="cpu"), "cuda")


def test_build_inputs_runs_with_a_seeded_generator():
    """Calls the REAL build_inputs path, which is what the val loop does.

    The earlier version of this test only checked _check_generator's comparison
    logic in isolation, so it passed while build_inputs still raised: randint had
    no `device=`, so it wanted a CPU generator while prepare_diffusion_concat
    wanted a CUDA one, and no single generator could satisfy both. A test that
    never calls the function under test cannot catch that -- so call it.

    Runs on CPU (where both requirements happen to coincide) and, when a GPU is
    present, on CUDA too, which is the combination that actually failed.
    """
    from models.detector import CELocDetector

    class _Stub:
        """Minimal stand-in: build_inputs only touches these three attributes,
        so CLIP is never constructed (no network, no 600 MB download in CI)."""
        num_timesteps = 1000
        snr_scale = 2.0
        alphas_cumprod = torch.as_tensor(
            diffusion_np.cosine_alphas_cumprod(1000), dtype=torch.float32)

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for devname in devices:
        dev = torch.device(devname)
        stub = _Stub()
        stub.alphas_cumprod = _Stub.alphas_cumprod.to(dev)
        gt = [torch.as_tensor(_boxes(12), dtype=torch.float32).to(dev),
              torch.as_tensor(_boxes(40), dtype=torch.float32).to(dev)]
        g = torch.Generator(device=dev).manual_seed(1234)

        x_t, t_batch, is_gt = CELocDetector.build_inputs(
            stub, gt, 100, [0.75, 1.0], generator=g)

        assert x_t.shape == (2, 100, 4), f"{devname}: wrong shape {x_t.shape}"
        assert x_t.device.type == dev.type
        assert (t_batch == t_batch[0]).all(), "t must be ONE value for the whole batch"
        assert int(is_gt[0].sum()) == 12 and int(is_gt[1].sum()) == 40

        # same seed -> same result, which is what makes val loss comparable
        g2 = torch.Generator(device=dev).manual_seed(1234)
        x_t2, _, _ = CELocDetector.build_inputs(stub, gt, 100, [0.75, 1.0], generator=g2)
        assert torch.equal(x_t, x_t2), f"{devname}: same seed gave different results"

    # and without a generator (the training path) it must also work
    stub = _Stub()
    gt = [torch.as_tensor(_boxes(5), dtype=torch.float32)]
    x_t, _, _ = CELocDetector.build_inputs(stub, gt, 32, [1.0])
    assert x_t.shape == (1, 32, 4)


def test_prepare_diffusion_concat_uses_a_generator_on_the_GT_device():
    """The real code path that crashed: GT on device X -> placeholders on X too ->
    the generator must be on X. Checks on CPU that a CPU generator passes through
    cleanly (a regression lock on the other direction: do not "fix" this by
    dropping the generator entirely)."""
    gt = torch.as_tensor(_boxes(12), dtype=torch.float32)
    ab = torch.as_tensor(diffusion_np.cosine_alphas_cumprod(1000), dtype=torch.float64)
    g = torch.Generator(device="cpu").manual_seed(1234)

    a, _, _ = diffusion_math.prepare_diffusion_concat(gt, 100, 500, ab, 2.0,
                                                      valid_h=0.75, generator=g)
    g2 = torch.Generator(device="cpu").manual_seed(1234)
    b, _, _ = diffusion_math.prepare_diffusion_concat(gt, 100, 500, ab, 2.0,
                                                      valid_h=0.75, generator=g2)
    assert torch.equal(a, b), \
        "the same seed must give the same result — otherwise val loss is not comparable across epochs"

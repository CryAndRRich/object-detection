"""Verification steps 1-2 of docs/diffu-grounding-dino-plan.md.

Runs on CPU, needs no checkpoint and no dataset:

    python tests/test_diffusion.py        # or: pytest tests/test_diffusion.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.diffusion import RefPointDiffusion, build_timestep_modules, make_beta_schedule  # noqa: E402
from models.diffusion.timestep import sinusoidal_timestep_embedding  # noqa: E402
from util.misc import inverse_sigmoid  # noqa: E402

NUM_QUERIES = 900


def reference_cosine_betas(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Independent transcription of Nichol & Dhariwal (ICML 2021) eq. 17.

    Deliberately written out longhand here so the test is an oracle rather than a
    restatement of the implementation under test. Same formula DiffusionDet's
    ``cosine_beta_schedule`` uses.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# --------------------------------------------------------------------------- #
# 1. schedule
# --------------------------------------------------------------------------- #
def test_cosine_schedule_matches_reference():
    for timesteps in (100, 1000):
        ours = make_beta_schedule("cosine", timesteps)
        ref = reference_cosine_betas(timesteps)
        assert torch.allclose(ours, ref, atol=0, rtol=0), f"betas differ at T={timesteps}"


def test_schedules_are_valid():
    for name in ("cosine", "linear", "sqrt"):
        betas = make_beta_schedule(name, 1000)
        assert betas.shape == (1000,)
        assert (betas >= 0).all() and (betas <= 0.999).all(), f"{name}: betas out of range"

        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        assert (alphas_cumprod.diff() <= 0).all(), f"{name}: alphas_cumprod must be non-increasing"
        assert alphas_cumprod[0] > 0.99, f"{name}: barely any signal left at t=0"
        assert alphas_cumprod[-1] < 0.02, f"{name}: signal not destroyed at t=T-1"


def test_buffers_are_consistent():
    diff = RefPointDiffusion(num_timesteps=1000)
    a = diff.alphas_cumprod

    assert torch.allclose(diff.sqrt_alphas_cumprod, a.sqrt(), rtol=1e-6, atol=0)
    assert torch.allclose(diff.alphas_cumprod_prev[1:], a[:-1], atol=0, rtol=0)

    # The buffers are derived in float64 and stored as float32. Re-deriving them
    # here from the float32 ``alphas_cumprod`` goes through ``1 - a`` with
    # ``a ~ 0.99996`` at t=0, which is catastrophic cancellation: the oracle loses
    # ~3 decimal digits, not the buffer. Hence the loose relative tolerance -- the
    # point of this test is that no buffer is off by a factor or a sign.
    assert torch.allclose(diff.sqrt_one_minus_alphas_cumprod, (1 - a).sqrt(), rtol=1e-2, atol=1e-6)
    assert torch.allclose(diff.sqrt_recip_alphas_cumprod, (1.0 / a).sqrt(), rtol=1e-4, atol=0)
    assert torch.allclose(diff.sqrt_recipm1_alphas_cumprod, (1.0 / a - 1).sqrt(), rtol=1e-2, atol=1e-6)

    # q_sample's two coefficients must satisfy a^2 + b^2 = 1 at every t; this is
    # the invariant that actually matters and it holds in fp32.
    norm = diff.sqrt_alphas_cumprod**2 + diff.sqrt_one_minus_alphas_cumprod**2
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)
    assert diff.alphas_cumprod_prev[0].item() == 1.0
    for name in ("betas", "alphas_cumprod", "posterior_variance", "loss_weights"):
        buf = getattr(diff, name)
        assert torch.isfinite(buf).all(), f"buffer {name} has non-finite entries"


def test_loss_weights():
    diff = RefPointDiffusion(loss_weight_mode="diffudino", normalize_loss_weight=True)
    w = diff.loss_weights
    assert (w > 0).all(), "weights must be positive"
    assert abs(w.mean().item() - 1.0) < 1e-5, "normalized weights should average to 1"
    assert w[0] > w[-1], "diffudino weighting must decay with t (less weight on noisier samples)"

    raw = RefPointDiffusion(loss_weight_mode="diffudino", normalize_loss_weight=False).loss_weights
    assert abs(raw[0].item() - 0.5) < 1e-2, "unnormalized w(0) should be ~0.5*sqrt(1)/(2-1)"

    off = RefPointDiffusion(loss_weight_mode="none").loss_weights
    assert torch.allclose(off, torch.ones_like(off))

    t = torch.tensor([0, 500, 999])
    assert diff.loss_weight(t).shape == (3,)


# --------------------------------------------------------------------------- #
# 2. training-time reference points
# --------------------------------------------------------------------------- #
def _gt():
    return torch.tensor(
        [
            [0.25, 0.25, 0.20, 0.30],
            [0.70, 0.60, 0.10, 0.15],
            [0.50, 0.50, 0.90, 0.80],
        ]
    )


def test_refpoints_shapes_and_range():
    diff = RefPointDiffusion()
    gt = _gt()
    refpoints, noise, t = diff.prepare_diffusion_refpoints(gt, NUM_QUERIES)

    assert refpoints.shape == (NUM_QUERIES, 4)
    assert noise.shape == (NUM_QUERIES, 4)
    assert t.shape == (1,) and 0 <= t.item() < diff.num_timesteps
    assert torch.isfinite(refpoints).all(), "reference points must be finite"

    # The decoder immediately sigmoids these, so check the pre-logit box space is
    # strictly inside (box_eps, 1 - box_eps) -- the guard that keeps logit finite.
    boxes = refpoints.sigmoid()
    assert boxes.min() >= diff.box_eps - 1e-6, f"box min {boxes.min():.6f} below box_eps"
    assert boxes.max() <= 1.0 - diff.box_eps + 1e-6, f"box max {boxes.max():.6f} above 1-box_eps"


def test_refpoints_at_t0_recover_gt():
    diff = RefPointDiffusion()
    gt = _gt()
    t = torch.zeros(1, dtype=torch.long)
    refpoints, _, _ = diff.prepare_diffusion_refpoints(gt, NUM_QUERIES, t=t)
    recovered = refpoints[: gt.shape[0]].sigmoid()
    err = (recovered - gt).abs().max().item()
    assert err < 0.02, f"at t=0 the first num_gt queries should be ~gt, max err {err:.4f}"


def test_refpoints_at_tmax_are_noise():
    diff = RefPointDiffusion()
    gt = _gt()
    t = torch.full((1,), diff.num_timesteps - 1, dtype=torch.long)
    refpoints, _, _ = diff.prepare_diffusion_refpoints(gt, NUM_QUERIES, t=t)
    recovered = refpoints[: gt.shape[0]].sigmoid()
    err = (recovered - gt).abs().max().item()
    assert err > 0.05, f"at t=T-1 the gt should be destroyed, but max err is only {err:.4f}"

    # Almost everything gets clamped to the latent bounds at maximum noise, so the
    # boxes pile up at the two extremes; just assert we spread across the range.
    boxes = refpoints.sigmoid()
    assert boxes.std() > 0.2, f"pure-noise boxes look degenerate, std={boxes.std():.4f}"


def test_refpoints_empty_gt():
    """An image with no annotation must not crash and must not produce NaN."""
    diff = RefPointDiffusion()
    for t_val in (0, 500, 999):
        t = torch.full((1,), t_val, dtype=torch.long)
        refpoints, noise, _ = diff.prepare_diffusion_refpoints(torch.zeros(0, 4), NUM_QUERIES, t=t)
        assert refpoints.shape == (NUM_QUERIES, 4)
        assert torch.isfinite(refpoints).all(), f"non-finite refpoints for empty gt at t={t_val}"
        assert torch.isfinite(noise).all()

    # At t=0 the single fake box [0.5, 0.5, 1, 1] should come back.
    refpoints, _, _ = diff.prepare_diffusion_refpoints(
        torch.zeros(0, 4), NUM_QUERIES, t=torch.zeros(1, dtype=torch.long)
    )
    fake = refpoints[0].sigmoid()
    expected = torch.tensor([0.5, 0.5, 1.0 - diff.box_eps, 1.0 - diff.box_eps])
    assert (fake - expected).abs().max() < 0.02, f"fake box is {fake.tolist()}"


def test_refpoints_more_gt_than_queries():
    """Subsampling path: 50 gt boxes into 8 queries."""
    diff = RefPointDiffusion()
    gt = torch.rand(50, 4).clamp(0.05, 0.95)
    refpoints, noise, _ = diff.prepare_diffusion_refpoints(gt, 8)
    assert refpoints.shape == (8, 4) and noise.shape == (8, 4)
    assert torch.isfinite(refpoints).all()


def test_pad_modes():
    gt = _gt()
    for mode in ("normal", "center", "sigmoid_normal"):
        diff = RefPointDiffusion(pad_mode=mode)
        refpoints, _, _ = diff.prepare_diffusion_refpoints(gt, 64, t=torch.zeros(1, dtype=torch.long))
        assert torch.isfinite(refpoints).all(), f"pad_mode={mode} produced non-finite values"
        pad_boxes = refpoints[gt.shape[0] :].sigmoid()
        assert (pad_boxes[:, 2:] > 0).all(), f"pad_mode={mode} produced non-positive width/height"


def test_batch_helper_draws_independent_timesteps():
    diff = RefPointDiffusion()
    gt_list = [_gt(), torch.zeros(0, 4), torch.rand(2000, 4).clamp(0.05, 0.95)]
    refpoints, noise, t = diff.prepare_diffusion_refpoints_batch(gt_list, NUM_QUERIES)
    assert refpoints.shape == (3, NUM_QUERIES, 4)
    assert noise.shape == (3, NUM_QUERIES, 4)
    assert t.shape == (3,)
    assert torch.isfinite(refpoints).all()


# --------------------------------------------------------------------------- #
# forward/backward consistency of the diffusion math
# --------------------------------------------------------------------------- #
def test_noise_start_roundtrip():
    """``predict_noise_from_start`` must invert ``predict_start_from_noise``."""
    diff = RefPointDiffusion()
    torch.manual_seed(0)
    x_start = torch.randn(4, 16, 4)
    noise = torch.randn(4, 16, 4)
    # Stay away from t=T-1 where sqrt_recipm1_alphas_cumprod ~ 1e4 and the
    # inversion is numerically ill-conditioned in fp32.
    t = torch.tensor([0, 100, 400, 700])

    x_t = diff.q_sample(x_start, t, noise)
    assert torch.allclose(diff.predict_start_from_noise(x_t, t, noise), x_start, atol=1e-3)
    assert torch.allclose(diff.predict_noise_from_start(x_t, t, x_start), noise, atol=1e-3)


def test_space_conversions_roundtrip():
    diff = RefPointDiffusion()
    boxes = torch.rand(32, 4).clamp(diff.box_eps, 1 - diff.box_eps)
    assert torch.allclose(diff.latent_to_boxes(diff.boxes_to_latent(boxes)), boxes, atol=1e-6)


def test_ddim_time_pairs():
    for steps in (1, 3, 5, 10):
        diff = RefPointDiffusion(sampling_timesteps=steps)
        pairs = diff.ddim_time_pairs()
        assert len(pairs) == steps, f"{steps} sampling steps should mean {steps} decoder evaluations"
        assert pairs[0][0] == diff.num_timesteps - 1, "sampling must start at t=T-1"
        assert pairs[-1][1] < 0, "the last pair must carry the stop sentinel"
        times = [p[0] for p in pairs]
        assert times == sorted(times, reverse=True), "timesteps must decrease"


def test_ddim_step():
    diff = RefPointDiffusion(sampling_timesteps=3)
    x = torch.randn(2, 16, 4)
    x_start = torch.randn(2, 16, 4)
    pred_noise = torch.randn(2, 16, 4)

    out = diff.ddim_step(x, x_start, pred_noise, time=332, time_next=-1)
    assert torch.equal(out, x_start), "a negative t_next must return the predicted x_0 as-is"

    out = diff.ddim_step(x, x_start, pred_noise, time=999, time_next=665)
    assert out.shape == x.shape and torch.isfinite(out).all()

    deterministic = RefPointDiffusion(ddim_eta=0.0)
    a = deterministic.ddim_step(x, x_start, pred_noise, 999, 665)
    b = deterministic.ddim_step(x, x_start, pred_noise, 999, 665)
    assert torch.allclose(a, b), "eta=0 must make the sampler deterministic"


def test_full_ddim_loop_is_stable():
    """Drive the sampler with a perfect oracle denoiser and check it converges."""
    diff = RefPointDiffusion(sampling_timesteps=3)
    torch.manual_seed(0)
    target = diff.boxes_to_latent(torch.rand(2, 32, 4).clamp(0.1, 0.9))

    x = diff.init_latent(2, 32, device=torch.device("cpu"))
    for time, time_next in diff.ddim_time_pairs():
        t = torch.full((2,), time, dtype=torch.long)
        x_start = target  # oracle: always predicts the right x_0
        pred_noise = diff.predict_noise_from_start(x, t, x_start)
        x = diff.ddim_step(x, x_start, pred_noise, time, time_next)
        assert torch.isfinite(x).all(), f"sampler diverged at t={time}"

    assert torch.allclose(x, target, atol=1e-4), "with an oracle denoiser DDIM must land on the target"


def test_q_sample_stays_fp32_under_autocast():
    """AMP must not silently downcast the schedule math (see force_fp32)."""
    if not torch.cuda.is_available():
        return
    diff = RefPointDiffusion().cuda()
    x_start = torch.randn(2, 16, 4, device="cuda")
    t = torch.tensor([10, 900], device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = diff.q_sample(x_start, t)
    assert out.dtype == torch.float32, f"q_sample leaked out of fp32: {out.dtype}"


# --------------------------------------------------------------------------- #
# timestep conditioning modules
# --------------------------------------------------------------------------- #
def test_sinusoidal_embedding():
    t = torch.tensor([0, 1, 500, 999])
    for dim in (256, 255):
        emb = sinusoidal_timestep_embedding(t, dim)
        assert emb.shape == (4, dim)
        assert torch.isfinite(emb).all()
    emb = sinusoidal_timestep_embedding(t, 256)
    assert not torch.allclose(emb[0], emb[2]), "different timesteps must embed differently"


def test_timestep_injectors_are_identity_at_init():
    """Critical for finetuning: a fresh injector must not perturb the queries."""
    for mode in ("film", "add"):
        encoder, injectors = build_timestep_modules(mode, d_model=256, num_layers=6)
        assert len(injectors) == 6
        t_emb = encoder(torch.tensor([3, 700]))
        assert t_emb.shape == (2, 256 * 4 if mode == "film" else 256)

        for layout in ((900, 2, 256), (2, 900, 256)):
            x = torch.randn(*layout)
            out = injectors[0](x, t_emb)
            assert out.shape == x.shape, f"{mode} changed the query shape for layout {layout}"
            assert torch.allclose(out, x, atol=1e-6), f"{mode} is not the identity at init ({layout})"


def test_film_residual_variant():
    """The literal DiffuDINO form doubles the query at init -- assert it, loudly."""
    _, injectors = build_timestep_modules("film", d_model=64, num_layers=1, film_residual=True)
    x = torch.randn(10, 2, 64)
    t_emb = torch.randn(2, 256)
    assert torch.allclose(injectors[0](x, t_emb), 2 * x, atol=1e-6)


def test_timestep_injector_gradients_flow():
    encoder, injectors = build_timestep_modules("film", d_model=64, num_layers=2)
    x = torch.randn(10, 2, 64)
    t = torch.tensor([5, 600])
    out = injectors[0](x, encoder(t))
    out.pow(2).mean().backward()

    grads = [p.grad for p in list(encoder.parameters()) + list(injectors[0].parameters()) if p.grad is not None]
    assert grads, "no gradients reached the timestep modules"
    assert any(g.abs().sum() > 0 for g in grads), "timestep modules received only zero gradients"


def test_shared_injector():
    _, injectors = build_timestep_modules("film", d_model=64, num_layers=6, share_across_layers=True)
    assert len(injectors) == 1


# --------------------------------------------------------------------------- #
def main():
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = []
    for name, fn in tests:
        try:
            torch.manual_seed(1234)
            fn()
            print(f"  ok    {name}")
        except Exception as exc:  # noqa: BLE001 - a test runner should report, not raise
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

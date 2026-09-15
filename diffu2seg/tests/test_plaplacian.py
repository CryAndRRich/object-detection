"""p-Laplacian: the matmul expansions, the clamp, and convergence.

THE MOST IMPORTANT FILE IN THE TEST SUITE. A wrong expansion still produces a
finite f, a plausible mask, a plausible box, and an oracle_recall number. No
assert fires anywhere downstream -- exactly the "sai am tham" failure mode that
CLAUDE.md warns about for the xyxy/cxcywh mix-up. The only defence is checking
against a literal triple loop in fp64.

Run:  python -m pytest tests/test_plaplacian.py -q
      python tests/test_plaplacian.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d2s.plaplacian import compute_g, plaplacian_propagate  # noqa: E402

TOL = 1e-12          # fp64: the expansion is exact algebra, not an approximation


def _random_affinity(n, seed=0, dtype=torch.float64):
    """Row-stochastic (N, N), like softmax(QK^T) coming out of SD2."""
    g = torch.Generator().manual_seed(seed)
    A = torch.rand(n, n, generator=g, dtype=dtype) + 1e-3
    return A / A.sum(dim=1, keepdim=True)


def _random_f(k, n, seed=1, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(k, n, generator=g, dtype=dtype)


def _naive_g(A, f):
    """g_i = sqrt(sum_j A_ij (f_j - f_i)^2), written out literally."""
    K, N = f.shape
    out = torch.zeros(K, N, dtype=f.dtype)
    for k in range(K):
        for i in range(N):
            s = 0.0
            for j in range(N):
                s += A[i, j].item() * (f[k, j].item() - f[k, i].item()) ** 2
            out[k, i] = s ** 0.5
    return out


def test_g_expansion_matches_naive():
    """(A f^2) - 2 f (A f) + f^2 (A 1) == the triple loop."""
    A, f = _random_affinity(12), _random_f(3, 12)
    fast = compute_g(A, f, eps=0.0)
    slow = _naive_g(A, f)
    assert (fast - slow).abs().max().item() < TOL


def test_gamma_expansion_matches_naive():
    """The SECOND expansion -- the one that hides inside a plain-looking average.

    sum_j gamma_ij f_j  and  sum_j gamma_ij, with gamma_ij = A_ij (gp_i + gp_j).
    Easy to miss because the update reads like a weighted mean; just as fatal
    in memory terms, and just as silent when wrong.
    """
    A, f = _random_affinity(10, seed=4), _random_f(2, 10, seed=5)
    p, lam = 1.6, 1e-5

    g = compute_g(A, f, eps=1e-8)
    gp = g.pow(p - 2.0)
    row_sum = A.sum(dim=1)

    fast_num = lam * f + gp * (f @ A.T) + (gp * f) @ A.T
    fast_den = lam + gp * row_sum + gp @ A.T

    K, N = f.shape
    slow_num = torch.zeros(K, N, dtype=f.dtype)
    slow_den = torch.zeros(K, N, dtype=f.dtype)
    for k in range(K):
        for i in range(N):
            acc_n, acc_d = 0.0, 0.0
            for j in range(N):
                gamma = A[i, j].item() * (gp[k, i].item() + gp[k, j].item())
                acc_n += gamma * f[k, j].item()
                acc_d += gamma
            slow_num[k, i] = lam * f[k, i].item() + acc_n
            slow_den[k, i] = lam + acc_d

    assert (fast_num - slow_num).abs().max().item() < TOL
    assert (fast_den - slow_den).abs().max().item() < TOL


def test_p2_collapses_to_linear_diffusion():
    """At p=2, gp = g^0 = 1 and gamma = 2A, so one step is a one-liner.

    NEGATIVE CONTROL: writing `gp = g.pow(p)` instead of `g.pow(p - 2)` -- an
    easy slip -- breaks this, because g^2 != 1.
    """
    A, f0 = _random_affinity(9, seed=7, dtype=torch.float32), None
    f0 = torch.zeros(2, 9, dtype=torch.float32)
    f0[0, 0] = 1.0
    f0[1, 5] = 1.0
    lam = 1e-3

    f, _, _ = plaplacian_propagate(A, f0, p=2.0, lam=lam, tau_prop=0.0, max_iter=1)
    row_sum = A.sum(dim=1)
    expected = (lam * f0 + 2.0 * (f0 @ A.T)) / (lam + 2.0 * row_sum)
    assert (f - expected).abs().max().item() < 1e-6


def test_flat_field_does_not_produce_nan():
    """g == 0 everywhere; with p-2 = -0.4 the unclamped form is inf -> NaN.

    NEGATIVE CONTROL included: with g_eps=0 the result MUST be non-finite. If
    this half ever starts passing, the clamp has been silently neutralised and
    the guard above is no longer proving anything.
    """
    N = 8
    A = _random_affinity(N, seed=11, dtype=torch.float32)
    f_flat = torch.full((1, N), 0.3, dtype=torch.float32)

    ok = compute_g(A, f_flat, eps=1e-8)
    assert torch.isfinite(ok).all() and (ok > 0).all()
    assert torch.isfinite(ok.pow(1.6 - 2.0)).all()

    bad = compute_g(A, f_flat, eps=0.0)
    assert not torch.isfinite(bad.pow(1.6 - 2.0)).all(), \
        "clamp guard is no longer being tested"


def test_propagate_stays_finite_on_flat_seed():
    N = 16
    A = _random_affinity(N, seed=13, dtype=torch.float32)
    f0 = torch.zeros(3, N, dtype=torch.float32)
    f0[:, 0] = 1.0
    f, n_iter, _ = plaplacian_propagate(A, f0, p=1.6, lam=1e-5,
                                        tau_prop=1e-4, max_iter=50)
    assert torch.isfinite(f).all()
    assert 1 <= n_iter <= 50


def test_residual_decreases_and_converges():
    A = _random_affinity(20, seed=17, dtype=torch.float32)
    f0 = torch.zeros(2, 20, dtype=torch.float32)
    f0[0, 3] = 1.0
    f0[1, 11] = 1.0
    f, n_iter, res = plaplacian_propagate(A, f0, p=1.6, lam=1e-3, tau_prop=1e-10,
                                          max_iter=300, return_history=True)
    assert n_iter < 300, "should converge well inside the cap on a dense graph"
    assert res[-1] <= res[0]
    assert res[-1] <= 1e-10


def test_large_lam_pins_solution_to_seed():
    """lam -> large means the anchor dominates: f ~ f0. Checks lam's position."""
    N = 12
    A = _random_affinity(N, seed=19, dtype=torch.float32)
    f0 = torch.zeros(1, N, dtype=torch.float32)
    f0[0, 4] = 1.0
    f, _, _ = plaplacian_propagate(A, f0, p=1.6, lam=1e6, tau_prop=1e-12, max_iter=50)
    assert (f - f0).abs().max().item() < 1e-3


def test_output_is_bounded_by_seed_range():
    """A row-stochastic, f0 in [0,1] -> f stays in [0,1] (maximum principle)."""
    N = 14
    A = _random_affinity(N, seed=23, dtype=torch.float32)
    f0 = torch.zeros(4, N, dtype=torch.float32)
    for k in range(4):
        f0[k, k * 3] = 1.0
    f, _, _ = plaplacian_propagate(A, f0, p=1.6, lam=1e-4, tau_prop=1e-8, max_iter=100)
    assert f.min().item() >= -1e-6
    assert f.max().item() <= 1.0 + 1e-6


def test_seed_keeps_the_maximum():
    """The anchored cell must stay the strongest -- otherwise thresholding a map
    would not even select the object the prompt was placed on."""
    N = 24
    A = _random_affinity(N, seed=29, dtype=torch.float32)
    seed_idx = 7
    f0 = torch.zeros(1, N, dtype=torch.float32)
    f0[0, seed_idx] = 1.0
    f, _, _ = plaplacian_propagate(A, f0, p=1.6, lam=1e-2, tau_prop=1e-10, max_iter=200)
    assert int(f[0].argmax().item()) == seed_idx


def test_rejects_fp16():
    """fp16 has eps ~6e-8; g**(-0.4) on such values overflows."""
    N = 8
    A = _random_affinity(N, dtype=torch.float32).half()
    f0 = torch.zeros(1, N, dtype=torch.float16)
    f0[0, 0] = 1.0
    try:
        plaplacian_propagate(A, f0, p=1.6, lam=1e-5, tau_prop=1e-4, max_iter=5)
    except AssertionError:
        return
    raise AssertionError("fp16 input must be rejected")


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

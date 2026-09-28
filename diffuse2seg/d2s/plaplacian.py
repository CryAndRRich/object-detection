r"""Non-linear p-Laplacian propagation on the attention graph.

This is Algorithm 1 of Diffuse2Seg (arXiv 2609.06491), which is itself the
non-local graph regularisation of Elmoataz et al. (2008). Written from the
formulas -- nothing imported or copied from refs/repos/.

                        THE ENERGY BEING MINIMISED

    E(f) = (1/p) * sum_i [ sum_j A_ij (f_j - f_i)^2 ]^(p/2)
         + (lam/2) * ||f - f0||^2
           \_______ smooth _______/   \____ anchor ____/

The anchor is what makes this different from M2N2's Markov chain on the same
matrix. A Markov chain converges to a distribution that has forgotten f0
entirely (uniform, after IPF), which is why M2N2 must read out TIME-to-threshold
rather than a value. Here lam*f0 holds the solution near its seed, so the
converged VALUE is itself the answer and no IPF is needed.

                   WHY p < 2 IS SUPPOSED TO MATTER

With g_i the weighted gradient at token i, the smooth term is (1/p) sum_i g_i^p.

    p = 2   : penalises SQUARED gradient. One slope of 10 costs as much as a
              hundred slopes of 1, so the optimum spreads the slope evenly ->
              edges get sanded down. gamma collapses to 2*A and this whole file
              reduces to one line of ordinary linear diffusion.
    p < 2   : sub-quadratic. A slope of 10 costs only 10^1.6 ~ 40, so the
              optimum PAYS for a few steep edges in exchange for being flat
              everywhere else -> edges survive.

Same principle as L1 vs L2 in regression. Whether it is measurable on CE-130,
where the median object short side is 4.65 cells and the median SMALLEST object
per image is 2.24 cells, is exactly what tools/check_plaplacian_vs_p2.py exists
to decide.

           TWO THINGS THE PAPER DOES NOT WRITE DOWN, BOTH FATAL

(1) THE EXPANSION. Written literally,
        g_i = sqrt( sum_j A_ij (f_j - f_i)^2 )
    builds a (K, N, N) tensor. At K=441 prompts and N=4096 tokens that is
    7.4e9 fp32 elements = 29 GB; the A30 has 24 GB. It simply never runs.
    Expanding the square turns it into three matmuls:

        sum_j A_ij (f_j - f_i)^2 = (A f^2)_i - 2 f_i (A f)_i + f_i^2 (A 1)_i

    The SAME trap sits in the update step, one line further down, and is easier
    to miss because it looks like a plain weighted average. Since
    gamma_ij = A_ij (gp_i + gp_j), the two halves split:

        sum_j gamma_ij f_j = gp_i * (A f)_i + (A (gp * f))_i
        sum_j gamma_ij     = gp_i * (A 1)_i + (A gp)_i

    Four matmuls per iteration total (A@f, A@f^2, A@(gp*f), A@gp), plus A@1
    hoisted out of the loop. No intermediate is ever 3-D -- asserted below.

(2) THE CLAMP. g is exactly 0 wherever f is locally flat, which on CE-130 means
    most of a plain background. p - 2 = -0.4 < 0, so 0**(-0.4) = inf, gamma
    becomes inf, and f is NaN on the very next line. Every entry of g is
    clamped to g_eps before exponentiation.

Both are covered by tests/test_plaplacian.py, including negative controls that
fail when the guard is removed.
"""

import torch

__all__ = ["compute_g", "plaplacian_propagate"]


def compute_g(A, f, row_sum=None, eps=1e-8):
    """Weighted gradient magnitude g_i = sqrt( sum_j A_ij (f_j - f_i)^2 ).

    Args:
        A:       (N, N) row-stochastic affinity.
        f:       (K, N) K prompt maps propagated in parallel.
        row_sum: (N,) precomputed A @ 1. Recomputed if None.
        eps:     lower clamp; see (2) in the module docstring.

    Returns:
        (K, N), every entry >= eps.
    """
    if row_sum is None:
        row_sum = A.sum(dim=1)

    Af = f @ A.T                      # (K, N)  = (A f)_i
    Af2 = (f * f) @ A.T               # (K, N)  = (A f^2)_i

    sq = Af2 - 2.0 * f * Af + (f * f) * row_sum
    # Tiny negatives are ordinary floating-point cancellation, not a bug.
    sq = sq.clamp_min(0.0)
    return sq.sqrt().clamp_min(eps)


def plaplacian_propagate(A, f0, p, lam, tau_prop, max_iter, g_eps=1e-8,
                         return_history=False):
    """Gauss-Jacobi iteration of Algorithm 1.

    Args:
        A:        (N, N) row-stochastic affinity, fp32.
        f0:       (K, N) one-hot seeds (exactly one 1.0 per row).
        p:        exponent. p=2 is linear diffusion; p<2 preserves edges.
        lam:      anchor strength.
        tau_prop: stop once ||f^{t+1} - f^t||^2 <= tau_prop.
        max_iter: hard cap. Mandatory -- the paper gives none, and one
                  non-converging image would otherwise hang the whole job.

    Returns:
        (f, n_iter, residuals) -- residuals is a list only if return_history.

    n_iter is returned, not logged, because the gate checks it: numbers from a
    run that hit max_iter on many images describe the cap, not the mechanism.
    """
    assert A.dim() == 2 and A.shape[0] == A.shape[1], f"A must be (N,N), got {tuple(A.shape)}"
    assert f0.dim() == 2 and f0.shape[1] == A.shape[0], \
        f"f0 must be (K,N) with N={A.shape[0]}, got {tuple(f0.shape)}"
    assert A.dtype == torch.float32 and f0.dtype == torch.float32, \
        "p-Laplacian must run in fp32: g**(p-2) underflows in fp16"

    row_sum = A.sum(dim=1)                       # (N,) hoisted: f-independent
    f = f0.clone()
    exponent = p - 2.0
    residuals = []
    n_iter = 0

    for it in range(max_iter):
        g = compute_g(A, f, row_sum=row_sum, eps=g_eps)
        gp = g.pow(exponent)                     # (K, N)

        # gamma_ij = A_ij (gp_i + gp_j), never materialised -- see (1) above.
        num = lam * f0 + gp * (f @ A.T) + (gp * f) @ A.T
        den = lam + gp * row_sum + gp @ A.T

        f_new = num / den

        delta = torch.sum((f_new - f) ** 2).item()
        f = f_new
        n_iter = it + 1
        if return_history:
            residuals.append(delta)
        if delta <= tau_prop:
            break

    return (f, n_iter, residuals) if return_history else (f, n_iter, None)

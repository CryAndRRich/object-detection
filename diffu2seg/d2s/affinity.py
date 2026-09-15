r"""(h, w, h, w) attention tensor -> the (N, N) graph p-Laplacian propagates on.

                    WHAT COMES OUT OF THE AGGREGATOR

`extract_attention` returns a 4-D tensor whose last two axes sum to 1 for every
(i, j). Flattened to (N, N) with N = h*w, that is exactly a ROW-STOCHASTIC
matrix -- each row a probability distribution over the other tokens. Verified
numerically: every row sums to 1.0.

That property is what the whole method rests on. A[i, j] is not a similarity
score to be thresholded, it is "how much of token i's attention goes to j", and
a matrix of those is a weighted graph over the image.

                            WHAT IS *NOT* DONE HERE

NO IPF. M2N2 runs iterative proportional fitting to make A doubly stochastic,
because its Markov chain p_t = p_0 A^t converges to a stationary distribution
that depends on A, and it needs that limit to be the same (uniform) for every
image before "time to reach threshold" is comparable across images.

The p-Laplacian has no such requirement: its energy has an anchor term lam*f0
that fixes the solution near its seed, so there is no drift to a
stationary distribution and nothing to normalise away. Running IPF here would be
wasted work at best -- 15 sweeps over a matrix that is 0.07 GB at r=64 -- and a
silent distortion of the graph at worst.

NO SYMMETRISATION BY DEFAULT. Attention is genuinely directional and the energy
does not require A to be symmetric. The option exists for experiments; it is off.
"""

import torch

__all__ = ["to_affinity", "blend_timesteps", "change_temperature"]


def change_temperature(x, temperature, dim=-1):
    """softmax(log(x) / T) -- sharpen (T<1) or flatten (T>1) a distribution.

    Adapted from M2N2's utils.change_temperature. The clamp is an addition:
    SD2's attention contains exact zeros, and log(0) = -inf propagates into
    nan through the softmax.
    """
    x = x.clamp_min(torch.finfo(x.dtype).tiny)
    return torch.softmax(torch.log(x) / temperature, dim=dim)


def to_affinity(attn_hwhw, tau_att=0.55, symmetrize=False, dtype=torch.float32):
    """(h, w, h, w) -> (N, N) row-stochastic affinity, N = h*w.

    tau_att < 1 sharpens: attention mass concentrates on the strongest links, so
    propagation crosses object boundaries less readily. 0.55 is the paper's.
    """
    assert attn_hwhw.dim() == 4, f"expected (h,w,h,w), got {tuple(attn_hwhw.shape)}"
    h, w, h2, w2 = attn_hwhw.shape
    assert (h, w) == (h2, w2), "attention tensor must be square in both pairs"

    A = attn_hwhw.to(dtype).reshape(h * w, h * w)

    if tau_att is not None and tau_att != 1.0:
        A = change_temperature(A, tau_att, dim=-1)

    if symmetrize:
        A = 0.5 * (A + A.T)

    # Renormalise regardless: temperature and symmetrisation both break the
    # row-sum, and the maximum-principle bound f in [0,1] depends on it.
    A = A / A.sum(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
    return A


def blend_timesteps(attns, weights):
    """Weighted mean of several (h, w, h, w) tensors, renormalised per row.

    The paper mixes two UNet layers at w1=0.85 / w2=0.15; the same mechanism
    serves for mixing timesteps. STAGE 1 PASSES A SINGLE TIMESTEP -- blending is
    a second variable, and the project rule is one variable per step.
    """
    assert len(attns) == len(weights), "one weight per attention tensor"
    assert abs(sum(weights) - 1.0) < 1e-9, f"weights must sum to 1, got {sum(weights)}"

    out = attns[0] * weights[0]
    for a, wgt in zip(attns[1:], weights[1:]):
        out = out + a * wgt

    h, w = out.shape[0], out.shape[1]
    denom = out.reshape(h, w, -1).sum(dim=2)[:, :, None, None]
    return out / denom.clamp_min(torch.finfo(out.dtype).tiny)

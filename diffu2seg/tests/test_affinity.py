"""Affinity: row-stochasticity, temperature, and the absence of IPF.

Run:  python -m pytest tests/test_affinity.py -q
      python tests/test_affinity.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d2s.affinity import blend_timesteps, change_temperature, to_affinity  # noqa: E402

H = W = 6
N = H * W


def _fake_attn(seed=0):
    """(h, w, h, w) with the last two axes summing to 1, like SD2's output."""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(H, W, H, W, generator=g, dtype=torch.float32) + 1e-4
    return a / a.reshape(H, W, -1).sum(dim=2)[:, :, None, None]


def test_reshape_gives_row_stochastic_matrix():
    """The property everything downstream assumes.

    Normalising the last two axes of (h,w,h,w) is the same as making each row of
    the flattened (N,N) sum to 1 -- so the tensor is already a transition matrix
    and needs no extra normalisation to be a graph.
    """
    A = to_affinity(_fake_attn(), tau_att=None)
    assert A.shape == (N, N)
    assert torch.allclose(A.sum(dim=1), torch.ones(N), atol=1e-6)
    assert (A >= 0).all()


def test_row_stochastic_after_temperature():
    """Temperature re-runs a softmax, so rows must be renormalised, and are."""
    A = to_affinity(_fake_attn(1), tau_att=0.55)
    assert torch.allclose(A.sum(dim=1), torch.ones(N), atol=1e-6)


def test_row_stochastic_after_symmetrize():
    A = to_affinity(_fake_attn(2), tau_att=0.55, symmetrize=True)
    assert torch.allclose(A.sum(dim=1), torch.ones(N), atol=1e-6)


def test_temperature_below_one_sharpens():
    """tau < 1 concentrates mass: the largest entry of a row grows."""
    attn = _fake_attn(3)
    plain = to_affinity(attn, tau_att=None)
    sharp = to_affinity(attn, tau_att=0.55)
    assert sharp.max(dim=1).values.mean() > plain.max(dim=1).values.mean()


def test_temperature_above_one_flattens():
    attn = _fake_attn(4)
    plain = to_affinity(attn, tau_att=None)
    flat = to_affinity(attn, tau_att=2.0)
    assert flat.max(dim=1).values.mean() < plain.max(dim=1).values.mean()


def test_zero_entries_do_not_produce_nan():
    """SD2 attention contains exact zeros and log(0) = -inf.

    NEGATIVE CONTROL for the clamp inside change_temperature.
    """
    x = torch.zeros(3, 4)
    x[:, 0] = 1.0
    out = change_temperature(x, 0.55)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.sum(dim=1), torch.ones(3), atol=1e-6)


def test_blend_is_a_weighted_mean():
    a, b = _fake_attn(5), _fake_attn(6)
    out = blend_timesteps([a, b], [0.85, 0.15])
    assert torch.allclose(out.reshape(H, W, -1).sum(dim=2), torch.ones(H, W), atol=1e-6)

    same = blend_timesteps([a], [1.0])
    assert torch.allclose(same, a, atol=1e-6)


def test_blend_rejects_weights_that_do_not_sum_to_one():
    try:
        blend_timesteps([_fake_attn(7), _fake_attn(8)], [0.5, 0.2])
    except AssertionError:
        return
    raise AssertionError("weights not summing to 1 must be rejected")


def test_output_is_fp32():
    """p-Laplacian rejects fp16: g**(p-2) has a negative exponent."""
    assert to_affinity(_fake_attn(9).half(), tau_att=0.55).dtype == torch.float32


def test_rejects_non_square_tensor():
    try:
        to_affinity(torch.rand(4, 5, 4, 4))
    except AssertionError:
        return
    raise AssertionError("mismatched (h,w) pairs must be rejected")


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

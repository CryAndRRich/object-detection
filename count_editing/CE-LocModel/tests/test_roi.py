"""EXPERIMENT B — RoI feature sampling. Invariants, not smoke tests.

The one that matters most is `test_B_equals_A_at_step_zero`: if it fails, the
A-vs-B comparison silently changes two things at once and any conclusion drawn
from it is worthless.
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.box_transformer import BoxTransformer  # noqa: E402
from models.roi_sampler import RoIFeatureSampler, box_grid_points  # noqa: E402


def test_sample_points_lie_inside_their_box():
    boxes = torch.tensor([[[0.5, 0.5, 0.20, 0.10],
                           [0.1, 0.8, 0.05, 0.05],
                           [0.9, 0.2, 0.40, 0.30]]])
    pts = box_grid_points(boxes, 3)
    for i in range(boxes.shape[1]):
        cx, cy, w, h = boxes[0, i]
        x, y = pts[0, i, :, 0], pts[0, i, :, 1]
        assert (x >= cx - w / 2 - 1e-6).all() and (x <= cx + w / 2 + 1e-6).all()
        assert (y >= cy - h / 2 - 1e-6).all() and (y <= cy + h / 2 + 1e-6).all()


def test_sample_points_SCALE_with_box_size():
    """The whole point of experiment B.

    Sampling only the centre gives an identical vector however large the box is,
    so it carries no size information -- measured AUC 0.000 at telling a correct
    box from one twice too big, versus 0.896 for a 3x3 grid. Boxes at a
    near-constant size were exactly experiment A's failure, so if the points ever
    stop scaling with (w,h), B degenerates back into A.
    """
    small = torch.tensor([[[0.5, 0.5, 0.1, 0.1]]])
    big = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]])
    ps, pb = box_grid_points(small, 3), box_grid_points(big, 3)
    spread_s = (ps[0, 0].max(0).values - ps[0, 0].min(0).values)
    spread_b = (pb[0, 0].max(0).values - pb[0, 0].min(0).values)
    assert torch.allclose(spread_b / spread_s, torch.tensor([4.0, 4.0]), atol=1e-4)


def test_k1_carries_NO_size_information():
    """[NEGATIVE CONTROL] Proves the previous test is testing something real:
    with k=1 the sample points do NOT move when the box grows."""
    small = torch.tensor([[[0.5, 0.5, 0.1, 0.1]]])
    big = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]])
    assert torch.equal(box_grid_points(small, 1), box_grid_points(big, 1))


def test_output_is_exactly_zero_at_init():
    s = RoIFeatureSampler(64, 32, 3)
    out = s(torch.randn(2, 64, 64), torch.rand(2, 10, 4).clamp(0.1, 0.9))
    assert out.abs().max().item() == 0.0
    assert s.branch_norm() == 0.0


def test_zero_init_still_receives_gradient():
    """A zero-initialised LAYER is not a frozen one: for y = Wx + b with W = 0,
    dL/dW = delta * x^T is non-zero, so it starts learning on the first step. (A
    zero-initialised NETWORK would be stuck by symmetry -- this is not that.)"""
    s = RoIFeatureSampler(64, 32, 3)
    out = s(torch.randn(2, 64, 64), torch.rand(2, 10, 4).clamp(0.1, 0.9))
    out.pow(2).sum().add(out.sum()).backward()
    assert s.out.weight.grad.abs().max() > 0

    opt = torch.optim.SGD(s.parameters(), lr=0.1)
    pt, bx = torch.randn(2, 64, 64), torch.rand(2, 10, 4).clamp(0.1, 0.9)
    for _ in range(3):
        loss = (s(pt, bx) - 1.0).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert s.branch_norm() > 0, "still pinned at zero after 3 steps"


def _pair(d=64, roi_dim=48, k=3):
    torch.manual_seed(0)
    a = BoxTransformer(d_model=d, n_layer=2, n_head=4, dropout=0.0).eval()
    torch.manual_seed(0)
    b = BoxTransformer(d_model=d, n_layer=2, n_head=4, dropout=0.0,
                       roi_k=k, roi_dim=roi_dim).eval()
    return a, b


def test_B_equals_A_at_step_zero():
    """THE critical invariant. B must start as a bit-exact copy of A.

    This also guards the construction ORDER: RoIFeatureSampler is built last, so
    it does not consume RNG draws that would shift box_head/score_head. Building
    it earlier makes this fail with a difference of ~0.4 in the boxes -- not
    because zero-init broke, but because the two models no longer start from the
    same weights, which would quietly turn a one-variable comparison into a
    two-variable one.
    """
    a, b = _pair()
    boxes = torch.rand(2, 20, 4).clamp(0.05, 0.95)
    t = torch.randint(0, 1000, (2,))
    mem = torch.randn(2, 50, 64)
    praw = torch.randn(2, 64, 48)
    with torch.no_grad():
        pa, la = a(boxes, t, mem)
        pb, lb = b(boxes, t, mem, patch_raw=praw)
    assert (pa - pb).abs().max().item() == 0.0
    assert (la - lb).abs().max().item() == 0.0


def test_roi_branch_changes_output_once_trained():
    """[NEGATIVE CONTROL for the test above] Once the branch has weights, B must
    STOP matching A -- otherwise the previous test would pass on a model where the
    RoI features are wired up to nothing."""
    a, b = _pair()
    torch.nn.init.normal_(b.roi.out.weight, std=0.02)
    boxes = torch.rand(2, 20, 4).clamp(0.05, 0.95)
    t = torch.randint(0, 1000, (2,))
    mem, praw = torch.randn(2, 50, 64), torch.randn(2, 64, 48)
    with torch.no_grad():
        pa, _ = a(boxes, t, mem)
        pb, _ = b(boxes, t, mem, patch_raw=praw)
    assert (pa - pb).abs().max() > 1e-4


def test_permutation_equivariance_holds_with_roi():
    """Boxes stay a SET, not a sequence, with the RoI branch active."""
    _, b = _pair()
    torch.nn.init.normal_(b.roi.out.weight, std=0.02)
    boxes = torch.rand(2, 20, 4).clamp(0.05, 0.95)
    t = torch.randint(0, 1000, (2,))
    mem, praw = torch.randn(2, 50, 64), torch.randn(2, 64, 48)
    with torch.no_grad():
        p1, l1 = b(boxes, t, mem, patch_raw=praw)
        perm = torch.randperm(boxes.shape[1])
        p2, l2 = b(boxes[:, perm], t, mem, patch_raw=praw)
    assert (p1[:, perm] - p2).abs().max() < 1e-5
    assert (l1[:, perm] - l2).abs().max() < 1e-5


def test_missing_patch_raw_fails_loudly():
    """Silently running B without patch features would look like a weak result
    rather than a bug, so it must raise."""
    _, b = _pair()
    with pytest.raises(ValueError, match="patch_raw"):
        b(torch.rand(1, 5, 4), torch.randint(0, 1000, (1,)), torch.randn(1, 10, 64))


def test_boxes_at_different_places_get_different_roi_features():
    """The signal experiment A could not extract: two boxes over different parts
    of the image must read different features."""
    s = RoIFeatureSampler(48, 32, 3)
    torch.nn.init.normal_(s.out.weight, std=0.05)
    g = 16
    fmap = torch.randn(1, g * g, 48)
    left = torch.tensor([[[0.15, 0.15, 0.1, 0.1]]])
    right = torch.tensor([[[0.85, 0.85, 0.1, 0.1]]])
    with torch.no_grad():
        assert (s(fmap, left) - s(fmap, right)).abs().max() > 1e-4


def test_same_centre_different_size_gives_different_features():
    """And the signal that fixes the size error: same centre, different (w,h)."""
    s = RoIFeatureSampler(48, 32, 3)
    torch.nn.init.normal_(s.out.weight, std=0.05)
    fmap = torch.randn(1, 256, 48)
    small = torch.tensor([[[0.5, 0.5, 0.08, 0.08]]])
    large = torch.tensor([[[0.5, 0.5, 0.40, 0.40]]])
    with torch.no_grad():
        assert (s(fmap, small) - s(fmap, large)).abs().max() > 1e-4


def test_parameter_count_matches_the_design_choice():
    """'Project first, then mix' was chosen over one big Linear to save
    parameters while keeping centre and edges distinguishable."""
    s = RoIFeatureSampler(768, 256, 3)
    n = sum(p.numel() for p in s.parameters())
    assert n == (768 * 256 + 256) + (9 * 256 * 256 + 256)
    assert n < 1_000_000, f"{n} -- the single-Linear variant would be 1.77M"

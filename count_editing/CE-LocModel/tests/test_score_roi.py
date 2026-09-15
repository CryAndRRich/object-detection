"""EXPERIMENT E1 — the score head reads the RoI feature of the PREDICTED box.

E1 must be a ONE-VARIABLE change against A: the box path is untouched, only the
score head's input changes. The failure mode worth guarding is not a crash -- it is
a run that trains fine, produces plausible boxes, and answers a DIFFERENT question:

  * the RoI feature never reaches the score head   -> E1 is silently A
  * the RoI feature is also added to `tgt`         -> E1 is silently B+E1
  * the score gradient reaches box_head            -> boxes move to suit scoring,
                                                      `oracle_recall` shifts, and
                                                      the comparison with A is void
  * score_roi combined with refine_rounds          -> silently ignored (the refine
                                                      branch returns first)

Every one of those keeps the loss decreasing, so only an explicit test catches it.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.box_transformer import BoxTransformer  # noqa: E402

D, ROI_DIM, K = 64, 48, 3
KW = dict(d_model=D, n_layer=2, n_head=4, dropout=0.0)


def _mk(**kw):
    """Build with a FIXED seed so two models differing only in flags start from the
    same weights -- otherwise 'bit-identical' tests compare noise."""
    torch.manual_seed(0)
    return BoxTransformer(**KW, roi_dim=ROI_DIM, **kw).eval()


def _batch(b=2, n=10, m=20, p=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    return dict(
        boxes=torch.rand(b, n, 4, generator=g).clamp(0.1, 0.9),
        t=torch.randint(0, 1000, (b,), generator=g),
        mem=torch.randn(b, m, D, generator=g),
        praw=torch.randn(b, p, ROI_DIM, generator=g),
    )


# --------------------------------------------------------------------------
# THE invariant: E1's boxes are A's boxes.
# --------------------------------------------------------------------------

def test_E1_boxes_are_bit_exact_to_A():
    """The whole design rests on this. A/B/C1 measured oracle_recall 0.1197/0.1201/
    0.1384; E1 must reproduce A's 0.1197 EXACTLY, because it computes boxes through
    the identical path. If this fails, any coverage difference reported later is a
    bug, not a finding."""
    a, e = _mk(), _mk(roi_k=K, roi_to_tgt=False, score_roi=True)
    d = _batch()
    with torch.no_grad():
        box_a, _ = a(d["boxes"], d["t"], d["mem"])
        box_e, _ = e(d["boxes"], d["t"], d["mem"], patch_raw=d["praw"])
    assert (box_a - box_e).abs().max().item() == 0.0


def test_E1_logits_are_NOT_A_logits():
    """NEGATIVE CONTROL for the test above. If the RoI branch were never wired in,
    E1 would equal A in BOTH outputs and the bit-exact test would pass vacuously.
    The scores must differ -- E1's are zero-initialised, A's come from a random
    Linear -- which proves a different head produced them."""
    a, e = _mk(), _mk(roi_k=K, roi_to_tgt=False, score_roi=True)
    d = _batch()
    with torch.no_grad():
        _, lg_a = a(d["boxes"], d["t"], d["mem"])
        _, lg_e = e(d["boxes"], d["t"], d["mem"], patch_raw=d["praw"])
    assert not torch.allclose(lg_a, lg_e)


def test_E1_logits_start_nonzero_because_the_head_is_NOT_zero_init():
    """`score_from_roi` is deliberately randomly initialised -- zeroing it would
    deadlock the RoI branch (see test_roi_branch_actually_LEARNS...). So E1's
    scores are NOT zero and NOT A's at step 0; only the BOXES are bit-exact to A.
    Documented as a test so nobody 'restores symmetry' by zeroing this head."""
    e = _mk(roi_k=K, roi_to_tgt=False, score_roi=True)
    d = _batch()
    with torch.no_grad():
        _, lg = e(d["boxes"], d["t"], d["mem"], patch_raw=d["praw"])
    assert lg.abs().max().item() > 0.0
    # ...but the RoI CONTRIBUTION is zero at step 0, because roi.out is zeroed.
    # That is what keeps the initial score a pure function of the decoder token,
    # i.e. E1 starts from A's score behaviour and grows the RoI path from there.
    assert e.roi.branch_norm() == 0.0


# --------------------------------------------------------------------------
# detach(): the score must not be able to move the box.
# --------------------------------------------------------------------------

def test_score_gradient_does_NOT_reach_the_box_path():
    """Without .detach() the score loss flows through the sampling coordinates into
    box_head, and the network learns to MOVE BOXES where scoring is easy. That is
    the score<->coordinate feedback loop measured in
    docs/bai-hoc-ce-loc-detection.md §4 (label stability ~55 %)."""
    e = _mk(roi_k=K, roi_to_tgt=False, score_roi=True)
    e.train()
    # UN-ZERO both the head and the RoI projection first. With the shipped
    # zero-init, dL/d(box) is exactly 0 whether or not .detach() is there, so the
    # assertion below would pass on BROKEN code -- verified by deleting .detach()
    # and watching this test still go green. The weights must be real for the
    # gradient path to exist at all.
    with torch.no_grad():
        e.score_from_roi.weight.normal_(0.0, 0.05)
        e.roi.out.weight.normal_(0.0, 0.05)
        e.roi.proj_point.weight.normal_(0.0, 0.05)

    d = _batch()
    _, lg = e(d["boxes"], d["t"], d["mem"], patch_raw=d["praw"])
    lg.sum().backward()

    assert e.box_head.weight.grad is None or e.box_head.weight.grad.abs().max() == 0, \
        "score loss reached box_head -- the .detach() on the predicted box is missing"
    # and the RoI branch itself MUST receive gradient, else nothing is learned
    assert e.score_from_roi.weight.grad is not None
    assert e.score_from_roi.weight.grad.abs().max() > 0


def test_box_loss_DOES_reach_the_box_path():
    """Negative control for the test above: detach must not have frozen the box
    path wholesale. A loss on the BOXES still has to train box_head."""
    e = _mk(roi_k=K, roi_to_tgt=False, score_roi=True)
    e.train()
    d = _batch()
    box, _ = e(d["boxes"], d["t"], d["mem"], patch_raw=d["praw"])
    box.sum().backward()
    assert e.box_head.weight.grad.abs().max() > 0


# --------------------------------------------------------------------------
# E1 is not B.
# --------------------------------------------------------------------------

def test_roi_to_tgt_false_leaves_the_decoder_input_untouched():
    """B adds the RoI feature to `tgt`; E1 must not. Compared through the BOXES,
    which is what `tgt` determines."""
    e = _mk(roi_k=K, roi_to_tgt=False, score_roi=True)
    b = _mk(roi_k=K, roi_to_tgt=True, score_roi=False)
    d = _batch()
    with torch.no_grad():
        box_e, _ = e(d["boxes"], d["t"], d["mem"], patch_raw=d["praw"])
        box_b, _ = b(d["boxes"], d["t"], d["mem"], patch_raw=d["praw"])
    # B's RoI branch is zero-initialised too, so at step 0 both equal A. Give B's
    # branch a real weight to prove the flag is what separates them.
    with torch.no_grad():
        b.roi.out.weight.normal_(0.0, 0.05)
        box_b2, _ = b(d["boxes"], d["t"], d["mem"], patch_raw=d["praw"])
        e.roi.out.weight.copy_(b.roi.out.weight)
        box_e2, _ = e(d["boxes"], d["t"], d["mem"], patch_raw=d["praw"])
    assert not torch.allclose(box_b, box_b2), "B's flag did nothing"
    assert torch.allclose(box_e, box_e2, atol=0), \
        "E1's boxes changed when the RoI weights changed -> RoI is leaking into tgt"


def test_B_defaults_are_unchanged():
    """Every pre-E1 config (A, B, A.1, A.2, C1) must keep its meaning: roi_k>0 with
    no flags given is still EXPERIMENT B."""
    b = _mk(roi_k=K)
    assert b.roi_to_tgt is True and b.score_roi is False
    assert b.score_from_roi is None


# --------------------------------------------------------------------------
# The score head must see the PREDICTED box, not the input box.
# --------------------------------------------------------------------------

def test_score_reads_the_PREDICTED_box_not_x_t():
    """D samples RoI at the CURRENT stage's box; B sampled at the noisy input box.
    E1 must sample at what box_head produced. Verified by forcing box_head to a
    constant: the logits must then stop depending on the input boxes entirely."""
    e = _mk(roi_k=K, roi_to_tgt=False, score_roi=True)
    with torch.no_grad():
        # Un-zero the WHOLE RoI path, not just the head. `roi.out` ships
        # zero-initialised, so leaving it makes `rf` identically 0 and the
        # comparison below is 0 == 0 -- it passed even when the code sampled x_t.
        # Verified by patching the source to sample boxes_norm and watching this
        # test stay green before the fix.
        e.score_from_roi.weight.normal_(0.0, 0.05)
        e.roi.out.weight.normal_(0.0, 0.05)
        e.roi.proj_point.weight.normal_(0.0, 0.05)
        # make box_head output constant -> predicted box identical for every slot
        e.box_head.weight.zero_()
        e.box_head.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 0.0]))

    d1 = _batch(seed=1)
    d2 = dict(d1)
    d2["boxes"] = _batch(seed=99)["boxes"]           # totally different input boxes
    with torch.no_grad():
        _, lg1 = e(d1["boxes"], d1["t"], d1["mem"], patch_raw=d1["praw"])
        _, lg2 = e(d2["boxes"], d2["t"], d2["mem"], patch_raw=d2["praw"])
    # The decoder token still differs (tgt encodes the input box), so logits are not
    # required to be equal. What must hold is that the ROI HALF is identical: check
    # by zeroing the token half of the head.
    with torch.no_grad():
        e.score_from_roi.weight[:, :D].zero_()       # keep only the RoI half
        _, r1 = e(d1["boxes"], d1["t"], d1["mem"], patch_raw=d1["praw"])
        _, r2 = e(d2["boxes"], d2["t"], d2["mem"], patch_raw=d2["praw"])
    assert torch.allclose(r1, r2, atol=1e-6), \
        "the RoI half of the score depends on x_t -> it is sampling the INPUT box"


def test_roi_half_of_the_head_actually_matters():
    """Negative control: if the RoI half contributed nothing, the test above would
    pass trivially (0 == 0)."""
    e = _mk(roi_k=K, roi_to_tgt=False, score_roi=True)
    d = _batch()
    with torch.no_grad():
        e.score_from_roi.weight.normal_(0.0, 0.05)
        e.roi.out.weight.normal_(0.0, 0.05)
        _, full = e(d["boxes"], d["t"], d["mem"], patch_raw=d["praw"])
        e.score_from_roi.weight[:, D:].zero_()       # drop the RoI half
        _, tok_only = e(d["boxes"], d["t"], d["mem"], patch_raw=d["praw"])
    assert not torch.allclose(full, tok_only), "the RoI half is inert"


# --------------------------------------------------------------------------
# Refusals: combinations that would train happily and measure nothing.
# --------------------------------------------------------------------------

def test_score_roi_without_roi_k_is_refused():
    with pytest.raises(ValueError, match="roi_k"):
        BoxTransformer(**KW, roi_dim=ROI_DIM, score_roi=True)


def test_score_roi_with_refine_is_refused():
    """`forward` returns from the refine branch BEFORE the E1 head, so this would
    silently be plain C1 while the log says E1."""
    with pytest.raises(ValueError, match="refine_rounds"):
        BoxTransformer(d_model=D, n_layer=2, n_head=4, dropout=0.0, roi_dim=ROI_DIM,
                       roi_k=K, score_roi=True, refine_rounds=2)


def test_missing_patch_raw_raises_instead_of_silently_skipping():
    e = _mk(roi_k=K, roi_to_tgt=False, score_roi=True)
    d = _batch()
    with pytest.raises(ValueError, match="patch_raw"):
        e(d["boxes"], d["t"], d["mem"])


# --------------------------------------------------------------------------
# Shapes and multi-class.
# --------------------------------------------------------------------------

def test_shapes_1d_and_multiclass():
    d = _batch()
    e1 = _mk(roi_k=K, roi_to_tgt=False, score_roi=True)
    with torch.no_grad():
        box, lg = e1(d["boxes"], d["t"], d["mem"], patch_raw=d["praw"])
    assert box.shape == (2, 10, 4) and lg.shape == (2, 10)

    torch.manual_seed(0)
    e80 = BoxTransformer(**KW, roi_dim=ROI_DIM, roi_k=K, roi_to_tgt=False,
                         score_roi=True, n_class=80).eval()
    with torch.no_grad():
        _, lg80 = e80(d["boxes"], d["t"], d["mem"], patch_raw=d["praw"])
    assert lg80.shape == (2, 10, 80)


def test_boxes_stay_in_range():
    e = _mk(roi_k=K, roi_to_tgt=False, score_roi=True)
    d = _batch()
    with torch.no_grad():
        e.score_from_roi.weight.normal_(0.0, 0.5)
        box, _ = e(d["boxes"], d["t"], d["mem"], patch_raw=d["praw"])
    assert box.min() >= 0.0 and box.max() <= 1.0


def test_head_is_built_last_so_RNG_order_is_preserved():
    """`nn.Linear.__init__` consumes RNG draws even when zeroed right after. If
    score_from_roi were constructed before box_head, E1 and A would start from
    DIFFERENT weights and test_E1_boxes_are_bit_exact_to_A would fail for a reason
    unrelated to the experiment."""
    a, e = _mk(), _mk(roi_k=K, roi_to_tgt=False, score_roi=True)
    for name in ("box_head", "score_head", "box_proj"):
        wa = getattr(a, name).weight
        we = getattr(e, name).weight
        assert torch.equal(wa, we), f"{name} differs -> construction order shifted RNG"


# --------------------------------------------------------------------------
# The zero-init deadlock. Found by running SGD, not by reading the code.
# --------------------------------------------------------------------------

def test_roi_branch_actually_LEARNS_no_zero_init_deadlock():
    """The bug this file exists for.

    `roi.out` ships zero-initialised (EXPERIMENT B's design, so that B == A at step
    0). If `score_from_roi` is ALSO zero-initialised, the two lock each other at
    zero forever:

        dL/d(score_from_roi.W[:, d_model:])  proportional to  rf              = 0
        dL/d(roi.out.weight)                 proportional to  W[:, d_model:]  = 0

    Neither can move until the other does. Measured before the fix: over 3 SGD
    steps the token half of the head moved to 0.0499 while the RoI half and
    roi.out stayed at exactly 0.00000000 -- so E1 would train as plain A with a
    constant 0.5 score, with the loss falling and nothing to warn anyone.

    Only ONE of the pair may be zero-initialised. This test pins that down.
    """
    e = _mk(roi_k=K, roi_to_tgt=False, score_roi=True)
    e.train()
    opt = torch.optim.SGD(e.parameters(), lr=0.01)
    d = _batch()
    for _ in range(3):
        _, lg = e(d["boxes"], d["t"], d["mem"], patch_raw=d["praw"])
        loss = (lg - 1.0).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    roi_half = e.score_from_roi.weight[:, D:].abs().max().item()
    assert roi_half > 0, ("the RoI half of the score head is pinned at zero -- "
                          "score_from_roi and roi.out are both zero-initialised")
    assert e.roi.branch_norm() > 0, ("roi.out never left zero -- the RoI features "
                                     "are multiplied by a dead branch")


def test_exactly_one_of_the_pair_is_zero_initialised():
    """States the contract directly, so a future change to either init has to
    confront it. `roi.out` zero / `score_from_roi` random is the shipped choice:
    roi.out's `branch_norm()` is logged every epoch and is the cheap early read on
    whether the RoI signal is being used at all."""
    e = _mk(roi_k=K, roi_to_tgt=False, score_roi=True)
    roi_out_zero = e.roi.out.weight.abs().max().item() == 0.0
    head_zero = e.score_from_roi.weight.abs().max().item() == 0.0
    assert roi_out_zero != head_zero, (
        f"roi.out zero={roi_out_zero}, score_from_roi zero={head_zero}: both zero "
        f"deadlocks, neither zero loses the 'starts as A' property somewhere")
    assert roi_out_zero, "roi.out is the one that should be zeroed (branch_norm log)"

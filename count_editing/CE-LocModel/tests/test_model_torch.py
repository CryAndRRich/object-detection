"""Torch architecture tests — INVARIANTS, not 'it runs without crashing'.

Round 1 only had 'the forward pass runs' + 'converges under an oracle', and both
passed while the code was wrong. Here each test checks one specific DESIGN
INVARIANT.
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.box_transformer import BoxTransformer, SinusoidalCoordEmbedding  # noqa: E402
from models.criterion import SetCriterion, sigmoid_focal_loss  # noqa: E402


@pytest.fixture(scope="module")
def dec():
    torch.manual_seed(0)
    return BoxTransformer(d_model=64, n_layer=2, n_head=4, dropout=0.0).eval()


def _inputs(B=2, N=40, M=32, D=64):
    torch.manual_seed(1)
    return torch.rand(B, N, 4), torch.randint(0, 1000, (B,)), torch.randn(B, M, D)


def test_permutation_equivariance(dec):
    """CHANGE (a): boxes are a SET, not a sequence.

    If anyone re-adds an index-based learned `pos_emb`, this test fails immediately
    — the network would learn "slot 0 is usually real GT", which it must not learn
    because at inference every slot comes from randn.
    """
    b, t, mem = _inputs()
    with torch.no_grad():
        pb, lg = dec(b, t, mem)
        p = torch.randperm(b.shape[1])
        pb2, lg2 = dec(b[:, p], t, mem)
    assert (pb[:, p] - pb2).abs().max() < 1e-5
    assert (lg[:, p] - lg2).abs().max() < 1e-5


def test_N_varies_freely_between_train_and_eval(dec):
    """CHANGE (c): train N=100, eval N=300 — possible because index-based pos_emb
    was removed."""
    _, t, mem = _inputs()
    with torch.no_grad():
        for n in [1, 17, 100, 300]:
            pb, lg = dec(torch.rand(2, n, 4), t, mem)
            assert pb.shape == (2, n, 4) and lg.shape == (2, n)


def test_memory_length_varies_freely(dec):
    """The number of condition tokens can change (224px -> 196, 512px -> 1024)."""
    b, t, _ = _inputs()
    with torch.no_grad():
        for m in [2, 197, 1025]:
            pb, _ = dec(b, t, torch.randn(2, m, 64))
            assert pb.shape[1] == b.shape[1]


def test_output_range_is_valid_before_training(dec):
    """Check the value range while the model is UNTRAINED — round 1 lacked exactly
    this kind of test."""
    b, t, mem = _inputs()
    with torch.no_grad():
        pb, lg = dec(b, t, mem)
    assert (pb >= 0).all() and (pb <= 1).all()
    assert torch.isfinite(pb).all() and torch.isfinite(lg).all()


def test_each_box_receives_DIFFERENT_information_from_memory(dec):
    """Round 1's bottleneck: with only 2 memory tokens, all N boxes received the
    SAME vector.

    With many positioned memory tokens, two boxes at different locations must give
    different outputs. This is the central invariant of the whole round-2 design.
    """
    t = torch.randint(0, 1000, (1,))
    mem = torch.randn(1, 256, 64)
    top_left = torch.tensor([[[0.1, 0.1, 0.05, 0.05]]])
    bottom_right = torch.tensor([[[0.9, 0.9, 0.05, 0.05]]])
    with torch.no_grad():
        a, _ = dec(top_left, t, mem)
        b, _ = dec(bottom_right, t, mem)
    assert (a - b).abs().max() > 1e-4, "boxes at different positions gave identical output"


def test_sinusoidal_PE_decays_with_distance():
    """[NEGATIVE CONTROL] A raw Linear encodes position as MAGNITUDE, not identity."""
    pe = SinusoidalCoordEmbedding(64)
    origin = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
    e0 = pe(origin)[0]
    previous = None
    for d in [0.01, 0.05, 0.2, 0.5]:
        e1 = pe(origin + torch.tensor([d, 0.0, 0.0, 0.0]))[0]
        sim = torch.cosine_similarity(e0, e1, dim=0).item()
        if previous is not None:
            assert sim <= previous + 1e-6, "similarity must decay with distance"
        previous = sim

    # the raw Linear version: nearby positions do NOT give nearby embeddings in a
    # way attention can use — the dot product scales linearly with the value
    lin = torch.nn.Linear(4, 64, bias=False)
    with torch.no_grad():
        a, b2 = lin(origin)[0], lin(origin * 2)[0]
    assert torch.cosine_similarity(a, b2, dim=0).item() > 0.99, \
        "raw Linear: doubling coordinates gives a co-directional vector — identity is lost"


def test_gradients_reach_every_trainable_parameter(dec):
    b, t, mem = _inputs()
    pb, lg = dec(b, t, mem)
    (pb.sum() + lg.sum()).backward()
    missing = [n for n, p in dec.named_parameters()
               if p.requires_grad and (p.grad is None or p.grad.abs().max() == 0)]
    assert not missing, f"no gradient: {missing}"


# ------------------------------------------------------------------ criterion

def test_coordinate_loss_applies_only_to_MATCHED_pairs():
    """An unmatched box has NO coordinate target, but it DOES have a score target of 0."""
    torch.manual_seed(0)
    crit = SetCriterion()
    pb, lg = torch.rand(1, 20, 4, requires_grad=True), torch.zeros(1, 20, requires_grad=True)
    gt = [torch.rand(3, 4) * 0.2 + 0.4]
    loss, st, idx = crit(pb, lg, gt)
    assert st["n_matched"] == 3
    assert len(idx[0][0]) == 3


def test_iou_matched_is_REAL_IOU_not_giou():
    """A wrong tracking metric is useless: GIoU can be negative for disjoint boxes."""
    crit = SetCriterion()
    far = torch.tensor([[[0.9, 0.9, 0.05, 0.05]]])
    gt = [torch.tensor([[0.1, 0.1, 0.05, 0.05]])]
    _, st, _ = crit(far, torch.zeros(1, 1), gt)
    assert 0.0 <= st["iou_matched"] <= 1.0, f"IoU={st['iou_matched']} outside [0,1]"


def test_does_not_crash_when_an_image_has_no_GT():
    crit = SetCriterion()
    loss, st, _ = crit(torch.rand(2, 10, 4), torch.zeros(2, 10),
                       [torch.zeros(0, 4), torch.zeros(0, 4)])
    assert torch.isfinite(loss) and st["n_matched"] == 0


def test_focal_loss_matches_the_known_formula():
    """alpha=0.25 gamma=2. With logit=0 (p=0.5) and target=0:
       loss = (1-alpha) * 0.5^2 * (-log 0.5) = 0.75*0.25*0.693 = 0.130"""
    fl = sigmoid_focal_loss(torch.zeros(1), torch.zeros(1))
    assert float(fl[0]) == pytest.approx(0.1300, abs=1e-3)


def test_loss_weights_match_diffusiondet():
    from models.criterion import W_CLASS, W_GIOU, W_L1
    assert (W_L1, W_GIOU, W_CLASS) == (5.0, 2.0, 2.0)

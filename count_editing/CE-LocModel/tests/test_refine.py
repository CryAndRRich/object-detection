"""EXPERIMENT C1 — six supervised refinement rounds.

C1 must be a ONE-VARIABLE change against A: same CLIP, same diffusion, same
matcher, same loss weights, same N, and the SAME six attention layers. Most of
these tests therefore assert that nothing else moved.

The failure mode worth guarding is not a crash. It is a run that trains fine,
produces valid boxes, and answers a different question than the one asked --
`anchor` silently confused with `box`, or inference reading the wrong round.
"""

import os
import sys

import pytest
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.box_transformer import BoxTransformer, update_box  # noqa: E402
from models.criterion import SetCriterion, loss_from_output  # noqa: E402

KW = dict(d_model=64, n_layer=3, n_head=4, dropout=0.0)


def make(rounds=0, seed=0):
    torch.manual_seed(seed)
    return BoxTransformer(**KW, refine_rounds=rounds).eval()


def batch(n=7, b=2, seed=1):
    torch.manual_seed(seed)
    return (torch.rand(b, n, 4) * 0.5 + 0.25,
            torch.zeros(b, dtype=torch.long),
            torch.randn(b, 11, KW["d_model"]))


# ------------------------------------------------------------- update_box

def test_zero_delta_returns_the_anchor_exactly():
    """The property that makes C1 step 0 bit-identical to A: exp(0)=1 and 0*w=0."""
    a = torch.tensor([[[0.3, 0.4, 0.069, 0.061]]])
    assert torch.equal(update_box(a, torch.zeros_like(a)), a)


def test_object_normalized_formula():
    """cx' = cx + d_cx*w and w' = w*exp(d_w) — V-DETR's, not a plain residual.
    The centre step is in units of the box's OWN size, so a tiny box moves gently."""
    a = torch.tensor([[[0.3, 0.4, 0.069, 0.061]]])
    r = update_box(a, torch.tensor([[[0.5, 0.0, 0.0, 0.0]]]))
    assert abs(float(r[0, 0, 0]) - (0.3 + 0.5 * 0.069)) < 1e-7
    r = update_box(a, torch.tensor([[[0.0, 0.0, 0.6931472, 0.0]]]))
    assert abs(float(r[0, 0, 2]) - 2 * 0.069) < 1e-5


def test_clamps_keep_boxes_valid():
    """exp() has no upper bound and the centre can be pushed anywhere; both must be
    contained or a single bad round produces boxes outside the image."""
    a = torch.tensor([[[0.3, 0.4, 0.069, 0.061]]])
    assert abs(float(update_box(a, torch.tensor([[[0., 0., -10., 0.]]]))[0, 0, 2])
               - 0.005) < 1e-6          # MIN_WH, the smallest real CE-130 box is 0.0059
    r = update_box(a, torch.tensor([[[99., 99., 0., 0.]]]))
    assert float(r[0, 0, 0]) == 1.0 and float(r[0, 0, 1]) == 1.0
    r = update_box(a, torch.tensor([[[-99., -99., 0., 0.]]]))
    assert float(r[0, 0, 0]) == 0.0 and float(r[0, 0, 1]) == 0.0


# --------------------------------------------------- one variable vs A

def test_C1_at_step_zero_is_bit_identical_to_A():
    """Zero-init on box_delta, and the refine modules built LAST so they do not
    consume RNG draws that would shift every earlier layer (`nn.Linear.__init__`
    consumes RNG even when the weights are zeroed immediately afterwards —
    verified — which is the same trap tests/test_roi.py locks for EXPERIMENT B)."""
    a, c = make(0), make(3)
    sa, sc = a.state_dict(), c.state_dict()
    shared = [k for k in sa if k in sc]
    assert len(shared) > 50
    assert all(torch.equal(sa[k], sc[k]) for k in shared)

    b, t, mem = batch()
    with torch.no_grad():
        box_a, _ = a(b, t, mem)
        outs = c(b, t, mem)
    # every round returns the anchor exactly at init
    assert all(torch.equal(o[0], b) for o in outs)
    assert box_a.shape == outs[-1][0].shape


def test_box_delta_is_zero_initialised_but_score_is_not():
    """Zeroing the score head too would make every score identical at step 0, which
    is a different (and worse) starting point, not a neutral one."""
    c = make(3)
    assert all(h.weight.abs().max() == 0 and h.bias.abs().max() == 0
               for h in c.box_delta)
    assert any(h.weight.abs().max() > 0 for h in c.round_score)


def test_refine_rounds_zero_keeps_the_old_path():
    b, t, mem = batch()
    with torch.no_grad():
        out = make(0)(b, t, mem)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[0].shape == (2, 7, 4) and out[1].shape == (2, 7)


def test_rounds_must_match_layers():
    """One read per EXISTING layer. C1 adds no attention layer, so any other number
    would silently mean something else."""
    with pytest.raises(ValueError, match="must equal n_layer"):
        BoxTransformer(**KW, refine_rounds=5)


# ------------------------------------- the silent bug: anchor vs box

def test_constant_delta_gives_IDENTICAL_rounds():
    """THE test that distinguishes a fixed anchor from a compounding one.

    `Delta = 0 -> every round equals x0` passes for BOTH implementations, so it
    cannot catch the bug on its own. With a constant NON-ZERO delta, regressing
    from a fixed anchor yields six identical boxes, while `box = update(box, d)`
    yields six different ones.
    """
    c = make(3)
    with torch.no_grad():
        for h in c.box_delta:
            torch.nn.init.constant_(h.weight, 0.0)
            torch.nn.init.constant_(h.bias, 0.05)
        outs = c(*batch())
    first = outs[0][0]
    assert all(torch.allclose(o[0], first, atol=1e-7) for o in outs)
    assert not torch.allclose(first, batch()[0])       # it did move


def test_each_round_uses_its_own_head():
    """Per-layer heads (V-DETR's `mlp_sep` defaults to True). Giving each head a
    different bias must produce a different box per round."""
    c = make(3)
    with torch.no_grad():
        for i, h in enumerate(c.box_delta):
            torch.nn.init.constant_(h.weight, 0.0)
            torch.nn.init.constant_(h.bias, 0.01 * (i + 1))
        outs = c(*batch())
    cx = [float(o[0][0, 0, 0]) for o in outs]
    assert len(set(cx)) == 3, f"rounds share a head: {cx}"


# ------------------------------------------------------------- criterion

def test_single_round_list_equals_the_old_path():
    """A one-element list must reproduce A/B's loss exactly, or the deep-supervision
    branch has changed what the loss means."""
    crit = SetCriterion("hungarian")
    torch.manual_seed(0)
    b, lg = torch.rand(2, 10, 4) * 0.5 + 0.25, torch.randn(2, 10)
    gt = [b[0, :2].clone(), b[1, :3].clone()]
    l1, s1, _ = crit(b, lg, gt)
    l2, s2, _ = crit([(b, lg)], gt)
    assert torch.allclose(l1, l2)
    for k in ("loss_l1", "loss_giou", "loss_ce", "iou_matched", "n_matched"):
        assert abs(s1[k] - s2[k]) < 1e-9


def test_rounds_loss_is_the_mean_and_keeps_per_round_stats():
    crit = SetCriterion("hungarian")
    torch.manual_seed(0)
    b, lg = torch.rand(2, 10, 4) * 0.5 + 0.25, torch.randn(2, 10)
    gt = [b[0, :2].clone(), b[1, :3].clone()]
    rounds = [(b, lg), (b * 0.99, lg), (b * 0.98, lg)]
    total, st, _ = crit(rounds, gt)
    singles = [float(crit(x, y, gt)[0]) for x, y in rounds]
    assert abs(float(total) - sum(singles) / 3) < 1e-6
    assert len(st["iou_matched_per_round"]) == 3
    # headline numbers describe the round inference actually uses
    assert abs(st["loss_final"] - singles[-1]) < 1e-6


def test_loss_from_output_handles_both_shapes():
    """Four tools were found unpacking `(boxes, logits)` unconditionally, which
    raises on C1 only when that tool is run — i.e. on the server."""
    crit = SetCriterion("hungarian")
    torch.manual_seed(0)
    b, lg = torch.rand(1, 8, 4) * 0.5 + 0.25, torch.randn(1, 8)
    gt = [b[0, :2].clone()]
    l1, _, _, g1 = loss_from_output(crit, (b, lg), gt)
    l2, _, _, g2 = loss_from_output(crit, [(b, lg)], gt)
    assert torch.allclose(l1, l2) and torch.equal(g1, g2)


# ------------------------------------------------------------- config

def test_c1_config_differs_from_A_in_one_key():
    with open("config/experiment_a.yaml") as f:
        a = yaml.safe_load(f)
    with open("config/experiment_c1.yaml") as f:
        c = yaml.safe_load(f)
    for s in ("data", "diffusion", "loss", "matcher", "eval"):
        assert a[s] == c[s], f"{s} differs"
    d = {"n_class": 1, "use_text": True, "roi_k": 0, "refine_rounds": 0}
    diff = {k for k in set(a["model"]) | set(c["model"])
            if a["model"].get(k, d.get(k)) != c["model"].get(k, d.get(k))}
    assert diff == {"refine_rounds"}
    assert c["model"]["refine_rounds"] == c["model"]["n_layer"]


# ------------------- the round the checkpoint is chosen on

def test_mean_and_final_loss_genuinely_differ():
    """Documents WHY selection must not use the mean.

    Round 1 regresses straight from the noisy anchor, so it is always the worst.
    Averaging it into the selection criterion means the checkpoint is picked on a
    number no forward pass ever produces.
    """
    crit = SetCriterion("hungarian")
    torch.manual_seed(0)
    gt = [torch.rand(6, 4) * 0.4 + 0.3]
    anchor = torch.rand(1, 50, 4) * 0.6 + 0.2
    good = torch.cat([gt[0], torch.rand(44, 4) * 0.4 + 0.3]).unsqueeze(0)
    rounds = [(anchor * (1 - (r + 1) / 6) + good * ((r + 1) / 6), torch.randn(1, 50))
              for r in range(6)]
    total, st, _ = crit(rounds, gt)
    assert st["loss_final"] < float(total) * 0.8, \
        f"mean {float(total):.3f} vs final {st['loss_final']:.3f} — expected a real gap"
    assert st["iou_matched_final"] > st["iou_matched"]


def test_selecting_on_the_mean_would_pick_the_wrong_checkpoint():
    """A concrete counter-example, so the rule is not just asserted but shown.

    Model X is strictly better at inference (final 2.0 vs 3.5) yet loses on the
    mean. Selecting on `loss_final` is what makes the saved checkpoint the good one.
    """
    X = [9.5, 7.0, 6.4, 5.6, 4.8, 2.0]
    Y = [4.0, 3.9, 3.8, 3.7, 3.6, 3.5]
    assert sum(Y) / 6 < sum(X) / 6            # the mean prefers Y ...
    assert X[-1] < Y[-1]                      # ... but X is what inference uses


def test_train_selects_and_reports_on_the_final_round():
    """Guards the four places that must agree on ONE quantity: console log,
    history entry, checkpoint selection, and val_rising_streak. They were found
    reading `val["loss"]` (the mean) while inference used the last round."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "train.py")).read()
    assert 'val_sel = val.get("loss_final", val["loss"])' in src
    assert "if val_sel < best:" in src
    assert '"select_loss": val_sel' in src
    # summary, rising streak AND the end-of-run print all read select_loss
    assert src.count('e.get("select_loss", e["val"]["loss"])') >= 3
    assert 'if val["loss"] < best' not in src

    # NOTHING may read the raw mean outside a deliberate fallback. The earlier
    # version of this test only banned one exact string, so it kept passing while
    # `min(history, key=lambda e: e["val"]["loss"])` still picked the epoch the
    # end-of-run summary printed -- a different epoch than best.pth actually held.
    # Counting is what closes that hole: every remaining occurrence must sit inside
    # a `.get("select_loss", ...)` fallback or a comment.
    import re
    for line in src.splitlines():
        code = line.split("#")[0]
        if '["val"]["loss"]' in code or re.search(r'\bval\["loss"\]', code):
            deliberate = ('select_loss' in code or 'loss_final' in code
                          # the one field whose NAME says it reports the mean
                          or 'mean_over_rounds' in code)
            assert deliberate, \
                f"reads the six-round mean outside a fallback: {line.strip()}"
    assert 'min(history, key=sel_of)' in src or 'min(history, key=sel)' in src


def test_run_val_keeps_per_round_stats():
    """`run_val` had a fixed 6-key whitelist, so every `*_final` and `*_per_round`
    value was dropped — leaving the val IoU curve, which is exactly what C1 must be
    read on, missing from history.json."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "train.py")).read()
    assert 'k.endswith("_final")' in src and 'k.endswith("_per_round")' in src
    assert '"val_iou_per_round": val.get("iou_matched_per_round")' in src


def test_eval_exposes_a_round_flag():
    """`--round` measures whether AP improves round by round; the per-round IoU in
    the training log cannot show that on its own."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "eval.py")).read()
    assert '"--round"' in src and "return_all_rounds=True" in src
    assert "_round{a.round}" in src        # separate output file, never overwrites


# ---------------------------------------------------------------------------
# EXPERIMENT C1b — one supervised round, SAME depth.
#
# C1's six rounds measured monotonically WORSE the further they went on every
# score-free metric (oracle_recall 0.1384 -> 0.1328, mean_bestIoU 0.2314 ->
# 0.2157, AP50 0.0168 -> 0.0151), while the loss they were selected on improved.
# C1b asks whether one round is as good as six -- but only if the ONLY thing that
# changes is the number of supervised rounds. If `refine_rounds=1` also truncated
# the decoder to one layer, a worse result would be explained by lost depth and
# the experiment would answer nothing.
# ---------------------------------------------------------------------------

def test_one_round_still_runs_every_layer():
    """The decoder must be as DEEP with refine_rounds=1 as with 0 or n_layer."""
    m = make(rounds=1)
    assert len(m.decoder.layers) == KW["n_layer"], (
        "refine_rounds=1 changed the decoder depth; then C1b vs C1 would compare "
        "one-round-shallow against six-round-deep and confound two variables")


def test_one_round_returns_exactly_one_pair():
    m = make(rounds=1)
    h = torch.randn(2, 5, KW["d_model"])
    mem = torch.randn(2, 7, KW["d_model"])
    anchor = torch.rand(2, 5, 4) * 0.5 + 0.25
    outs = m._forward_refine(h, mem, anchor)
    assert len(outs) == 1
    assert outs[0][0].shape == (2, 5, 4)


def test_one_round_reads_the_LAST_layer_not_the_first():
    """Reading after layer 0 and throwing the rest away would be a different (and
    much weaker) model than reading after the last layer.

    This CANNOT be tested on the output boxes: `box_delta` is zero-initialised on
    purpose (so C1 step 0 is bit-identical to A), which makes `update_box` return
    the anchor no matter which layer's features it is handed. Both a correct and a
    truncated implementation emit exactly the anchor, and the assertion would pass
    for both. So compare the FEATURES the head consumes, and give `box_delta` a
    non-zero weight before comparing boxes.
    """
    m = make(rounds=1)
    h = torch.randn(2, 5, KW["d_model"])
    mem = torch.randn(2, 7, KW["d_model"])
    anchor = torch.rand(2, 5, 4) * 0.5 + 0.25

    # make the head actually depend on its input, else everything returns `anchor`
    with torch.no_grad():
        m.box_delta[0].weight.normal_(0.0, 0.02)

    got = m._forward_refine(h, mem, anchor)[0][0]

    hh = h                                   # ... to the END (correct)
    for layer in m.decoder.layers:
        hh = layer(hh, mem)
    want_last = update_box(anchor, m.box_delta[0](m.ln_f(hh)))

    h1 = m.decoder.layers[0](h, mem)         # ... to layer 0 only (naive truncation)
    want_first = update_box(anchor, m.box_delta[0](m.ln_f(h1)))

    assert torch.allclose(got, want_last, atol=1e-6)
    # negative control: with a non-zero head the two are genuinely different, so
    # the assertion above has the power to fail on a truncated decoder
    assert not torch.allclose(want_last, want_first, atol=1e-6)


def test_zero_init_means_boxes_alone_CANNOT_detect_truncation():
    """Documents why the test above perturbs the head: at init every round returns
    the anchor exactly, so any box-only assertion is vacuous."""
    m = make(rounds=1)
    h = torch.randn(2, 5, KW["d_model"])
    mem = torch.randn(2, 7, KW["d_model"])
    anchor = torch.rand(2, 5, 4) * 0.5 + 0.25
    assert torch.allclose(m._forward_refine(h, mem, anchor)[0][0], anchor, atol=1e-7)


def test_one_round_head_count_matches_rounds_not_layers():
    m = make(rounds=1)
    assert len(m.box_delta) == 1 and len(m.round_score) == 1


def test_refine_rounds_between_2_and_nlayer_minus_1_is_still_rejected():
    """Only 0, 1 and n_layer are defined. A value like 3 would silently read boxes
    off layers 0,1,2 and drop the rest."""
    with pytest.raises(ValueError):
        BoxTransformer(**KW, refine_rounds=2)


# ---------------------------------------------------------------------------
# `run_val` must not build a graph.
#
# The decorator was lost once, by inserting a new @torch.no_grad() function
# directly ABOVE run_val: the decorator stayed with the newcomer and run_val was
# left bare. Nothing failed at import, nothing failed for 239 training batches --
# it crashed only at the first `.numpy()` inside validation, three minutes in,
# on the GPU. The cost is not only the crash: without no_grad the whole val pass
# holds its activations.
# ---------------------------------------------------------------------------

def test_run_val_is_wrapped_in_no_grad():
    """The decorator must sit on run_val itself. `torch.no_grad()` records the
    function it wraps in `__wrapped__`, so its absence means run_val is bare."""
    import train as train_mod
    assert getattr(train_mod.run_val, "__wrapped__", None) is not None, (
        "train.run_val is not wrapped — @torch.no_grad() is missing. Validation "
        "would build a graph and crash at .numpy() three minutes into training.")


def test_run_val_really_runs_without_grad():
    """End-to-end: run the actual run_val on a two-image fake loader with
    grad enabled outside, and assert it neither crashes at `.numpy()` nor
    returns anything carrying a graph."""
    import train as train_mod

    d, N, G = KW["d_model"], 4, 3

    class _Crit:
        def __call__(self, pb, lg, tg, labels=None):
            st = {"loss": 1.0, "loss_l1": 0.1, "loss_giou": 0.2, "loss_ce": 0.3,
                  "iou_matched": 0.4, "n_matched": 2.0}
            return None, st, None

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)          # a real trainable parameter
            self.decoder = type("D", (), {"roi": None, "score_roi": False})()
        def build_inputs(self, tg, N, valid_h, generator=None):
            return torch.rand(len(tg), N, 4), torch.zeros(len(tg), dtype=torch.long), None
        def forward(self, x_t, tt, **kw):
            b = self.lin(x_t).sigmoid()               # requires grad unless no_grad
            return b, torch.randn(b.shape[0], N, requires_grad=True) * self.lin.weight.sum()

    batch = {"boxes": [torch.rand(G, 4) * 0.4 + 0.3, torch.rand(G, 4) * 0.4 + 0.3],
             "labels": [torch.zeros(G, dtype=torch.long)] * 2,
             "valid_h": torch.ones(2),
             "pixel_values": torch.zeros(2, 3, 8, 8),
             "text": ["a", "b"]}

    with torch.enable_grad():                         # the state train.py is in
        out = train_mod.run_val(_Model(), [batch], _Crit(), N, torch.device("cpu"))

    assert "oracle_recall_final" in out
    assert "score" in out

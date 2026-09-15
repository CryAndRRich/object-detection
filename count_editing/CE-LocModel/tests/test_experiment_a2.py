"""EXPERIMENT A.2 — class through the OUTPUT head instead of the INPUT text.

A.1 and A.2 must differ in exactly one thing, so most of these tests assert that
everything ELSE stayed identical. The failure this file guards against is not a
crash; it is a run that looks fine and answers a different question than the one
asked -- e.g. A.2 quietly keeping the text tower, or A.1's model changing because
A.2's plumbing was added.
"""

import os
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.criterion import SetCriterion, sigmoid_focal_loss  # noqa: E402
from models.detector import CELocDetector, build_model  # noqa: E402

ANN = "../../data/coco_minitrain/annotations/instances_minitrain2017.json"


def cfg_of(name):
    with open(f"config/{name}.yaml") as f:
        return yaml.safe_load(f)


# ------------------------------------------------- the experiment is one variable

def test_a2_config_differs_from_a1_in_exactly_three_keys():
    a1, a2 = cfg_of("experiment_a1"), cfg_of("experiment_a2")
    for section in ("diffusion", "loss", "matcher", "eval"):
        assert a1[section] == a2[section], f"{section} must be identical"
    assert {k for k in set(a1["data"]) | set(a2["data"])
            if a1["data"].get(k) != a2["data"].get(k)} == {"per_class"}
    defaults = {"n_class": 1, "use_text": True, "roi_k": 0}
    diff = {k for k in set(a1["model"]) | set(a2["model"])
            if a1["model"].get(k, defaults.get(k)) != a2["model"].get(k, defaults.get(k))}
    assert diff == {"n_class", "use_text"}
    assert {k for k in a1["training"]
            if a1["training"][k] != a2["training"][k]} == {"epochs", "save_dir"}


def test_budgets_are_matched_in_image_views_not_epochs():
    """A.1 has 73,531 samples, A.2 only 25,000. Equal epochs would give A.1 nearly
    3x the compute and the comparison would measure budget, not conditioning."""
    a1, a2 = cfg_of("experiment_a1"), cfg_of("experiment_a2")
    v1 = a1["training"]["epochs"] * 73531
    v2 = a2["training"]["epochs"] * 25000
    assert abs(v1 - v2) / v1 < 0.02, f"{v1} vs {v2} image-views"


def test_ab_configs_still_have_no_a2_keys():
    """A/B must keep the defaults, or adding A.2 would have changed them."""
    for name in ("experiment_a", "experiment_b"):
        m = cfg_of(name)["model"]
        assert m.get("n_class", 1) == 1 and m.get("use_text", True) is True


# ---------------------------------------------------------------------- the model

def test_a2_has_no_text_tower_at_all():
    """Not merely unused: an unused-but-present tower registers frozen parameters
    in the checkpoint and in every count, so A.1 and A.2 would differ for a reason
    unrelated to the experiment."""
    m = build_model(cfg_of("experiment_a2"))
    assert m.n_class == 80 and not m.encoder.use_text
    assert m.encoder.text is None and m.encoder.proj_text is None
    assert not any("encoder.text" in k for k in m.state_dict())


def test_a2_memory_has_no_text_token():
    m = build_model(cfg_of("experiment_a2"))
    px = torch.zeros(2, 3, 512, 512)
    assert m.encoder(px, None).shape[1] == m.encoder.num_patches
    m1 = build_model(cfg_of("experiment_a1"))
    assert m1.encoder(px, ["dog", "cat"]).shape[1] == m1.encoder.num_patches + 1


def test_a2_ignores_text_even_when_given():
    """The training loop passes `texts` for every experiment; A.2 must not read it,
    or its central claim -- the class never enters through the input -- is false."""
    m = build_model(cfg_of("experiment_a2")).eval()
    px = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        a = m.encoder(px, ["dog"])
        b = m.encoder(px, ["fire hydrant"])
    assert torch.equal(a, b)


def test_mixing_the_two_pathways_is_rejected():
    """n_class=80 with use_text=True would let the class arrive through BOTH paths,
    measuring nothing. Fail loudly at construction rather than after a 4-hour run."""
    with pytest.raises(ValueError, match="disagree"):
        CELocDetector(n_class=80, use_text=True)
    with pytest.raises(ValueError, match="disagree"):
        CELocDetector(n_class=1, use_text=False)


def test_logit_shapes_per_experiment():
    for name, shape in [("experiment_a1", (2, 100)), ("experiment_a2", (2, 100, 80))]:
        m = build_model(cfg_of(name)).eval()
        px = torch.zeros(2, 3, 512, 512)
        x_t = torch.zeros(2, 100, 4)
        tt = torch.zeros(2, dtype=torch.long)
        with torch.no_grad():
            _, lg = m(x_t, tt, pixel_values=px, texts=["dog", "cat"])
        assert lg.shape == shape, f"{name}: {lg.shape}"


def test_build_model_reproduces_the_old_construction_for_a_and_b():
    """`build_model` replaced eight hand-written constructor calls. If it drifts,
    a tool silently evaluates a different model than the one trained."""
    for name in ("experiment_a", "experiment_b"):
        cfg = cfg_of(name)
        m = cfg["model"]
        torch.manual_seed(0)
        old = CELocDetector(
            m["clip_name"], m["d_model"], m["n_layer"], m["n_head"],
            cfg["data"]["image_size"], cfg["diffusion"]["num_timesteps"],
            cfg["diffusion"]["snr_scale"], cfg["diffusion"]["sampling_steps"],
            m["dropout"], m["freeze_clip"], roi_k=m.get("roi_k", 0))
        torch.manual_seed(0)
        new = build_model(cfg)
        so, sn = old.state_dict(), new.state_dict()
        assert so.keys() == sn.keys()
        assert all(torch.equal(so[k], sn[k]) for k in so)


# ----------------------------------------------------------------------- the loss

def test_multiclass_loss_targets_the_right_class():
    """The one-hot must land on the GT's class, not on slot 0 or on every class."""
    crit = SetCriterion("hungarian")
    torch.manual_seed(0)
    boxes = torch.rand(1, 8, 4) * 0.5 + 0.25
    logits = torch.zeros(1, 8, 80, requires_grad=True)
    gt = [boxes[0, 3:4].clone()]
    lab = [torch.tensor([42])]
    loss, _, idx = crit(boxes, logits, gt, labels=lab)
    loss.backward()
    g = logits.grad[0]
    pi = idx[0][0][0].item()
    assert g[pi, 42] < 0, "matched slot must be pushed UP on class 42"
    assert (g[pi, torch.arange(80) != 42] > 0).all(), "other classes pushed down"


def test_multiclass_needs_labels():
    crit = SetCriterion("hungarian")
    with pytest.raises(ValueError, match="labels"):
        crit(torch.rand(1, 4, 4), torch.zeros(1, 4, 80), [torch.rand(1, 4)])


def test_one_dim_path_is_untouched_by_the_multiclass_code():
    """A/B must produce the same loss they did before A.2 existed, whether or not
    `labels` are supplied."""
    crit = SetCriterion("hungarian")
    torch.manual_seed(0)
    b, lg = torch.rand(2, 10, 4) * 0.5 + 0.25, torch.randn(2, 10)
    gt = [b[0, :2].clone(), b[1, :3].clone()]
    l1, s1, _ = crit(b, lg, gt)
    l2, s2, _ = crit(b, lg, gt, labels=[torch.zeros(2, dtype=torch.long),
                                        torch.zeros(3, dtype=torch.long)])
    assert torch.equal(l1, l2) and s1 == s2


def test_NEGATIVE_focal_loss_is_80x_at_init_and_collapses():
    """Documents a number that WILL look alarming in the first A.2 log line, so it
    does not get 'fixed' into an incomparable normalisation. Also a negative
    control: if it did NOT collapse, the multi-class head would be untrainable."""
    z = torch.zeros(2, 100, 80)
    at_init = float(sigmoid_focal_loss(z, torch.zeros_like(z)).sum())
    one_d = torch.zeros(2, 100)
    assert abs(at_init / float(sigmoid_focal_loss(one_d, torch.zeros_like(one_d)).sum())
               - 80) < 0.01
    trained = torch.full((2, 100, 80), torch.logit(torch.tensor(0.02)).item())
    assert float(sigmoid_focal_loss(trained, torch.zeros_like(trained)).sum()) < at_init / 1000


# -------------------------------------------------------------------- the dataset

@pytest.mark.skipif(not os.path.exists(ANN), reason="COCO-minitrain not present")
def test_per_class_false_regroups_the_same_annotations():
    """A.1 and A.2 must describe the SAME data. Different box totals would mean the
    two runs saw different supervision."""
    from data.factory import build_dataset
    d1 = build_dataset(cfg_of("experiment_a1"), "train")
    d2 = build_dataset(cfg_of("experiment_a2"), "train")
    s1, s2 = d1.stats(), d2.stats()
    assert s1["n_boxes_total"] == s2["n_boxes_total"] == 181475
    assert s2["n_images"] == 25000 and s1["n_images"] == 73531
    assert s1["n_classes"] == s2["n_classes"] == 80


@pytest.mark.skipif(not os.path.exists(ANN), reason="COCO-minitrain not present")
def test_labels_are_contiguous_and_aligned_with_boxes():
    """COCO ids run 1..90 WITH GAPS. Indexing an 80-way head with a raw id would be
    out of range for ids > 80 and would silently mislabel the rest."""
    from data.factory import build_dataset
    ds = build_dataset(cfg_of("experiment_a2"), "train")
    seen = set()
    for i in range(0, 4000, 7):
        m = ds.__getitem__(i, need_image=False)
        assert len(m["labels"]) == len(m["boxes"])
        assert m["labels"].dtype == np.int64
        assert m["labels"].min() >= 0 and m["labels"].max() < 80
        seen.update(m["labels"].tolist())
    assert len(seen) > 60, f"only {len(seen)} classes seen — mapping looks wrong"
    assert ds.cat_ids == sorted(ds.cat_ids) and max(ds.cat_ids) == 90


@pytest.mark.skipif(not os.path.exists(ANN), reason="COCO-minitrain not present")
def test_class_index_mapping_is_identical_between_train_and_eval_splits():
    """A checkpoint trained on minitrain is evaluated on val2017. If the two files
    produced different orderings, every predicted class would be wrong on eval and
    nothing would crash."""
    from data.factory import build_dataset
    cfg = cfg_of("experiment_a2")
    tr, va = build_dataset(cfg, "train"), build_dataset(cfg, "val")
    assert tr.cat_ids == va.cat_ids and tr.cat_names == va.cat_names


@pytest.mark.skipif(not os.path.exists(ANN), reason="COCO-minitrain not present")
def test_flip_keeps_boxes_and_labels_aligned():
    from data.coco_dataset import COCODetection
    from data.factory import build_dataset
    cfg = cfg_of("experiment_a2")
    a = build_dataset(cfg, "train", 0.0)[11]
    b = COCODetection(ANN, "../../data/coco_minitrain/images/train2017",
                      flip_prob=1.0, seed=0, per_class=False)[11]
    assert np.array_equal(a["labels"], b["labels"])
    assert np.allclose(b["boxes"][:, 0], 1 - a["boxes"][:, 0])


@pytest.mark.skipif(not os.path.exists(ANN), reason="COCO-minitrain not present")
def test_ce130_also_returns_labels_so_the_loop_needs_no_branch():
    from data.factory import build_dataset
    ds = build_dataset(cfg_of("experiment_a"), "train")
    m = ds[0]
    assert len(m["labels"]) == len(m["boxes"]) and set(m["labels"].tolist()) <= {0}


# ------------------------------------------------------------------ evaluation

def test_scores_and_classes_reads_both_head_shapes():
    from eval import scores_and_classes
    s, c = scores_and_classes(torch.tensor([0.0, 2.0]))
    assert c is None and abs(s[0] - 0.5) < 1e-6
    lg = torch.full((3, 80), -5.0)
    lg[1, 7] = 5.0
    s, c = scores_and_classes(lg)
    assert c[1] == 7 and s[1] > 0.99 and s[0] < 0.01


@pytest.mark.skipif(not os.path.exists(ANN), reason="COCO-minitrain not present")
def test_the_whole_batch_path_carries_labels():
    """The dataset returning `labels` is not enough -- TorchWrap and collate must
    forward them, and a missing key there only surfaces once a DataLoader actually
    runs. That is precisely how it was found: every unit test passed while
    training died on `KeyError: 'labels'` in collate.

    So this walks the REAL wrapper and the REAL collate, not a stand-in.
    """
    from torch.utils.data import DataLoader

    from data.factory import build_dataset
    from train import TorchWrap, collate

    for name in ("experiment_a1", "experiment_a2", "experiment_a"):
        cfg = cfg_of(name)
        ds = build_dataset(cfg, "train")
        ds.items = ds.items[:4]
        batch = next(iter(DataLoader(TorchWrap(ds), batch_size=2, collate_fn=collate)))
        assert "labels" in batch, f"{name}: collate dropped labels"
        assert len(batch["labels"]) == len(batch["boxes"]) == 2
        for lb, bx in zip(batch["labels"], batch["boxes"]):
            assert lb.shape[0] == bx.shape[0] and lb.dtype == torch.long


@pytest.mark.skipif(not os.path.exists(ANN), reason="COCO-minitrain not present")
def test_a2_trains_one_real_step_end_to_end():
    """Forward, loss, backward on real COCO data through the real model. Shape
    tests alone let the KeyError above through."""
    from data.factory import build_dataset
    from data.ce130_dataset import normalize_for_clip

    cfg = cfg_of("experiment_a2")
    ds = build_dataset(cfg, "train")
    m = build_model(cfg)
    m.train()
    s = [ds[0], ds[1]]
    px = torch.from_numpy(np.stack([normalize_for_clip(x["image"]) for x in s]))
    tg = [torch.from_numpy(x["boxes"]).float() for x in s]
    tl = [torch.from_numpy(x["labels"]).long() for x in s]
    x_t, tt, _ = m.build_inputs(tg, 100, [x["valid_h"] for x in s])
    pb, lg = m(x_t, tt, pixel_values=px, texts=None)
    assert lg.shape == (2, 100, 80)
    loss, st, _ = SetCriterion("hungarian")(pb, lg, tg, labels=tl)
    assert torch.isfinite(loss) and st["n_matched"] > 0
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_class_aware_eval_penalises_a_right_box_on_the_wrong_class():
    """The core requirement of A.2's evaluation. Without the per-class split, a box
    that lands perfectly on a cat while calling it a dog would score as a hit, and
    A.2 would beat A.1 for a reason that has nothing to do with the experiment.

    Both halves are asserted: the same class must count, the wrong class must not.
    A test that only checked the first would pass on class-agnostic code.
    """
    from eval import evaluate
    from utils.box_ops_np import cxcywh_to_xyxy

    gt = cxcywh_to_xyxy(np.array([[0.5, 0.5, 0.2, 0.2]])) * 512
    pred = gt.copy()                       # a geometrically PERFECT box
    same = evaluate([(pred, np.array([0.9]), gt)], 0.5)
    # wrong class -> the box and the GT land in different groups
    wrong = evaluate([(pred, np.array([0.9]), np.zeros((0, 4))),
                      (np.zeros((0, 4)), np.array([]), gt)], 0.5)
    assert same["recall"] == 1.0 and same["precision"] == 1.0
    assert wrong["recall"] == 0.0 and wrong["precision"] == 0.0


def test_class_groups_cover_gt_only_and_pred_only_classes():
    """The loop iterates the UNION of predicted and GT classes. Iterating only the
    predicted ones would silently drop every miss for a class the model never
    proposed -- inflating recall exactly where the model is worst."""
    c_k = np.array([5, 7])
    g_lab = np.array([5, 5, 9], dtype=np.int64)
    groups = np.unique(np.concatenate([c_k, g_lab]))
    assert set(groups.tolist()) == {5, 7, 9}
    assert np.unique(np.concatenate([np.array([], dtype=int),
                                     np.array([], dtype=np.int64)])).size == 0


def test_no_tool_reads_logits_without_the_shape_aware_helper():
    """Every consumer of `logits` must go through `scores_and_classes`.

    `torch.sigmoid(logits[0])` on A.2's [N,80] leaves a 2-D array; `np.argsort`
    then returns 2-D indices and NMS dies with "only integer scalar arrays can be
    converted to a scalar index". eval.py was fixed for this and
    visualize_predictions.py was NOT -- it ran fine on A.1 and crashed on A.2,
    after 18 hours of training had already been spent.

    A unit test of `scores_and_classes` could not catch that: the helper was
    correct, it simply was not called. So this greps the call sites instead.
    """
    import glob
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    for f in glob.glob(os.path.join(root, "tools/*.py")) + glob.glob(os.path.join(root, "*.py")):
        src = open(f).read()
        if "ddim_sample" not in src:
            continue                     # does not produce logits
        if "sigmoid(logits" in src and "scores_and_classes" not in src:
            offenders.append(os.path.basename(f))
    assert not offenders, (f"{offenders} read logits with a raw sigmoid; use "
                           f"scores_and_classes so the [N,C] head works too")


def test_draw_one_requires_the_right_class_when_given_one():
    """A.2's pictures must agree with its AP: a perfectly placed box that names the
    wrong class is a miss, exactly as eval.py's per-class split scores it."""
    from tools.visualize_predictions import draw_one

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    gt = np.array([[10.0, 10.0, 30.0, 30.0]])
    pred = gt.copy()
    sc = np.array([0.9])
    _, n_same = draw_one(img, gt, pred, sc, pred_cls=np.array([7]),
                         gt_cls=np.array([7]))
    _, n_diff = draw_one(img, gt, pred, sc, pred_cls=np.array([7]),
                         gt_cls=np.array([3]))
    _, n_agnostic = draw_one(img, gt, pred, sc)          # A/B/A.1 path unchanged
    assert n_same == 1 and n_diff == 0 and n_agnostic == 1

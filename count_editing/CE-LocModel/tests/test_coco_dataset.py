"""EXPERIMENT A.1 dataset — runs against the REAL annotation file, not fixtures.

The bugs worth catching here are all data-shape bugs (a wrong box format, a
dropped category, an image counted twice), and a fixture would have to reproduce
the very structure being checked. So these read the actual 112 MB JSON; the scan
takes ~5 s and is cached across tests in the module-level loader.

NEGATIVE CONTROLS are included: a test suite that cannot demonstrate it would
FAIL on a known-bad input proves nothing. Round 1 shipped tests that passed while
the code was wrong.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.coco_dataset import COCODetection  # noqa: E402
from data.factory import build_dataset, dataset_kind  # noqa: E402
from utils.box_ops_np import box_iou, cxcywh_to_xyxy, scale_to_canvas  # noqa: E402

ROOT = "../../data"
ANN = f"{ROOT}/coco_minitrain/annotations/instances_minitrain2017.json"
IMG = f"{ROOT}/coco_minitrain/images/train2017"

pytestmark = pytest.mark.skipif(not os.path.exists(ANN),
                                reason="COCO-minitrain not present")

_ds = None


def ds():
    global _ds
    if _ds is None:
        _ds = COCODetection(ANN, IMG)
    return _ds


# ------------------------------------------------------------------ inventory

def test_pair_counts_match_the_published_dataset():
    """25,000 images / 181,475 non-crowd boxes / 80 classes -> 73,531 pairs.

    Hard-coded on purpose: if a future change silently drops or duplicates
    samples, every downstream number moves and nothing else would notice.
    """
    st = ds().stats()
    assert st["n_images"] == 73531
    assert st["n_source_images"] == 25000
    assert st["n_boxes_total"] == 181475
    assert st["n_classes"] == 80


def test_iscrowd_annotations_are_dropped():
    """2,071 crowd annotations carry deliberately sloppy boxes."""
    import json
    with open(ANN) as f:
        d = json.load(f)
    assert len(d["annotations"]) - ds().stats()["n_boxes_total"] == 2071


def test_one_image_becomes_one_sample_per_class():
    """The whole framing of A.1: the model sees ONE text per forward pass, so a
    3-class image must become 3 samples over the same pixels."""
    from collections import defaultdict
    g = defaultdict(list)
    for j, it in enumerate(ds().items[:5000]):
        g[it["image_id"].split("_")[0]].append(j)
    multi = [v for v in g.values() if len(v) >= 3]
    assert multi, "expected some image with >=3 classes"
    idx = multi[0]
    assert len({ds().items[j]["img_path"] for j in idx}) == 1
    texts = [ds().items[j]["text"] for j in idx]
    assert len(set(texts)) == len(texts), f"classes must be distinct: {texts}"


def test_image_ids_are_unique_per_pair():
    """The token cache and every per-image log key off image_id. One source image
    produces ~3 samples, so an id of just the image number would collide and one
    class's tokens would be served for another's."""
    ids = [it["image_id"] for it in ds().items]
    assert len(set(ids)) == len(ids)


# ------------------------------------------------------------------- geometry

def test_boxes_are_valid_after_the_full_transform():
    rs = np.random.RandomState(0)
    b = np.concatenate([ds().__getitem__(int(i), need_image=False)["boxes"]
                        for i in rs.choice(len(ds()), 300, replace=False)])
    assert (b[:, 2] > 0).all() and (b[:, 3] > 0).all()
    x1, y1, x2, y2 = cxcywh_to_xyxy(b).T
    assert x1.min() >= -1e-6 and y1.min() >= -1e-6
    assert x2.max() <= 1 + 1e-6 and y2.max() <= 1 + 1e-6


def test_NEGATIVE_forgetting_the_xywh_conversion_is_caught():
    """COCO's bbox is [x, y, w, h] top-left. Feeding it to `scale_to_canvas`
    (which expects xyxy) yields boxes that are still on the canvas and trip no
    assertion -- the exact silent-failure mode that has cost this project twice.

    Proves the geometry tests above CAN fail: the wrong reading must not match.
    """
    import json
    with open(ANN) as f:
        d = json.load(f)
    imgs = {i["id"]: i for i in d["images"]}
    a = next(x for x in d["annotations"] if not x.get("iscrowd", 0))
    im = imgs[a["image_id"]]
    x, y, w, h = a["bbox"]
    right, _, _ = scale_to_canvas(np.array([[x, y, x + w, y + h]]), im["width"], im["height"])
    wrong, _, _ = scale_to_canvas(np.array([[x, y, w, h]]), im["width"], im["height"])
    assert box_iou(cxcywh_to_xyxy(right), cxcywh_to_xyxy(wrong))[0][0, 0] < 0.9


def test_portrait_images_pad_on_the_right_not_the_bottom():
    """Unlike CE-130 (every image 384px tall), 23.3 % of COCO is portrait. valid_h
    is then 1.0 and valid_w < 1.0."""
    port = [i for i, it in enumerate(ds().items[:6000])
            if it["orig_wh"][1] > it["orig_wh"][0]]
    land = [i for i, it in enumerate(ds().items[:6000])
            if it["orig_wh"][0] > it["orig_wh"][1]]
    assert port and land
    m = ds().__getitem__(port[0], need_image=False)
    assert abs(m["valid_h"] - 1.0) < 1e-6 and m["valid_w"] < 1.0
    m = ds().__getitem__(land[0], need_image=False)
    assert abs(m["valid_w"] - 1.0) < 1e-6 and m["valid_h"] < 1.0


def test_canvas_matches_the_ce130_contract():
    m = ds()[0]
    assert m["image"].shape == (512, 512, 3) and m["image"].dtype == np.uint8
    for k in ("image", "boxes", "text", "valid_h", "image_id", "orig_size", "flipped"):
        assert k in m


def test_need_image_false_gives_identical_boxes():
    """The cached path skips the JPEG decode; if it also changed the geometry,
    training and evaluation would silently disagree."""
    a = ds().__getitem__(7, need_image=True)
    b = ds().__getitem__(7, need_image=False)
    assert np.array_equal(a["boxes"], b["boxes"])
    assert b["image"] is None and abs(a["valid_h"] - b["valid_h"]) < 1e-12


def test_flip_is_a_true_mirror():
    a = COCODetection(ANN, IMG, flip_prob=0.0)[5]
    b = COCODetection(ANN, IMG, flip_prob=1.0, seed=0)[5]
    assert b["flipped"] and not a["flipped"]
    assert np.allclose(b["boxes"][:, 0], 1 - a["boxes"][:, 0])
    assert np.allclose(b["boxes"][:, 2:], a["boxes"][:, 2:])
    assert np.array_equal(b["image"][:, ::-1], a["image"])


# -------------------------------------------------------------------- factory

def test_factory_defaults_to_ce130_so_old_configs_are_untouched():
    import yaml
    with open("config/experiment_a.yaml") as f:
        cfg = yaml.safe_load(f)
    assert dataset_kind(cfg) == "ce130"


def test_factory_never_augments_the_eval_split():
    """flip_prob is supplied by the caller, so an augmentation setting in a config
    cannot leak into evaluation."""
    import yaml
    with open("config/experiment_a1.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["data"]["flip_prob"] == 0.5
    assert build_dataset(cfg, "val").flip_prob == 0.0
    assert build_dataset(cfg, "train", cfg["data"]["flip_prob"]).flip_prob == 0.5


def test_a1_config_differs_from_a_ONLY_in_data_and_budget():
    """A.1 is a control: model, loss, matcher, diffusion and eval must be byte
    identical to A, or a difference in the result cannot be attributed."""
    import yaml
    with open("config/experiment_a.yaml") as f:
        a = yaml.safe_load(f)
    with open("config/experiment_a1.yaml") as f:
        c = yaml.safe_load(f)
    for section in ("diffusion", "loss", "matcher", "eval"):
        assert a[section] == c[section], f"{section} differs from A"
    # A.1 spells out n_class/use_text that A leaves implicit, so compare EFFECTIVE
    # values rather than raw keys -- otherwise stating a default would read as a
    # change of model.
    defaults = {"n_class": 1, "use_text": True, "roi_k": 0}
    keys = set(a["model"]) | set(c["model"])
    for k in keys:
        d = defaults.get(k)
        assert a["model"].get(k, d) == c["model"].get(k, d), f"model.{k} differs from A"
    changed = {k for k in a["training"] if a["training"][k] != c["training"][k]}
    assert changed == {"batch_size", "epochs", "save_dir"}, changed


def test_coco_val_and_test_resolve_to_the_same_file():
    """COCO has no third split. Documented, not accidental -- eval prints the
    resolved path so it cannot be mistaken for a held-out test set."""
    import yaml
    with open("config/experiment_a1.yaml") as f:
        cfg = yaml.safe_load(f)
    assert build_dataset(cfg, "val").ann_file == build_dataset(cfg, "test").ann_file


def test_min_boxes_filter():
    """min_boxes=3 moves COCO towards CE-130 density. A SEPARATE experiment --
    the test exists so the knob is known to work, not so it gets turned on here."""
    d = COCODetection(ANN, IMG, min_boxes=3)
    n = [len(it["boxes_xyxy_px"]) for it in d.items]
    assert min(n) >= 3 and len(d) == 18057


def test_every_entry_point_parses():
    """`tools/overfit_one.py` and `tools/profile_and_memory.py` were found with a
    stray `,` on its own line -- a SyntaxError sitting in the repo unnoticed
    because no test ever imported them and they are only run by hand on the
    server. Compiling every script here costs milliseconds and turns "the tool
    crashes on the server" into a local failure."""
    import py_compile
    import glob
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = (glob.glob(os.path.join(root, "*.py"))
             + glob.glob(os.path.join(root, "tools/*.py"))
             + glob.glob(os.path.join(root, "data/*.py"))
             + glob.glob(os.path.join(root, "models/*.py"))
             + glob.glob(os.path.join(root, "utils/*.py")))
    assert len(files) > 15, f"expected to find the scripts, got {len(files)}"
    for f in files:
        py_compile.compile(f, doraise=True)


def test_every_tool_imports_not_just_compiles():
    """`py_compile` above catches SyntaxError but NOT NameError: a name used inside
    a function is only resolved when that function RUNS. tools/preflight.py shipped
    with `tl` referenced in four nested callbacks that never received it -- every
    local check passed and it failed on the server, the one place it is ever run.

    Importing is a weak guarantee (module level only), so the real coverage is
    `tools/check_before_train.py`, which executes preflight end to end on --limit
    data. This test is the cheap first line.
    """
    import glob
    import importlib
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in sorted(glob.glob(os.path.join(root, "tools/*.py"))):
        importlib.import_module("tools." + os.path.basename(f)[:-3])


def test_preflight_exposes_a_limit_flag_for_smoke_runs():
    """Without --limit, preflight can only be exercised on the server, which is
    exactly what let a NameError through."""
    import argparse
    import tools.preflight as pf
    src = open(pf.__file__).read()
    assert '"--limit"' in src, "preflight lost its smoke-mode flag"
    assert "SMOKE MODE" in src

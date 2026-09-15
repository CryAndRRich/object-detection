"""CE-130 dataset tests on REAL DATA — numpy + PIL, no torch needed.

Locks in the data findings so nobody accidentally breaks them again.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection, CLIP_MEAN, normalize_for_clip  # noqa: E402
from utils.box_ops_np import decode_diffusion, encode_diffusion  # noqa: E402
from utils.diffusion_np import make_placeholders  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "..", "..", "data", "all_phase2_V2")
needs_data = pytest.mark.skipif(not os.path.isdir(ROOT), reason="all_phase2_V2 not present")


@needs_data
@pytest.mark.parametrize("split,n_images,n_classes",
                         [("train", 1911, 72), ("val", 908, 28), ("test", 779, 28)])
def test_dedupe_gives_right_image_and_class_counts(split, n_images, n_classes):
    """Branches _b1/_b2/_b3 share one ground_truth.jpg -> must dedupe by image."""
    st = CE130Detection(ROOT, split).stats()
    assert st["n_images"] == n_images
    assert st["n_classes"] == n_classes


@needs_data
def test_classes_are_completely_disjoint_across_splits():
    """Inherited from the FSC-147 split (class-agnostic counting) -> ZERO-SHOT task.

    Consequence: the text encoder MUST be frozen, and these numbers must not be
    compared against closed-set detectors.
    """
    sets = {sp: {it["text"] for it in CE130Detection(ROOT, sp).items}
            for sp in ["train", "val", "test"]}
    assert not (sets["train"] & sets["val"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["val"] & sets["test"])


@needs_data
def test_do_NOT_subtract_inpainted_bboxes():
    """ground_truth.jpg is the ORIGINAL image with nothing removed -> keep all_bboxes.

    Pixel evidence: the diff against inpainted_turn_1.png over inpainted_bboxes[0]
    is 51.96/255 versus 1.41/255 for the whole image. Round 1 subtracted them ->
    threw away 7-8 % of REAL objects.
    """
    import json
    ds = CE130Detection(ROOT, "train")
    st = ds.stats()
    assert st["n_boxes_total"] > 71000, "looks like inpainted_bboxes is being subtracted"

    # inpainted_bboxes must be CONTAINED IN all_bboxes of the same branch
    it = next(x for x in ds.items if x["image_id"] == "1074")
    br = os.path.dirname(it["img_path"])
    with open(os.path.join(br, "annotation.json")) as f:
        ann = json.load(f)
    from utils.box_ops_np import box_iou
    for b in ann.get("inpainted_bboxes", []):
        iou = box_iou(np.array([b], dtype=float), it["boxes_xyxy_px"])[0]
        assert iou.max() > 0.5, "inpainted_bboxes must be present in all_bboxes"


@needs_data
def test_every_image_is_384_tall_so_padding_is_always_at_the_bottom():
    """This invariant is what lets cy be bounded by a SINGLE scalar threshold."""
    ds = CE130Detection(ROOT, "train")
    for i in np.linspace(0, len(ds) - 1, 40).astype(int):
        m = ds[int(i)]
        W, H = m["orig_size"]
        assert H == 384 and W >= H
        assert 0.0 < m["valid_h"] <= 1.0


@needs_data
def test_boxes_are_within_the_valid_range():
    ds = CE130Detection(ROOT, "train")
    for i in np.linspace(0, len(ds) - 1, 60).astype(int):
        b = ds[int(i)]["boxes"]
        if len(b) == 0:
            continue
        assert (b[:, 2] > 0).all() and (b[:, 3] > 0).all(), "degenerate boxes remain"
        assert (b[:, 0] > -0.05).all() and (b[:, 0] < 1.05).all()
        assert (b[:, 1] > -0.05).all() and (b[:, 1] < 1.05).all()


@needs_data
def test_padded_region_normalises_to_near_zero():
    """[NEGATIVE CONTROL] CLIP-mean padding -> ~0.000. Black padding -> -1.79 sigma
    (a dark block creating a fake edge)."""
    ds = CE130Detection(ROOT, "train")
    m = next(ds[int(i)] for i in range(len(ds)) if ds[int(i)]["valid_h"] < 0.8)
    x = normalize_for_clip(m["image"])
    nh = int(round(m["valid_h"] * 512))
    assert abs(x[:, nh + 2:, :].mean()) < 0.02

    from data.ce130_dataset import CLIP_STD
    black = float(((0.0 - CLIP_MEAN) / CLIP_STD).min())
    assert black < -1.4, "black padding must be far from 0 — proving the test discriminates"


@needs_data
def test_six_step_round_trip_on_real_data():
    """Real boxes through encode -> decode must come back to themselves."""
    ds = CE130Detection(ROOT, "train")
    for i in np.linspace(0, len(ds) - 1, 30).astype(int):
        b = ds[int(i)]["boxes"]
        if len(b) == 0:
            continue
        back = decode_diffusion(encode_diffusion(b, 2.0), 2.0)
        assert np.abs(back - b).max() < 1e-12


@needs_data
def test_placeholders_match_real_object_size_and_stay_out_of_padding():
    """The two CE-130 fixes, measured on real data."""
    ds = CE130Detection(ROOT, "train")
    rng = np.random.default_rng(0)
    in_pad = total = 0
    ratios = []
    for i in np.linspace(0, len(ds) - 1, 50).astype(int):
        m = ds[int(i)]
        if len(m["boxes"]) == 0:
            continue
        med = (np.median(m["boxes"][:, 2]), np.median(m["boxes"][:, 3]))
        ph = make_placeholders(80, med, m["valid_h"], rng)
        total += len(ph)
        in_pad += int((ph[:, 1] > m["valid_h"]).sum())
        ratios.append(np.median(ph[:, 2]) / med[0])
    assert in_pad == 0, f"{in_pad}/{total} placeholders landed in the padding"
    assert 0.7 < np.median(ratios) < 1.4, "placeholder size is far off real objects"


@needs_data
def test_flip_keeps_image_and_boxes_in_sync():
    a = CE130Detection(ROOT, "train", flip_prob=0.0)
    b = CE130Detection(ROOT, "train", flip_prob=1.0, seed=0)
    ma, mb = a[5], b[5]
    assert np.array_equal(mb["image"], ma["image"][:, ::-1])
    assert np.allclose(mb["boxes"][:, 0], 1.0 - ma["boxes"][:, 0])
    assert np.allclose(mb["boxes"][:, 1:], ma["boxes"][:, 1:])

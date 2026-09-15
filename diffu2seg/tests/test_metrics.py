"""Metrics: the two accumulation traps, and the absence of score_AUC.

Run:  python -m pytest tests/test_metrics.py -q
      python tests/test_metrics.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics import quality_one_image, summarise  # noqa: E402


def _box(cx, cy, w, h):
    return np.array([[cx, cy, w, h]], dtype=np.float64)


def test_empty_prediction_counts_n_gt_not_zero():
    """TRAP 2. An image with no predictions still contributes its GT count.

    NEGATIVE CONTROL: the whole point is that n_gt is 5 and NOT 0. Returning
    (0, 0) would silently delete five missed objects from the denominator.
    """
    gt = np.tile(_box(0.5, 0.5, 0.1, 0.1), (5, 1))
    best, hits, n_gt = quality_one_image(np.zeros((0, 4)), gt)

    assert n_gt == 5, "missed objects must stay in the denominator"
    assert n_gt != 0
    assert hits == 0
    assert best.shape == (5,)
    assert np.all(best == 0.0)


def test_no_gt_contributes_nothing():
    best, hits, n_gt = quality_one_image(_box(0.5, 0.5, 0.1, 0.1), np.zeros((0, 4)))
    assert (len(best), hits, n_gt) == (0, 0, 0)


def test_raw_accumulation_differs_from_mean_of_ratios():
    """TRAP 1. Proves the trap is real, not folklore.

    Image 1: 1 GT, hit.        Image 2: 100 GT, 1 hit.
    Raw:  (1 + 1) / (1 + 100) = 0.0198
    Mean of ratios: (1.0 + 0.01) / 2 = 0.505   <- 25x larger, and wrong
    """
    hits, n_gt, best_all = 0, 0, []

    b1, h1, g1 = 1.0, 1, 1
    best_all.append(np.array([b1]))
    hits += h1
    n_gt += g1

    b2 = np.zeros(100)
    b2[0] = 1.0
    best_all.append(b2)
    hits += 1
    n_gt += 100

    out = summarise(best_all, hits, n_gt)

    assert abs(out["oracle_recall"] - 2.0 / 101.0) < 1e-12
    mean_of_ratios = (1.0 / 1 + 1.0 / 100) / 2
    assert abs(out["oracle_recall"] - mean_of_ratios) > 0.4, "the trap must bite"


def test_oracle_recall_is_score_free():
    """Shuffling prediction order changes nothing: there is no ranking here."""
    gt = np.concatenate([_box(0.25, 0.25, 0.1, 0.1), _box(0.75, 0.75, 0.1, 0.1)])
    pred = np.concatenate(
        [_box(0.25, 0.25, 0.1, 0.1), _box(0.9, 0.1, 0.05, 0.05), _box(0.75, 0.75, 0.1, 0.1)]
    )

    _, hits_a, n_a = quality_one_image(pred, gt)
    _, hits_b, n_b = quality_one_image(pred[::-1], gt)
    assert (hits_a, n_a) == (hits_b, n_b) == (2, 2)


def test_perfect_prediction_gives_recall_one():
    gt = np.concatenate([_box(0.3, 0.3, 0.2, 0.2), _box(0.7, 0.7, 0.2, 0.2)])
    best, hits, n_gt = quality_one_image(gt.copy(), gt)
    assert hits == n_gt == 2
    assert np.allclose(best, 1.0)
    assert abs(summarise([best], hits, n_gt)["mean_bestIoU"] - 1.0) < 1e-12


def test_iou_threshold_is_half():
    """A box overlapping exactly 1/3 is not a hit; a near-exact one is."""
    gt = _box(0.5, 0.5, 0.2, 0.2)
    shifted = _box(0.6, 0.5, 0.2, 0.2)          # IoU = 1/3
    _, hits, _ = quality_one_image(shifted, gt)
    assert hits == 0

    _, hits2, _ = quality_one_image(_box(0.505, 0.5, 0.2, 0.2), gt)
    assert hits2 == 1


def test_no_score_auc_key():
    """Locks the module docstring's promise.

    score_AUC here would be a proxy, and a proxy under that name invites a
    comparison against A/B/C1's 0.4965-0.4988 that it cannot support.
    """
    out = summarise([np.array([1.0])], 1, 1)
    assert "score_AUC" not in out
    assert "score_AUC_n_images" not in out
    assert set(out) == {
        "oracle_recall", "mean_bestIoU", "median_bestIoU",
        "n_gt", "n_hit", "n_pred_total", "n_images",
    }


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Chấm box baseline: mẫu + hình học trùng đường eval CE-Loc (`CE130AddDataset` / `CoCountAddDataset`), box gốc × s = canvas, mốc prior
trùng eval, lỗi khi thiếu box / K lệch, CLI `score.py` (+ `--first`)."""

import json
import sys

import numpy as np
import pytest

from baseline.data import canvas_scale, geometry, load_samples
from baseline.scoring import records_from_boxes, score_records
from ce_localization.data.cocount import CoCountAddDataset, load_review
from ce_localization.data.turns import CE130AddDataset, collate_add
from ce_localization.engine.add_eval import add_metrics, prior_records, prior_unit_boxes
from tests.baseline.helpers import fake_ce130, fake_cocount


def _eval_geo(ds):
    """Hình học canvas như `predict_add` dựng từ Dataset eval của CE-Loc."""
    out = []
    for i in range(len(ds)):
        b = collate_add([ds[i]])
        out.append({"image_id": b["image_id"][0], "wh": b["whwh"][0, :2].numpy().astype(np.float64),
                    **{k: b[k][0].numpy().astype(np.float64) for k in ("holes", "objects", "objects_all", "exemplars") if k in b}})
    return out


def test_ce130_geometry_scale_prior_match_eval(tmp_path):
    root, samples_root, ti = fake_ce130(str(tmp_path))
    samples, index = load_samples("ce130", turn_index=ti, samples_root=samples_root)
    ev = _eval_geo(CE130AddDataset(index, root, samples_root, "test_all"))
    assert [s["image_id"] for s in samples] == [r["image_id"] for r in ev]
    geo = lambda s: geometry(s, index=index, samples_root=samples_root)  # noqa: E731
    K = 3
    recs, k = records_from_boxes(samples, {s["image_id"]: [list(s["holes"][-1])] * K for s in samples}, geo)
    for s, r, e in zip(samples, recs, ev):
        for key in ("wh", "holes", "objects"):
            np.testing.assert_array_equal(r[key], e[key])
        np.testing.assert_allclose(r["boxes"][0], e["holes"][-1], atol=1e-4)                  # lỗ gốc × s = lỗ canvas
    res, pri, dump = score_records(recs, "ce130", k, index=index)
    assert res["inpainted"]["hit50@3_latest"] == 1.0 and len(dump["inpainted"]) == len(samples) and "cnll_median" in res["inpainted"]
    ref = add_metrics(prior_records([{**e, "t": 1, "boxes": np.zeros((K, 4))} for e in ev], prior_unit_boxes(index, "train"), K, seed=0))
    assert pri["inpainted"] == ref
    with pytest.raises(ValueError, match="thiếu box"):
        records_from_boxes(samples, {samples[0]["image_id"]: [[0, 0, 5, 5]]}, geo)
    with pytest.raises(ValueError, match="cùng số box"):
        records_from_boxes(samples, {s["image_id"]: [[0, 0, 5, 5]] * (1 + i % 2) for i, s in enumerate(samples)}, geo)


def test_cocount_geometry_and_keys(tmp_path):
    croot = str(tmp_path / "cocount")
    rv = load_review(fake_cocount(croot))
    samples, _ = load_samples("cocount", cocount_root=croot, review=rv, min_obj=5)
    ds = CoCountAddDataset(croot, review=rv, min_obj=5)
    assert [s["image_id"] for s in samples] == ds.keys and len(samples) == 3                 # file < 5 vật sau rà bị loại
    geo = lambda s: geometry(s, cocount_root=croot, review=rv)  # noqa: E731
    for s, e in zip(samples, _eval_geo(ds)):
        g = geo(s)
        for key in ("wh", "holes", "objects", "objects_all", "exemplars"):
            np.testing.assert_array_equal(g[key], e[key])
    recs, K = records_from_boxes(samples, {s["image_id"]: [[1, 1, 20, 20], [30, 40, 45, 52]] for s in samples}, geo)
    res, _, _ = score_records(recs, "cocount", K, obj_size=["mean", "exemplar"])
    assert set(res) == {"cocount", "cocount_objsize", "cocount_objsize_exemplar"} and "best_iou@2_latest" not in res["cocount"]


def test_score_cli_first(tmp_path, monkeypatch):
    _, samples_root, ti = fake_ce130(str(tmp_path))
    samples, _ = load_samples("ce130", turn_index=ti, samples_root=samples_root)
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps({"image_id": s["image_id"], "boxes": [list(s["holes"][0]), [0, 0, 10, 10]]}) + "\n" for s in samples))
    import baseline.score as score
    for first, k in ((None, 2), (1, 1)):
        out = tmp_path / f"o{k}" / "s.json"
        monkeypatch.setattr(sys, "argv", ["score.py", "--dataset", "ce130", "--turn-index", ti, "--samples-root", samples_root,
                                          "--boxes", str(raw), "--out", str(out), "--meta", "method=test"]
                            + (["--first", str(first)] if first else []))
        score.main()
        d = json.load(open(out))
        assert d["method"] == "test" and d["n_samples"] == k and d["cnll_calib"] == "mu" and "inpainted" in d["prior"]
        assert d["results"]["inpainted"][f"hit50@{k}_any"] == 1.0
        assert len(json.load(open(str(out).replace(".json", "_boxes.json")))["results"]["inpainted"]) == len(samples)

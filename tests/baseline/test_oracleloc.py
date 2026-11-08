"""OracleLoc (ZETA7): cỡ box = μ, lưới trọn trong ảnh, box[0] tiêu chí `cnll` = min C-NLL trên cả lưới (đúng code chấm), `euclid` =
min ‖z − μ‖; ảnh < 5 vật: box giữa ảnh; trọn luồng `predict.py` không --ckpt -> hai bộ JSON chấm được."""

import json
import sys

import numpy as np

from baseline.methods.oracleloc import grid_boxes, search
from ce_localization.engine.add_eval import cnll_fit, cnll_features, cnll
from tests.baseline.helpers import fake_ce130, tiny_cfg
from tests.fakes import fake_cocount_review


def _scene():
    rng = np.random.default_rng(0)
    xy = np.stack(np.meshgrid(np.arange(4) * 60.0 + 20, np.arange(3) * 50.0 + 30), -1).reshape(-1, 2) + rng.normal(0, 4, (12, 2))
    wh = rng.normal([30, 24], 2, (12, 2))
    return np.concatenate([xy, xy + wh], 1), (300.0, 220.0)


def test_grid_inside_and_search_minimises():
    objs, wh = _scene()
    g = grid_boxes(wh, 31.5, 20.25, 2)
    assert g[:, 0].min() == 0 and g[:, 1].min() == 0 and np.isclose(g[:, 2].max(), wh[0]) and np.isclose(g[:, 3].max(), wh[1])
    assert np.allclose(g[:, 2] - g[:, 0], 31.5) and np.allclose(g[:, 3] - g[:, 1], 20.25)
    res = search(objs, wh, 30, 2)
    mu, inv = cnll_fit(objs, objs, wh)
    cand = grid_boxes(wh, mu[0] * wh[0], mu[1] * wh[1], 2)
    c_all = cnll(cand, objs, wh)
    for crit, b in res.items():
        assert b.shape == (30, 4)
        assert np.allclose(b[:, 2] - b[:, 0], mu[0] * wh[0]) and np.allclose(b[:, 3] - b[:, 1], mu[1] * wh[1])   # cỡ = μ
    assert np.isclose(cnll(res["cnll"][:1], objs, wh)[0], c_all.min())                                    # đúng min C-NLL lưới
    e_all = np.linalg.norm(cnll_features(cand, objs, wh) - mu, axis=1)
    assert np.isclose(np.linalg.norm(cnll_features(res["euclid"][:1], objs, wh) - mu), e_all.min())
    assert cnll(res["cnll"][:1], objs, wh)[0] <= cnll(res["euclid"][:1], objs, wh)[0] + 1e-12


def test_few_objects_center_box():
    objs = np.array([[0, 0, 10, 20], [50, 50, 70, 60.0]])
    b = search(objs, (100.0, 80.0), 5, 2)["cnll"]
    assert b.shape == (5, 4) and np.allclose(b[0], [42.5, 32.5, 57.5, 47.5]) and (b == b[0]).all()          # cỡ TB 15 × 15, giữa ảnh


def test_predict_without_ckpt(tmp_path, monkeypatch):
    import baseline.predict as predict
    root, samples, ti = fake_ce130(str(tmp_path / "data"))
    review = fake_cocount_review(str(tmp_path / "data" / "cocount"))
    cfg_path, cfg = tiny_cfg(tmp_path, "zeta7")
    data = ["--turn-index", ti, "--samples-root", samples, "--cocount-root", tmp_path / "data" / "cocount", "--cocount-review", review]
    for ds, key in (("ce130", "inpainted"), ("cocount", "cocount")):
        out = tmp_path / "o" / ds / "grid.json"
        monkeypatch.setattr(sys, "argv", ["predict", "--config", cfg_path, "--dataset", ds, "--out", str(out)] + [str(x) for x in data])
        predict.main()
        got = {}
        for c in ("euclid", "cnll"):
            d = json.load(open(tmp_path / "o" / ds / f"grid_{c}.json"))
            raw = [json.loads(x) for x in open(tmp_path / "o" / ds / f"grid_{c}_raw.jsonl")]
            assert d["criterion"] == c and d["method"] == "oracleloc" and d["ckpt"] is None
            assert all(len(r["boxes"]) == cfg["predict"]["k"] for r in raw)
            got[c] = d["results"][key]["cnll"]
        assert got["cnll"] <= got["euclid"] + 1e-6

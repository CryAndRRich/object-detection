"""Tool đọc dump của eval (không model): `cnll_outliers.py`, `plot_cnll_worst.py` (CE-CoCount + CE-130)."""

import json
import os
import sys

import numpy as np


def test_cnll_outliers_tool(tmp_path, monkeypatch):
    """sel = min C-NLL từng mẫu; trung bình bỏ p % lớn nhất; outlier trùng giữa model; ghi summary + 3 hình."""
    from ce_localization.tools import cnll_outliers as co
    rng = np.random.default_rng(0)
    rows = lambda bump: [{"image_id": f"im{i}", "t": 1, "n_obj": 5 + i % 7, "boxes": [],  # noqa: E731
                          "cnll": (rng.gamma(1.0, 1.0, 4) + (5000.0 if i in bump else 0)).tolist(),
                          "cnll_gt": rng.gamma(1.0, 0.5, 2).tolist()} for i in range(200)]
    paths = []
    for name, bump in (("A", {0, 1, 2}), ("B", {0, 1, 3})):
        p = tmp_path / f"{name}_boxes.json"
        p.write_text(json.dumps({"results": {"inpainted": rows(bump)}}))
        paths.append(f"{name}={p}")
    ps = co.per_sample(json.loads((tmp_path / "A_boxes.json").read_text())["results"]["inpainted"])
    assert len(ps) == 200 and ps["im0|1"][0] > 5000 and ps["im5|1"][0] < 50
    s = co.summarize([v[0] for v in ps.values()])
    assert s["mean"] > s["mean_trim1"] > 0 and s["median"] < 5 and 0.6 < s["top1pct_share"] < 0.7   # 1 % = 2 / 3 mẫu bị đẩy
    out = str(tmp_path / "o")
    monkeypatch.setattr(sys, "argv", ["cnll_outliers.py", "--dump", *paths, "--top", "3", "--out", out])
    co.main()
    r = json.load(open(os.path.join(out, "summary.json")))
    assert r["outliers_in_all_models"] == ["im0|1", "im1|1"] and r["overlap_matrix"][0][1] == 2 / 3
    assert sorted(os.listdir(out)) == ["outliers_heatmap.png", "sorted_sel.png", "summary.json", "trimmed_mean.png"]


def _rows(rng, items):
    """items: [(image_id, t, nw, nh, record cho record_cnll)] -> dòng dump như eval.py (box 2 số lẻ, C-NLL tính trên box đã làm tròn)."""
    from ce_localization.engine.add_eval import record_cnll
    rows = []
    for iid, t, nw, nh, rec in items:
        xy = rng.uniform(0, 1, (6, 2)) * [nw, nh]
        boxes = np.concatenate([xy, xy + rng.uniform(3, 30, (6, 2))], 1).round(2)
        row = {"image_id": iid, "t": t, "wh": [float(nw), float(nh)], "boxes": boxes.tolist()}
        rc = record_cnll({**rec, "boxes": boxes, "wh": (float(nw), float(nh))})
        if rc is not None:
            row["cnll"] = rc[0].tolist()
        rows.append(row)
    return rows


def test_plot_cnll_worst_cocount_and_ce130(tmp_path, monkeypatch):
    """Hình học dựng lại (CE-CoCount qua bản rà tay; CE-130 từ turn_index) == dataset lúc eval; C-NLL từng box == dump (`record_cnll`,
    F1all với lớp kia); xếp theo box tốt nhất giảm dần; `--top-pct` vẽ đúng số mẫu, mỗi mẫu một hình + rank.json."""
    from ce_localization.data.cocount import CoCountAddDataset, load_review
    from ce_localization.data.turns import CE130AddDataset
    from ce_localization.tools import plot_cnll_worst as pw
    from tests.ce_localization.helpers import _fake_cocount, _fake_review, _fake_turn_index
    rng = np.random.default_rng(0)
    cc = _fake_cocount(str(tmp_path / "cc"))
    rvp = _fake_review(cc, str(tmp_path / "rv.jsonl"))
    ds = CoCountAddDataset(cc, 128, review=load_review(rvp), min_obj=5)
    items = []
    for i in range(len(ds)):
        s = ds[i]
        rec = {k: s[k].numpy().astype(np.float64) for k in ("objects", "objects_all", "holes")}
        items.append((s["image_id"], 0, s["valid_hw"][1], s["valid_hw"][0], rec))
    base = str(tmp_path / "d")
    os.makedirs(base)
    root, samples, tidx, index, _ = _fake_turn_index(base)
    ce = CE130AddDataset(index, root, samples, "train", 128)
    items_ce = []
    for i in range(len(ce)):
        s = ce[i]
        g = pw.ce130_geometry(index, samples, s["image_id"], 128)
        assert np.array_equal(g["objects"], s["objects"].double().numpy()) and np.array_equal(g["holes"], s["holes"].double().numpy())
        items_ce.append((s["image_id"], s["t"], s["valid_hw"][1], s["valid_hw"][0],
                         {k: s[k].numpy().astype(np.float64) for k in ("objects", "holes")}))
    for name, rows, extra in (("cocount", _rows(rng, items), ["--cocount-root", cc, "--cocount-review", rvp]),
                              ("inpainted", _rows(rng, items_ce), ["--dataset", "ce130", "--turn-index", tidx, "--samples-root", samples])):
        dump = str(tmp_path / f"{name}_boxes.json")
        json.dump({"results": {name: rows}}, open(dump, "w"))
        out = str(tmp_path / f"viz_{name}")
        monkeypatch.setattr(sys, "argv", ["plot_cnll_worst.py", dump, *extra, "--image-size", "128", "--top-pct", "50", "--out", out])
        pw.main()
        r = json.load(open(os.path.join(out, "rank.json")))["samples"]
        ref = {x["image_id"]: min(x["cnll"]) for x in rows if "cnll" in x}
        assert len(r) == len(ref) > 0 and all(np.isclose(x["min"], ref[x["image_id"]], rtol=1e-9) for x in r)
        assert [x["min"] for x in r] == sorted((x["min"] for x in r), reverse=True)
        top = int(np.ceil(len(r) * 0.5))
        assert sorted(os.listdir(out)) == ["rank.json"] + [f"worst_{x['rank']:02d}_{x['image_id']}.png" for x in r[:top]]

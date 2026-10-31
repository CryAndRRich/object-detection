"""Tool đọc dump của eval (không model): `cnll_outliers.py`, `plot_snapshots.py`, `plot_snapshot_compare.py`, `plot_cnll_worst.py` (CE-CoCount + CE-130), `rescore_cnll.py`."""

import argparse
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


def test_plot_snapshots_tool(tmp_path, monkeypatch):
    """Mỗi run / epoch (tên epNNN): TB / trung vị / bỏ 1 % của C-NLL box thấp nhất mỗi ảnh, mốc GT, hit50 từ JSON chỉ số; ghi hình."""
    from ce_localization.tools import plot_snapshots as ps
    rng = np.random.default_rng(1)
    specs = []
    for name in ("A", "B"):
        d = tmp_path / name
        d.mkdir()
        for ep, bump in ((10, 1000.0), (20, 0.0)):
            rows = [{"image_id": f"im{i}", "t": 1, "n_obj": 6, "cnll": (rng.gamma(1.0, 1.0, 3) + (bump if i == 0 else 0)).tolist(),
                     "cnll_gt": [0.5]} for i in range(100)]
            (d / f"ep{ep:03d}_boxes.json").write_text(json.dumps({"results": {"inpainted": rows}}))
            (d / f"ep{ep:03d}.json").write_text(json.dumps({"n_samples": 3, "results": {"inpainted": {"hit50@3_any": ep / 100,
                                                                                                    "mean_iou_any": 0.1}}}))
            if ep == 10:
                sel = np.sort([min(r["cnll"]) for r in rows])
        specs.append(f"{name}={d}")
    out = str(tmp_path / "o")
    monkeypatch.setattr(sys, "argv", ["plot_snapshots.py", *sum([["--run", x] for x in specs], []), "--out", out])
    ps.main()
    r = json.load(open(os.path.join(out, "summary.json")))
    e10 = r["B"][0]
    assert [x["epoch"] for x in r["A"]] == [10, 20] and e10["n"] == 100 and e10["gt"] == 0.5 and e10["hit50"] == 0.1
    assert np.isclose(e10["mean"], sel.mean()) and np.isclose(e10["median"], np.median(sel)) and np.isclose(e10["trim1"], sel[:-1].mean())
    assert e10["mean"] > r["B"][1]["mean"] and sorted(os.listdir(out)) == ["snapshots.png", "summary.json"]


def test_plot_snapshot_compare_tool(tmp_path, monkeypatch):
    """history.json có resume (elapsed đặt lại) -> thời lượng = tổng các phiên, iter trùng giữ bản sau; đọc cả khoá `<key>_mock` (DELTA6);
    run thiếu snapshot / dump vẫn vẽ; tập (CE-130 train) chỉ một run có vẫn thành ô / hàng, không run nào có thì bỏ; ghi đủ 2 hình."""
    from ce_localization.tools import plot_snapshot_compare as pc
    rng = np.random.default_rng(2)
    W, S = tmp_path / "w", tmp_path / "s"
    for run, ipe, mock in (("a", 10, True), ("b", 20, False)):
        hist, el = [], 0.0
        for it in range(5, 41 * ipe // 10 + 1, 5):
            el = 0.0 if it == 20 * ipe // 10 + 5 else el + 3.0          # phiên 2 bắt đầu sau iter 2 epoch
            hist.append({"iter": it, "loss": 1 / it, "loss_eps": 0.5 / it, "loss_aux": 0.2 / it, "aux_ratio": 0.9, "elapsed_sec": el})
            if it % ipe == 0:
                hist.append({"iter": it, "epoch": it // ipe, "epoch_train_loss": 1 / it})
        hist.append({**hist[0], "loss": 99.0})                          # iter trùng sau resume
        (W / run).mkdir(parents=True)
        (W / run / "history.json").write_text(json.dumps({"config": {"training": {"batch_size": 8, "lr": 1e-4}}, "history": hist}))
        for sub, key in (("ce130_test_all", "inpainted"), ("cocount_review5", "cocount")) + ((("ce130_train", "inpainted"),) if run == "b" else ()):
            (S / run / sub).mkdir(parents=True)
            k = key + "_mock" if mock else key
            for ep in ((1, 2, 4) if run == "a" else (2, 4)):
                rows = [{"image_id": f"im{i}", "cnll": (rng.gamma(1.0, 2.0, 30) - 1).tolist()} for i in range(20)]
                (S / run / sub / f"ep{ep:03d}_boxes.json").write_text(json.dumps({"results": {k: rows}}))
                (S / run / sub / f"ep{ep:03d}.json").write_text(json.dumps({"results": {k: {"cnll": 1.0 + ep, "cnll_gt": 2.0}}}))
    ep, ipe, h, sec, sess, _ = pc.load_history(str(W / "a" / "history.json"))
    assert ipe == 10 and sess == 2 and h["loss"][0] == 99.0 and len(ep) == len(set(ep)) == 8
    assert sec == 3.0 * 4 + 3.0 * 3                                       # phiên 1: iter 5..20, phiên 2: 25..40 (đặt lại ở 25)
    assert pc.result(str(S / "a" / "ce130_test_all" / "ep001.json"), "inpainted")["cnll"] == 2.0
    ns = argparse.Namespace(snap_root=str(S), epochs=[1, 2, 4], density_epochs=[2, 4])
    names = lambda runs: [st[0] for st in pc.present_sets([{"run": r} for r in runs], ns)]
    assert names(["a", "b"]) == ["ce130_train", "ce130_test_all", "cocount_review5"]
    assert names(["a"]) == ["ce130_test_all", "cocount_review5"]
    out = tmp_path / "o" / "cmp"
    monkeypatch.setattr(sys, "argv", ["plot_snapshot_compare.py", "--out-prefix", str(out), "--weights-root", str(W), "--snap-root", str(S),
                                      "--run", "a=Base", "--run", "b=+ More", "--epochs", "1", "2", "4", "--density-epochs", "2", "4",
                                      "--smooth-iters", "10"])
    pc.main()
    assert sorted(os.listdir(tmp_path / "o")) == ["cmp_cnll_all_boxes_density.png", "cmp_overview.png"]
    monkeypatch.setattr(sys, "argv", ["plot_snapshot_compare.py", "--out-prefix", str(out), "--run", "a"])
    with __import__("pytest").raises(SystemExit):
        pc.main()


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
    F1all với lớp kia); xếp theo box tốt nhất giảm dần; `--top-pct` vẽ đúng số mẫu, mỗi mẫu một hình + rank.json; box vẽ = <= 5 box C-NLL
    thấp nhất không trùng nhau (IoU <= 0,5); nhiều dump: vẽ đúng giao top p % của mọi dump (lưới GT + từng setting) + intersect.json; `--trend`: xếp theo C-NLL dump cuối −
    dump đầu giảm dần, vẽ đúng `--top` mẫu + trend.json."""
    from ce_localization.utils.box_ops_np import box_iou
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
        bx = {x["image_id"]: np.asarray(x["boxes"], dtype=np.float64) for x in rows}
        for x in r:
            sb = bx[x["image_id"]][x["sel"]]
            iou = box_iou(sb, sb)[0] - np.eye(len(sb))
            assert 0 < len(x["sel"]) <= 5 and iou.max() <= 0.5 and np.isclose(x["sel_cnll"][0], x["min"])
        rows2 = _rows(rng, items if name == "cocount" else items_ce)
        dump2 = str(tmp_path / f"{name}_boxes2.json")
        json.dump({"results": {name: rows2}}, open(dump2, "w"))
        out2 = str(tmp_path / f"inter_{name}")
        monkeypatch.setattr(sys, "argv", ["plot_cnll_worst.py", f"A={dump}", f"B={dump2}", *extra, "--image-size", "128",
                                          "--intersect-pct", "50", "--out", out2])
        pw.main()
        ref2 = {x["image_id"]: min(x["cnll"]) for x in rows2 if "cnll" in x}
        k = int(round(len(ref) / 2))
        want = set(sorted(ref, key=ref.get, reverse=True)[:k]) & set(sorted(ref2, key=ref2.get, reverse=True)[:k])
        got = json.load(open(os.path.join(out2, "intersect.json")))["samples"]
        assert {x["image_id"] for x in got} == want and len(os.listdir(out2)) == len(want) + 1
        rows3 = _rows(rng, items if name == "cocount" else items_ce)
        dump3 = str(tmp_path / f"{name}_boxes3.json")
        json.dump({"results": {name: rows3}}, open(dump3, "w"))
        out3 = str(tmp_path / f"trend_{name}")
        monkeypatch.setattr(sys, "argv", ["plot_cnll_worst.py", "--trend", f"ep010={dump}", f"ep020={dump2}", f"ep030={dump3}", *extra,
                                          "--image-size", "128", "--top", "2", "--select", "30", "--out", out3])
        pw.main()
        ref3 = {x["image_id"]: min(x["cnll"]) for x in rows3 if "cnll" in x}
        tr = json.load(open(os.path.join(out3, "trend.json")))["samples"]
        ids = sorted(set(ref) & set(ref2) & set(ref3))
        rise = {i: ref3[i] - ref[i] for i in ids}
        assert [x["image_id"] for x in tr] == sorted(ids, key=lambda i: -rise[i]) and all(np.isclose(x["rise"], rise[x["image_id"]]) for x in tr)
        assert all(np.isclose(x["min"]["ep020"], ref2[x["image_id"]]) and -1 <= x["rho"] <= 1 for x in tr)
        assert sorted(os.listdir(out3)) == sorted(["trend.json"] + [f"trend_{j:02d}_{x['image_id']}.png" for j, x in enumerate(tr[:2], 1)])


def test_rescore_cnll_shift():
    """C-NLL cũ (trừ min NLL trên vật) + `calib_shift` (½·min D² của vật) == C-NLL mới (trừ NLL tại μ); lượng ∈ [0, 2]; < 5 vật -> None."""
    from scipy.stats import multivariate_normal
    from ce_localization.engine.add_eval import CNLL_RIDGE, _features, cnll
    from ce_localization.tools.rescore_cnll import calib_shift
    rng = np.random.default_rng(3)
    wh = (200.0, 150.0)
    for n in (5, 9, 30):
        xy = rng.uniform(0, 160, (n, 2))
        objs = np.concatenate([xy, xy + rng.uniform(8, 30, (n, 2))], 1)
        boxes = np.concatenate([xy[:4] + 5, xy[:4] + 25], 1)
        z = _features(objs, objs, wh, is_objects=True)
        mvn = multivariate_normal(z.mean(0), np.cov(z.T, bias=True) + CNLL_RIDGE * np.eye(4))
        old = -mvn.logpdf(_features(boxes, objs, wh)) - (-mvn.logpdf(z)).min()
        s = calib_shift(objs, wh)
        assert 0 <= s <= 2 and np.allclose(old + s, cnll(boxes, objs, wh), atol=1e-6)
    assert calib_shift(objs[:4], wh) is None


def test_rescore_cnll_shift_for_keys():
    """Khoá file tổng -> lượng dời: khớp đúng khoá; `*_objsize*` / prior không hậu tố sampler dùng khoá cùng gốc; không có -> None."""
    from ce_localization.tools.rescore_cnll import shift_for
    one = {"cocount": (10, 0.3), "cocount_objsize": (10, 0.3)}
    two = {"inpainted_mock": (6, 0.4), "inpainted_ddim": (6, 0.4), "cocount_mock": (5, 0.2), "cocount_ddim_objsize": (5, 0.2)}
    assert shift_for("cocount_objsize_exemplar", one) == (10, 0.3)
    assert shift_for("inpainted", two) == (6, 0.4) and shift_for("inpainted_ddim", two) == (6, 0.4)
    assert shift_for("cocount_objsize", two) == (5, 0.2) and shift_for("cocount_mock_objsize_exemplar", two) == (5, 0.2)
    assert shift_for("original", two) == (None, None)

"""CE-CoCount (docs/EXPERIMENT_GAMMA.md mục 18): dataset (`data/cocount.py`), `add_metrics` với `objects_all` / `latest=False`,
C-NLL của box GT (`tools/gt_cnll.py`), TRỌN LUỒNG `eval.py --dataset cocount` cho checkpoint kiểu bài, CE-Loc + box vật (GAMMA4)
và CE-Loc + refine (GAMMA2) — checkpoint giả dựng thẳng từ config, không train."""

import json
import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

NAMES = ("INTRA_FOO_TOM1_TOM2_00008_00006_0_200", "INTER_OFF_PEN0_PNC0_00007_00005_0_40")


def _fake_cocount(root, seed=0):
    """CE-CoCount giả: mỗi frame hai file `_positive` / `_negative`, vật hai lớp xen kẽ, 10 loc_bbox ở chỗ trống."""
    rng = np.random.default_rng(seed)
    for d in ("Image", "Anno", "Anno_with_exam_bbox"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    for fi, stem in enumerate(NAMES):
        W, H = (200, 120) if fi == 0 else (90, 160)
        Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(os.path.join(root, "Image", stem + "_positive.jpg"))
        Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(os.path.join(root, "Image", stem + "_negative.jpg"))
        na, nb = int(stem.split("_")[4]), int(stem.split("_")[5])
        for pol, n, cls, off in (("positive", na, "normal tomato(Big round tomato).", 0), ("negative", nb, "baby tomato.", 9)):
            objs = [[float(5 + 11 * k), float(5 + off), float(13 + 11 * k), float(13 + off)] for k in range(n)]
            loc = [[float(4 + 8 * k), float(40 + (k % 2) * 10), float(12 + 8 * k), float(48 + (k % 2) * 10)] for k in range(10)]
            anno = {"class_name": cls, "loc_bbox": loc, "exam_bbox": objs[:3], "source_img_name": stem,
                    "counting_anno": {"count": n, "points": [[(o[0] + o[2]) / 2, (o[1] + o[3]) / 2] for o in objs],
                                      "annotated_count": n, "exemplars": [list(map(int, o)) for o in objs[:3]]}}
            with open(os.path.join(root, "Anno", f"{stem}_{pol}.json"), "w") as f:
                json.dump(anno, f)
            with open(os.path.join(root, "Anno_with_exam_bbox", f"{stem}_{pol}.json"), "w") as f:
                json.dump({"class_name": cls, "loc_bbox": [], "exam_bbox": [{"bbox": o, "score": 0.9} for o in objs],
                           "source_img_name": stem, "points": anno["counting_anno"]["points"]}, f)
    return root


def test_clean_class_and_twin():
    from ce_localization.data.cocount import clean_class, twin_name
    assert clean_class("normal tomato(Big round tomato).") == "normal tomato" and clean_class("paper clip.") == "paper clip"
    assert twin_name("A_B_positive") == "A_B_negative" and twin_name("A_B_negative") == "A_B_positive"


def test_filter_objects_drops_giant_sam_boxes(tmp_path):
    """Box SAM khoanh cả đống vật (cạnh > k × cạnh TB exemplar) bị bỏ ở cả lớp này lẫn lớp kia; không lọc thì giữ nguyên."""
    from ce_localization.data.cocount import CoCountAddDataset, filter_objects, read_cocount
    ex = np.array([[0, 0, 8, 8], [10, 0, 18, 8]], float)
    objs = np.array([[0, 0, 8, 8], [0, 0, 20, 20], [0, 0, 30, 30], [0, 0, 100, 90]], float)   # cạnh 8 / 20 / 30 / ~95; mốc 8
    assert len(filter_objects(objs, ex, 3)) == 2 and len(filter_objects(objs, ex, 2)) == 1
    assert filter_objects(objs, ex, None) is objs and filter_objects(objs, np.zeros((0, 4)), 3) is objs
    root = _fake_cocount(str(tmp_path / "cc"))
    name, twin = NAMES[0] + "_positive", NAMES[0] + "_negative"
    for n in (name, twin):                                                    # thêm 2 box khổng lồ vào mỗi lớp
        f = os.path.join(root, "Anno_with_exam_bbox", n + ".json")
        e = json.load(open(f))
        e["exam_bbox"] += [{"bbox": [0, 0, 150, 110], "score": 0.9}, {"bbox": [1, 1, 149, 109], "score": 0.9}]
        json.dump(e, open(f, "w"))
    raw, filt = read_cocount(root, name), read_cocount(root, name, 3)
    assert len(raw["objects"]) == 10 and len(filt["objects"]) == 8 and len(raw["objects_all"]) - len(filt["objects_all"]) == 4
    assert np.allclose(filt["exemplars"], raw["exemplars"])
    ds = CoCountAddDataset(root, 128, density=None, max_obj_ratio=3)
    assert ds[ds.keys.index(name)]["objects"].shape == (8, 4)


def _fake_review(root, path, drop_auto=None):
    """Bản rà giả (khuôn `box_review/app.py`): mỗi file loại tay box 0; `drop_auto` {tên: [chỉ số]} = lọc trùng tự động."""
    with open(path, "w") as f:
        for n in sorted(x[:-5] for x in os.listdir(os.path.join(root, "Anno_with_exam_bbox"))):
            m = len(json.load(open(os.path.join(root, "Anno_with_exam_bbox", n + ".json")))["exam_bbox"])
            f.write(json.dumps({"id": n, "bad": {n: [0]}, "auto": {n: (drop_auto or {}).get(n, [])}, "n": {n: m}, "share": {},
                                "time": "", "sec": 1.0}) + "\n")
    return path


def test_review_filter_and_min_obj(tmp_path):
    """Bản rà tay: bỏ `bad` ∪ `auto` theo chỉ số ở CẢ hai lớp (giữ thứ tự), exemplar không đổi; `min_obj` chỉ giữ file còn đủ box
    vật cùng lớp; bản rà thiếu file / lệch số box / dùng chung với max_obj_ratio -> lỗi."""
    from ce_localization.data.cocount import CoCountAddDataset, load_review, read_cocount
    from ce_localization.tools.gt_cnll import cocount_gt
    root = _fake_cocount(str(tmp_path / "cc"))
    name, twin = NAMES[0] + "_positive", NAMES[0] + "_negative"
    rv = load_review(_fake_review(root, str(tmp_path / "rv.jsonl"), {name: [2, 5]}))
    assert rv[name] == (8, {0, 2, 5}) and rv[twin] == (6, {0})
    raw, r = read_cocount(root, name), read_cocount(root, name, review=rv)
    assert np.array_equal(r["objects"], raw["objects"][[1, 3, 4, 6, 7]])
    assert np.array_equal(r["objects_all"], np.concatenate([raw["objects"][[1, 3, 4, 6, 7]], read_cocount(root, twin)["objects"][1:]]))
    assert np.array_equal(r["exemplars"], raw["exemplars"]) and np.array_equal(r["loc"], raw["loc"])
    ds = CoCountAddDataset(root, 128, density=None, review=rv, min_obj=5)        # còn 5 / 5 / 6 / 4 vật cùng lớp
    assert ds.n_all == 4 and sorted(ds.keys) == sorted([name, twin, NAMES[1] + "_positive"])
    assert ds[ds.keys.index(name)]["objects"].shape == (5, 4)
    assert len(CoCountAddDataset(root, 128, density=None, review=rv)) == 4
    assert sorted(len(x[1]) for x in cocount_gt(root, review=rv, min_obj=5)) == [5, 5, 6]          # tool C-NLL box GT: cùng tập
    with pytest.raises(ValueError, match="MỘT"):
        read_cocount(root, name, 3, rv)
    with pytest.raises(ValueError, match="chưa có"):
        read_cocount(root, name, review={k: v for k, v in rv.items() if k != twin})
    with pytest.raises(ValueError, match="không khớp"):
        read_cocount(root, name, review={**rv, name: (9, set())})


def test_cocount_dataset_fields_and_density(tmp_path):
    from ce_localization.data.cocount import CoCountAddDataset, read_cocount
    from ce_localization.data.turns import collate_add, to_device_add
    root = _fake_cocount(str(tmp_path / "cc"))
    ds = CoCountAddDataset(root, 128, density="empty")
    assert len(ds) == 4 and ds.classes() == ["baby tomato", "normal tomato"]
    it = ds[ds.keys.index(NAMES[0] + "_positive")]
    r = read_cocount(root, NAMES[0] + "_positive")
    assert len(r["objects"]) == 8 and len(r["objects_all"]) == 14 and len(r["loc"]) == 10
    scale = 128 / 200
    assert it["holes"].shape == (10, 4) and torch.allclose(it["holes"][0], torch.tensor(r["loc"][0]).float() * scale)
    assert it["objects"].shape == (8, 4) and it["objects_all"].shape == (14, 4) and it["t"] == 0
    assert len(r["exemplars"]) == 3 and torch.allclose(it["exemplars"], torch.tensor(r["exemplars"]).float() * scale)
    assert it["text"] == "normal tomato" and it["valid_hw"] == (int(120 * scale), 128)
    img = it["image"]                                                         # [4, T, T] uint8 (view HWC), kênh 4 = density trống
    nh, nw = it["valid_hw"]
    assert img.shape == (4, 128, 128) and int(img[3, :nh, :nw].unique().item()) == 14 and int(img[3, nh:, :].max()) == 0
    b = to_device_add(collate_add([ds[0], ds[1]]), torch.device("cpu"))
    assert len(b["objects_all"]) == 2 and len(b["exemplars"]) == 2 and b["images"].shape == (2, 4, 128, 128)
    assert CoCountAddDataset(root, 128, density=None)[0]["image"].shape[0] == 3
    with pytest.raises(ValueError):
        CoCountAddDataset(root, 128, density="sample")


def test_add_metrics_objects_all_and_no_latest():
    from ce_localization.engine.add_eval import add_metrics
    objs = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10], [60, 0, 70, 10], [80, 0, 90, 10]], float)
    other = np.array([[0, 50, 10, 60]], float)
    rec = {"image_id": "x", "t": 0, "wh": np.array([100.0, 100.0]), "holes": np.array([[0, 30, 10, 40], [50, 30, 60, 40]], float),
           "boxes": np.array([[0, 30, 10, 40], [0, 50, 10, 60]], float), "objects": objs}
    base = add_metrics([rec])
    assert base["on_object"] == 0 and "best_iou@2_latest" in base and "by_turn" in base
    res = add_metrics([dict(rec, objects_all=np.concatenate([objs, other]))], latest=False)
    assert res["on_object"] == 0.5                                            # box 2 đè vật lớp KIA
    assert not any("latest" in k for k in res) and "by_turn" not in res
    assert res["best_iou@2_any"] == 1.0 and res["hole_cover"] == 0.5
    assert res["cnll_F1_n1_median"] == base["cnll_F1_n1_median"]             # C-NLL vẫn trên vật cùng lớp
    assert not any("F1all" in k for k in base) and res["n_cnll"] == 1         # F1all chỉ khi có `objects_all` (CE-CoCount)
    assert res["cnll_F1all_n1_median"] > res["cnll_F1_n1_median"]             # box 2 đè vật lớp kia: F1all phạt


def test_resize_to_objects():
    from ce_localization.engine.add_eval import resize_to_objects
    objs = np.array([[0, 0, 10, 20], [50, 50, 70, 60]], float)                 # TB w 15, TB h 15
    rec = {"image_id": "x", "t": 0, "wh": np.array([100.0, 100.0]), "boxes": np.array([[10, 10, 30, 50], [0, 0, 1, 1]], float),
           "objects": objs, "objects_all": np.concatenate([objs, [[0, 0, 99, 99]]])}
    out = resize_to_objects([rec, dict(rec, objects=np.zeros((0, 4)))])
    assert np.allclose(out[0]["boxes"], [[12.5, 22.5, 27.5, 37.5], [-7, -7, 8, 8]])  # giữ tâm, không kẹp, chỉ vật cùng lớp
    assert out[1]["boxes"] is rec["boxes"] and rec["boxes"][0, 2] == 30          # 0 vật: giữ nguyên; record gốc không đổi
    from ce_localization.engine.add_eval import object_size
    many = np.array([[0, 0, 10, 10]] * 7 + [[0, 0, 12, 8], [0, 0, 200, 300]], float)   # 1 box SAM khoanh cả đống
    r = dict(rec, objects=many, exemplars=np.array([[0, 0, 4, 6], [0, 0, 6, 8]], float))
    assert object_size(r, "mean")[0] == pytest.approx((70 + 12 + 200) / 9)
    assert object_size(r, "median") == (10.0, 10.0)
    assert object_size(r, "trim") == pytest.approx((82 / 8, 78 / 8))           # bỏ đúng box outlier
    assert object_size(r, "small") == pytest.approx((82 / 8, 78 / 8))           # cạnh <= trung vị 10: 7 box 10×10 + box 12×8 (cạnh 9,8)
    assert object_size(r, "exemplar") == (5.0, 7.0) and object_size(rec, "exemplar") is None
    assert resize_to_objects([rec], "exemplar")[0] is rec                       # không có box mẫu: giữ nguyên
    with pytest.raises(ValueError):
        object_size(r, "max")


def test_gt_cnll_tool(tmp_path, monkeypatch):
    from ce_localization.tools import gt_cnll as g
    from tests.ce_localization.helpers import _fake_turn_index
    base = str(tmp_path / "d")
    os.makedirs(base)
    _, _, tpath, index, _ = _fake_turn_index(base)
    ce = g.ce130_gt(index, "test", "ce130")
    keys = index.keys("test", "ce130")
    assert len(ce) == len(keys)
    for (gt, objs, wh), k in zip(ce, keys):
        e = index.turns[k]
        b = index.branches[e["branch"]]
        assert len(gt) == e["t"] and len(objs) == len(b["objects"]) - e["t"] and wh == tuple(b["wh"])
    s = g.summarize(ce, latest=True)
    assert s["n_samples"] == len(keys) and 0 < s["n_cnll"] <= len(keys)
    assert s["cnll_F1_sel_median"] <= s["cnll_F1_any_n1_median"] and np.isfinite(s["cnll_F2_latest_n1_median"])
    root = _fake_cocount(str(tmp_path / "cc"))
    cc = g.cocount_gt(root)
    assert len(cc) == 4 and all(len(x[0]) == 10 for x in cc) and cc[0][2] in ((200, 120), (90, 160))
    assert [len(x[1]) for x in g.cocount_gt(root, 3)] == [len(x[1]) for x in cc]          # dữ liệu giả không có box khổng lồ
    assert all(len(x) == 4 and len(x[3]) > 0 for x in cc)                            # + vật lớp kia (F1all)
    sc = g.summarize(cc)
    assert sc["n_cnll"] == 4 and not any("latest" in k for k in sc) and np.isfinite(sc["cnll_F1all_any_n1_median"])
    assert not any("F1all" in k for k in s)                                             # CE-130: một lớp / ảnh
    rz = g.resize_gt(cc)
    for (b, o, *_), (b0, o0, *_) in zip(rz, cc):
        assert np.allclose((b[:, :2] + b[:, 2:]) / 2, (b0[:, :2] + b0[:, 2:]) / 2) and o is o0
        assert np.allclose(b[:, 2] - b[:, 0], (o[:, 2] - o[:, 0]).mean())
    out = str(tmp_path / "gt.json")
    monkeypatch.setattr(sys, "argv", ["gt_cnll.py", "--turn-index", tpath, "--split-source", "ce130", "--cocount-root", root,
                                      "--cocount-obj-filter", "3", "--out", out])
    g.main()
    with open(out) as f:
        res = json.load(f)
    assert set(res) == {"ce130", "cocount", "cocount_objsize"} and res["cocount"]["n_cnll"] == 4 and res["cocount"]["obj_filter"] == 3


@pytest.mark.parametrize("kind", ["paper", "obj_rand", "pr", "review"])
def test_eval_cocount_full_flow(tmp_path, monkeypatch, kind):
    """eval.py --dataset cocount: khoá `cocount*`, đủ mẫu, không chỉ số `_latest`, on_object trên cả hai lớp, dump box chạy được;
    `obj_rand` chạy HAI sampler (ddpm + ddim của DELTA5) ⇒ khoá hậu tố `_ddpm` / `_ddim` kèm `_noobj` / `_objsize*`; `review` (2026-10-22):
    box vật theo bản rà tay + chỉ file còn >= 6 vật cùng lớp (2 / 4 file), đổi cỡ box theo box vật ĐÃ rà."""
    import ce_localization.eval as ea
    import ce_localization.train as ta
    from ce_localization.models.detector import build_model
    from tests.ce_localization.helpers import _fake_paper_ckpt, _fake_text_table, _fake_turn_index, _gamma2_cfg, _gamma_cfg
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    root = _fake_cocount(str(tmp_path / "cc"))
    monkeypatch.setattr(ta, "build_text_table", lambda names, cfg, dev, state_dict=None: _fake_text_table(names, cfg, dev))
    ck = str(tmp_path / "ck.pth")
    rv = _fake_review(root, str(tmp_path / "rv.jsonl")) if kind == "review" else None
    if kind == "paper":
        cfg_path, _ = _gamma_cfg(tmp_path, base, "density")
        _fake_paper_ckpt(ck, T=100)
        extra = ["--config", cfg_path, "--add-samplers", "mock"]
    else:
        _, cfg = _gamma2_cfg(tmp_path, base, "obj_rand" if kind == "review" else kind)
        torch.manual_seed(0)
        model = build_model(cfg, pretrained_backbone=False)
        torch.save({"model": model.state_dict(), "config": cfg, "iter": 0}, ck)
        extra = (["--refine-t", "none", "5", "--proposer-sampler", "ddpm"] if kind == "pr" else
                 ["--add-samplers", "ddpm", "--cocount-review", rv, "--cocount-min-obj", "6"] if kind == "review" else
                 ["--add-samplers", "ddpm", "ddim", "--cocount-obj-filter", "3"])     # config thu nhỏ T = 20: mock 100 bước không chạy
    out, dump = str(tmp_path / "res.json"), str(tmp_path / "boxes.json")
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", ck, "--dataset", "cocount", "--cocount-root", root, "--n-samples", "3",
                                      "--num-workers", "0", "--out", out, "--dump-boxes", dump, "--device", "cpu", "--obj-size", "mean",
                                      "exemplar"]
                        + extra)
    ea.main()
    with open(out) as f:
        res = json.load(f)
    want = {"paper": {"cocount"}, "obj_rand": {"cocount" + s + o for s in ("_ddpm", "_ddim") for o in ("", "_noobj")},
            "pr": {"cocount_ce", "cocount_t5"}, "review": {"cocount", "cocount_noobj"}}[kind]
    want |= {k + sfx for k in want for sfx in ("_objsize", "_objsize_exemplar")}
    assert set(res["results"]) == want and res["dataset"] == "cocount" and res["density"] == {"cocount": "empty"}
    assert res["cocount_obj_filter"] == (3 if kind.startswith("obj") else None)
    assert res.get("obj_k") == (3 if kind in ("obj_rand", "review") else None)
    n = 2 if kind == "review" else 4                                          # rà: còn 7 / 5 / 6 / 4 vật -> giữ 7 và 6
    assert (res["cocount_files"], res["cocount_files_all"], res["cocount_review"], res["cocount_min_obj"]) == \
        (n, 4, rv, 6 if kind == "review" else None)
    for k, r in res["results"].items():
        assert r["n"] == n and 0 <= r["best_iou@3_any"] <= 1 and 0 <= r["on_object"] <= 1 and r["n_cnll"] == n
        assert not any("latest" in kk for kk in r) and r["n_paper_train"] == 0
    with open(dump) as f:
        dd = json.load(f)
    assert set(dd["results"]) == want and all(np.asarray(x["boxes"]).shape == (3, 4) for x in dd["results"][sorted(want)[0]])
    from ce_localization.data.cocount import load_review, read_cocount
    review = load_review(rv) if rv else None
    for k in want - {k for k in want if "_objsize" in k}:                    # cùng box: giữ tâm, cỡ = TB box vật cùng lớp / box mẫu
        for sfx, field in (("_objsize", "objects"), ("_objsize_exemplar", "exemplars")):
            for x, y in zip(dd["results"][k], dd["results"][k + sfx]):
                b, c = np.asarray(x["boxes"]), np.asarray(y["boxes"])
                with Image.open(os.path.join(root, "Image", x["image_id"] + ".jpg")) as im:
                    o = read_cocount(root, x["image_id"], review=review)[field] * (x["wh"][0] / im.size[0])   # pixel gốc -> canvas
                assert np.allclose(b[:, :2] + b[:, 2:], c[:, :2] + c[:, 2:], atol=0.02)
                assert np.allclose(c[:, 2] - c[:, 0], (o[:, 2] - o[:, 0]).mean(), atol=0.02)
                assert np.allclose(c[:, 3] - c[:, 1], (o[:, 3] - o[:, 1]).mean(), atol=0.02)


def test_rescore_cocount_from_dump(tmp_path, monkeypatch):
    """`tools/rescore_cocount.py`: dump box (kiểu `eval.py --dump-boxes`, làm tròn) + annotation -> dựng lại record y `CoCountAddDataset`
    + prior cùng seed ⇒ chỉ số cũ khớp (`--check`), có thêm `cnll_F1all_*`; khoá `_objsize` (box trong dump đã đổi cỡ) + prior đổi cỡ."""
    from ce_localization.data.cocount import CoCountAddDataset
    from ce_localization.engine.add_eval import add_metrics, prior_records, prior_unit_boxes, resize_to_objects
    from ce_localization.tools import rescore_cocount as rc
    from tests.ce_localization.helpers import _fake_turn_index
    base = str(tmp_path / "d")
    os.makedirs(base)
    _, _, tpath, index, _ = _fake_turn_index(base)
    root = _fake_cocount(str(tmp_path / "cc"))
    ds = CoCountAddDataset(root, 128, max_obj_ratio=3)
    rng = np.random.default_rng(0)
    rec = []
    for i in range(len(ds)):
        it = ds[i]
        nh, nw = it["valid_hw"]
        xy = rng.uniform(0, 1, (5, 2)) * [nw, nh]
        rec.append({"image_id": it["image_id"], "t": 0, "wh": np.array([nw, nh], dtype=np.float64),
                    "boxes": np.concatenate([xy, xy + 12], 1), "holes": it["holes"].numpy().astype(np.float64),
                    **{k: it[k].numpy().astype(np.float64) for k in ("objects", "objects_all", "exemplars")}})
    unit = prior_unit_boxes(index, "train", "ce130")
    old = {"results": {}, "prior": {}}
    dump = {"ckpt": "x", "iter": 1, "split": "test", "n_samples": 5, "seed": 0, "results": {}}
    for key, post in (("cocount", None), ("cocount_objsize", "mean")):
        r_ = rec if post is None else resize_to_objects(rec, post)
        pr = prior_records(r_, unit, 5, seed=0)
        old["results"][key] = {k: v for k, v in add_metrics(r_, latest=False).items() if "F1all" not in k}
        old["prior"][key] = add_metrics(pr if post is None else resize_to_objects(pr, post), latest=False)
        dump["results"][key] = [{"image_id": r["image_id"], "t": 0, "wh": r["wh"].round(3).tolist(),
                                 "boxes": r["boxes"].round(2).tolist()} for r in r_]
    d = tmp_path / "out"
    d.mkdir()
    with open(d / "boxes_m.json", "w") as f:
        json.dump(dump, f)
    with open(d / "test_m.json", "w") as f:
        json.dump(old, f)
    monkeypatch.setattr(sys, "argv", ["rescore_cocount.py", str(d / "boxes_m.json"), "--cocount-root", root, "--cocount-obj-filter", "3",
                                      "--image-size", "128", "--turn-index", tpath, "--prior-source", "ce130", "--check"])
    rc.main()                                                                      # lệch thì sys.exit
    with open(d / "rescore_m.json") as f:
        new = json.load(f)
    assert set(new["results"]) == set(new["prior"]) == {"cocount", "cocount_objsize"}
    for key in ("cocount", "cocount_objsize"):
        assert np.isfinite(new["results"][key]["cnll_F1all_n1_median"]) and "cnll_F1all_sel_mean" in new["prior"][key]
    assert rc.size_stat("cocount_noobj_objsize") == "mean" and rc.size_stat("cocount_objsize_exemplar") == "exemplar"
    assert rc.size_stat("cocount_norefine") is None
    old["results"]["cocount"]["on_object"] += 0.5                                 # chỉ số cũ lệch -> --check báo lỗi
    with open(d / "test_m.json", "w") as f:
        json.dump(old, f)
    with pytest.raises(SystemExit):
        rc.main()


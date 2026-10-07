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


def test_cocount_dataset_fields_and_density(tmp_path):
    from ce_localization.data.cocount import CoCountAddDataset, read_cocount
    from ce_localization.data.turns import collate_add, to_device_add
    root = _fake_cocount(str(tmp_path / "cc"))
    ds = CoCountAddDataset(root, 128, style="paper", density="empty")
    assert len(ds) == 4 and ds.classes() == ["baby tomato", "normal tomato"]
    it = ds[ds.keys.index(NAMES[0] + "_positive")]
    r = read_cocount(root, NAMES[0] + "_positive")
    assert len(r["objects"]) == 8 and len(r["objects_all"]) == 14 and len(r["loc"]) == 10
    scale = 128 / 200
    assert it["holes"].shape == (10, 4) and torch.allclose(it["holes"][0], torch.tensor(r["loc"][0]).float() * scale)
    assert it["objects"].shape == (8, 4) and it["objects_all"].shape == (14, 4) and it["t"] == 0
    assert it["text"] == "normal tomato" and it["valid_hw"] == (int(120 * scale), 128)
    img = it["image"]                                                         # [4, T, T] uint8 (view HWC), kênh 4 = density trống
    nh, nw = it["valid_hw"]
    assert img.shape == (4, 128, 128) and int(img[3, :nh, :nw].unique().item()) == 14 and int(img[3, nh:, :].max()) == 0
    b = to_device_add(collate_add([ds[0], ds[1]]), torch.device("cpu"))
    assert len(b["objects_all"]) == 2 and b["images"].shape == (2, 4, 128, 128)
    ours = CoCountAddDataset(root, 128, style="ours", density="empty")[0]["image"]
    assert ours.dtype == torch.float32 and ours.shape == (4, 128, 128) and float(ours[3].abs().max()) == 0
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
    sc = g.summarize(cc)
    assert sc["n_cnll"] == 4 and not any("latest" in k for k in sc)
    out = str(tmp_path / "gt.json")
    monkeypatch.setattr(sys, "argv", ["gt_cnll.py", "--turn-index", tpath, "--split-source", "ce130", "--cocount-root", root,
                                      "--out", out])
    g.main()
    with open(out) as f:
        res = json.load(f)
    assert set(res) == {"ce130", "cocount"} and res["cocount"]["n_cnll"] == 4


@pytest.mark.parametrize("kind", ["paper", "obj", "pr"])
def test_eval_cocount_full_flow(tmp_path, monkeypatch, kind):
    """eval.py --dataset cocount: khoá `cocount*`, đủ mẫu, không chỉ số `_latest`, on_object trên cả hai lớp, dump box chạy được."""
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
    if kind == "paper":
        cfg_path, _ = _gamma_cfg(tmp_path, base, "density")
        _fake_paper_ckpt(ck, T=100)
        extra = ["--config", cfg_path, "--add-samplers", "mock"]
    else:
        _, cfg = _gamma2_cfg(tmp_path, base, kind)
        torch.manual_seed(0)
        model = build_model(cfg, pretrained_backbone=False)
        torch.save({"model": model.state_dict(), "config": cfg, "iter": 0}, ck)
        extra = (["--refine-t", "none", "5", "--proposer-sampler", "ddpm"] if kind == "pr" else
                 ["--add-samplers", "ddpm"])                                  # config thu nhỏ T = 20: mock 100 bước không chạy
    out, dump = str(tmp_path / "res.json"), str(tmp_path / "boxes.json")
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", ck, "--dataset", "cocount", "--cocount-root", root, "--n-samples", "3",
                                      "--num-workers", "0", "--out", out, "--dump-boxes", dump, "--device", "cpu"] + extra)
    ea.main()
    with open(out) as f:
        res = json.load(f)
    want = {"paper": {"cocount"}, "obj": {"cocount", "cocount_noobj"}, "pr": {"cocount_ce", "cocount_t5"}}[kind]
    assert set(res["results"]) == want and res["dataset"] == "cocount" and res["density"] == {"cocount": "empty"}
    for k, r in res["results"].items():
        assert r["n"] == 4 and 0 <= r["best_iou@3_any"] <= 1 and 0 <= r["on_object"] <= 1 and r["n_cnll"] == 4
        assert not any("latest" in kk for kk in r) and r["n_paper_train"] == 0
    with open(dump) as f:
        dd = json.load(f)
    assert set(dd["results"]) == want and all(np.asarray(x["boxes"]).shape == (3, 4) for x in dd["results"][sorted(want)[0]])

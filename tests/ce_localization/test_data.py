"""Dữ liệu (`data/`): letterbox + quét CE-130, density (giải mã jet, letterbox, chỉ mục, chọn bản),
điểm density + cỡ giả, dataset với đích box / điểm, tool dựng điểm density, tool trần AP tâm + cỡ (G1).
"""

import json
import os

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from ce_localization.data.dataset import CE130Dataset, collate, letterbox, scale_boxes, scale_points
from ce_localization.data.density import JET, decode_jet, letterbox_density
from ce_localization.data.points import (PointTable, find_peaks, knn_distance, match_points_to_boxes,
                                         pseudo_boxes, pseudo_sizes)
from tests.ce_localization.helpers import (CFG_B0, PSEUDO, _make_branch, _fake_ce130, _fake_density, _gauss,
                                           _write_points, _run_g0)


def resize_and_pad(img, density, target):
    """`ObjectPlacementDataset.resize_and_pad` của CE-Loc gốc, chép nguyên công thức
    (refs/repos/Count-Editing/CE-LocModel/data/dataset.py:27-46) làm đối chứng."""
    w, h = img.size
    scale = min(target / w, target / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), resample=Image.BILINEAR)
    density = density.resize((nw, nh), resample=Image.NEAREST)
    padded_img = Image.new("RGB", (target, target), (0, 0, 0))
    padded_img.paste(img, (0, 0))
    padded_density = Image.new("L", (target, target), 0)
    padded_density.paste(density, (0, 0))
    return padded_img, padded_density, scale


def test_letterbox_matches_celoc_original():
    """Canvas + scale y như `resize_and_pad` của CE-Loc gốc."""
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (384, 683, 3), dtype=np.uint8))
    canvas, scale, nw, nh = letterbox(img, 512)
    ref, _, ref_scale = resize_and_pad(img, Image.new("L", img.size, 0), 512)
    assert np.array_equal(canvas, np.asarray(ref)) and scale == ref_scale
    assert (nw, nh) == (512, int(384 * 512 / 683))
    assert canvas[nh:].max() == 0                               # đệm ĐEN ở đáy


def test_scale_boxes_clips_and_drops():
    b = scale_boxes([[10, 10, 20, 20], [190, 10, 250, 20], [210, 5, 230, 9]], 0.5, 100, 75)
    assert np.allclose(b, [[5, 5, 10, 10], [95, 5, 100, 10]])  # box ngoài vùng thật bị bỏ


def test_dataset_item_and_collate(tmp_path):
    root = str(tmp_path / "ce")
    _make_branch(root, "train", "1", [[10, 20, 60, 80], [100, 30, 150, 120]], "egg")
    ds = CE130Dataset(root, "train", 128)
    s = ds[0]
    assert s["image"].shape == (3, 128, 128) and s["valid_hw"] == (96, 128)
    assert torch.allclose(s["boxes"], torch.tensor([[10, 20, 60, 80], [100, 30, 150, 120]]) * 0.64)
    pad = s["image"][:, 96:]                                     # đen sau chuẩn hoá ImageNet
    assert torch.allclose(pad[0], torch.full_like(pad[0], -0.485 / 0.229))
    b = collate([s, s])
    assert b["whwh"].tolist() == [[128, 96, 128, 96]] * 2 and b["valid_hw"].tolist() == [[96, 128]] * 2


def test_jet_lut_matches_matplotlib_and_decode_roundtrips():
    mpl = pytest.importorskip("matplotlib")
    ref = mpl.colormaps["jet"](np.linspace(0, 1, 256), bytes=True)[:, :3]
    assert np.array_equal(JET, ref)
    assert tuple(JET[0]) == (0, 0, 127) and tuple(JET[255]) == (127, 0, 0)    # nền / đậm nhất
    lv = np.random.default_rng(0).integers(0, 256, (37, 53)).astype(np.uint8)
    dec, dist = decode_jet(JET[lv])
    assert dist == 0 and np.array_equal(JET[dec], JET[lv])                    # màu trùng => mức ±1
    assert np.abs(dec.astype(int) - lv).max() <= 3                          # mức 29–32 cùng (0,0,255)


def test_letterbox_density_matches_celoc_original_nearest():
    """Cùng resize NEAREST + dán góc trên-trái của `resize_and_pad` gốc, chỉ khác bước giải mã."""
    rng = np.random.default_rng(0)
    lv = rng.integers(0, 256, (384, 683)).astype(np.uint8)
    img = Image.fromarray(rng.integers(0, 255, (384, 683, 3), dtype=np.uint8))
    canvas, scale, nw, nh = letterbox(img, 512)
    _, ref, _ = resize_and_pad(img, Image.fromarray(lv), 512)          # uint8 2D -> "L"
    got = letterbox_density(lv, nw, nh, 512)
    assert got.dtype == np.float32 and got.shape == (512, 512)
    assert np.array_equal(got, np.asarray(ref, dtype=np.float32) / 255.0)
    assert got[nh:].max() == 0


def test_density_index_pick_modes(tmp_path):
    _fake_ce130(str(tmp_path / "all_phase2_V2"))
    di = _fake_density(str(tmp_path))
    multi = [i for i, v in di.variants.items() if len(v) == 3][0]
    single = [i for i, v in di.variants.items() if len(v) == 1][0]
    areas = [a for _, a, _ in di.variants[multi]]
    assert areas == sorted(areas, reverse=True) and areas[0] > areas[-1]
    assert di.pick(multi, "full") == (di.variants[multi][0][0], "full")
    assert di.pick(multi, "partial") == (di.variants[multi][-1][0], "partial")
    assert di.pick(single, "partial") == (di.variants[single][0][0], "full")   # 1 bản -> full
    assert di.pick(multi, "empty") == (None, "empty")
    cnt, seen = {"full": 0, "partial": 0, "empty": 0}, set()
    for k in range(3000):
        rel, kind = di.pick(multi, "mix", np.random.default_rng([0, 0, k]))
        cnt[kind] += 1
        if kind == "partial":
            assert rel != di.variants[multi][0][0]
            seen.add(rel)
    assert all(abs(c / 3000 - 1 / 3) < 0.03 for c in cnt.values()), cnt
    assert seen == {r for r, _, _ in di.variants[multi][1:]}                  # rút đều các bản thiếu vật
    kinds = {di.pick(single, "mix", np.random.default_rng([0, 0, k]))[1] for k in range(50)}
    assert kinds == {"full", "empty"}


def test_dataset_density_channel_and_mix_reproducible(tmp_path):
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root)
    di = _fake_density(str(tmp_path))
    ds = CE130Dataset(root, "train", 128, density="full", density_index=di)
    s = ds[1]
    nh, nw = s["valid_hw"]
    assert s["image"].shape == (4, 128, 128) and s["density_kind"] == "full"
    den = s["image"][3]
    assert den.max() == 1.0 and den[nh:].abs().max() == 0 and den.min() >= 0
    x1, y1, x2, y2 = s["boxes"][0].tolist()                    # blob ở tâm box đầu
    assert den[int((y1 + y2) / 2), int((x1 + x2) / 2)] > 0
    ref = CE130Dataset(root, "train", 128)[1]["image"]
    assert torch.equal(s["image"][:3], ref)                     # 3 kênh đầu y như ALPHA0
    assert CE130Dataset(root, "train", 128, density="empty", density_index=di)[1]["image"][3].abs().max() == 0

    def kinds(epoch):
        d = CE130Dataset(root, "train", 128, density="mix", density_index=di, seed=0)
        d.epoch = epoch
        return [d[i]["density_kind"] for i in range(len(d))], [d[i]["image"][3].sum().item() for i in range(len(d))]
    assert kinds(0) == kinds(0)                                 # tái lập (resume / worker)
    assert len({tuple(kinds(e)[0]) for e in range(8)}) > 1      # đổi theo epoch
    b = collate([s, s])
    assert b["images"].shape == (2, 4, 128, 128) and b["density_kind"] == ["full", "full"]
    with pytest.raises(ValueError):
        CE130Dataset(root, "train", 128, density="mix")           # thiếu chỉ mục


def test_find_peaks_separate_blobs_background_and_merged():
    c = [(10.5, 12.5), (40.5, 12.5), (25.5, 30.5)]
    lv = _gauss(48, 56, c)
    p = find_peaks(lv, tau=8, radius=2)
    assert len(p) == 3
    for cx, cy in c:
        assert np.min(np.hypot(p[:, 0] - cx, p[:, 1] - cy)) < 0.51           # đúng tâm pixel
    assert len(find_peaks(np.zeros((20, 20), np.uint8), 8, 2)) == 0          # nền -> 0 điểm
    assert len(find_peaks(_gauss(40, 40, [(15.5, 20.5), (18.5, 20.5)]), 8, 2)) == 1   # 2 vật sát -> gộp 1
    lo = _gauss(48, 56, c, peak=0.1)                                          # đỉnh thấp hơn tau -> bỏ
    assert len(find_peaks(lo, tau=64, radius=2)) == 0


def test_find_peaks_plateau_is_one_point_at_centroid_and_survives_jet():
    lv = np.zeros((20, 30), np.uint8)
    lv[5:8, 10:14] = 200                                                       # đỉnh phẳng 3×4
    lv[4, 9:15] = lv[8, 9:15] = 100
    p = find_peaks(lv, 8, 1)
    assert p.tolist() == [[12.0, 6.5]]                     # trọng tâm cột 10..13, hàng 5..7 (+0,5: pixel liên tục)
    dec, _ = decode_jet(JET[_gauss(48, 56, [(10.5, 12.5), (40.5, 30.5)])])     # qua mã màu jet thật
    assert len(find_peaks(dec, 8, 2)) == 2


def test_knn_and_pseudo_sizes_by_hand():
    p = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [10.0, 10.0]])
    d = knn_distance(p, k=3)
    assert d[0] == pytest.approx((3 + 4 + np.hypot(10, 10)) / 3)
    assert d[1] == pytest.approx((3 + 5 + np.hypot(7, 10)) / 3)
    assert knn_distance(p[:2], k=3).tolist() == [3.0, 3.0]                     # < k láng giềng
    assert np.isnan(knn_distance(p[:1])).all()
    s = pseudo_sizes(p, 3, beta=2.0, s_min=5.0, s_max=12.0)
    assert np.allclose(s, np.clip(2.0 * d, 5.0, 12.0))
    assert pseudo_sizes(p[:1], 3, 1.0, 5.0, 12.0).tolist() == [12.0]           # 1 điểm -> s_max
    b = pseudo_boxes(p, s)
    assert np.allclose((b[:, :2] + b[:, 2:]) / 2, p) and np.allclose(b[:, 2] - b[:, 0], s)
    assert (b[:, 0] < 0).any()                                                 # KHÔNG kẹp: giữ tâm


def test_scale_points_drops_outside_valid_region():
    p = scale_points([[10, 10], [150, 20], [20, 149]], 0.5, 70, 60)
    assert p.tolist() == [[5.0, 5.0]]


def test_match_points_to_boxes_one_to_one_inside_only():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [50, 50, 60, 60]], float)
    pts = np.array([[4.0, 5.0], [9.0, 5.0], [30.0, 30.0]])
    m = match_points_to_boxes(pts, boxes)
    assert sorted(m["pairs"]) == [(0, 0), (1, 1)]                              # gần tâm box nhất, 1-1
    assert m["inside"][2].sum() == 0 and m["inside"][:, 2].sum() == 0
    assert match_points_to_boxes(np.zeros((0, 2)), boxes)["pairs"] == []


def test_dataset_point_targets_are_pseudo_boxes_gt_only_for_scoring(tmp_path):
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root, n_train=2, n_val=1, n_test=1)
    ref = CE130Dataset(root, "train", 128)
    ids = [it["image_id"] for it in ref.items]
    pts = {ids[0]: [[20.0, 30.0], [100.0, 50.0], [60.0, 120.0], [199.0, 10.0]], ids[1]: []}
    pt = _write_points(str(tmp_path / "p.json"), pts)
    ds = CE130Dataset(root, "train", 128, targets="point", points=pt, pseudo=PSEUDO)
    s, s_ref = ds[0], ref[0]
    nh, nw = s["valid_hw"]
    exp = scale_points(pts[ids[0]], 0.64, nw, nh)
    b = s["boxes"].double().numpy()
    assert np.allclose((b[:, :2] + b[:, 2:]) / 2, exp, atol=1e-4)             # tâm box giả = điểm × scale
    w = b[:, 2] - b[:, 0]
    assert np.all(w >= PSEUDO["min_frac"] * nh - 1e-4) and np.all(w <= PSEUDO["max_frac"] * nh + 1e-4)
    assert torch.equal(s["gt_boxes"], s_ref["boxes"])                          # box GT giữ nguyên để chấm
    assert torch.equal(s["image"], s_ref["image"])
    assert len(ds[1]["boxes"]) == 0 and len(ds[1]["gt_boxes"]) > 0             # ảnh density trống
    assert torch.equal(ref[0]["gt_boxes"], ref[0]["boxes"])                    # ALPHA: gt == đích
    bt = collate([s, ds[1]])
    assert len(bt["gt_boxes"]) == 2 and len(bt["boxes"][1]) == 0
    with pytest.raises(ValueError):                                            # thiếu điểm của một ảnh
        CE130Dataset(root, "train", 128, targets="point", points=_write_points(
            str(tmp_path / "q.json"), {ids[0]: []}), pseudo=PSEUDO)


def test_g0_tool_writes_points_and_report(tmp_path, monkeypatch, capsys):
    base = str(tmp_path)
    _fake_ce130(os.path.join(base, "all_phase2_V2"))
    _fake_density(base)
    out, rep = str(tmp_path / "density_points.json"), str(tmp_path / "rep.json")
    _run_g0(monkeypatch, base, out, rep)
    log = capsys.readouterr().out
    assert "yaml: pseudo_size:" in log
    pt = PointTable(out)
    ds = CE130Dataset(os.path.join(base, "all_phase2_V2"), "train", 128)
    assert all(it["image_id"] in pt for it in ds.items)
    with open(rep) as f:
        r = json.load(f)
    tr = r["chosen"]["train"]["all"]
    # blob giả vẽ tại tâm MỌI box (bản full) -> nhãn gần như đủ và đúng
    assert tr["precision"] > 0.8 and tr["recall"] > 0.6 and set(r["chosen"]) == {"train", "val", "test"}
    assert {"beta", "min_frac", "max_frac"} <= set(r["pseudo_size"])

    # --config-in/--config-out: bản config đã điền; phần còn lại y hệt beta0.yaml
    cout = str(tmp_path / "cfg_out.yaml")
    _run_g0(monkeypatch, base, out, rep, ["--config-in", CFG_B0, "--config-out", cout])
    with open(cout) as f:
        c = yaml.safe_load(f)
    with open(CFG_B0) as f:
        c0 = yaml.safe_load(f)
    ps = c["data"].pop("pseudo_size")
    assert ps["beta"] == pytest.approx(r["pseudo_size"]["beta"], abs=1e-4) and ps["knn"] == 3
    assert c["data"].pop("points") == os.path.abspath(out)
    c0["data"].pop("pseudo_size"), c0["data"].pop("points")
    assert c == c0
    _run_g0(monkeypatch, base, out, rep, ["--config-in", CFG_B0, "--config-out", cout,
                                          "--pseudo-size", "3", "0.8", "0.01", "0.4"])
    with open(cout) as f:
        assert yaml.safe_load(f)["data"]["pseudo_size"] == {"knn": 3, "beta": 0.8, "min_frac": 0.01, "max_frac": 0.4}


def test_g1_box_sets_by_hand():
    """Bộ box của cửa G1: tâm gt + cỡ gt = chính GT; đỉnh không ghép được lấy cỡ chung của ảnh;
    knn_sq = đúng box giả BETA0; img_sq / img_wh / knn_rel_sq theo trung bình nhân của GT."""
    from ce_localization.tools.point_box_ceiling import SIZES, box_sets
    gt = np.array([[0, 0, 10, 40], [50, 50, 90, 60], [100, 0, 120, 20]], dtype=float)   # s = 20, 20, 20
    c = (gt[:, :2] + gt[:, 2:]) / 2
    peaks = np.vstack([c[:2] + 1.0, [[300.0, 300.0]]])        # ghép GT0, GT1; đỉnh thứ 3 ngoài mọi box
    b = box_sets(gt, peaks, 400, PSEUDO)
    assert set(b) == {(src, sz) for src in ("gt", "peaks") for sz in SIZES}
    assert np.allclose(b[("gt", "gt_wh")], gt)
    assert np.allclose(b[("gt", "obj_sq")], np.concatenate([c - 10, c + 10], 1))
    assert np.allclose(b[("gt", "img_sq")], b[("gt", "obj_sq")])                         # mọi s = 20
    iw, ih = np.exp(np.log([10, 40, 20]).mean()), np.exp(np.log([40, 10, 20]).mean())
    assert np.allclose(b[("gt", "img_wh")][:, 2:] - b[("gt", "img_wh")][:, :2], [[iw, ih]] * 3)
    k = pseudo_sizes(c, PSEUDO["knn"], PSEUDO["beta"], PSEUDO["min_frac"] * 400, PSEUDO["max_frac"] * 400)
    assert np.allclose(b[("gt", "knn_sq")], pseudo_boxes(c, k))
    rel = b[("gt", "knn_rel_sq")][:, 2] - b[("gt", "knn_rel_sq")][:, 0]
    assert np.allclose(rel / rel[0], k / k[0]) and np.isclose(np.exp(np.log(rel).mean()), 20)
    pw = b[("peaks", "gt_wh")]
    assert np.allclose(pw[:2, 2:] - pw[:2, :2], [[10, 40], [40, 10]])                   # cỡ GT được ghép
    assert np.allclose(pw[2, 2:] - pw[2, :2], [iw, ih])                                 # không ghép -> cỡ ảnh
    assert np.allclose((pw[:, :2] + pw[:, 2:]) / 2, peaks)                              # tâm luôn = đỉnh
    empty = box_sets(gt, np.zeros((0, 2)), 400, PSEUDO)
    assert all(len(empty[("peaks", sz)]) == 0 for sz in SIZES)


def test_g1_tool_full_run(tmp_path, monkeypatch):
    """Trọn luồng cửa G1 trên CE-130 giả: điểm từ G0 -> báo cáo; tâm gt + cỡ gt chấm đúng AP = 1."""
    import sys
    import ce_localization.tools.point_box_ceiling as g1
    base = str(tmp_path)
    _fake_ce130(os.path.join(base, "all_phase2_V2"))
    _fake_density(base)
    pts, rep = str(tmp_path / "density_points.json"), str(tmp_path / "g1.json")
    _run_g0(monkeypatch, base, pts, str(tmp_path / "g0.json"))
    monkeypatch.setattr(sys, "argv", ["point_box_ceiling.py", "--ce130", os.path.join(base, "all_phase2_V2"),
                                      "--config", CFG_B0, "--points", pts, "--workers", "0", "--report", rep])
    g1.main()
    with open(rep) as f:
        r = json.load(f)
    assert set(r["splits"]) == {"train", "val", "test"}
    for split, res in r["splits"].items():
        assert len(res) == 12
        assert res["gt/gt_wh"]["AP50"] == pytest.approx(1.0) and res["gt/gt_wh"]["AP75"] == pytest.approx(1.0)
        for v in res.values():
            assert 0.0 <= v["AP75"] <= v["AP50"] <= 1.0 and set(v["by_density@0.5"]) == {"<=30 vật", "31-100 vật", ">100 vật"}


# ----------------------------------------------------------------------------- GAMMA (bài add)

def test_turn_index_matches_every_turn_one_to_one(tmp_path):
    from tests.ce_localization.helpers import _fake_turn_index
    root, samples, _, index, r = _fake_turn_index(str(tmp_path))
    assert r["ok"] and r["n_matched"] == r["n_turns"] == r["n_samples"] and r["n_holes_unassigned"] == 0
    assert r["per_split"]["train"] == len(index.keys("train")) > 0 and set(r["per_split"]) == {"train", "val", "test"}
    for k, e in index.turns.items():                       # file samples ghép đúng = cùng pixel với ảnh lượt t
        name, t = k.rsplit("_t", 1)
        a = np.asarray(Image.open(os.path.join(root, e["branch"], f"inpainted_turn_{t}.png")).convert("RGB"))
        assert np.array_equal(a, np.asarray(Image.open(os.path.join(samples, e["sample"])).convert("RGB")))
        assert e["branch"].endswith(name) and e["t"] == int(t) and e["density"].replace("density", "images") == e["sample"]
    b = index.branches["train/1001_b1"]
    assert b["removed"] == [0, 0, 1, 2, 3, 4, 0] and len(b["holes"]) == 4


def test_turn_index_reports_target_mismatch_and_missing_sample(tmp_path):
    import json as _json
    from ce_localization.data.turns import build_turn_index
    from tests.ce_localization.helpers import _fake_ce130_turns
    root, samples = _fake_ce130_turns(str(tmp_path))
    ann = os.path.join(samples, "train", "annotation", "1000_1.json")
    with open(ann) as f:
        a = _json.load(f)
    a["target_bbox"][0] += 5
    with open(ann, "w") as f:
        _json.dump(a, f)
    os.remove(os.path.join(samples, "test", "images", "3000_1.png"))
    r = build_turn_index(root, samples, workers=0, log=lambda *x: None)["report"]
    assert not r["ok"] and len(r["target_mismatch"]) == 1 and len(r["match_issues"]) == 1


@pytest.mark.parametrize("image", ["inpainted", "original"])
def test_add_dataset_item(tmp_path, image):
    from ce_localization.data.density import DensityIndex
    from ce_localization.data.turns import CE130AddDataset, collate_add
    from tests.ce_localization.helpers import _fake_turn_index
    root, samples, _, index, _ = _fake_turn_index(str(tmp_path))
    dens = "sample" if image == "inpainted" else "full"
    dindex = DensityIndex(str(tmp_path / "density_index.json"), samples)
    ds = CE130AddDataset(index, root, samples, "train", 128, density=dens, density_index=dindex, image=image)
    i = ds.keys.index("1001_b1_t2")
    s = ds[i]
    sc = 128 / 200                                           # ảnh 200×150 -> nw 128, nh 96
    b = index.branches["train/1001_b1"]
    assert s["image"].shape == (4, 128, 128) and s["valid_hw"] == (96, 128) and s["t"] == 2
    assert torch.allclose(s["holes"], torch.tensor(b["holes"][:2], dtype=torch.float32) * sc, atol=1e-4)
    assert torch.equal(s["target"], s["holes"][-1])
    n_obj = 7 if image == "original" else 5                  # lượt 2 đã xoá vật 2, 3
    assert len(s["objects"]) == n_obj and s["image"][3].max() > 0
    if image == "original":
        canvas, _, _, _ = letterbox(Image.open(os.path.join(root, "train/1001_b1/ground_truth.jpg")).convert("RGB"), 128)
        exp = (canvas.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        assert np.allclose(s["image"][:3].numpy(), exp.transpose(2, 0, 1), atol=1e-5)
    bt = collate_add([s, ds[0]])
    assert bt["images"].shape == (2, 4, 128, 128) and bt["target"].shape == (2, 4)
    assert bt["whwh"][0].tolist() == [128, 96, 128, 96] and len(bt["holes"]) == 2
    rgb = CE130AddDataset(index, root, samples, "test", 128)
    assert rgb[0]["image"].shape == (3, 128, 128) and rgb.classes() == ["cup"]


@pytest.mark.parametrize("image", ["inpainted", "original"])
def test_visualize_gamma_draws(tmp_path, monkeypatch, image):
    import sys
    import ce_localization.tools.visualize_data as viz
    from tests.ce_localization.helpers import _fake_turn_index, _gamma_cfg
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    cfg_path, _ = _gamma_cfg(tmp_path, base, "density")
    out = str(tmp_path / "viz")
    monkeypatch.setattr(sys, "argv", ["visualize_data.py", "--config", cfg_path, "--split", "train", "--n", "3",
                                      "--image", image, "--out", out])
    viz.main()
    assert len([f for f in os.listdir(out) if f.endswith(f"_{image}.png")]) == 3


def test_image_inputs_paper_style_matches_original_preprocessing(tmp_path):
    """style 'paper' = `resize_and_pad` + `to_tensor` của CE-Loc gốc, density `.convert("L")`."""
    from ce_localization.data.turns import image_inputs
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (150, 200, 3), dtype=np.uint8))
    den = Image.fromarray(JET[rng.integers(0, 256, (150, 200)).astype(np.uint8)])
    dp = str(tmp_path / "d.png")
    den.save(dp)
    x, scale, nw, nh = image_inputs(img, dp, 128, "paper")
    ri, rd, rs = resize_and_pad(img, Image.open(dp).convert("L"), 128)
    assert x.shape == (4, 128, 128) and scale == rs and (nw, nh) == (128, 96)
    assert np.allclose(x[:3].numpy(), np.asarray(ri, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    assert np.array_equal(x[3].numpy(), np.asarray(rd, dtype=np.float32) / 255.0)
    xo, _, _, _ = image_inputs(img, None, 128, "ours")
    assert xo.shape == (3, 128, 128) and xo.min() < 0                    # chuẩn hoá ImageNet


def test_image_inputs_blank_density_is_jet_background():
    """Density trống đúng kiểu dữ liệu (PNG toàn nền jet (0,0,127)): đầu vào của bài -> 14/255 trong vùng ảnh, 0 ở đệm;
    đầu vào 'ours' (giải mã jet) -> 0."""
    from ce_localization.data.turns import image_inputs
    img = Image.new("RGB", (200, 150), (90, 90, 90))
    x, _, nw, nh = image_inputs(img, Image.new("RGB", (200, 150), (0, 0, 127)), 128, "paper")
    assert np.allclose(x[3, :nh, :nw].numpy(), 14 / 255.0) and x[3, nh:].abs().max() == 0
    xo, _, _, _ = image_inputs(img, Image.new("RGB", (200, 150), (0, 0, 127)), 128, "ours")
    assert xo[3].abs().max() == 0


def test_add_dataset_paper_style_matches_image_inputs(tmp_path):
    """`style="paper"` (GAMMA "CE-Loc gốc + R-50") == `image_inputs(..., "paper")` (đã kiểm với `resize_and_pad` của bài):
    ảnh `to_tensor`, density `.convert("L")`; box / lỗ không đổi theo style; density `empty` = PNG jet trống -> 14/255."""
    from ce_localization.data.turns import CE130AddDataset, image_inputs
    from tests.ce_localization.helpers import _fake_turn_index
    root, samples, _, index, _ = _fake_turn_index(str(tmp_path))
    ours = CE130AddDataset(index, root, samples, "train", 128, density="sample")
    pap = CE130AddDataset(index, root, samples, "train", 128, density="sample", style="paper")
    a, b = ours[0], pap[0]
    e, _ = pap.entry(0)
    ref = image_inputs(Image.open(os.path.join(samples, e["sample"])).convert("RGB"),
                       os.path.join(samples, e["density"]), 128, "paper")[0]
    assert b["image"].dtype == torch.float32 and torch.allclose(b["image"], ref)
    assert 0 <= float(b["image"][:3].min()) and float(b["image"][:3].max()) <= 1
    for k in ("target", "holes", "objects"):
        assert torch.equal(a[k], b[k]), k
    emp = CE130AddDataset(index, root, samples, "train", 128, density="empty", style="paper")[0]["image"][3]
    nh, nw = b["valid_hw"]
    assert torch.allclose(emp[:nh, :nw], torch.tensor(14 / 255)) and float(emp[nh:].abs().sum()) == 0
    with pytest.raises(ValueError):
        CE130AddDataset(index, root, samples, "train", 128, style="x")

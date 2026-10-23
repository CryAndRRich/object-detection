"""Dữ liệu: letterbox kiểu bài, chỉ mục (nhánh, lượt), dataset CE-130 (inpaint / gốc), cache uint8, CE-CoCount (lọc box, bản rà tay)."""

import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from ce_localization.data.density import JET, DensityIndex, decode_jet
from ce_localization.data.turns import (AddCache, CE130AddDataset, build_add_cache, build_turn_index, collate_add, convert_add_cache,
                                        letterbox, paper_to_float, scale_boxes, to_device_add)
from tests.ce_localization.helpers import COCOUNT_NAMES, _fake_ce130_turns, _fake_cocount, _fake_review, _fake_turn_index


def _paper_resize_and_pad(img, density, target=512):
    """`ObjectPlacementDataset.resize_and_pad` của CE-Loc gốc (bản sao)."""
    w, h = img.size
    scale = min(target / w, target / h)
    nw, nh = int(w * scale), int(h * scale)
    img, density = img.resize((nw, nh), Image.BILINEAR), density.resize((nw, nh), Image.NEAREST)
    pi, pd = Image.new("RGB", (target, target), (0, 0, 0)), Image.new("L", (target, target), 0)
    pi.paste(img, (0, 0))
    pd.paste(density, (0, 0))
    return pi, pd, scale


@pytest.fixture(scope="module")
def ce130(tmp_path_factory):
    base = str(tmp_path_factory.mktemp("ce130"))
    return (base, *_fake_turn_index(base))


def test_letterbox_and_scale_boxes():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (384, 683, 3), dtype=np.uint8))
    canvas, scale, nw, nh = letterbox(img, 512)
    ref, _, ref_scale = _paper_resize_and_pad(img, Image.new("L", img.size, 0), 512)
    assert np.array_equal(canvas, np.asarray(ref)) and scale == ref_scale and (nw, nh) == (512, int(384 * 512 / 683))
    b = scale_boxes([[10, 10, 20, 20], [190, 10, 250, 20], [210, 5, 230, 9]], 0.5, 100, 75)
    assert np.allclose(b, [[5, 5, 10, 10], [95, 5, 100, 10]])           # box ngoài vùng thật bị bỏ


def test_turn_index_and_splits(ce130, tmp_path):
    """Chỉ mục ghép một-một bằng pixel; split như CE-Loc gốc (train = samples/train, val / test = samples/test theo split CE-130 của ảnh
    gốc, test_all = toàn bộ samples/test); lệch target_bbox / thiếu file -> report không ok."""
    _, root, samples, _, index, r = ce130
    assert r["ok"] and r["n_matched"] == r["n_turns"] == r["n_samples"] and r["n_holes_unassigned"] == 0
    for k, e in index.turns.items():
        name, t = k.rsplit("_t", 1)
        a = np.asarray(Image.open(os.path.join(root, e["branch"], f"inpainted_turn_{t}.png")).convert("RGB"))
        assert np.array_equal(a, np.asarray(Image.open(os.path.join(samples, e["sample"])).convert("RGB")))
    assert index.branches["train/1001_b1"]["removed"] == [0, 0, 1, 2, 3, 4, 0]
    tr, va, te, al = (index.keys(s) for s in ("train", "val", "test", "test_all"))
    assert tr and va and te and all(index.turns[k]["sample"].startswith("train/") for k in tr)
    assert all(k.startswith("2001") for k in va) and al == sorted(va + te) and not set(al) & set(tr)
    with pytest.raises(ValueError):
        index.keys("train_all")
    root2, samples2 = _fake_ce130_turns(str(tmp_path))
    ann = os.path.join(samples2, "train", "annotation", "1000_1.json")
    a = json.load(open(ann))
    a["target_bbox"][0] += 5
    json.dump(a, open(ann, "w"))
    os.remove(os.path.join(samples2, "test", "images", "3000_1.png"))
    r2 = build_turn_index(root2, samples2, workers=0, log=lambda *x: None)["report"]
    assert not r2["ok"] and len(r2["target_mismatch"]) == 1 and len(r2["match_issues"]) == 1


@pytest.mark.parametrize("image", ["inpainted", "original"])
def test_add_dataset_item_paper_inputs(ce130, image):
    """Mẫu (nhánh, lượt): t lỗ đầu, đích = lỗ mới nhất, vật còn lại (ảnh gốc: mọi vật); đầu vào = `resize_and_pad` + `to_tensor` của bài,
    density `.convert("L")` (ảnh inpaint: của mẫu; ảnh gốc: bản `full`). /255 sau `to_device_add` trùng từng bit; NHWC liền."""
    base, root, samples, _, index, _ = ce130
    dindex = DensityIndex(os.path.join(base, "density_index.json"), samples)
    ds = CE130AddDataset(index, root, samples, "train", 128, image=image, density_index=dindex)
    s = ds[ds.keys.index("1001_b1_t2")]
    b, e = index.branches["train/1001_b1"], index.turns["1001_b1_t2"]
    assert s["image"].shape == (4, 128, 128) and s["image"].dtype == torch.uint8 and s["valid_hw"] == (96, 128) and s["t"] == 2
    assert torch.allclose(s["holes"], torch.tensor(b["holes"][:2], dtype=torch.float32) * 0.64, atol=1e-4)
    assert torch.equal(s["target"], s["holes"][-1]) and len(s["objects"]) == (7 if image == "original" else 5)
    src = os.path.join(samples, e["sample"]) if image == "inpainted" else os.path.join(root, e["branch"], "ground_truth.jpg")
    den = os.path.join(samples, e["density"]) if image == "inpainted" else dindex.full("1001")
    ri, rd, _ = _paper_resize_and_pad(Image.open(src).convert("RGB"), Image.open(den).convert("L"), 128)
    ref = torch.from_numpy(np.concatenate([np.asarray(ri, np.float32).transpose(2, 0, 1), np.asarray(rd, np.float32)[None]]) / 255.0)
    bt = to_device_add(collate_add([s, ds[0]]), torch.device("cpu"))
    assert torch.equal(bt["images"][0], ref) and bt["images"].is_contiguous(memory_format=torch.channels_last)
    assert bt["whwh"][0].tolist() == [128, 96, 128, 96] and len(bt["holes"]) == 2
    assert torch.equal(paper_to_float(torch.arange(256, dtype=torch.uint8)),
                       torch.from_numpy(np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0))


def test_uint8_cache_bitwise(ce130, tmp_path):
    """Cache zlib / raw (+ bung zlib -> raw) cho đúng từng bit mẫu đọc từ PNG; ảnh gốc không dùng cache."""
    _, root, samples, _, index, _ = ce130
    tr = index.keys("train")
    plain = CE130AddDataset(index, root, samples, "train", 128)
    zdir, rdir = str(tmp_path / "z"), str(tmp_path / "r")
    assert build_add_cache(index, samples, tr, zdir, 128, workers=0, log=lambda *a: None, fmt="zlib") == len(tr)
    assert convert_add_cache(zdir, rdir, workers=0, log=lambda *a: None) == len(tr)
    for cdir, fmt in ((zdir, "zlib"), (rdir, "raw")):
        cache = AddCache(cdir)
        cached = CE130AddDataset(index, root, samples, "train", 128, cache=cache)
        assert cache.format == fmt and cached.cache is not None
        for i in range(len(plain)):
            a, b = plain[i], cached[i]
            assert all(torch.equal(a[k], b[k]) for k in ("image", "target", "holes", "objects")) and a["valid_hw"] == b["valid_hw"]
    with pytest.raises(ValueError):
        convert_add_cache(rdir, str(tmp_path / "x"), workers=0)


def test_density_jet_and_full():
    lv = np.arange(256, dtype=np.uint8).reshape(16, 16)
    dec, dist = decode_jet(JET[lv])
    assert dist == 0 and np.abs(dec.astype(int) - lv).max() <= 3       # mức 29–32 cùng màu (0, 0, 255)


def test_cocount_dataset_filters(tmp_path):
    """CE-CoCount: lớp sạch / tên cặp; density trống = 14/255 trên vùng thật; filt3 bỏ box SAM khổng lồ ở cả hai lớp; bản rà tay bỏ `bad`
    ∪ `auto` ở cả hai lớp, `min_obj` giữ file đủ vật; hình học `cocount_geometry` == dataset; lệch bản rà -> lỗi."""
    from ce_localization.data.cocount import CoCountAddDataset, clean_class, cocount_geometry, load_review, read_cocount, twin_name
    assert clean_class("normal tomato(Big round tomato).") == "normal tomato" and twin_name("A_B_positive") == "A_B_negative"
    root = _fake_cocount(str(tmp_path / "cc"))
    name = COCOUNT_NAMES[0] + "_positive"
    ds = CoCountAddDataset(root, 128)
    s = ds[ds.keys.index(name)]
    nh, nw = s["valid_hw"]
    assert bool((s["image"][3, :nh, :nw] == 14).all()) and int(s["image"][3, nh:].sum()) == 0 and s["t"] == 0
    assert len(s["holes"]) == 10 and len(s["objects"]) == 8 and len(s["objects_all"]) == 14 and s["text"] == "normal tomato"
    g = cocount_geometry(root, name, 128)
    assert np.array_equal(g["objects"], s["objects"].double().numpy()) and np.array_equal(g["wh"], [nw, nh])
    for n in (name, twin_name(name)):                                  # thêm 2 box khổng lồ mỗi lớp
        f = os.path.join(root, "Anno_with_exam_bbox", n + ".json")
        e = json.load(open(f))
        e["exam_bbox"] += [{"bbox": [0, 0, 150, 110], "score": 0.9}, {"bbox": [1, 1, 149, 109], "score": 0.9}]
        json.dump(e, open(f, "w"))
    raw, filt = read_cocount(root, name), read_cocount(root, name, 3)
    assert len(raw["objects"]) == 10 and len(filt["objects"]) == 8 and len(raw["objects_all"]) - len(filt["objects_all"]) == 4
    rv = load_review(_fake_review(root, str(tmp_path / "rv.jsonl"), drop_auto={name: [9]}))
    r = read_cocount(root, name, review=rv)
    assert np.array_equal(r["objects"], raw["objects"][1:9]) and len(r["objects_all"]) == 8 + 7
    ds5 = CoCountAddDataset(root, 128, review=rv, min_obj=7)
    assert ds5.n_all == 4 and ds5.keys == sorted([name, twin_name(name)])         # frame 2 còn 6 / 4 vật
    with pytest.raises(ValueError, match="chọn MỘT"):
        CoCountAddDataset(root, 128, max_obj_ratio=3, review=rv)
    rv[name] = (3, set())
    with pytest.raises(ValueError, match="không khớp"):
        read_cocount(root, name, review=rv)

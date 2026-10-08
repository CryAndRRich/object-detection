"""User study bài ADD (`ce_localization/user_study/`, docs/EXPERIMENT_GAMMA.md mục 17): luật chọn 4 box, dựng items.json,
phiên chấm (không gradio), chỉ số; TRỌN LUỒNG eval.py --dump-boxes -> build -> chấm -> score."""

import json
import os
import sys

import numpy as np
import pytest
from PIL import Image


def _cluster(c, n, rng, jitter=1.0):
    return [[c[0] + rng.normal(0, jitter), c[1] + rng.normal(0, jitter), c[2] + rng.normal(0, jitter),
             c[3] + rng.normal(0, jitter)] for _ in range(n)]


def test_select_boxes_vote_nms_rank_and_fill():
    from ce_localization.user_study.selection import select_boxes
    rng = np.random.default_rng(0)
    A, B, C, D = [10, 10, 50, 50], [100, 100, 160, 140], [20, 120, 60, 180], [150, 10, 190, 40]
    boxes = (_cluster(B, 8, rng) + _cluster(A, 10, rng) + _cluster(C, 5, rng) + [D]
             + [[70, 70, 70.5, 90], [300, 300, 400, 400], [80, 80, 60, 90]])        # suy biến / ngoài ảnh / ngược
    out = select_boxes(boxes, (200, 200))
    assert out["n_valid"] == 24 and out["n_distinct"] == 4
    assert out["vote"] == [10, 8, 5, 1] and out["nms"] == [0.3] * 4
    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in out["boxes"]]
    for c, ref in zip(centers, (A, B, C, D)):
        assert abs(c[0] - (ref[0] + ref[2]) / 2) < 5 and abs(c[1] - (ref[1] + ref[3]) / 2) < 5
    assert all(boxes[i] == pytest.approx(b, abs=0.01) for i, b in zip(out["idx"], out["boxes"]))
    two = select_boxes(_cluster(A, 20, rng) + _cluster(B, 10, rng), (200, 200))        # 2 cụm -> nới NMS cho đủ 4
    assert len(two["idx"]) == 4 and two["n_distinct"] == 2 and two["nms"][:2] == [0.3, 0.3] and two["nms"][2] > 0.3
    assert select_boxes([[300, 300, 400, 400]], (200, 200)) == {"idx": [], "boxes": [], "vote": [], "nms": [], "n_valid": 0,
                                                              "n_distinct": 0}
    clipped = select_boxes([[-20, -20, 30, 30]], (200, 200))
    assert clipped["boxes"] == [[0, 0, 30, 30]]


def test_screen_order_per_model_and_repeats():
    """Dãy màn mỗi model: đủ mọi mẫu, xáo, màn chấm lại sau bản gốc >= min_gap, tất định, KHÔNG phụ thuộc model khác."""
    from ce_localization.user_study.build import screen_order
    ids = [f"i{j}" for j in range(40)]
    sc = screen_order(ids, "a", seed=1, repeat=0.1, min_gap=0.05)
    orig = [x for x in sc if x["repeat_of"] is None]
    assert sorted(x["image_id"] for x in orig) == sorted(ids) and all(x["id"] == f"a|{x['image_id']}" for x in orig)
    assert len({x["id"] for x in sc}) == len(sc)
    by_id = {x["id"]: x for x in sc}
    reps = [x for x in sc if x["repeat_of"] is not None]
    assert len(reps) == 4
    for x in reps:
        o = by_id[x["repeat_of"]]
        assert o["image_id"] == x["image_id"] and x["id"] == o["id"] + "|repeat" and x["order"] >= o["order"] + 0.05
    seq = [x["image_id"] for x in sorted(orig, key=lambda x: x["order"])]
    assert seq != sorted(ids)                                                          # đã xáo
    assert sc == screen_order(ids, "a", seed=1, repeat=0.1, min_gap=0.05)
    assert [x["order"] for x in screen_order(ids, "b", seed=1)] != [x["order"] for x in sc[:40]]   # model khác: thứ tự khác


def test_box_svg_colors_order_and_labels():
    """SVG box: toạ độ pixel ảnh gốc (viewBox W × H), màu theo nhãn, box to vẽ trước (box nhỏ nằm trên, bấm trúng), data-box = số
    box (khớp phím / nút us_tog<i>), tắt box vật thì không vẽ."""
    import re
    from ce_localization.user_study.app import COLOR, as_reasons, box_svg, toggle
    boxes = [[100, 100, 1000, 900], [200, 200, 400, 400]]
    svg = box_svg((2200, 1100), [[1500, 100, 1800, 400]], boxes, [[], ["wrong_size", "on_object"]])
    assert svg.startswith('<svg viewBox="0 0 2200 1100"')
    assert svg.count(f'stroke="{COLOR["obj"]}"') == 1 and 'pointer-events="none"' in svg
    groups = re.findall(r'<g data-box="(\d)"><rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)" '
                        r'fill="transparent" stroke="([^"]+)"', svg)
    assert [g[0] for g in groups] == ["1", "2"]                               # box 1 (to) trước, box 2 (nhỏ) đè lên
    assert [float(v) for v in groups[0][1:5]] == [100, 100, 900, 800] and groups[0][5] == COLOR["ok"]
    assert groups[1][5] == COLOR["bad"]
    big_last = box_svg((2200, 1100), [], [boxes[1], boxes[0]], [[], []], show_objects=False)
    assert [g[0] for g in re.findall(r'<g data-box="(\d)">', big_last)] == ["2", "1"] and COLOR["obj"] not in big_last
    assert toggle([]) == ["on_object"] and toggle(["implausible", "other"]) == []
    assert as_reasons(["other", "on_object", "x"]) == ["on_object", "other"]               # thứ tự REASONS, bỏ lạ
    assert as_reasons("ok") == [] and as_reasons("wrong_size") == ["wrong_size"]          # dạng cũ một chuỗi


def test_rating_store_one_line_per_screen(tmp_path):
    """Mỗi màn một dòng: màn mới ghi nối, lưu lại ghi đè đúng dòng đó (giữ thứ tự); file cũ có dòng trùng được gộp khi mở (bản sau cùng)."""
    from ce_localization.user_study.app import RatingStore
    p = str(tmp_path / "r" / "ratings.jsonl")
    st = RatingStore(p)
    for i, lab in (("a", [[]]), ("b", [["other"]]), ("c", [[]])):
        st.save({"id": i, "labels": lab})
    st.save({"id": "b", "labels": [["wrong_size"]]})                          # lưu lại: ghi đè, vẫn ở vị trí 2
    lines = [json.loads(x) for x in open(p).read().splitlines()]
    assert [x["id"] for x in lines] == ["a", "b", "c"] and lines[1]["labels"] == [["wrong_size"]]
    assert not os.path.exists(p + ".tmp")
    with open(p, "a") as f:                                                   # file kiểu cũ: ghi nối cả bản lưu lại
        f.write(json.dumps({"id": "a", "labels": [["on_object"]]}) + "\n")
        f.write(json.dumps({"id": "d", "labels": [[]]}) + "\n")
    st2 = RatingStore(p)
    lines = [json.loads(x) for x in open(p).read().splitlines()]
    assert [x["id"] for x in lines] == ["a", "b", "c", "d"] and lines[0]["labels"] == [["on_object"]]
    assert st2.latest["a"]["labels"] == [["on_object"]] and len(st2.latest) == 4


def test_auc():
    from ce_localization.user_study.score import auc
    assert auc([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert auc([4, 3, 2, 1], [False, False, True, True]) == 0.0
    assert auc([1, 1, 1, 1], [False, True, False, True]) == 0.5
    assert np.isnan(auc([1, 2], [True, True]))


def test_full_flow_dump_build_rate_score(tmp_path, monkeypatch):
    """eval.py (checkpoint kiểu bài giả) --dump-boxes -> build.py (2 "model" = 2 khoá dump) -> phiên chấm: bấm box, lý do, lưu,
    tắt mở lại tiếp đúng màn, quay lại sửa -> score.py."""
    import ce_localization.eval as ea
    import ce_localization.train as ta
    from ce_localization.data.turns import TurnIndex
    from ce_localization.user_study import build as bu
    from ce_localization.user_study import score as su
    from ce_localization.user_study.app import Session
    from tests.ce_localization.helpers import _fake_paper_ckpt, _fake_text_table, _fake_turn_index, _gamma_cfg
    base = str(tmp_path / "d")
    os.makedirs(base)
    _, samples, tpath, index, _ = _fake_turn_index(base)
    cfg_path, _ = _gamma_cfg(tmp_path, base, "density")
    ck = str(tmp_path / "best_model.pth")
    _fake_paper_ckpt(ck, T=100)
    monkeypatch.setattr(ta, "build_text_table", lambda names, cfg, dev, state_dict=None: _fake_text_table(names, cfg, dev))
    dump = str(tmp_path / "us" / "boxes.json")
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", ck, "--config", cfg_path, "--split", "test", "--n-samples", "6",
                                      "--add-samplers", "ddpm", "mock", "--image", "inpainted", "--num-workers", "0",
                                      "--dump-boxes", dump, "--device", "cpu"])
    ea.main()
    with open(dump) as f:
        dd = json.load(f)
    keys = index.keys("test")
    assert set(dd["results"]) == {"inpainted_ddpm", "inpainted_mock"} and dd["n_samples"] == 6
    recs = dd["results"]["inpainted_mock"]
    assert [r["image_id"] for r in recs] == keys and all(np.asarray(r["boxes"]).shape == (6, 4) for r in recs)

    items = str(tmp_path / "us" / "items.json")
    both = str(tmp_path / "us" / "items_both.json")
    m_ddpm = ["--model", "m_ddpm", dump, "inpainted_ddpm", "Model DDPM"]
    m_mock = ["--model", "m_mock", dump, "inpainted_mock", "Model mock"]
    common = ["build.py", "--turn-index", tpath, "--repeat", "0.25", "--min-gap", "0.1"]
    monkeypatch.setattr(sys, "argv", common + ["--out", both] + m_ddpm + m_mock)
    bu.main()
    monkeypatch.setattr(sys, "argv", common + ["--out", items] + m_ddpm)      # dựng một model ...
    bu.main()
    with open(items) as f:
        d1 = json.load(f)
    monkeypatch.setattr(sys, "argv", ["build.py", "--turn-index", tpath, "--add-to", items] + m_mock)   # ... rồi thêm model sau
    bu.main()
    with pytest.raises(ValueError):                                           # thêm trùng model
        bu.build_items(index, [("m_mock", {}, {})], base=json.load(open(items)))
    with open(items) as f:
        d = json.load(f)
    with open(both) as f:
        assert json.load(f) == d                                              # thêm sau = dựng cùng lúc, từng byte
    assert [x for x in d["screens"] if x["model"] == "m_ddpm"] == d1["screens"]   # màn model cũ không đổi
    assert set(d["items"]) == set(keys) and d["k"] == 4 and d["models"]["m_mock"]["name"] == "Model mock"
    for iid, it in d["items"].items():
        e = index.turns[iid]
        b = index.branches[e["branch"]]
        assert it["image"] == e["sample"] and len(it["holes"]) == e["t"] and it["wh"] == b["wh"]
        assert len(it["objects"]) == len(b["objects"]) - e["t"]               # vật bị xoá tới lượt t không hiện
        r = next(x for x in recs if x["image_id"] == iid)
        px = bu.to_image_px(r["boxes"], r["wh"], b["wh"])
        sel = it["models"]["m_mock"]
        W, H = b["wh"]
        for i, bx in zip(sel["idx"], sel["boxes"]):                           # box hiện = box mẫu idx quy về pixel ảnh, kẹp
            want = [min(max(px[i][0], 0), W), min(max(px[i][1], 0), H), min(max(px[i][2], 0), W), min(max(px[i][3], 0), H)]
            assert bx == pytest.approx(want, abs=0.02)
    n_orig = 2 * len(keys)
    assert sum(x["repeat_of"] is None for x in d["screens"]) == n_orig

    ratings = str(tmp_path / "us" / "ratings.jsonl")
    sess = Session(items, samples, models=["m_mock"], cache_dir=str(tmp_path / "img"))   # chấm riêng một model
    nv = len(sess.view)
    assert nv == sum(x["model"] == "m_mock" for x in d["screens"]) and sess.pos == 0 and not sess.done
    assert all(sess.screens[j]["model"] == "m_mock" for j in sess.view)
    rated = 0
    while not sess.done and rated < 6:
        n = len(sess.labels)
        it = d["items"][sess.screen()[0]["image_id"]]
        page = sess.html(True)
        bg = sess.background(it["image"])
        assert os.path.dirname(bg) == sess.cache_dir and max(Image.open(bg).size) == 1100   # ảnh nền WebP, ảnh nhỏ cũng phóng lên
        assert f'class="us_bg" src="/gradio_api/file={bg}"' in page and f'viewBox="0 0 {it["wh"][0]} {it["wh"][1]}"' in page
        assert page.count("data-box=") == n
        nxt_imgs = {d["items"][sess.screens[sess.view[p]]["image_id"]]["image"]
                    for p in range(sess.pos + 1, min(sess.pos + 4, len(sess.view)))} - {it["image"]}
        assert all(os.path.exists(sess.background(x)) and sess.background(x) in page for x in nxt_imgs)   # tải trước 3 màn sau
        if n:
            sess.toggle(0)                                                     # = bấm box 1 / phím 1
            assert sess.labels[0] == ["on_object"] and f'stroke="#eb1e1e"' in sess.html(True)
            if n > 1:
                sess.set_reasons(1, ["implausible", "wrong_size", "on_object"])       # nhiều lý do một box
                assert sess.labels[1] == ["on_object", "wrong_size", "implausible"]
        if rated == 0:
            h = sess.header()
            assert "Model mock" in h and sess.screen()[0]["image_id"].rsplit("_t", 1)[0] in h and "not saved" in h
            assert "·" not in h and "—" not in h
        sess.save()                                                            # lưu rồi sang màn sau
        rated += 1
        assert sess.pos == rated
    assert len(open(ratings).read().splitlines()) == rated
    p_now = sess.pos
    sess.next()
    sess.next()                                                                # Next = bỏ qua, KHÔNG lưu
    assert sess.pos == p_now + 2 and len(open(ratings).read().splitlines()) == rated
    sess.save()
    sess.first_unsaved()                                                       # về màn bị bỏ qua đầu tiên
    assert sess.pos == p_now and not sess.saved(p_now) and sess.saved(p_now + 2)
    rated += 1
    sess.set_models(["m_ddpm"])                                                # đổi model: dãy khác, từ màn chưa lưu đầu
    assert sess.pos == 0 and all(sess.screens[j]["model"] == "m_ddpm" for j in sess.view) and "Model DDPM" in sess.header()
    sess.save()
    rated += 1
    sess.set_models(None)                                                      # mọi model: hai dãy xen nhau
    assert len(sess.view) == len(d["screens"]) and len({sess.screens[j]["model"] for j in sess.view[:10]}) == 2
    with pytest.raises(ValueError):
        sess.set_models(["khong_co"])
    sess2 = Session(items, samples, models=["m_mock"], cache_dir=str(tmp_path / "img"))  # mở lại: màn chưa lưu đầu
    assert sess2.pos == p_now and len(sess2.store.latest) == rated
    sess2.prev()
    sid = sess2.screen()[0]["id"]
    assert sess2.labels == sess2.store.latest[sid]["labels"] and "(saved)" in sess2.header()
    nb = len(sess2.labels)
    sess2.labels = [[] for _ in range(nb)]
    sess2.save()                                                               # sửa: bản sau thay bản trước
    assert sess2.pos == p_now and su.load_ratings(ratings)[sid]["labels"] == [[]] * nb
    assert len(open(ratings).read().splitlines()) == len(sess2.store.latest) == rated     # lưu lại: ghi đè, không thêm dòng
    sess2.go(0)
    sess2.prev()
    assert sess2.pos == 0                                                      # không lùi quá màn đầu
    sess2.set_models(None)
    while not sess2.done:
        sess2.first_unsaved()
        sess2.save()
    assert len(sess2.store.latest) == len(d["screens"]) and "ALL SCREENS" in sess2.header()
    last = len(sess2.view) - 1
    sess2.go(last)
    sess2.next()
    sess2.save()
    assert sess2.pos == last                                                   # màn cuối: đứng yên

    monkeypatch.setattr(sys, "argv", ["score.py", "--items", items, "--boot", "50"])
    su.main()
    with open(str(tmp_path / "us" / "score.json")) as f:
        res = json.load(f)
    assert res["n_screens_rated"] == n_orig and res["n_repeat_rated"] == len(d["screens"]) - n_orig
    for m in ("m_ddpm", "m_mock"):
        r = res["models"][m]
        assert r["n_screens"] == len(keys) and 0 <= r["box_ok"][0] <= 1 and r["box_ok"][1] <= r["box_ok"][0] <= r["box_ok"][2]
        assert r["in_hole"]["n"] + r["out_hole"]["n"] == r["n_boxes"]
        assert r["short"] == np.mean([len(d["items"][i]["models"][m]["idx"]) < 4 for i in keys])
    assert res["models"]["m_ddpm"]["short"] > 0                               # model giả: màn thiếu / 0 box vẫn được tính
    both = sum(all(d["items"][i]["models"][m]["idx"] for m in ("m_ddpm", "m_mock")) for i in keys)
    assert res["pairs"]["m_ddpm - m_mock"]["n_images"] == both
    assert 0 <= res["consistency"]["agree"] <= 1
    rr = res["models"]["m_mock"]
    if rr["n_reasons"] == rr["n_reasons"]:                                     # có box không ổn
        assert rr["n_reasons"] >= 1 and sum(rr["reasons"].values()) == pytest.approx(rr["n_reasons"])


def test_default_ratings_path():
    from ce_localization.user_study.app import default_ratings_path
    assert default_ratings_path("/a/items.json") == "/a/ratings.jsonl"
    assert default_ratings_path("/a/items_objsize.json") == "/a/ratings_objsize.jsonl"
    assert default_ratings_path("/a/items.json", "rater2") == "/a/ratings_rater2.jsonl"
    assert default_ratings_path("/a/foo.json") == "/a/foo_ratings.jsonl"


def test_cocount_flow_two_datasets(tmp_path, monkeypatch):
    """eval.py --dataset cocount --obj-size --dump-boxes -> build.py --cocount-root (cocount + cocount_objsize = 2 items.json) ->
    hai phiên chấm (nhãn riêng từng bộ, vật lớp kia nét đứt) -> UI một ô Dataset đổi bộ -> score.py."""
    import ce_localization.eval as ea
    import ce_localization.train as ta
    from ce_localization.data.cocount import read_cocount
    from ce_localization.user_study import build as bu
    from ce_localization.user_study import score as su
    from ce_localization.user_study.app import Session, box_svg, build_ui
    from tests.ce_localization.helpers import _fake_paper_ckpt, _fake_text_table, _fake_turn_index, _gamma_cfg
    from tests.ce_localization.test_cocount import _fake_cocount
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    root = _fake_cocount(str(tmp_path / "cc"))
    cfg_path, _ = _gamma_cfg(tmp_path, base, "density")
    ck = str(tmp_path / "best_model.pth")
    _fake_paper_ckpt(ck, T=100)
    monkeypatch.setattr(ta, "build_text_table", lambda names, cfg, dev, state_dict=None: _fake_text_table(names, cfg, dev))
    us = tmp_path / "us"
    dump = str(us / "boxes.json")
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", ck, "--config", cfg_path, "--dataset", "cocount", "--cocount-root", root,
                                      "--n-samples", "6", "--add-samplers", "mock", "--obj-size", "--num-workers", "0",
                                      "--dump-boxes", dump, "--device", "cpu"])
    ea.main()
    items, items_rs = str(us / "items.json"), str(us / "items_objsize.json")
    for out, key in ((items, "cocount"), (items_rs, "cocount_objsize")):
        monkeypatch.setattr(sys, "argv", ["build.py", "--cocount-root", root, "--out", out, "--model", "paper", dump, key, "CE-Loc (paper)"])
        bu.main()
    with pytest.raises(SystemExit):                                           # thêm model CE-130 vào items CoCount: chặn
        monkeypatch.setattr(sys, "argv", ["build.py", "--add-to", items, "--model", "x", dump, "cocount", "X"])
        bu.main()
    d, drs = json.load(open(items)), json.load(open(items_rs))
    assert d["dataset"] == "cocount" and len(d["items"]) == 4 and set(d["items"]) == set(drs["items"])
    for iid, it in d["items"].items():
        r = read_cocount(root, iid)
        W, H = Image.open(os.path.join(root, it["image"])).size
        assert it["image"] == f"Image/{iid}.jpg" and it["wh"] == [W, H] and it["t"] == 0 and len(it["holes"]) == 10
        assert len(it["objects"]) == len(r["objects"]) and len(it["objects_other"]) == len(r["objects_all"]) - len(r["objects"])
        for b in drs["items"][iid]["models"]["paper"]["boxes"]:              # box resize: cỡ = TB vật cùng lớp (trừ box bị kẹp mép)
            if 0 < b[0] and 0 < b[1] and b[2] < W and b[3] < H:
                o = np.asarray(it["objects"])
                assert b[2] - b[0] == pytest.approx((o[:, 2] - o[:, 0]).mean(), abs=0.05)

    svg = box_svg((100, 100), [[0, 0, 5, 5]], [], [], others=[[10, 10, 20, 20], [30, 30, 40, 40]])
    assert svg.count("stroke-dasharray") == 2 and svg.count(f'stroke="#1e90ff"') == 3
    cache = str(tmp_path / "img")
    s1 = Session(items, root, cache_dir=cache, name="CE-CoCount")
    s2 = Session(items_rs, root, cache_dir=cache, name="CE-CoCount (box resize)")
    assert s1.store.path == str(us / "ratings.jsonl") and s2.store.path == str(us / "ratings_objsize.jsonl")
    sc, it = s1.screen()
    h = s1.header()
    assert "Dataset: **CE-CoCount**" in h and sc["image_id"] in h and f"{len(it['objects_other'])} of the other class" in h
    assert s1.html(True).count("stroke-dasharray") == len(it["objects_other"])
    assert os.path.exists(s1.background(it["image"]))
    s1.save()
    s2.save()
    s2.save()
    assert len(open(s1.store.path).read().splitlines()) == 1 and len(open(s2.store.path).read().splitlines()) == 2

    gr = pytest.importorskip("gradio")
    demo, _ = build_ui({"CE-CoCount": s1, "CE-CoCount (box resize)": s2}, gr)
    fns = {getattr(f, "api_name", None): f for f in demo.fns.values()} if isinstance(demo.fns, dict) else {}
    assert not fns or "dataset" in fns

    monkeypatch.setattr(sys, "argv", ["score.py", "--items", items_rs, "--ratings", s2.store.path, "--boot", "20"])
    su.main()
    res = json.load(open(str(us / "score.json")))
    assert res["models"]["paper"]["n_screens"] == 2

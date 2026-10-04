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


def test_screen_order_blind_shuffle_and_repeats():
    from ce_localization.user_study.build import screen_order
    ids, models = [f"i{j}" for j in range(40)], ["a", "b", "c"]
    sc = screen_order(ids, models, seed=1, repeat=0.1, min_gap=10)
    orig = [s for s in sc if s["repeat_of"] is None]
    assert sorted((s["image_id"], s["model"]) for s in orig) == sorted((i, m) for i in ids for m in models)
    reps = [(p, s) for p, s in enumerate(sc) if s["repeat_of"] is not None]
    assert len(reps) == 12
    for p, s in reps:
        o = sc[s["repeat_of"]]
        assert o["repeat_of"] is None and (o["image_id"], o["model"]) == (s["image_id"], s["model"])
        assert p - s["repeat_of"] >= 10
    assert [s["model"] for s in orig[:12]] != sorted(s["model"] for s in orig[:12])     # đã xáo
    assert sc == screen_order(ids, models, seed=1, repeat=0.1, min_gap=10)


def test_render_colors_and_box_at():
    from ce_localization.user_study.app import COLOR, as_reasons, box_at, render, toggle
    img = Image.new("RGB", (2200, 1100), (0, 0, 0))
    boxes = [[100, 100, 1000, 900], [200, 200, 400, 400]]
    im, r = render(img, [[1500, 100, 1800, 400]], boxes, [[], ["wrong_size", "on_object"]], max_side=1100)
    assert im.size == (1100, 550) and r == 0.5
    px = np.asarray(im)
    assert tuple(px[300, 50]) == COLOR["ok"]                    # cạnh trái box 1 (x = 100 * 0,5) — dưới nhãn số
    assert tuple(px[150, 100]) == COLOR["bad"]
    assert tuple(px[150, 750]) == COLOR["obj"]
    assert box_at(boxes, 300, 300) == 1 and box_at(boxes, 900, 800) == 0 and box_at(boxes, 1500, 50) is None
    assert toggle([]) == ["on_object"] and toggle(["implausible", "other"]) == []
    assert as_reasons(["other", "on_object", "x"]) == ["on_object", "other"]               # thứ tự REASONS, bỏ lạ
    assert as_reasons("ok") == [] and as_reasons("wrong_size") == ["wrong_size"]          # dạng cũ một chuỗi


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
    monkeypatch.setattr(sys, "argv", ["build.py", "--turn-index", tpath, "--out", items, "--repeat", "0.25", "--min-gap", "2",
                                      "--model", "m_ddpm", dump, "inpainted_ddpm", "Model DDPM", "--model", "m_mock", dump, "inpainted_mock",
                                      "Model mock"])
    bu.main()
    with open(items) as f:
        d = json.load(f)
    assert set(d["items"]) == set(keys) and d["k"] == 4
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
    assert sum(s["repeat_of"] is None for s in d["screens"]) == n_orig

    ratings = str(tmp_path / "us" / "ratings.jsonl")
    sess = Session(items, samples)
    assert sess.s == 0 and not sess.done
    rated = 0
    while not sess.done and rated < 6:
        n = len(sess.labels)
        im = sess.image(True)
        W, H = d["items"][sess.screen()[0]["image_id"]]["wh"]
        assert max(im.size) == 1100 and im.size == (round(W * sess.r), round(H * sess.r))     # ảnh nhỏ cũng phóng lên
        if n:
            b0 = sess.sel()["boxes"][0]
            hit = sess.click((b0[0] + b0[2]) / 2 * sess.r, (b0[1] + b0[3]) / 2 * sess.r)
            assert hit is not None and sess.labels.count(["on_object"]) == 1
            if n > 1:
                sess.set_reasons(1, ["implausible", "wrong_size", "on_object"])       # nhiều lý do một box
                assert sess.labels[1] == ["on_object", "wrong_size", "implausible"]
        if rated == 0:
            h = sess.header()
            sc = sess.screen()[0]
            assert "Model DDPM" in h or "Model mock" in h
            assert sc["image_id"].rsplit("_t", 1)[0] in h and "not saved" in h
            assert "·" not in h and "—" not in h
        sess.save()                                                            # lưu rồi sang màn sau
        rated += 1
        assert sess.s == rated
    assert len(open(ratings).read().splitlines()) == rated
    s_now = sess.s
    sess.next()
    sess.next()                                                                # Next = bỏ qua, KHÔNG lưu
    assert sess.s == s_now + 2 and len(open(ratings).read().splitlines()) == rated
    sess.save()
    sess.first_unsaved()                                                       # về màn bị bỏ qua đầu tiên
    assert sess.s == s_now and s_now not in sess.store.latest and s_now + 2 in sess.store.latest
    rated += 1
    sess2 = Session(items, samples)                                            # tắt mở lại: về màn chưa lưu đầu tiên
    assert sess2.s == s_now and len(sess2.store.latest) == rated
    sess2.prev()
    assert sess2.s == s_now - 1 and sess2.labels == sess2.store.latest[s_now - 1]["labels"] and "(saved)" in sess2.header()
    sess2.labels = [[] for _ in sess2.labels]
    sess2.save()                                                               # sửa: bản sau thay bản trước
    assert sess2.s == s_now and su.load_ratings(ratings)[s_now - 1]["labels"] == [[]] * len(sess2.labels)
    sess2.go(0)
    sess2.prev()
    assert sess2.s == 0                                                        # không lùi quá màn đầu
    while not sess2.done:
        sess2.first_unsaved()
        sess2.save()
    assert len(sess2.store.latest) == len(d["screens"]) and "ALL SCREENS SAVED" in sess2.header()
    last = len(d["screens"]) - 1
    sess2.go(last)
    sess2.next()
    sess2.save()
    assert sess2.s == last                                                     # màn cuối: đứng yên

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

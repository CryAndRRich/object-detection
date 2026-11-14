"""Web chấm box định vị (`ce_localization/box_review/loc_judge.py`) trên dữ liệu giả: prepare chọn box C-NLL thấp nhất + đổi về pixel
ảnh gốc (dump canvas 512 / VLM ảnh gốc), box cả ảnh khi model không trả box hợp lệ, bỏ ảnh không C-NLL, các run phải cùng tập ảnh;
box vật; hub chỉ giữ một run mỗi model (finetune nếu có); chấm ổn / không ổn, tiến độ cộng dồn, ảnh cache (dựng nền), HTML, merge,
dựng giao diện."""

import json
import os

import pytest
from PIL import Image

from ce_localization.box_review import loc_judge
from ce_localization.box_review.loc_judge import JudgeHub, judged, merge, pick_runs, prepare

RUNS = (("CE-Loc", "zeroshot", "celoc/x/{d}/ep010_boxes.json"), ("Qwen", "zeroshot", "vlm/q/{d}/test.jsonl"),
        ("Qwen", "finetune", "vlm/q-sft/{d}/test.jsonl"))
CE = {"1_b1_t1": "test/images/1_1.png", "1_b1_t2": "test/images/1_2.png", "2_b1_t1": "test/images/2_1.png",
      "3_b1_t1": "test/images/3_1.png"}                      # 3_b1_t1: không C-NLL (< 5 vật) ở mọi run
CC = ["F_1_positive", "F_1_negative"]
NAN = float("nan")


def _write(path, obj, jsonl=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        if jsonl:
            f.write("".join(json.dumps(r) + "\n" for r in obj))
        else:
            json.dump(obj, f)


def fake(root, drop_vlm=None):
    data, out = os.path.join(root, "data"), os.path.join(root, "output")
    turns, branches = {}, {}
    for i, sample in CE.items():
        br = "test/" + i.rsplit("_t", 1)[0]
        turns[i] = {"branch": br, "t": int(i[-1]), "sample": sample}
        branches[br] = {"class": "apple", "wh": [100, 50], "objects": [[1, 1, 11, 11], [20, 20, 30, 30], [40, 5, 50, 15]],
                        "removed": [0, 1, 2]}                  # lượt t: còn vật removed 0 hoặc > t
        os.makedirs(os.path.join(data, "samples", "test", "images"), exist_ok=True)
        Image.new("RGB", (100, 50), (90, 120, 160)).save(os.path.join(data, "samples", sample))
    _write(os.path.join(data, "turn_index.json"), {"turns": turns, "branches": branches})
    for i in CC:
        os.makedirs(os.path.join(data, "cocount", "Image"), exist_ok=True)
        Image.new("RGB", (30, 60), (20, 200, 20)).save(os.path.join(data, "cocount", "Image", i + ".jpg"))
        _write(os.path.join(data, "cocount", "Anno", i + ".json"),
               {"class_name": "sunflower seed(big).", "loc_bbox": [], "counting_anno": {"count": 2, "exemplars": []}})
        _write(os.path.join(data, "cocount", "Anno_with_exam_bbox", i + ".json"),
               {"exam_bbox": [{"bbox": [2, 3, 8, 9]}, {"bbox": [10, 20, 16, 26]}]})
    # CE-Loc: canvas 512 — CE-130 100 × 50 -> 512 × 256 (s = 5,12), CoCount 30 × 60 -> 256 × 512 (s = 8,533)
    ce_rows = [{"image_id": i, "wh": [512.0, 256.0], "boxes": [[51.2, 25.6, 102.4, 51.2], [0, 0, 512, 256]],
                "cnll": [NAN, NAN] if i == "3_b1_t1" else [1.0, 2.0]} for i in CE]
    s = 512 / 60
    cc_rows = [{"image_id": i, "wh": [256.0, 512.0], "boxes": [[0, 0, 10, 10], [s * 3, s * 6, s * 9, s * 12]], "cnll": [3.0, 0.5]}
               for i in CC]
    _write(os.path.join(out, "celoc/x/ce130_test_all/ep010_boxes.json"), {"results": {"inpainted": ce_rows}})
    _write(os.path.join(out, "celoc/x/cocount_review5/ep010_boxes.json"), {"results": {"cocount": cc_rows}})
    for run in ("q", "q-sft"):
        for d, ids, wh in (("ce130_test_all", list(CE), [100, 50]), ("cocount_review5", CC, [30, 60])):
            rows = []
            for i in ids:
                if (run, i) == drop_vlm:
                    continue
                if i == "3_b1_t1":
                    boxes = [{"box": [1, 1, 5, 5], "cnll": None}]
                elif (run, i) == ("q", "2_b1_t1"):
                    boxes = [{"box": [0, 0, 100, 50], "score": None, "fallback": "full_image", "cnll": 99.0}]
                else:
                    boxes = [{"box": [1, 2, 3, 4], "cnll": 5.0}, {"box": [2, 2, 120, 30], "cnll": 0.25}]
                rows.append({"image_id": i, "orig_wh": wh, "boxes": boxes})
            _write(os.path.join(out, "vlm", run, d, "test.jsonl"), rows, jsonl=True)
    return out, data


@pytest.fixture
def setup(tmp_path):
    out, data = fake(str(tmp_path))
    man = prepare(out, str(tmp_path / "judge"), data, runs=RUNS, log=lambda m: None)
    return man, str(tmp_path / "judge"), data


def test_prepare_min_cnll_box(setup, tmp_path):
    man, _, _ = setup
    items = {it["key"]: it for it in man["items"]}
    assert len(items) == len(man["items"]) == (3 + 2) * 3 and man["runs"] == [list(r[:2]) for r in RUNS]
    assert not any(it["id"] == "3_b1_t1" for it in items.values())                               # không C-NLL: không màn
    a = items["ce130|CE-Loc|zeroshot|1_b1_t1"]
    assert a["box"] == [10.0, 5.0, 20.0, 10.0] and a["cnll"] == 1.0 and a["wh"] == [100, 50] and not a["invalid"]
    assert a["src"] == "samples/test/images/1_1.png" and a["cls"] == "apple"
    c = items["cocount|CE-Loc|zeroshot|F_1_negative"]
    assert c["box"] == [3.0, 6.0, 9.0, 12.0] and c["cls"] == "sunflower seed" and c["wh"] == [30, 60]
    assert man["objects"]["ce130|1_b1_t1"] == [[1.0, 1.0, 11.0, 11.0], [40.0, 5.0, 50.0, 15.0]]                  # vật 2 xoá ở lượt 1
    assert man["objects"]["ce130|1_b1_t2"] == [[1.0, 1.0, 11.0, 11.0]]
    assert man["objects"]["cocount|F_1_positive"] == [[2.0, 3.0, 8.0, 9.0], [10.0, 20.0, 16.0, 26.0]]
    v = items["ce130|Qwen|finetune|1_b1_t2"]
    assert v["box"] == [2.0, 2.0, 100.0, 30.0] and v["cnll"] == 0.25                                # kẹp trong ảnh
    f = items["ce130|Qwen|zeroshot|2_b1_t1"]
    assert f["invalid"] and f["box"] == [0.0, 0.0, 100.0, 50.0]
    out, data = fake(str(tmp_path / "bad"), drop_vlm=("q-sft", "1_b1_t2"))
    with pytest.raises(ValueError, match="tập ảnh"):
        prepare(out, str(tmp_path / "bad_out"), data, runs=RUNS, log=lambda m: None)


def test_pick_runs():
    runs = (("A", "zeroshot", "a"), ("B", "zeroshot", "b0"), ("B", "finetune", "b1"), ("C", "finetune", "c"))
    assert pick_runs(runs) == (("A", "zeroshot", "a"), ("B", "finetune", "b1"), ("C", "finetune", "c"))
    assert len(loc_judge.RUNS) == 11 and ("RePlan", "finetune") in [r[:2] for r in loc_judge.RUNS]


def test_hub_judge_flow(setup):
    man, out, data = setup
    hub = JudgeHub(man, out, data)
    assert hub.runs == [("CE-Loc", "zeroshot"), ("Qwen", "finetune")]                             # Qwen zero-shot bỏ (có finetune)
    assert hub.labels == {"CE-Loc": "CE-Loc", "Qwen": "Qwen | Finetune"}
    assert set(hub.sessions) == {(d, u) for d in ("CE130", "CoCount") for u in ("user1", "user2")}
    a, b = hub.sessions[("CE130", "user1")], hub.sessions[("CE130", "user2")]
    assert {it["id"] for it in a.items}.isdisjoint({it["id"] for it in b.items})                  # chia theo ảnh, không trùng
    assert not any(it["setting"] == "zeroshot" and it["model"] == "Qwen" for it in a.items)
    f = {"model": "Qwen"}
    head, page = hub.act("CE130", "user1", "", "show", f)
    first = a.cur()
    assert first["model"] == "Qwen" and first["setting"] == "finetune" and "screen 1 /" in head and "Qwen | Finetune" in page
    assert all(ord(ch) < 128 for ch in page) and 'data-box="' in page
    objs = ",".join(str(v) for o in man["objects"][f"ce130|{first['id']}"] for v in o)
    assert objs and f'data-objs="{objs}"' in page
    assert os.path.exists(os.path.join(out, "cache", first["img"]))                               # ảnh đang hiện: dựng ngay
    assert 'data-pre=""' in page                                                                  # ảnh sau chưa cache: không tải trước
    hub.warm(log=lambda m: None)
    assert all(hub.url(it["img"], False) for a2 in hub.sessions.values() for it in a2.items)      # luồng nền dựng đủ
    hub.act("CE130", "user1", json.dumps({"key": first["key"]}), "fail", f)
    rec = a.store.latest[first["key"]]
    assert rec["ok"] is False and rec["box"] == first["box"] and judged(rec) and a.n_saved() == 1
    a.go(a.vindex[a.index[first["key"]]])
    a.save(True)                                                                                  # chấm lại: không đếm hai lần
    assert a.ndone[("Qwen", "finetune")] == 1
    prog = hub.progress("CE130")
    assert "| Qwen \\| Finetune | 1/" in prog and "| CE-Loc | 0/" in prog and all(ord(ch) < 128 for ch in prog)
    head, page = hub.act("CE130", "user1", "", "show", {"model": "Nope"})                          # model không có
    assert "no screens" in head and page == ""


def test_merge_finished_runs(setup):
    man, out, data = setup
    hub = JudgeHub(man, out, data)
    for (d, u), a in hub.sessions.items():
        for p, it in enumerate(a.items):
            if d == "CE130" and not (u == "user2" and it["model"] == "CE-Loc"):
                a.go(a.vindex[p])
                a.save(it["id"] != "2_b1_t1")
    got = merge(out, log=lambda m: None)
    assert got == {"ce130_qwen_finetune.jsonl": 3}                                                 # CE-Loc thiếu phần user2
    rows = [json.loads(line) for line in open(os.path.join(out, "merged", "ce130_qwen_finetune.jsonl"))]
    assert [r["image_id"] for r in rows] == ["1_b1_t1", "1_b1_t2", "2_b1_t1"]
    assert list(rows[0]) == list(loc_judge.MERGE_KEEP) + ["ok"] and [r["ok"] for r in rows] == [True, True, False]


@pytest.mark.slow
def test_loc_judge_build_ui(setup):
    gr = pytest.importorskip("gradio")
    man, out, data = setup
    demo, _ = loc_judge.build_ui(JudgeHub(man, out, data), gr)
    assert demo is not None

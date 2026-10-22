"""Web rà box tay (`ce_localization/box_review/`): nguồn CE-CoCount theo file + CE-130 theo nhánh trên dữ liệu giả, SVG (box đã loại
biến mất), phiên rà (loại box, lưu một dòng mỗi màn, nháp, hoàn tác cả lần lưu, điền sẵn box vật giữa các nhánh cùng ảnh), dựng giao diện."""

import json
import os

import pytest
from PIL import Image

from ce_localization.box_review.app import ReviewStore, Session, box_svg, build_ui
from ce_localization.box_review.sources import ce130_screens, cocount_screens, dedup_boxes


def _fake_cocount(root):
    """Hai frame × hai lớp; frame A lớp positive 4 box (box 1 / 2 trùng nhau), lớp negative 1 box; frame B lớp negative 0 box."""
    for sub in ("Anno", "Anno_with_exam_bbox", "Image"):
        os.makedirs(os.path.join(root, sub))
    files = {"INTRA_A_1_2_0_0_positive": [[0, 0, 10, 10], [20, 0, 40, 20], [20, 0, 40, 20], [0, 40, 45, 80]],
             "INTRA_A_1_2_0_0_negative": [[50, 50, 60, 60]],
             "INTRA_B_1_2_0_0_positive": [[1, 1, 5, 5]],
             "INTRA_B_1_2_0_0_negative": []}
    for n, boxes in files.items():
        with open(os.path.join(root, "Anno", n + ".json"), "w") as f:
            json.dump({"class_name": "apple.", "loc_bbox": [[0, 0, 5, 5]], "counting_anno": {"count": len(boxes), "points": []}}, f)
        with open(os.path.join(root, "Anno_with_exam_bbox", n + ".json"), "w") as f:
            json.dump({"exam_bbox": [{"bbox": b, "score": 0.9} for b in boxes]}, f)
        Image.new("RGB", (100, 80)).save(os.path.join(root, "Image", n + ".jpg"))
    return root


def _fake_ce130(root):
    """train: ảnh 7 hai nhánh (b1: 2 lượt, b2: 1 lượt); vật 0 bị xoá ở lượt 1 của b1."""
    objs = [[0, 0, 10, 10], [20, 20, 40, 40], [50, 50, 70, 70], [50, 50, 70, 71]]
    for br, holes in (("7_b1", [[0, 0, 10, 10], [80, 0, 90, 5]]), ("7_b2", [[20, 20, 40, 40]])):
        d = os.path.join(root, "train", br)
        os.makedirs(d)
        with open(os.path.join(d, "annotation.json"), "w") as f:
            json.dump({"img": br, "all_bboxes": objs, "inpainted_bboxes": holes, "class_based_caption": "buffalo"}, f)
        Image.new("RGB", (100, 80)).save(os.path.join(d, "ground_truth.jpg"))
        for t in range(1, len(holes) + 1):
            Image.new("RGB", (100, 80)).save(os.path.join(d, f"inpainted_turn_{t}.png"))
    return root


def test_cocount_screens_one_per_file(tmp_path):
    sc = cocount_screens(_fake_cocount(str(tmp_path / "cc")))
    assert [s["id"] for s in sc] == ["INTRA_A_1_2_0_0_negative", "INTRA_A_1_2_0_0_positive", "INTRA_B_1_2_0_0_negative",
                                     "INTRA_B_1_2_0_0_positive"]
    a = sc[1]
    assert len(a["groups"]) == 1 and a["groups"][0]["key"] == a["id"] and a["title"] == "## apple"     # tên vật (bỏ dấu chấm)
    assert a["groups"][0]["boxes"][3] == [0, 40, 45, 80] and len(a["groups"][0]["boxes"]) == 4      # đủ, đúng thứ tự file
    assert a["views"][0][1].endswith("Image/INTRA_A_1_2_0_0_positive.jpg")
    assert sc[0]["groups"][0]["color"] != a["groups"][0]["color"] and sc[2]["groups"][0]["boxes"] == []


def test_ce130_screens_holes_tags_gone(tmp_path):
    sc = ce130_screens(_fake_ce130(str(tmp_path / "ce")), "train")
    assert [s["id"] for s in sc] == ["7_b1", "7_b2"]
    b1 = sc[0]
    assert [v[0] for v in b1["views"]] == ["original", "turn 1", "turn 2"] and b1["wh"] == [100, 80]
    obj, hole = b1["groups"]
    assert obj["share"] == "7" and hole["share"] is None and hole["tags"] == {0: "H1", 1: "H2"}
    assert obj["gone"] == {0: 1} and len(obj["boxes"]) == 4
    only = ce130_screens(str(tmp_path / "ce"), "train", holes=False)
    assert [g["key"] for g in only[0]["groups"]] == ["objects"]


def test_box_svg_removed_boxes_disappear(tmp_path):
    sc = cocount_screens(_fake_cocount(str(tmp_path / "cc")))[1]
    key = sc["groups"][0]["key"]
    s = box_svg(sc, (100, 80), {key: set()})
    assert s.count("data-b=") == 4 and s.index('data-b="0:3"') < s.index('data-b="0:0"')      # box to vẽ trước
    assert "<text" not in s                                                                    # không chữ trên ảnh
    s = box_svg(sc, (100, 80), {key: {1, 3}})
    assert s.count("data-b=") == 2 and 'data-b="0:1"' not in s and 'data-b="0:2"' in s         # box đã loại không vẽ
    ce = ce130_screens(_fake_ce130(str(tmp_path / "ce")), "train")[0]
    s = box_svg(ce, (100, 80), {"objects": set(), "holes": set()}, view=1)
    assert ">H1</text>" in s and 'data-b="0:0" x="0.0" y="0.0" width="10.0" height="10.0" fill="transparent" stroke="#1e90ff" ' \
                                  'stroke-width="1.5" stroke-opacity="0.35"' in s                     # vật đã xoá ở turn 1: mờ
    assert box_svg(ce, (100, 80), {"objects": set(), "holes": set()}, show={"g1"}).count("data-b=") == 2


def test_session_remove_save_undo_resume(tmp_path):
    out = str(tmp_path / "out" / "r.jsonl")
    sc = cocount_screens(_fake_cocount(str(tmp_path / "cc")))
    s = Session(sc, out)
    assert s.pos == 0
    s.go(1)
    key = sc[1]["id"]
    s.click("0:1|123")
    s.click("0:2|124")                                      # box trùng bên dưới: bấm tiếp để loại
    s.click("0:1|125")                                      # đã loại: bỏ qua (box không còn trên ảnh)
    s.click("9:9|1")
    s.click("rác")
    assert s.marks[key] == {1, 2} and s.status() == "modified, NOT saved" and len(s.undo_stack) == 2
    s.save()
    assert s.pos == 2 and ReviewStore(out).get(key)["bad"] == {key: [1, 2]} and ReviewStore(out).get(key)["n"] == {key: 4}
    s.undo()                                                # hoàn tác lần lưu: về màn đó, bỏ dòng, giữ box đã loại làm nháp
    assert s.pos == 1 and ReviewStore(out).get(key) is None and s.marks[key] == {1, 2}
    s.undo()                                                # rồi hoàn tác lần loại gần nhất: box hiện lại
    assert s.marks[key] == {1} and 'data-b="0:2"' in s.html()
    s.next()
    s.prev()
    assert s.marks[key] == {1}                              # nháp giữ khi sang màn khác
    s.save()
    s.go(1)
    s.mark(0, 0)
    s.save()                                                # lưu lại màn đã lưu: ghi đè, vẫn một dòng
    with open(out) as f:
        lines = [json.loads(x) for x in f if x.strip()]
    assert len(lines) == 1 and lines[0]["bad"][key] == [0, 1]
    s.undo()                                                # hoàn tác lần ghi đè: trả lại bản trước
    assert ReviewStore(out).get(key)["bad"][key] == [1]
    for _ in range(3):                                      # lần loại 0, lần lưu đầu, lần loại 1
        s.undo()
    assert ReviewStore(out).get(key) is None and s.marks[key] == set()
    s.undo()
    assert s.msg == "nothing to undo"
    s.go(0)
    s.save()
    assert Session(sc, out).pos == 1                        # mở lại: tiếp từ màn chưa lưu đầu tiên
    assert s.header().startswith("2 / 4 · ")


def test_session_ce130_share_prefill_and_view(tmp_path):
    sc = ce130_screens(_fake_ce130(str(tmp_path / "ce")), "train")
    s = Session(sc, str(tmp_path / "r.jsonl"))
    s.next_view()
    s.mark(0, 1)
    s.mark(1, 1)
    assert s.view == 1                                      # vẫn màn cũ: giữ view
    s.save()
    assert s.pos == 1 and s.view == 0
    assert s.marks == {"objects": {1}, "holes": set()} and s.prefill == "7_b1"      # box vật điền sẵn từ nhánh cùng ảnh
    rec = ReviewStore(str(tmp_path / "r.jsonl")).get("7_b1")
    assert rec["bad"] == {"objects": [1], "holes": [1]} and rec["share"] == {"objects": "7"}
    s.set_view("turn 1")
    assert s.view == 1
    page = s.html()
    assert "inpainted_turn_1.png" in page and 'data-key="7_b2|fit"' in page and "aspect-ratio: 100 / 80" in page
    assert "Branch **7_b2**" in s.header()


def test_dedup_boxes_keeps_smaller_or_median():
    a, b, c, far = [0, 0, 10, 10], [0, 0, 10, 10.5], [0, 0, 10, 10.8], [50, 50, 60, 60]     # IoU a-b 0,95, a-c 0,93, b-c 0,97
    assert dedup_boxes([b, a, far]) == {0}                              # 2 box: giữ box nhỏ hơn
    assert dedup_boxes([c, a, b, far]) == {0, 1}                        # 3 box: giữ trung vị (b)
    assert dedup_boxes([c, a, b, far], present={0, 2, 3}) == {0}        # chỉ xét box còn lại: c, b -> giữ b
    assert dedup_boxes([a, a]) == {1} and dedup_boxes([a, [0, 0, 10, 11.2]]) == set()          # bằng nhau: giữ chỉ số nhỏ; IoU 0,89
    assert dedup_boxes([[0, 0, 10, 10], [0, 0, 10, 10.4], [0, 0, 10, 10.8], [0, 0, 10, 11.2]]) == {0, 2, 3}   # 4 box: trung vị dưới


def test_session_dedup_hides_and_migrates_saved(tmp_path):
    sc = cocount_screens(_fake_cocount(str(tmp_path / "cc")))
    out = str(tmp_path / "r.jsonl")
    key, neg = sc[1]["id"], sc[0]["id"]
    ReviewStore(out).put({"id": key, "bad": {key: [1]}, "n": {key: 4}, "share": {}, "time": "t0", "sec": 1})   # rà trước khi lọc trùng
    ReviewStore(out).put({"id": neg, "bad": {neg: []}, "n": {neg: 1}, "share": {}, "time": "t0", "sec": 1})
    s = Session(sc, out, dedup=True)
    recs = ReviewStore(out).latest
    assert recs[key]["auto"] == {key: []} and recs[key]["bad"] == {key: [1]}     # người rà đã loại box 1: lọc trên box còn lại
    assert recs[neg]["auto"] == {neg: []} and len([f for f in os.listdir(tmp_path) if ".bak_" in f]) == 1
    sc2 = cocount_screens(str(tmp_path / "cc"))
    s2 = Session(sc2, str(tmp_path / "new.jsonl"), dedup=True)
    s2.go(1)
    assert s2.auto_cur == {key: {2}} and 'data-b="0:2"' not in s2.html() and 'data-b="0:1"' in s2.html()   # box trùng 1 / 2: giữ 1
    s2.mark(0, 2)                                           # box đã lọc: không bấm được
    assert s2.marks[key] == set()
    s2.mark(0, 0)
    s2.save()
    assert ReviewStore(str(tmp_path / "new.jsonl")).get(key) == {**ReviewStore(str(tmp_path / "new.jsonl")).get(key),
                                                               "bad": {key: [0]}, "auto": {key: [2]}}
    assert s.dedup and Session(sc2, str(tmp_path / "off.jsonl")).auto_cur == {sc2[0]["id"]: set()}


def test_build_ui(tmp_path):
    gr = pytest.importorskip("gradio")
    sc = cocount_screens(_fake_cocount(str(tmp_path / "cc")))
    demo, launch_kw = build_ui(Session(sc, str(tmp_path / "r.jsonl"), "CE-CoCount"), gr)
    assert demo is not None and isinstance(launch_kw, dict)

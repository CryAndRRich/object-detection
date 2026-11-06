"""ZETA5 / ZETA6: parse (pixel canvas, schema `candidates` tác giả xếp theo score — box[0] = confidence cao nhất, JSON chung, regex, suy nghĩ, rác), lọc / đảo cạnh,
lặp vòng + box cả ảnh; prompt = nguyên văn prompt tác giả (refs/inference_layout.py) với K; few-shot tác giả (`vlm/icl/author.json`)
dựng đúng lượt + parse lại ra đúng box; hash bỏ qua concurrency."""

import json

import numpy as np
import pytest
from PIL import Image

from vlm.parse import extract, fill, parse_boxes, parse_candidates, parse_scored, strip_think
from vlm.prompt import author_turns, box_schema, build_messages, encode_image, prompt_hash, system_text, user_text

ORIG, SENT = (2000, 1000), (1000, 500)


def _cfg(k=5):
    return {"k": k}


@pytest.mark.parametrize("text", [
    'ok ```json\n[{"bbox_2d": [100, 200, 300, 400], "label": "cup"}, {"bbox_2d": [0, 0, 500, 500], "label": "cup"}]\n```',
    "<|begin_of_box|>[[100, 200, 300, 400], [0, 0, 500, 500]]<|end_of_box|>",
    '[{"label": "cup", "y": 200, "x": 100, "y2": 400, "x2": 300}, {"label": "cup", "x1": 0, "y1": 0, "x2": 500, "y2": 500}]',
])
def test_generic_json_canvas_to_original_pixels(text):
    np.testing.assert_allclose(parse_boxes(text, _cfg(), SENT, ORIG), [[200, 400, 600, 800], [0, 0, 1000, 1000]])


def test_author_candidates_sorted_by_score_and_bad_key_kept():
    text = json.dumps({"candidates": [{"name": "cup", "box": [0, 0, 10, 10], "score": 0.2},
                                      {"name": "chair", "seashell": [20, 20, 30, 30], "score": 0.9},
                                      {"name": "cup", "box": [40, 40, 50, 50], "score": 0.5, "reason": "x [1,2,3,4]"}]})
    assert parse_boxes(text, _cfg(), (100, 100), (100, 100)) == [[20, 20, 30, 30], [40, 40, 50, 50], [0, 0, 10, 10]]
    assert parse_scored(text, _cfg(), (100, 100), (100, 100))[1] == [0.9, 0.5, 0.2]


def test_top_score_first_invalid_skipped_missing_score_last():
    text = json.dumps({"candidates": [{"box": [5, 5, 9, 9]}, {"box": [1, 1, 3, 3], "score": 0.3},
                                      {"box": [7, 7, 7, 9], "score": 0.99},                        # score cao nhất nhưng diện tích 0
                                      {"box": [2, 2, 4, 4], "score": 0.8}, {"box": [6, 6, 8, 8], "score": 0.8},  # bằng nhau: giữ thứ tự
                                      {"box": [3, 3, 6, 6], "score": float("nan")}]})
    boxes, scores = parse_scored(text, _cfg(), (100, 100), (100, 100))
    assert boxes == [[2, 2, 4, 4], [6, 6, 8, 8], [1, 1, 3, 3], [5, 5, 9, 9], [3, 3, 6, 6]]
    assert scores == [0.8, 0.8, 0.3, None, None]
    many = json.dumps({"candidates": [{"box": [i, i, i + 2, i + 2], "score": i / 10} for i in range(8)]})
    assert parse_scored(many, _cfg(k=3), (100, 100), (100, 100))[1] == [0.7, 0.6, 0.5]           # quá K: giữ K score cao nhất
    assert parse_scored("Boxes: (1, 2, 3, 4)", _cfg(), (100, 100), (100, 100)) == ([[1, 2, 3, 4]], [None])


def test_pixel_format_uses_canvas_and_think_is_stripped():
    text = '<think>maybe [[1,1,2,2]]</think>{"boxes": [[10, 20, 30, 40.5]]}'
    np.testing.assert_allclose(parse_boxes(text, _cfg(), SENT, ORIG), [[20, 40, 60, 81]])
    assert strip_think("reasoning [[1,2,3,4]] </think>[[5,6,7,8]]") == "[[5,6,7,8]]"            # template tự mở <think>


def test_fallback_regex():
    assert extract("Boxes: (1, 2, 3, 4) and (5,6,7,8.5) trailing") == [[1, 2, 3, 4], [5, 6, 7, 8.5]]
    assert extract("first [[1,2,3,4]] then broken [9,9") == [[1, 2, 3, 4]]


def test_invalid_dropped_reversed_fixed_truncated_to_k():
    text = json.dumps([[30, 40, 10, 20], [5, 5, 5, 9], [0, 0, 1, 1]] + [[0, 0, 2, 2]] * 10)
    boxes = parse_boxes(text, _cfg(k=4), (1000, 1000), (1000, 1000))
    assert boxes[0] == [10, 20, 30, 40] and len(boxes) == 4 and [5, 5, 5, 9] not in boxes
    assert parse_boxes("I cannot help with that.", _cfg(), SENT, ORIG) == []
    assert parse_boxes('{"boxes": [[1, 2, true, 4]]}', _cfg(), SENT, ORIG) == []


def test_fill_cycles_and_full_image_fallback():
    assert fill([[1, 2, 3, 4], [5, 6, 7, 8]], 5, ORIG) == [[1, 2, 3, 4], [5, 6, 7, 8]] * 2 + [[1, 2, 3, 4]]
    assert fill([], 3, ORIG) == [[0.0, 0.0, 2000.0, 1000.0]] * 3


def test_encode_image_original_bytes_or_resize(tmp_path):
    import base64
    p = tmp_path / "a.png"
    Image.fromarray(np.zeros((300, 2000, 3), np.uint8)).save(p)
    im = encode_image(str(p), max_side=1024)
    assert im["orig_wh"] == im["canvas_wh"] == (2000, 300) and im["sent_wh"] == (1024, 154) and im["mime"] == "image/jpeg"
    big = encode_image(str(p), max_side=4096)                                                  # không thu -> nguyên byte PNG như tác giả
    assert big["sent_wh"] == (2000, 300) and big["mime"] == "image/png" and base64.b64decode(big["b64"]) == p.read_bytes()
    q = tmp_path / "g.bmp"
    Image.fromarray(np.zeros((10, 10), np.uint8)).save(q)                                     # đuôi lạ -> mã hoá lại JPEG RGB
    assert encode_image(str(q))["mime"] == "image/jpeg"
    assert encode_image(str(p))["sent_wh"] == (2000, 300)                                     # mặc định không thu (như tác giả)


def test_author_schema():
    assert box_schema(30)["strict"] is False and box_schema(30)["schema"]["properties"]["candidates"]["maxItems"] == 30


AUTHOR_SYSTEM_TAIL = ("Avoid covering people, faces, text, and main objects. Prefer empty/low-saliency regions.\n"
                      "Keep boxes inside the canvas and minimize overlap with existing objects.\n")


def test_prompt_is_author_text():
    sys_, usr = system_text(), user_text(30, "cup", (640, 480))
    assert sys_.startswith("You propose plausible placements for inserting NEW objects into a given background image.\n")
    assert "cup" not in sys_ and "CRITICAL (NO reference boxes provided):" in sys_ and AUTHOR_SYSTEM_TAIL in sys_ and "±15%" in sys_
    assert "Return boxes in the given canvas coordinate system as [x1,y1,x2,y2].\n" in sys_
    assert sys_.endswith("Output ONLY JSON matching the schema.\n")
    assert usr.startswith("Canvas size: 640x480.\nObject(s) to insert: cup\nReturn top 30 placement candidates as boxes [x1,y1,x2,y2] "
                          "within this canvas.\n")
    assert "If perspective varies, match an instance at similar y-position." in usr


def test_build_messages_system_examples_then_query():
    im = {"b64": "QUJD", "mime": "image/png", "sent_wh": (10, 10), "orig_wh": (10, 10), "canvas_wh": (10, 10)}
    ex = [{"role": "user", "text": "apple", "image": "X", "mime": "image/png"}, {"role": "assistant", "text": "{}", "image": None,
                                                                                  "mime": None}]
    msgs = build_messages({"class": "cup"}, im, _cfg(30), ex)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"] and msgs[2]["image"] is None
    assert msgs[1]["text"] == "apple" and "cup" in msgs[3]["text"] and msgs[3]["mime"] == "image/png" and msgs[0]["image"] is None


def _fake_encode(path):
    wh = (633, 384) if "2_b1" in path else (576, 384)
    return {"b64": path, "mime": "image/png", "sent_wh": wh, "orig_wh": wh, "canvas_wh": wh}


def test_author_icl_turns():
    t = author_turns(_fake_encode)
    assert [x["role"] for x in t] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert [bool(x["image"]) for x in t] == [True, False, False, False, True, False]                 # lượt sửa chỉ có text
    assert t[0]["image"].endswith("data/all_phase2_V2/test/2_b1/ground_truth.jpg")
    assert t[2]["text"].startswith("Your seashell boxes are too small")
    assert "as boxes [x1,y1,x2,y2]." in t[0]["text"]
    fix = parse_boxes(t[3]["text"], _cfg(), (633, 384), (633, 384))
    np.testing.assert_allclose(fix, [[380, 154, 454, 234], [424, 137, 522, 233], [360, 290, 430, 405]], atol=1)    # đã sửa khoá chair
    car = parse_boxes(t[5]["text"], _cfg(), (576, 384), (576, 384))
    np.testing.assert_allclose(car, [[53, 180, 210, 305], [292, 61, 402, 112], [238, 118, 319, 194]], atol=1)
    assert json.loads(t[1]["text"])["candidates"][0] == {"name": "seashell", "box": [425, 280, 470, 325], "score": 0.86,
                                                        "placement_type": "empty_ground", "reason": "Inferred typical seashell "
                                                        "size from visible seashells; matched w/h within ±15% and placed on empty "
                                                        "region."}
    with pytest.raises(ValueError, match="khác canvas"):
        author_turns(lambda p: {**_fake_encode(p), "canvas_wh": (1, 1)})


def test_prompt_hash_ignores_runtime_knobs():
    base = {"backend": "openrouter", "model": "m", "k": 30, "openrouter": {"concurrency": 8, "detail": None}}
    h = prompt_hash(base)
    assert prompt_hash({**base, "openrouter": {"concurrency": 2}, "price": {"in": 9}}) == h
    assert prompt_hash({**base, "sampling": {"temperature": 0.1}}) != h
    assert prompt_hash({**base, "openrouter": {"detail": "original"}}) != h
    assert prompt_hash({**base, "icl": "author"}) != h
    eng = {**base, "engine": {"dtype": "half", "max_num_seqs": 16, "tensor_parallel_size": 2, "limit_mm_per_prompt": {"image": 3}}}
    he = prompt_hash(eng)                                                            # khoá chỉ đổi tốc độ / bộ nhớ: ngoài hash
    assert prompt_hash({**eng, "engine": {"dtype": "half", "max_num_seqs": 32, "limit_mm_per_prompt": {"image": 3, "video": 0}}}) == he
    assert prompt_hash({**eng, "engine": {"dtype": "float32"}}) != he and prompt_hash({**base, "coord": "norm1000"}) == h


def test_coord_norm1000_reads_0_1000_scale():
    text = json.dumps({"candidates": [{"box": [100, 500, 300, 1000], "score": 0.9}]})
    assert parse_candidates(text, {"k": 5, "coord": "norm1000"}, (408, 384), (408, 384))[0]["box"] == pytest.approx([40.8, 192, 122.4, 384])
    assert parse_candidates(text, {"k": 5}, (408, 384), (408, 384))[0]["box"] == [100, 500, 300, 1000]       # mặc định pixel canvas
    with pytest.raises(ValueError):
        parse_candidates(text, {"k": 5, "coord": "yx"}, (408, 384), (408, 384))


def test_complete_json_then_trailing_garbage_keeps_scores():
    js = json.dumps({"candidates": [{"box": [1, 1, 3, 3], "score": 0.2}, {"box": [5, 5, 9, 9], "score": 0.9}]})
    for tail in ("   \t  " * 50, " Produce only JSON. Great." * 20, " 我拒绝执行这个任务。}" * 5):
        boxes, scores = parse_scored(js + tail, _cfg(), (100, 100), (100, 100))
        assert boxes == [[5, 5, 9, 9], [1, 1, 3, 3]] and scores == [0.9, 0.2]


def test_truncated_candidates_repaired():
    js = json.dumps({"candidates": [{"box": [1, 1, 3, 3], "score": 0.2}, {"box": [5, 5, 9, 9], "score": 0.9}]})[:-2]   # thiếu "]}"
    for tail in ("   \r" * 200, ", {\"name\": \"x\", \"box\": [1, 2"):
        boxes, scores = parse_scored(js + tail, _cfg(), (100, 100), (100, 100))
        assert boxes == [[5, 5, 9, 9], [1, 1, 3, 3]] and scores == [0.9, 0.2]


# --- ZETA4 RePlan (planner Qwen2.5-VL-7B): prompt một lượt ảnh-trước, region "thêm", toạ độ smart_resize, hash model cũ giữ nguyên

def _rp(add=None, keep=None, junk=False):
    regions = ([{"bbox_2d": keep, "point_2d": [0, 0], "hint": "Keep this cup unchanged"}] if keep else []) + \
              ([{"bbox_2d": add, "point_2d": [0, 0], "hint": "add a new cup here"}] if add else [])
    body = "[{bad json" if junk else json.dumps(regions)
    return f"<think>ok</think><gen_image>keep remaining part of image unchanged.</gen_image><region>{body}</region>"


def test_qwen_resized_wh_matches_smart_resize():
    from vlm.parse import qwen_resized_wh
    assert qwen_resized_wh((540, 384)) == (532, 392)                 # làm tròn bội số 28
    assert qwen_resized_wh((3840, 2160)) == (3836, 2156)             # 8,3 MP < max 12,8 MP: chỉ làm tròn
    w, h = qwen_resized_wh((5000, 4000))                             # > max pixels: thu giữ tỉ lệ
    assert (w, h) == (4004, 3192) and w * h <= 12845056 and w % 28 == h % 28 == 0


def test_replan_greedy_box_first_keep_regions_skipped_samples_unscored():
    from vlm.parse import REPLAN_SEP
    cfg = {"k": 30, "prompt": "replan", "coord": "qwen_resized"}
    text = REPLAN_SEP.join([_rp(add=[266, 196, 532, 392], keep=[0, 0, 10, 10]), "no tags", _rp(junk=True), _rp(keep=[1, 1, 5, 5]),
                            _rp(add=[0, 0, 133, 98])])
    cands = parse_candidates(text, cfg, (540, 384), (540, 384))     # model thấy 532 × 392 -> đổi về 540 × 384
    assert [c["score"] for c in cands] == [1.0, None]
    np.testing.assert_allclose([c["box"] for c in cands], [[270, 192, 540, 384], [0, 0, 135, 96]], atol=1e-6)
    no_greedy = REPLAN_SEP.join([_rp(keep=[0, 0, 10, 10]), _rp(add=[0, 0, 133, 98])])
    assert [c["score"] for c in parse_candidates(no_greedy, cfg, (540, 384), (540, 384))] == [None]   # greedy hỏng -> n_noscore


def test_replan_messages_single_user_turn_image_first(tmp_path):
    from vlm.prompt import replan_template
    p = tmp_path / "a.png"
    Image.new("RGB", (40, 30)).save(p)
    msgs = build_messages({"class": "cup"}, encode_image(str(p)), {"k": 30, "prompt": "replan"})
    assert len(msgs) == 1 and msgs[0]["role"] == "user" and msgs[0]["image_first"] and msgs[0]["image"]
    assert msgs[0]["text"] == replan_template() + "\nAdd one more cup to the image."
    assert replan_template().endswith("The user instruction: ")       # nguyên byte template tác giả


def test_replan_keys_do_not_change_existing_hashes():
    from ce_localization.utils.config import load_config
    import os
    vlm_dir = os.path.join(os.path.dirname(__file__), "..", "..", "vlm", "config")
    assert prompt_hash(load_config(os.path.join(vlm_dir, "openrouter", "gpt-4.1-mini.yaml"))) == "31cfbb6932c5"
    assert prompt_hash(load_config(os.path.join(vlm_dir, "kaggle", "qwen3-vl-4b.yaml"))) == "2acb150694da"
    rp = load_config(os.path.join(vlm_dir, "server2", "replan-7b.yaml"))
    assert prompt_hash({**rp, "replan": {**rp["replan"], "samples": 9}}) != prompt_hash(rp)
    assert prompt_hash({**rp, "instruction": "Add a {cls}."}) != prompt_hash(rp)

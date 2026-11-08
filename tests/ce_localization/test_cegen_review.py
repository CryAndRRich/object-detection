"""Web chấm CE-Gen (`ce_localization/box_review/cegen.py`) trên dữ liệu giả: quét run (CE-130 có subset / Turn_1, CE-CoCount không),
tên add / remove, prepare (cache JPEG, output resize về cỡ input, input dùng chung một file, thiếu v2, thống kê), chia việc theo ảnh,
lưu / đọc lại (dòng sau thắng), chấm thành công cần box, thao tác áp đúng màn trình duyệt gửi, HTML, dựng giao diện."""

import json
import os

import numpy as np
import pytest
from PIL import Image

from ce_localization.box_review import cegen
from ce_localization.box_review.cegen import Annotator, Hub, Store, clean_box, find_runs, prepare, run_task, split_items

CE_IDS = ["2_b1", "10_b1", "10_b2"]
CC_IDS = ["F_1_positive", "F_1_negative"]


def _img(path, wh, color=(80, 120, 160)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", wh, color).save(path)


def _run(d, ids, wh, out_wh, v2=True, cands=((5, 5, 20, 20), (1, 1, 4, 4))):
    for i in ids:
        _img(os.path.join(d, "input", i + ".png"), wh)
        _img(os.path.join(d, "img", i + ".png"), out_wh, (200, 10, 10))
    if not v2:
        return
    os.makedirs(os.path.join(d, cegen.V2), exist_ok=True)
    with open(os.path.join(d, cegen.V2, "records.jsonl"), "w") as f:
        for i in ids:
            hm = os.path.join(d, cegen.V2, i, "heatmap.png")
            os.makedirs(os.path.dirname(hm))
            Image.fromarray(np.full(wh[::-1], 200, np.uint8)).save(hm)
            f.write(json.dumps({"id": i, "status": "processed", "heatmap_path": hm,
                                "candidates": [{"xyxy": list(c), "score": 1.0} for c in cands],
                                "quality_audit_for_review_only": {"reason": "pass"}}) + "\n")


def fake_src(root, v2=True, n_models=1):
    """CE-130: model M<k> × zeroshot / finetune × add / remove, run có Turn_1 + 2 subset; CE-CoCount: không subset."""
    names = {("zeroshot", "add"): "FSC_final_M_add", ("zeroshot", "remove"): "turbo_8_step_remove",
             ("finetune", "add"): "M-LoRA-add-ckpt9000", ("finetune", "remove"): "M_LoRA_remove_ckpt9000"}
    for m in range(n_models):
        for (st, task), name in names.items():
            base = os.path.join(root, "FSC_final", f"M{m}", st, name, "Turn_1")
            _run(os.path.join(base, "FSC_easy_overlap"), CE_IDS[:2], (40, 30), (80, 60), v2)
            _run(os.path.join(base, "FSC_difficult_no_overlap"), CE_IDS[2:], (40, 30), (100, 60), v2)
            _run(os.path.join(root, "processed_dataset", f"M{m}", st, name.replace("FSC_final", "Cocount")), CC_IDS, (30, 50),
                 (30, 50), v2)
    return root


def fake_refs(root):
    ti = os.path.join(root, "turn_index.json")
    with open(ti, "w") as f:
        json.dump({"branches": {"test/2_b1": {"class": "apple", "wh": [40, 30]}, "val/10_b1": {"class": "pear", "wh": [40, 30]},
                                "test/10_b2": {"class": "pear", "wh": [41, 30]}}}, f)
    cc = os.path.join(root, "cocount")
    for i in CC_IDS:
        os.makedirs(os.path.join(cc, "Anno"), exist_ok=True)
        with open(os.path.join(cc, "Anno", i + ".json"), "w") as f:
            json.dump({"class_name": "sunflower seed(big)."}, f)
        _img(os.path.join(cc, "Image", i + ".jpg"), (30, 50))
    return ti, cc


@pytest.fixture
def man(tmp_path):
    src = fake_src(str(tmp_path / "src"))
    ti, cc = fake_refs(str(tmp_path))
    return prepare(src, str(tmp_path / "out"), ti, cc, workers=1, top_k=1, expect_runs=4, log=lambda m: None)


def test_run_task_names_on_server():
    add = ["FSC_final_JoyAI_add_lora", "FSC_final_JoyAI_add", "LongCat-Image-Edit-LoRA-add-ckpt9000", "turbo_8_step_add",
           "FSC_final_Qwen_add_lora_lre-4", "FSC_final_add", "Cocount_JoyAI_add_lora", "CoCount-train_add"]
    rm = ["FSC_final_JoyAI_remove_lora", "LongCat-Image-Edit-LoRA-remove-ckpt9000", "turbo_8_step_remove", "FSC_final_remove",
          "LongCat_LoRA_remove_ckpt9000", "Qwen_cocount_remove", "CoCount-train_remove", "Cocount_Qwen_remove_lora"]
    assert [run_task(n) for n in add] == ["add"] * len(add)
    assert [run_task(n) for n in rm] == ["remove"] * len(rm)
    for bad in ("address_run", "add_remove", "plain"):
        with pytest.raises(ValueError):
            run_task(bad)


def test_find_runs_layout(tmp_path):
    src = fake_src(str(tmp_path / "src"))
    runs = find_runs(src)
    ce = [r for r in runs if r["ds"] == "ce130"]
    assert len(ce) == 8 and {r["subset"] for r in ce} == {"FSC_easy_overlap", "FSC_difficult_no_overlap"}
    assert all(r["run_dir"].endswith("Turn_1") for r in ce)
    cc = [r for r in runs if r["ds"] == "cocount"]
    assert len(cc) == 4 and all(r["subset"] == "" and r["run_dir"] == r["path"] for r in cc)
    assert {(r["setting"], r["task"]) for r in cc} == {(s, t) for s in cegen.SETTINGS for t in cegen.TASKS}


def test_clean_class_matches_reader():
    from ce_localization.data.cocount import clean_class
    for n in ("normal tomato(Big round tomato).", "sunflower seed.", "  a   b (x) .", "plain"):
        assert cegen.clean_class(n) == clean_class(n)


def test_prepare_cache_manifest(man, tmp_path):
    items = man["items"]
    assert len(items) == 4 * 3 + 4 * 2
    cache = man["cache"]
    ce = [it for it in items if it["ds"] == "ce130"]
    # thứ tự: ảnh (tự nhiên: 2_b1 < 10_b1 < 10_b2), trong ảnh: add trước remove, zeroshot trước finetune
    assert [it["id"] for it in ce[:4]] == ["2_b1"] * 4
    assert [(it["task"], it["setting"]) for it in ce[:4]] == [("add", "zeroshot"), ("add", "finetune"),
                                                             ("remove", "zeroshot"), ("remove", "finetune")]
    it = ce[0]
    assert it["cls"] == "apple" and it["wh"] == [40, 30] and it["out_wh"] == [80, 60] and it["cands"] == [[5, 5, 20, 20]]
    with Image.open(os.path.join(cache, it["out"])) as im:
        assert im.size == (40, 30) and im.format == "JPEG"                                   # output resize về cỡ input
    with Image.open(os.path.join(cache, it["hm"])) as im:
        assert np.asarray(im)[0, 0].tolist() != [200, 200, 200]                               # heatmap đã tô màu
    assert len({x["in"] for x in ce if x["id"] == "2_b1"}) == 1                              # 4 run chung một file input
    assert {x["cls"] for x in items if x["ds"] == "cocount"} == {"sunflower seed"}
    st = man["stats"]
    assert st["CE-130 M0 zeroshot add"]["out_aspect>2%"] == 1                                # 10_b2: 100x60 vs 40x30
    assert st["CE-130 M0 zeroshot add"]["input!=ref_size"] == 1                              # 10_b2: ref 41x30
    with open(os.path.join(str(tmp_path / "out"), "manifest.json")) as f:
        assert len(json.load(f)["items"]) == len(items)


def test_prepare_missing_v2_and_run_count(tmp_path):
    src = fake_src(str(tmp_path / "src"), v2=False)
    ti, cc = fake_refs(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        prepare(src, str(tmp_path / "o"), ti, cc, workers=1, expect_runs=4, log=lambda m: None)
    with pytest.raises(ValueError):
        prepare(src, str(tmp_path / "o"), ti, cc, workers=1, expect_runs=12, log=lambda m: None)
    m = prepare(src, str(tmp_path / "o"), ti, cc, workers=1, expect_runs=4, allow_missing_v2=True, log=lambda m: None)
    assert all(not it["cands"] and it["hm"] is None and it["status"] == "no_v2" for it in m["items"])


def test_split_items_by_image(man):
    items = [it for it in man["items"] if it["ds"] == "ce130"]
    a, b = split_items(items, 0), split_items(items, 1)
    assert len(a) + len(b) == len(items) and not {x["key"] for x in a} & {x["key"] for x in b}
    assert {x["id"] for x in a} == {"2_b1", "10_b2"} and {x["id"] for x in b} == {"10_b1"}


def test_clean_box():
    assert clean_box([30, 5, -3, 50], (40, 30)) == [0.0, 5.0, 30.0, 30.0]
    assert clean_box([1, 1, 2, 10], (40, 30)) is None and clean_box(None, (40, 30)) is None and clean_box([1, 2], (4, 4)) is None


def test_annotator_save_resume(man, tmp_path):
    items = [it for it in man["items"] if it["ds"] == "ce130"]
    path = str(tmp_path / "a.jsonl")
    a = Annotator(items, path, "x", "user1")
    assert a.pos == 0
    assert a.save(True, None) and a.pos == 0 and not a.store.latest                         # thành công thiếu box: không lưu
    assert a.save(True, [5, 5, 20, 20], "cand0") == "" and a.pos == 1
    assert a.save(False, [1, 1, 9, 9], "drawn") == "" and a.pos == 2
    recs = Store(path).latest
    assert recs[items[0]["key"]]["box"] == [5.0, 5.0, 20.0, 20.0] and recs[items[0]["key"]]["box_src"] == "cand0"
    assert recs[items[1]["key"]]["success"] is False and recs[items[1]["key"]]["box"] is None
    a.go(0)
    a.save(True, [0, 0, 10, 10], "evil<script>")
    r0 = Store(path).latest[items[0]["key"]]                                                # ghi đè: dòng sau thắng
    assert r0["box"] == [0.0, 0.0, 10.0, 10.0] and r0["box_src"] is None
    b = Annotator(items, path, "x", "user1")
    assert b.pos == 2 and b.n_saved() == 2                                                  # mở lại: màn chưa chấm đầu tiên


def test_hub_acts_on_shown_screen(man, tmp_path):
    hub = Hub(man, str(tmp_path / "res"))
    assert set(hub.sessions) == {("CE-130", "user1"), ("CE-130", "user2"), ("CE-CoCount", "user1"), ("CE-CoCount", "user2")}
    head, page = hub.act(None, "user1", "", "show")
    assert "Choose" in head and page == ""
    a = hub.sessions[("CE-130", "user1")]
    k3 = a.items[3]["key"]
    head, page = hub.act("CE-130", "user1", json.dumps({"key": k3, "box": [5, 5, 20, 20], "src": "cand0"}), "ok")
    assert k3 in a.store.latest and a.pos == 4 and a.items[0]["key"] not in a.store.latest
    assert 'class="cg_root"' in page and cegen.FILE_ROUTE in page and "REMOVE one apple" not in page
    head, page = hub.act("CE-130", "user1", "not json", "fail")
    assert a.items[4]["key"] in a.store.latest and a.store.latest[a.items[4]["key"]]["success"] is False
    hub.act("CE-130", "user1", "", "first")
    assert a.pos == 0
    head, page = hub.act("CE-130", "user1", "", "goto", 4)
    assert a.pos == 3 and "saved: <b" in page and "SUCCESS" in page
    assert hub.sessions[("CE-130", "user2")].pos == 0                                       # phiên user khác không đổi


def test_hub_filter_and_progress(tmp_path):
    src = fake_src(str(tmp_path / "src"), n_models=2)
    ti, cc = fake_refs(str(tmp_path))
    man = prepare(src, str(tmp_path / "out"), ti, cc, workers=1, top_k=1, expect_runs=8, log=lambda m: None)
    hub = Hub(man, str(tmp_path / "res"))
    assert hub.models == ["M0", "M1"]
    a = hub.sessions[("CE-CoCount", "user1")]
    f = {"model": "M1", "setting": "finetune", "task": "all"}
    head, page = hub.act("CE-CoCount", "user1", "", "show", filt=f)
    want = [it["key"] for it in a.items if it["model"] == "M1" and it["setting"] == "finetune"]
    assert [a.items[p]["key"] for p in a.view] == want and len(want) == 2 and "M1 · finetune" in head
    assert a.cur()["key"] == want[0] and "M1" in page and "finetune" in page
    pre = page.split('data-pre="')[1].split('"')[0]
    assert all(it["out"] not in pre for it in a.items if it["model"] == "M0")              # tải trước chỉ màn trong bộ lọc
    hub.act("CE-CoCount", "user1", json.dumps({"key": want[0], "box": [2, 2, 9, 9], "src": "drawn"}), "ok", filt=f)
    assert want[0] in a.store.latest and a.cur()["key"] == want[1]
    hub.act("CE-CoCount", "user1", json.dumps({"key": want[1]}), "fail", filt=f)
    assert set(a.store.latest) == set(want)
    other = next(it["key"] for it in a.items if it["model"] == "M0")                         # màn ngoài bộ lọc: không lưu
    head, _ = hub.act("CE-CoCount", "user1", json.dumps({"key": other, "box": [2, 2, 9, 9], "src": "drawn"}), "ok", filt=f)
    assert other not in a.store.latest and "not saved" in head
    head, page = hub.act("CE-CoCount", "user1", "", "show", filt={"model": "M9"})
    assert "no screens" in head and page == ""
    hub.act("CE-CoCount", "user1", "", "show", filt={})                                      # bỏ lọc: màn chưa chấm đầu tiên
    assert len(a.view) == len(a.items) and a.cur()["key"] not in a.store.latest
    prog = hub.progress("CE-CoCount")
    assert "| M1 · finetune · add | 1/1 | 0/1 | 50% |" in prog and "| M1 · finetune · remove | 1/1 | 0/1 | 50% |" in prog
    for (d, u), s in hub.sessions.items():                                                   # chấm hết user2 phần M1 · finetune
        if d == "CE-CoCount" and u == "user2":
            for it in s.items:
                if it["model"] == "M1" and it["setting"] == "finetune":
                    s.store.put({"key": it["key"], "success": False})
    assert "| M1 · finetune · add | 1/1 | 1/1 | **done** |" in hub.progress("CE-CoCount")


def test_html_prefill_and_prefetch(man, tmp_path):
    hub = Hub(man, str(tmp_path / "res"))
    a = hub.sessions[("CE-CoCount", "user1")]
    page = a.html(hub.url)
    it = a.cur()
    assert f'data-box="{",".join(str(v) for v in it["cands"][0])}"' in page and 'data-src="cand0"' in page
    assert "ADD one sunflower seed" in page and "data-pre=" in page
    pre = page.split('data-pre="')[1].split('"')[0]
    assert a.items[1]["out"] in pre


def test_cegen_build_ui(man, tmp_path):
    gr = pytest.importorskip("gradio")
    demo, _ = cegen.build_ui(Hub(man, str(tmp_path / "res")), gr)
    assert demo is not None

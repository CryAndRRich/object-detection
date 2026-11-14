#!/usr/bin/env python3
"""Web chấm box định vị (người dùng 2026-11-12): mỗi màn = (tập, model, zeroshot / finetune, ảnh) — ảnh model nhận + MỘT box viền đỏ
đậm = box C-NLL THẤP NHẤT trong các box model trả (mọi model cùng luật, kể cả VLM; model không trả box hợp lệ: box cả ảnh như lúc
chấm điểm), kèm box vật cùng lớp viền xanh mảnh (đúng bộ vật C-NLL fit Gaussian; phím h ẩn / hiện) để áng chừng. Người chấm chỉ
bấm Enter = ổn / f = không ổn. Chấm TRỌN tập test: CE-130 `test_all` (ảnh có C-NLL: 6.090 / 6.122 — 32 ảnh < 5 vật không có C-NLL
nên không có màn) và CE-CoCount `review5` (1.263) × 11 run (`RUNS` = `pick_runs(ALL_RUNS)`: mỗi model MỘT run, có finetune thì lấy
finetune — người dùng 2026-11-14; không OracleLoc, không DELTA7.1). RePlan hiện "RePlan (Qwen2.5-VL-7B planner)": chỉ planner được
finetune (LoRA SFT).

Ba lệnh (khuôn + giao diện chung với `cegen.py`: Store ghi nối, Annotator / Hub, JS zoom `VIEW_JS`):

  prepare   đọc dump box của 11 run (`--output-root` = `output/add`, chỉ đọc JSON, chạy ở LOCAL được — cần torch cho `letterbox_size`)
            -> `<out-dir>/manifest.json`: mỗi màn ảnh nguồn (tương đối `--data-dir`), box (pixel ẢNH GỐC, xyxy), C-NLL, `invalid`
            (model không trả box hợp lệ); `objects` = box vật mỗi ảnh (CoCount theo `cocount_review.jsonl`). Box dump CE-Loc / baseline là pixel canvas 512 (letterbox góc trên trái) -> chia tỉ lệ;
            box VLM `test.jsonl` đã là pixel ảnh gốc. Mọi run phải phủ đúng một tập ảnh. `--cache` dựng sẵn mọi ảnh JPEG.
  serve     app Gradio (`--share`; không torch — env `imgedit` được): menu Data / User / Model (tên kèm setting) + 5 nút; ảnh
            JPEG cache dựng từ ảnh trong `--data-dir` (`samples/`, `cocount/Image/`) ở luồng nền ngay lúc mở (ảnh đang hiện chưa có
            thì dựng ngay; chỉ tải trước ảnh đã có cache); bảng "Progress per run" hai tập hiện ngay. Chia việc theo ẢNH xen kẽ user1 / user2
            (mọi run của một ảnh về cùng một người, hai người không trùng màn). Hiện tên model (người dùng 2026-11-12: không chấm mù).
  merge     run nào xong hết (gộp 2 user phủ mọi ảnh của tập) -> `<out-dir>/merged/<tập>_<model>_<setting>.jsonl` (model viết
            thường, khoảng trắng -> `-`), mỗi dòng: dataset, model, setting, image_id, class, wh, box, cnll, invalid + "ok".

Phím: Enter ổn · f không ổn (cả hai lưu rồi sang màn sau; chấm lại = ghi đè) · ← / → màn trước / sau · + / - / lăn chuột phóng thu
quanh điểm chuột, 0 vừa khung · h ẩn / hiện box vật.

Kết quả: `<out-dir>/judge_<ce130|cocount>_<user>.jsonl` (chỉ GHI NỐI, dòng sau cùng của `key` thắng): {"key", "dataset", "model",
"setting", "image_id", "class", "box", "cnll", "invalid", "wh", "ok": true | false, "user", "time", "sec"}.

  .venv-cpu/bin/python box_review/loc_judge.py prepare           # local, từ ce_localization/
  python box_review/loc_judge.py serve --share                   # server 2: sau khi upload manifest.json
  python box_review/loc_judge.py merge
"""

import argparse
import collections
import functools
import html
import itertools
import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from urllib.parse import quote

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ce_localization.box_review.cegen import (BASE_CSS, FILE_ROUTE, LABELS, PREFETCH, USERS, VIEW_JS, Annotator, Hub,  # noqa: E402
                                              Store, _save_jpeg, clean_class, split_items)

OD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # object-detection/
OUTPUT_ROOT = os.path.join(os.path.dirname(OD), "output", "add")
OUT_DIR = os.path.join(OUTPUT_ROOT, "review", "loc_judge")
DATA_DIR = os.path.join(OD, "data")
DSETS = {"ce130": ("ce130_test_all", "inpainted"), "cocount": ("cocount_review5", "cocount")}   # thư mục tập, khoá dump CE-Loc / baseline
SETTING_LABEL = {"zeroshot": "Zeroshot", "finetune": "Finetune"}
# tên hiện trên giao diện: RePlan chỉ có planner Qwen2.5-VL-7B được finetune (LoRA SFT), không phải cả RePlan
DISPLAY = {"RePlan": "RePlan (Qwen2.5-VL-7B planner)"}
CHOOSE = "**Choose Data, User and Model above to start.**"
# (model, setting, dump tương đối --output-root; {d} = thư mục tập) — dump box của đúng hàng SCORE / snapshot báo cáo
ALL_RUNS = (
    ("CE-Loc", "zeroshot", "celoc/delta8/{d}/ep010_boxes.json"),
    ("BootPlace", "zeroshot", "baseline/zeta1/{d}/pretrained_boxes.json"),
    ("BootPlace", "finetune", "baseline/zeta1_1/{d}/ep006_boxes.json"),
    ("CSANet", "zeroshot", "baseline/zeta2/{d}/ep000_boxes.json"),
    ("CSANet", "finetune", "baseline/zeta2/{d}/ep010_boxes.json"),
    ("HiddenObjects", "zeroshot", "baseline/zeta3/{d}/ep000_boxes.json"),
    ("HiddenObjects", "finetune", "baseline/zeta3/{d}/ep009_boxes.json"),
    ("RePlan", "zeroshot", "vlm/replan-7b/{d}/test.jsonl"),
    ("RePlan", "finetune", "vlm/replan-7b-sft/ep003/{d}/test.jsonl"),
    ("GPT-6 Luna", "zeroshot", "vlm/gpt-6-luna/{d}/test.jsonl"),
    ("gpt-4.1-mini", "zeroshot", "vlm/gpt-4.1-mini/{d}/test.jsonl"),
    ("Qwen3-VL-4B", "zeroshot", "vlm/qwen3-vl-4b/{d}/test.jsonl"),
    ("Qwen3-VL-4B", "finetune", "vlm/qwen3-vl-4b-sft/ep003/{d}/test.jsonl"),
    ("GLM-5.3-Flash", "zeroshot", "vlm/glm-5.3-flash/{d}/test.jsonl"),
    ("InternVL3.5-8B", "zeroshot", "vlm/internvl3.5-8b/{d}/test.jsonl"),
    ("InternVL3.5-8B", "finetune", "vlm/internvl3.5-8b-sft/ep002/{d}/test.jsonl"),
    ("LLaVA-OV-2-8B", "zeroshot", "vlm/llava-ov2-8b/{d}/test.jsonl"),
    ("LLaVA-OV-2-8B", "finetune", "vlm/llava-ov2-8b-sft/ep002/{d}/test.jsonl"),
)


def pick_runs(runs):
    """Mỗi model MỘT run (người dùng 2026-11-14): có finetune thì lấy finetune, không có mới lấy zero-shot."""
    ft = {r[0] for r in runs if r[1] == "finetune"}
    return tuple(r for r in runs if r[1] == "finetune" or r[0] not in ft)


RUNS = pick_runs(ALL_RUNS)          # 11 run: CE-Loc, GPT-6 Luna, gpt-4.1-mini, GLM-5.3-Flash zero-shot; 7 model còn lại finetune


def run_label(model, setting):
    """Nhãn run trên giao diện: CE-Loc chỉ một bản (không zero-shot / finetune), còn lại kèm setting."""
    name = DISPLAY.get(model, model)
    return name if model == "CE-Loc" else f"{name} | {SETTING_LABEL[setting]}"


# ---------------------------------------------------------------- prepare

def _finite(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def _best(boxes, cnll):
    """-> (chỉ số box C-NLL thấp nhất trong số hữu hạn, cnll) | None khi không box nào có C-NLL."""
    ok = [(c, i) for i, c in enumerate(cnll) if _finite(c) and i < len(boxes)]
    if not ok:
        return None
    c, i = min(ok)
    return i, c


def _clip(box, wh):
    W, H = wh
    x1, y1, x2, y2 = (float(v) for v in box)
    return [round(min(max(x1, 0), W), 2), round(min(max(y1, 0), H), 2), round(min(max(x2, 0), W), 2), round(min(max(y2, 0), H), 2)]


def read_run(path, key, wh_of, image_size=512):
    """Dump một run -> {image_id: (box pixel ảnh gốc, cnll, invalid)} (ảnh không box nào có C-NLL: bỏ). `wh_of(image_id)` = cỡ ảnh
    gốc. `.json` (eval CE-Loc / baseline: `results[key]`, box canvas `image_size`) hoặc `test.jsonl` (VLM: box ảnh gốc, box cả ảnh
    đánh dấu `fallback` khi model không trả box hợp lệ)."""
    from ce_localization.data.turns import letterbox_size
    out = {}
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                wh = wh_of(r["image_id"])
                if r.get("orig_wh") and list(r["orig_wh"]) != list(wh):
                    raise ValueError(f"{path}: {r['image_id']} orig_wh {r['orig_wh']} khác ảnh {wh}")
                bs = r.get("boxes") or []
                best = _best([b["box"] for b in bs], [b.get("cnll") for b in bs])
                if best:
                    i, c = best
                    out[r["image_id"]] = (_clip(bs[i]["box"], wh), c, bool(bs[i].get("fallback")))
        return out
    with open(path) as f:
        rows = json.load(f)["results"][key]
    for x in rows:
        best = _best(x["boxes"], x.get("cnll") or [])
        if not best:
            continue
        wh = wh_of(x["image_id"])
        s, nw, nh = letterbox_size(wh, image_size)
        if abs(nw - x["wh"][0]) > 1e-3 or abs(nh - x["wh"][1]) > 1e-3:
            raise ValueError(f"{path}: {x['image_id']} vùng ảnh dump {x['wh']} khác ({nw}, {nh})")
        i, c = best
        out[x["image_id"]] = (_clip([v / s for v in x["boxes"][i]], wh), c, False)
    return out


def _boxes(a):
    return [[round(float(v), 1) for v in b] for b in a]


def _images(ds, data_dir, review=None):
    """-> hàm image_id -> (ảnh nguồn tương đối `data_dir`, ảnh cache tương đối, lớp, [W, H] ảnh gốc, box vật cùng lớp). Box vật =
    đúng bộ vật C-NLL dùng để fit Gaussian (pixel ảnh gốc, xyxy): CE-130 vật còn trong ảnh ở lượt t (`present_objects`), CoCount
    `exam_bbox` sau rà tay (`review` = `load_review(cocount_review.jsonl)`)."""
    if ds == "ce130":
        from ce_localization.data.turns import present_objects
        with open(os.path.join(data_dir, "turn_index.json")) as f:
            ti = json.load(f)

        def info(i):
            e = ti["turns"][i]
            b = ti["branches"][e["branch"]]
            src = "samples/" + e["sample"]
            return (src, "ce130/" + os.path.splitext(e["sample"])[0].replace("/", "_") + ".jpg", b["class"],
                    [int(v) for v in b["wh"]], _boxes(present_objects(b, e["t"])) if "objects" in b else [])
        return info

    def info(i):
        from ce_localization.data.cocount import read_cocount
        root = os.path.join(data_dir, "cocount")
        objs = (_boxes(read_cocount(root, i, review)["objects"])
                if os.path.exists(os.path.join(root, "Anno_with_exam_bbox", i + ".json")) else [])
        with open(os.path.join(root, "Anno", i + ".json")) as f:
            cls = clean_class(json.load(f)["class_name"])
        src = f"cocount/Image/{i}.jpg"
        with Image.open(os.path.join(data_dir, src)) as im:
            wh = list(im.size)
        return src, f"cocount/{i}.jpg", cls, wh, objs
    return info


def prepare(output_root, out_dir, data_dir, runs=RUNS, cache=False, review=None, log=print):
    """Đọc dump mọi run -> `<out_dir>/manifest.json` (+ cache JPEG nếu `cache`). `review`: file rà tay CoCount (mặc định
    `<data_dir>/cocount_review.jsonl`, không có thì box vật chưa rà). -> manifest."""
    from ce_localization.data.cocount import load_review
    t0 = time.time()
    review = review or os.path.join(data_dir, "cocount_review.jsonl")
    rv = load_review(review) if os.path.exists(review) else None
    items, objects, stats = [], {}, collections.OrderedDict()
    for ds, (sub, key) in DSETS.items():
        info = functools.lru_cache(maxsize=None)(_images(ds, data_dir, rv))
        order = None
        for model, setting, pat in runs:
            path = os.path.join(output_root, pat.format(d=sub))
            got = read_run(path, key, lambda i: info(i)[3])
            if order is None:
                order = list(got)
            elif set(got) != set(order):
                raise ValueError(f"{path}: tập ảnh có C-NLL ({len(got)}) khác run đầu ({len(order)}): "
                                 f"{sorted(set(got) ^ set(order))[:5]}")
            for i in order:
                box, c, bad = got[i]
                src, img, cls, wh, objs = info(i)
                objects[f"{ds}|{i}"] = objs
                items.append({"ds": ds, "model": model, "setting": setting, "id": i, "key": f"{ds}|{model}|{setting}|{i}",
                              "cls": cls, "wh": wh, "src": src, "img": img, "box": box, "cnll": round(c, 4), "invalid": bad})
            stats[f"{LABELS[ds]} {model} {setting}"] = {"n": len(order), "invalid": sum(got[i][2] for i in order)}
        log(f"[prepare] {LABELS[ds]}: {len(order)} ảnh × {len(runs)} run ({time.time() - t0:.0f}s)")
    for k, v in stats.items():
        log(f"  {k}: {v['n']} màn" + (f", {v['invalid']} box cả ảnh (model không trả box hợp lệ)" if v["invalid"] else ""))
    man = {"version": 2, "time": datetime.now().isoformat(timespec="seconds"), "runs": [[m, s] for m, s, _ in runs],
           "sources": {f"{m}|{s}": p for m, s, p in runs}, "items": items, "objects": objects}
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(man, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, os.path.join(out_dir, "manifest.json"))
    log(f"[prepare] {len(items)} màn -> {os.path.join(out_dir, 'manifest.json')} ({time.time() - t0:.0f}s)")
    if cache:
        imgs = {it["img"]: it["src"] for it in items}
        for k, (rel, src) in enumerate(imgs.items(), 1):
            ensure_jpeg(os.path.join(out_dir, "cache"), data_dir, rel, src)
            if k % 1000 == 0 or k == len(imgs):
                log(f"[prepare] cache {k}/{len(imgs)} ảnh ({time.time() - t0:.0f}s)")
    return man


def ensure_jpeg(cache, data_dir, rel, src):
    """Ảnh cache JPEG (cỡ gốc) của màn; chưa có thì dựng từ ảnh nguồn. -> đường dẫn tuyệt đối."""
    path = os.path.abspath(os.path.join(cache, rel))
    if not os.path.exists(path):
        with Image.open(os.path.join(data_dir, src)) as im:
            _save_jpeg(im.convert("RGB"), path, 90)
    return path


# ---------------------------------------------------------------- serve

def judged(rec):
    return bool(rec) and rec.get("ok") in (True, False)


class JudgeAnnotator(Annotator):
    """Phiên chấm một (tập, user); bộ lọc = model (mỗi model một run); xong = đã bấm ổn / không ổn. `ndone` = số màn đã chấm mỗi
    run, đếm một lần lúc mở rồi cộng dồn khi lưu (bảng tiến độ không duyệt lại cả file mỗi lần bấm)."""

    filter_keys = ("model",)
    rec_done = staticmethod(judged)

    def __init__(self, items, path, name, user, objects=None):
        super().__init__(items, path, name, user)
        self.objects = objects or {}
        self.ndone = collections.Counter((r["model"], r["setting"]) for k, r in self.store.latest.items()
                                         if k in self.index and judged(r))

    def save(self, ok, box=None, src=None):
        it = self.cur()
        if not judged(self.store.latest.get(it["key"])):
            self.ndone[(it["model"], it["setting"])] += 1
        self.store.put({"key": it["key"], "dataset": it["ds"], "model": it["model"], "setting": it["setting"], "image_id": it["id"],
                        "class": it["cls"], "box": it["box"], "cnll": it["cnll"], "invalid": it["invalid"], "wh": it["wh"],
                        "ok": bool(ok), "user": self.user, "time": datetime.now().isoformat(timespec="milliseconds"),
                        "sec": round(time.time() - self.t_show, 1)})
        self.go(self.pos + 1)
        return ""

    def html(self, url, nonce=0):
        it = self.cur()
        rec = self.store.latest.get(it["key"])
        W, H = it["wh"]
        if judged(rec):
            state = 'judged: ' + ('<b class="cg_ok">OK</b>' if rec["ok"] else '<b class="cg_bad">NOT OK</b>')
        else:
            state = "not judged"
        meta = [f"class <b>{html.escape(it['cls'] or '?')}</b>", f"id {html.escape(it['id'])}"]
        if it["invalid"]:
            meta.append('<span class="cg_bad">no valid box from the model: box = whole image</span>')
        objs = self.objects.get(f"{it['ds']}|{it['id']}", [])
        meta.append(f"{len(objs)} object boxes (blue)")
        meta.append(state)
        nxt = [self.view[i] for i in range(max(self.pos - 1, 0), min(self.pos + 1 + PREFETCH, len(self.view))) if i != self.pos]
        pre = "|".join(u for u in (url(self.items[p]["img"], False) for p in nxt) if u)     # chỉ ảnh đã có cache
        return (f'<div class="cg_root" data-n="{nonce}" data-key="{html.escape(it["key"])}" data-wh="{W},{H}" '
                f'data-img="{html.escape(it["ds"] + "|" + it["id"])}" data-box="{",".join(str(v) for v in it["box"])}" '
                f'data-objs="{",".join(str(v) for b in objs for v in b)}" data-pre="{html.escape(pre)}">'
                f'<div><span class="cg_task cg_model">{html.escape(run_label(it["model"], it["setting"]))}</span> '
                f'<span class="cg_meta">{" | ".join(meta)}</span></div><div class="cg_meta" id="cg_zoominfo"></div>'
                f'<div class="cg_row"><div class="cg_panel"><div class="cg_vp" style="aspect-ratio: {W} / {H}">'
                f'<div class="cg_wrap" style="width: {W}px; height: {H}px"><img src="{url(it["img"])}" draggable="false" alt="">'
                f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg"></svg></div></div></div></div>'
                "</div>")


class JudgeHub(Hub):
    """Mọi phiên (tập × user) dựng sẵn từ manifest; `act` dùng chung với `cegen.Hub`. Manifest cũ (cả zero-shot lẫn finetune một
    model) cũng chạy: chỉ giữ run `pick_runs`. Chia việc theo thứ tự ảnh của mọi run trong manifest (không đổi khi bớt run)."""

    def __init__(self, manifest, out_dir, data_dir, users=USERS):
        self.cache, self.data_dir = os.path.join(out_dir, "cache"), data_dir
        self.src = {it["img"]: it["src"] for it in manifest["items"]}
        self.runs = list(pick_runs([tuple(r) for r in manifest["runs"]]))
        keep = set(self.runs)
        self.models = [m for m, _ in self.runs]
        self.labels = {m: run_label(m, s) for m, s in self.runs}
        objects = manifest.get("objects", {})
        self.sessions = {}
        for ds in DSETS:
            items = [it for it in manifest["items"] if it["ds"] == ds]
            for k, u in enumerate(users):
                mine = [it for it in split_items(items, k, len(users)) if (it["model"], it["setting"]) in keep]
                if mine:
                    self.sessions[(LABELS[ds], u)] = JudgeAnnotator(mine, os.path.join(out_dir, f"judge_{ds}_{u}.jsonl"),
                                                                    f"{LABELS[ds]} {u}", u, objects)
        self.users = list(users)
        self.datasets = [LABELS[ds] for ds in DSETS if any(k[0] == LABELS[ds] for k in self.sessions)]
        self.renders = itertools.count(1)
        self.totals = {k: collections.Counter((it["model"], it["setting"]) for it in a.items) for k, a in self.sessions.items()}

    def url(self, rel, build=True):
        """URL ảnh cache; chưa có: `build` thì dựng ngay (ảnh đang hiện), không thì None (ảnh tải trước — luồng nền `warm` dựng)."""
        path = os.path.abspath(os.path.join(self.cache, rel))
        if not os.path.exists(path):
            if not build:
                return None
            path = ensure_jpeg(self.cache, self.data_dir, rel, self.src[rel])
        return FILE_ROUTE + quote(path)

    def warm(self, log=print):
        """Dựng sẵn mọi ảnh cache còn thiếu (chạy ở luồng nền lúc mở app), xen kẽ các phiên theo thứ tự chấm."""
        t0, order = time.time(), []
        for a in self.sessions.values():
            order.append(list(dict.fromkeys(it["img"] for it in a.items)))
        todo = [r for r in dict.fromkeys(itertools.chain.from_iterable(itertools.zip_longest(*order)))
                if r and not os.path.exists(os.path.join(self.cache, r))]
        for k, rel in enumerate(todo, 1):
            try:
                ensure_jpeg(self.cache, self.data_dir, rel, self.src[rel])
            except OSError as e:
                log(f"[warm] lỗi {rel}: {e}")
            if k % 500 == 0 or k == len(todo):
                log(f"[warm] cache {k}/{len(todo)} ảnh ({time.time() - t0:.0f}s)")

    def progress(self, data):
        """Bảng markdown: mỗi run của tập `data` — số màn đã chấm của từng user / tổng (tổng tính sẵn, chỉ đếm bản đã chấm)."""
        cnt = collections.defaultdict(collections.Counter)
        for (d, u), a in self.sessions.items():
            if d != data:
                continue
            for run, n in self.totals[(d, u)].items():
                cnt[run][u + "_n"] += n
            for run, n in a.ndone.items():
                cnt[run][u] += n
        if not cnt:
            return ""
        rows = ["| run | " + " | ".join(self.users) + " | all |", "|---|" + "---:|" * (len(self.users) + 1)]
        for m, st in self.runs:
            c = cnt.get((m, st))
            if not c:
                continue
            done, n = sum(c[u] for u in self.users), sum(c[u + "_n"] for u in self.users)
            rows.append(f"| {run_label(m, st).replace('|', chr(92) + '|')} | " + " | ".join(f"{c[u]}/{c[u + '_n']}" for u in self.users)
                        + f" | {'**done**' if done == n else f'{100 * done / n:.0f}%'} |")
        return f"**Progress per run: {data}** (both users; done = OK or Not OK)\n\n" + "\n".join(rows)


JUDGE_JS = """
function cgDraw() {
  const r = cgRoot(); if (!r) return;
  const vps = cgVps(), sc = vps.length ? cgScale(vps[0]) : 1, b = (r.dataset.box || '').split(',').map(Number);
  const o = (r.dataset.objs || '').split(',').filter((v) => v !== '').map(Number);
  let s = '';
  if (!window.__cg.hideObjs) {                         // box vật cùng lớp: viền xanh mảnh 1,5 px màn hình
    const w = (1.5 / sc).toFixed(3);
    for (let i = 0; i + 3 < o.length; i += 4) {
      s += `<rect class="cg_obj" x="${o[i]}" y="${o[i + 1]}" width="${o[i + 2] - o[i]}" height="${o[i + 3] - o[i + 1]}" stroke-width="${w}"/>`;
    }
  }
  if (b.length === 4) {                                // box model (C-NLL thấp nhất): viền đỏ 4 px màn hình, vẽ sau cùng (nằm trên)
    s += `<rect class="cg_pred" x="${b[0]}" y="${b[1]}" width="${b[2] - b[0]}" height="${b[3] - b[1]}" stroke-width="${(4 / sc).toFixed(3)}"/>`;
  }
  r.querySelectorAll('svg').forEach((svg) => { svg.innerHTML = s; });
  const info = document.getElementById('cg_zoominfo');
  if (info) info.textContent = cgZoomText() + ' | h = hide / show object boxes';
}
window.cgKey = function (e) {
  if (e.key === 'h' || e.key === 'H') { window.__cg.hideObjs = !window.__cg.hideObjs; cgDraw(); }
};
window.cgPayload = function () { return JSON.stringify({key: window.__cg.key}); };
"""
KEYS_JS = "<script>" + VIEW_JS + JUDGE_JS + "</script>"
CSS = BASE_CSS + (".cg_model {background: #2b3a55;} .cg_pred {fill: none; stroke: #ff1a1a;} .cg_obj {fill: none; stroke: #1e90ff;} "
                  "#cg_zoominfo {margin: 4px 0; font-family: monospace;}")


def build_ui(hub, gr):
    """Thanh trên: menu Data / User / Model (= một run) + 5 nút; dưới: ảnh + box; cuối: bảng tiến độ hai tập."""
    import inspect
    kw = {"title": "Box review", "css": CSS, "head": KEYS_JS, "fill_width": True}
    bparams = inspect.signature(gr.Blocks.__init__).parameters
    launch_kw = {k: v for k, v in kw.items() if k not in bparams and k in ("css", "head")}     # gradio >= 6: css / head ở launch
    dd = {"value": None, "filterable": False, "min_width": 140}
    with gr.Blocks(**{k: v for k, v in kw.items() if k in bparams}) as demo:
        with gr.Row(equal_height=True):
            data = gr.Dropdown(hub.datasets, label="Data", **dd)
            user = gr.Dropdown(hub.users, label="User", **dd)
            model = gr.Dropdown([(hub.labels[m], m) for m in hub.models], label="Model", **dd)
        with gr.Row():
            ok = gr.Button("OK (Enter)", variant="primary", elem_id="cg_ok", min_width=100)
            fail = gr.Button("Not OK (f)", variant="stop", elem_id="cg_fail", min_width=100)
            prev = gr.Button("<- Prev", elem_id="cg_prev", min_width=100)
            nxt = gr.Button("Next ->", elem_id="cg_next", min_width=100)
            first = gr.Button("Go to first not done", min_width=100)
        head = gr.Markdown(CHOOSE)
        img = gr.HTML()
        payload = gr.Textbox(elem_classes="cg_hidden")
        with gr.Row():
            progs = [gr.Markdown(hub.progress(d)) for d in hub.datasets]
        outs = [head, img, *progs]
        sel = [data, user, model]
        q = {"queue": False, "show_progress": "hidden"}
        js = "(d, u, m, p) => [d, u, m, window.cgPayload ? window.cgPayload() : '']"

        def run(d, u, m, p, action):
            if not (d and u and m):
                return CHOOSE, "", *[hub.progress(x) for x in hub.datasets]
            head_md, page = hub.act(d, u, p, action, {"model": m})
            return head_md, page, *[hub.progress(x) for x in hub.datasets]

        def mk(action):
            return lambda d, u, m, p: run(d, u, m, p, action)
        for b, action in ((ok, "ok"), (fail, "fail"), (prev, "prev"), (nxt, "next"), (first, "first")):
            b.click(mk(action), sel + [payload], outs, js=js, **q)
        for r in sel:
            r.input(lambda d, u, m: run(d, u, m, "", "show"), sel, outs, **q)
        demo.load(lambda: (CHOOSE, "", *[hub.progress(x) for x in hub.datasets]), None, outs, **q)
    return demo, launch_kw


def serve(out_dir, data_dir, host="0.0.0.0", port=7860, share=False):
    import gradio as gr
    path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(path):
        sys.exit(f"thiếu {path} — chạy `loc_judge.py prepare` trước")
    t0 = time.time()
    with open(path) as f:
        man = json.load(f)
    hub = JudgeHub(man, out_dir, data_dir)
    for (d, u), a in hub.sessions.items():
        print(f"[{d} {u}] {len(a.items)} màn, xong {a.n_saved()} -> {a.store.path}", flush=True)
    print(f"[serve] manifest {len(man['items'])} màn ({man['time']}), dựng phiên {time.time() - t0:.1f}s; ảnh nguồn {data_dir}",
          flush=True)
    demo, launch_kw = build_ui(hub, gr)
    os.makedirs(hub.cache, exist_ok=True)
    threading.Thread(target=hub.warm, kwargs={"log": lambda m: print(m, flush=True)}, daemon=True).start()
    demo.launch(server_name=host, server_port=port, share=share, allowed_paths=[os.path.abspath(hub.cache)], **launch_kw)


# ---------------------------------------------------------------- merge

MERGE_KEEP = ("dataset", "model", "setting", "image_id", "class", "wh", "box", "cnll", "invalid")


def merge(out_dir, dest=None, log=print):
    """Gộp `judge_<tập>_<user>.jsonl` -> một file mỗi run xong hết (xem docstring đầu file). -> {tên file: số dòng}."""
    dest = dest or os.path.join(out_dir, "merged")
    with open(os.path.join(out_dir, "manifest.json")) as f:
        man = json.load(f)
    written = {}
    for ds in DSETS:
        order = list(dict.fromkeys(it["id"] for it in man["items"] if it["ds"] == ds))
        files = sorted(f for f in os.listdir(out_dir) if f.startswith(f"judge_{ds}_") and f.endswith(".jsonl"))
        runs, owner = collections.defaultdict(dict), {}
        for fn in files:
            for k, rec in Store(os.path.join(out_dir, fn)).latest.items():
                if not judged(rec):
                    continue
                if owner.setdefault(k, fn) != fn:
                    raise ValueError(f"màn {k} được chấm ở cả {owner[k]} và {fn}")
                runs[(rec["model"], rec["setting"])][rec["image_id"]] = rec
        log(f"[merge] {LABELS[ds]}: {len(files)} file, {len(order)} ảnh")
        for m, st in pick_runs([tuple(r) for r in man["runs"]]):
            recs = runs.get((m, st), {})
            name = f"{ds}_{m.lower().replace(' ', '-')}_{st}.jsonl"
            if set(recs) != set(order):
                log(f"  {name}: chưa xong ({len(set(recs) & set(order))} / {len(order)}), bỏ qua")
                continue
            os.makedirs(dest, exist_ok=True)
            tmp = os.path.join(dest, name + ".tmp")
            with open(tmp, "w") as f:
                for i in order:
                    r = recs[i]
                    f.write(json.dumps({**{k: r[k] for k in MERGE_KEEP}, "ok": r["ok"]}, ensure_ascii=False) + "\n")
            os.replace(tmp, os.path.join(dest, name))
            written[name] = len(order)
            log(f"  {name}: {len(order)} dòng")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="đọc dump box -> manifest.json")
    p.add_argument("--output-root", default=OUTPUT_ROOT, help="thư mục output/add (chứa celoc/, baseline/, vlm/)")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--data-dir", default=DATA_DIR, help="object-detection/data (turn_index.json, samples/, cocount/)")
    p.add_argument("--cache", action="store_true", help="dựng sẵn mọi ảnh JPEG (mặc định: serve dựng dần lúc cần)")
    s = sub.add_parser("serve", help="mở app Gradio")
    s.add_argument("--out-dir", default=OUT_DIR)
    s.add_argument("--data-dir", default=DATA_DIR)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=7861)
    s.add_argument("--share", action="store_true", help="tạo link công khai *.gradio.live (hết hạn sau 72 giờ)")
    g = sub.add_parser("merge", help="gộp bản chấm các user, mỗi run xong hết một file")
    g.add_argument("--out-dir", default=OUT_DIR)
    g.add_argument("--dest", default=None, help="mặc định <out-dir>/merged")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)  # noqa: E731
    if a.cmd == "prepare":
        prepare(a.output_root, a.out_dir, a.data_dir, cache=a.cache, log=log)
    elif a.cmd == "serve":
        serve(a.out_dir, a.data_dir, a.host, a.port, a.share)
    else:
        merge(a.out_dir, a.dest, log=log)


if __name__ == "__main__":
    main()

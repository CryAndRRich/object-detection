#!/usr/bin/env python3
"""Web chấm kết quả CE-Gen (người dùng 2026-11-08): mỗi màn một (tập, model, zeroshot / finetune, add / remove, ảnh) — người chấm xem
ảnh input + ảnh output của model (+ heatmap khác biệt) và làm HAI việc độc lập: (1) kết luận THÀNH CÔNG (thêm / xoá đúng MỘT vật cùng
loại) hay THẤT BẠI, (2) box bao vật được thêm / bị xoá — TỰ VẼ, hoặc bấm `s` nhận box gợi ý số 1 khi nó đã khớp. Một màn XONG khi và
chỉ khi có đủ cả hai (người dùng chốt 2026-11-08; box gợi ý chưa được nhận thì không tính). Thành công / thất bại -> metric user study; box -> C-NLL.

Dữ liệu (server 2, thư mục `hachi/`): `<src>/FSC_final` (CE-130, lượt 1: 1.679 nhánh = 1.335 test + 344 val, chia 4 subset
`FSC_{easy,difficult}_{overlap,no_overlap}`) và `<src>/processed_dataset` (CE-CoCount, 1.484 file lớp); mỗi tập 3 model × zeroshot /
finetune × add / remove = 12 run, mỗi run (hoặc subset) có `img/` (output), `input/` và `results_diff_boxes_v2/` (`records.jsonl`:
box khác biệt `candidates` xếp theo điểm, pixel ẢNH INPUT, xyxy cạnh phải / dưới không tính; `<id>/heatmap.png`) — sinh bằng
`hachi/Image_Editing/code/osor/SAVP/core/eval_difference_boxes_strong_v2.py` (CPU, không model; chạy trong `core/`, tham số như
`eval_diffrence_boxes.sh`: `--input <run>/input --output <run>/img --save-dir <run>/results_diff_boxes_v2 --threshold 0.07
--min-area 200`). Box gợi ý = vùng khác biệt MẠNH (heatmap > 0,35) — heatmap chuẩn hoá theo phân vị 99 của từng ảnh nên output không
đổi gì vẫn có box gợi ý; box có thể chỉ phủ một phần vật ⇒ người chấm sửa.

KHÔNG import torch — chạy bằng env `imgedit` (numpy + Pillow; `serve` thêm gradio). Hai bước:

  prepare   quét 2 tập × 12 run, đọc `records.jsonl`, dựng SẴN cache JPEG (input, output resize về cỡ input, heatmap tô màu turbo —
            vài chục KB / ảnh thay cho PNG output 1–2 MB) + `manifest.json` (mọi màn, lớp, box gợi ý). Song song `--workers`, chạy lại
            chỉ làm phần còn thiếu (`--force` làm lại hết). In thống kê từng run (status, không có box gợi ý, output lệch tỉ lệ, cỡ
            input khác ảnh gốc).
  serve     app Gradio (`--share`: link *.gradio.live; luôn mở cả cổng LAN `--host`): thanh trên = 5 menu Data / User / Model /
            Setting / Task (= MỘT run; chấm hết một run trước để có số sớm) + 5 nút; 3 ảnh chia đều cả trang; cuối trang 2 bảng
            "Progress per run" (CE130, CoCount — số màn đã chấm của từng run, cả hai user; hiện ngay từ đầu). Chữ trên giao diện
            chỉ dùng ký tự ASCII (người dùng 2026-11-12). Mọi phiên
            (2 tập × 2 user) dựng sẵn lúc mở app; trình duyệt tải trước + giải mã sẵn ảnh 8 màn kế; vẽ / chọn box chạy hoàn toàn ở
            trình duyệt, chỉ lúc lưu / chuyển màn mới gọi server (một lượt đi về).

Chia việc: theo ẢNH (`image_id`), xen kẽ theo thứ tự ảnh -> user1 / user2; mỗi ảnh chấm đủ 12 run (6 add rồi 6 remove, liền nhau —
ảnh input chỉ tải một lần). Hiện tên model / zeroshot / finetune (không chấm mù — người dùng 2026-11-08).

Thao tác (box gợi ý: số 1 = khung đỏ liền, số 2.. = đỏ đứt nét; box đang chọn / đã vẽ = vàng):
  s           nhận box gợi ý số 1 (vùng khác biệt mạnh nhất) làm box của màn
  kéo chuột   vẽ box (trên cả 3 ảnh — cùng hệ toạ độ); vẽ lại = thay box
  lăn chuột   phóng / thu quanh con trỏ, + / - phóng / thu, 0 về vừa khung; chuột phải kéo (hoặc giữ Space + kéo) = dịch ảnh — 3
              ảnh cùng phóng, cùng dịch; sang màn cùng ảnh input (12 run liền nhau) giữ nguyên zoom, ảnh khác về vừa khung
  Enter       THÀNH CÔNG + lưu box vàng   } đủ kết luận + box: màn XONG, sang màn sau; chưa có box vàng: vẫn lưu kết luận,
  f           THẤT BẠI + lưu box vàng     } ở lại màn, báo "chưa xong" (vẽ / bấm s rồi bấm lại)
  r / x       về box đã lưu của màn / bỏ box
  h           ẩn / hiện heatmap
  ← / →       màn trước / sau (không lưu; lưu lại màn đã chấm = ghi đè)

Kết quả: `<out-dir>/annot_<ce130|cocount>_<user>.jsonl` — chỉ GHI NỐI, dòng sau cùng của mỗi `key` là bản chấm (tắt app mở lại: tiếp
từ màn chưa chấm đầu tiên). Mỗi dòng: {"key", "dataset", "model", "setting", "task", "subset", "image_id", "class", "success",
"box": [x1, y1, x2, y2] | null (pixel ẢNH INPUT, toạ độ liên tục 0..W / 0..H — cùng hệ với box vật CE-130 / CE-CoCount),
"box_src": "drawn" (tự vẽ) | "picked" (nhận box gợi ý số 1 bằng `s`) | null, "wh": [W, H], "user", "time", "sec"}. Bản chấm dùng
được = dòng sau cùng của key thoả `is_done` (có `success` và box drawn / picked); dòng cũ "box_src": "cand<k>" (box gợi ý tự chọn
sẵn, logic trước 2026-11-08 tối) KHÔNG tính là xong.

  python box_review/cegen.py prepare --workers 8
  python box_review/cegen.py serve --share
"""

import argparse
import collections
import functools
import hashlib
import html
import io
import itertools
import json
import os
import re
import statistics
import sys
import threading
import time
from datetime import datetime
from urllib.parse import quote

import numpy as np
from PIL import Image

OD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # object-detection/
SRC = "/mnt/disk1/aiotlab/hachi/Image_Editing/output"
OUT_DIR = os.path.join(os.path.dirname(OD), "output", "cegen", "annot")
DATASETS = {"FSC_final": "ce130", "processed_dataset": "cocount"}
LABELS = {"ce130": "CE130", "cocount": "CoCount"}
USERS = ("user1", "user2")
TASKS = ("add", "remove")
SETTINGS = ("zeroshot", "finetune")
V2 = "results_diff_boxes_v2"
SUBSET_RE = re.compile(r"FSC_(easy|difficult)_(no_)?overlap$")
TASK_RE = re.compile(r"(?:^|[_\-])(add|remove)(?=$|[_\-])", re.I)
SKIP_DIRS = {"img", "input", "mask_l1", "mask_l2", "visual"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PREFETCH = 8
FILTER_KEYS = ("model", "setting", "task")
BOX_SRC = ("drawn", "picked")        # tự vẽ / nhận box gợi ý số 1 (phím s)
FILE_ROUTE = "/gradio_api/file="
CHOOSE = "**Choose Data, User, Model, Setting and Task above to start.**"
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


# ---------------------------------------------------------------- prepare

def clean_class(name):
    """Như `ce_localization.data.cocount.clean_class` (module đó import torch): bỏ phần trong ngoặc, dấu chấm cuối, khoảng trắng thừa."""
    return re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", name)).strip().rstrip(".").strip()


def _turbo_lut():
    """Bảng màu turbo 256 × 3 uint8 (xấp xỉ đa thức của Google, cùng bảng `cv2.COLORMAP_TURBO` của preview.jpg)."""
    x = np.linspace(0, 1, 256)
    poly = lambda c: sum(ci * x ** i for i, ci in enumerate(c))  # noqa: E731
    r = poly([0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943])
    g = poly([0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604])
    b = poly([0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973])
    return (np.clip(np.stack([r, g, b], 1), 0, 1) * 255).round().astype(np.uint8)


TURBO = _turbo_lut()


def _nat(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def run_task(name):
    """Tên thư mục run -> "add" | "remove" (đúng một trong hai, đứng riêng giữa `_` / `-`)."""
    found = {m.lower() for m in TASK_RE.findall(name)}
    if len(found) != 1:
        raise ValueError(f"không rút được add / remove từ tên run {name!r}")
    return found.pop()


def find_runs(src):
    """Mọi thư mục có cả `img/` và `input/` dưới `<src>/<FSC_final|processed_dataset>/<model>/<zeroshot|finetune>/<run>/...` -> list
    dict(ds, model, setting, task, subset, run_dir, path); subset = tên `FSC_*` hoặc "". Một (tập, model, setting, task) đúng một run."""
    out, seen = [], {}
    for dname, ds in DATASETS.items():
        root = os.path.join(src, dname)
        if not os.path.isdir(root):
            raise FileNotFoundError(root)
        for dirpath, dirs, _ in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("results_diff"))
            if not (os.path.isdir(os.path.join(dirpath, "img")) and os.path.isdir(os.path.join(dirpath, "input"))):
                continue
            rel = os.path.relpath(dirpath, root).split(os.sep)
            if len(rel) < 3 or rel[1] not in SETTINGS:
                raise ValueError(f"không hiểu bố cục {dirpath}: cần <model>/<zeroshot|finetune>/<run>/...")
            subset = rel[-1] if SUBSET_RE.match(rel[-1]) else ""
            run_dir = os.path.dirname(dirpath) if subset else dirpath
            unit = (ds, rel[0], rel[1], run_task(rel[2]))
            if seen.setdefault(unit, run_dir) != run_dir:
                raise ValueError(f"hai run cùng {unit}: {seen[unit]} và {run_dir}")
            out.append(dict(ds=ds, model=rel[0], setting=rel[1], task=unit[3], subset=subset, run_dir=run_dir, path=dirpath))
    return out


def _list_images(d):
    out = {}
    for f in os.listdir(d):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXT:
            if stem in out:
                raise ValueError(f"trùng tên ảnh {stem} trong {d}")
            out[stem] = os.path.join(d, f)
    return out


def _load_records(path):
    f = os.path.join(path, V2, "records.jsonl")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        return {r["id"]: r for r in (json.loads(line) for line in fh if line.strip())}


def _save_jpeg(im, path, quality):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    im.save(tmp, format="JPEG", quality=quality, optimize=True)
    os.replace(tmp, path)              # nhiều worker cùng ghi một ảnh input: bản nào thay sau cũng nguyên vẹn


def _prep(job, cache, force=False):
    """Một màn: cache JPEG input (đặt tên theo md5 nội dung ⇒ 12 run chung ảnh input chỉ lưu một lần), output resize về cỡ input,
    heatmap tô màu. -> dict thêm wh, out_wh, in, out, hm (đường dẫn tương đối trong cache)."""
    with open(job["src_in"], "rb") as f:
        data = f.read()
    rel_in = f"{job['ds']}/input/{job['id']}_{hashlib.md5(data).hexdigest()[:12]}.jpg"
    with Image.open(io.BytesIO(data)) as im:
        W, H = im.size
        if force or not os.path.exists(os.path.join(cache, rel_in)):
            _save_jpeg(im.convert("RGB"), os.path.join(cache, rel_in), 90)
    base = f"{job['ds']}/{job['model']}/{job['setting']}/{job['task']}/{job['id']}"
    with Image.open(job["src_out"]) as im:
        out_wh = list(im.size)
        if force or not os.path.exists(os.path.join(cache, base + ".jpg")):
            im = im.convert("RGB")
            if im.size != (W, H):
                im = im.resize((W, H), LANCZOS, reducing_gap=3.0)
            _save_jpeg(im, os.path.join(cache, base + ".jpg"), 85)
    rel_hm = None
    if job["src_hm"] and os.path.exists(job["src_hm"]):
        rel_hm = base + "_hm.jpg"
        if force or not os.path.exists(os.path.join(cache, rel_hm)):
            with Image.open(job["src_hm"]) as im:
                g = im.convert("L")
                if g.size != (W, H):
                    g = g.resize((W, H), LANCZOS)
                _save_jpeg(Image.fromarray(TURBO[np.asarray(g)]), os.path.join(cache, rel_hm), 80)
    out = {k: v for k, v in job.items() if not k.startswith("src_")}
    out.update(wh=[W, H], out_wh=out_wh, **{"in": rel_in, "out": base + ".jpg", "hm": rel_hm})
    return out


def _fmt(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def _ref_sizes(turn_index, cocount_root, ids):
    """Lớp + cỡ ảnh gốc: CE-130 từ `turn_index.json` (nhánh `<split>/<id>`), CE-CoCount từ `Anno/<id>.json` + `Image/<id>.jpg`."""
    ref = {}
    if ids.get("ce130"):
        with open(turn_index) as f:
            br = json.load(f)["branches"]
        for k, v in br.items():
            ref.setdefault(("ce130", k.split("/", 1)[1]), (v["class"], v.get("wh")))
    for i in ids.get("cocount", ()):
        with open(os.path.join(cocount_root, "Anno", i + ".json")) as f:
            cls = clean_class(json.load(f)["class_name"])
        img = os.path.join(cocount_root, "Image", i + ".jpg")
        wh = None
        if os.path.exists(img):
            with Image.open(img) as im:
                wh = list(im.size)
        ref[("cocount", i)] = (cls, wh)
    return ref


def prepare(src, out_dir, turn_index, cocount_root, workers=8, top_k=5, force=False, allow_missing_v2=False, expect_runs=12,
            log=print):
    """Quét run, dựng cache + `<out_dir>/manifest.json`. -> manifest (dict)."""
    t0 = time.time()
    cache = os.path.join(out_dir, "cache")
    runs = find_runs(src)
    units = collections.defaultdict(list)
    for r in runs:
        units[(r["ds"], r["model"], r["setting"], r["task"])].append(r)
    for ds in DATASETS.values():
        n = sum(u[0] == ds for u in units)
        if expect_runs and n != expect_runs:
            raise ValueError(f"{LABELS[ds]}: {n} run, cần {expect_runs}")
    jobs, missing, ids = [], [], collections.defaultdict(set)
    for r in runs:
        imgs, inps = _list_images(os.path.join(r["path"], "img")), _list_images(os.path.join(r["path"], "input"))
        if set(imgs) != set(inps):
            raise ValueError(f"img / input lệch tên ở {r['path']}: {sorted(set(imgs) ^ set(inps))[:5]}")
        recs = _load_records(r["path"])
        if recs is None:
            missing.append(r["path"])
            recs = {}
        for i in sorted(imgs, key=_nat):
            rec = recs.get(i, {})
            cands = [c["xyxy"] for c in rec.get("candidates", [])[:top_k]]
            hm = rec.get("heatmap_path") or os.path.join(r["path"], V2, i, "heatmap.png")
            jobs.append(dict(ds=r["ds"], model=r["model"], setting=r["setting"], task=r["task"], subset=r["subset"], id=i,
                             key=f"{r['ds']}|{r['model']}|{r['setting']}|{r['task']}|{i}", cands=cands,
                             status=rec.get("status", "no_v2"),
                             audit=(rec.get("quality_audit_for_review_only") or {}).get("reason"),
                             src_in=inps[i], src_out=imgs[i], src_hm=hm if recs else None))
            ids[r["ds"]].add(i)
    if missing and not allow_missing_v2:
        raise FileNotFoundError(f"{len(missing)} thư mục chưa có {V2}/records.jsonl (sinh trước, xem README):\n  "
                                + "\n  ".join(missing))
    ref = _ref_sizes(turn_index, cocount_root, ids)
    log(f"[prepare] {len(runs)} thư mục, {len(units)} run, {len(jobs)} màn ({_fmt(time.time() - t0)}); cache -> {cache}")
    fn = functools.partial(_prep, cache=cache, force=force)
    done, items, t1 = 0, [], time.time()
    step = max(1, len(jobs) // 50)

    def tick():
        el = time.time() - t1
        eta = el / done * (len(jobs) - done) if done else 0
        log(f"[prepare] {done}/{len(jobs)} ({100 * done / max(len(jobs), 1):.0f}%) · {_fmt(el)} · ETA {_fmt(eta)} · "
            f"{datetime.now():%H:%M:%S}")
    log(f"[prepare] bắt đầu, {workers} worker · {datetime.now():%H:%M:%S}")
    if workers <= 1:
        for j in jobs:
            items.append(fn(j))
            done += 1
            if done % step == 0:
                tick()
    else:
        from multiprocessing import get_context
        with get_context("fork" if sys.platform.startswith("linux") else "spawn").Pool(workers) as pool:
            for it in pool.imap_unordered(fn, jobs, chunksize=8):
                items.append(it)
                done += 1
                if done % step == 0:
                    tick()
    for it in items:
        it["cls"], it["ref_wh"] = ref.get((it["ds"], it["id"]), (None, None))
    order = {}
    for ds in DATASETS.values():
        for k, i in enumerate(sorted(ids[ds], key=_nat)):
            order[(ds, i)] = k
    items.sort(key=lambda it: (list(DATASETS.values()).index(it["ds"]), order[(it["ds"], it["id"])], TASKS.index(it["task"]),
                               it["model"], SETTINGS.index(it["setting"])))
    stats = collections.OrderedDict()
    for it in items:
        s = stats.setdefault(f"{LABELS[it['ds']]} {it['model']} {it['setting']} {it['task']}", collections.Counter())
        s["n"] += 1
        s["status=" + it["status"]] += 1
        s["no_cand"] += not it["cands"]
        s["no_heatmap"] += it["hm"] is None
        (W, H), (ow, oh) = it["wh"], it["out_wh"]
        s["out_aspect>2%"] += abs((ow / oh) / (W / H) - 1) > 0.02
        s["input!=ref_size"] += it["ref_wh"] is not None and list(it["ref_wh"]) != it["wh"]
        s["no_class"] += it["cls"] is None
    log("[prepare] thống kê từng run:")
    for k, s in stats.items():
        log(f"  {k}: " + ", ".join(f"{a}={b}" for a, b in sorted(s.items()) if b))
    man = {"version": 1, "time": datetime.now().isoformat(timespec="seconds"), "src": os.path.abspath(src), "cache": cache,
           "top_k": top_k, "items": items, "stats": {k: dict(v) for k, v in stats.items()}}
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(man, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(out_dir, "manifest.json"))
    log(f"[prepare] xong {len(items)} màn trong {_fmt(time.time() - t0)} -> {os.path.join(out_dir, 'manifest.json')}")
    return man


# ---------------------------------------------------------------- serve

def split_items(items, user_idx, n_users=len(USERS)):
    """Phần việc của người thứ `user_idx` (0..n−1): đánh số ảnh (`id`) theo thứ tự xuất hiện, người k nhận ảnh k, k+n, k+2n…
    (mọi run của một ảnh về cùng một người)."""
    order = {}
    for it in items:
        order.setdefault(it["id"], len(order))
    return [it for it in items if order[it["id"]] % n_users == user_idx]


class Store:
    """jsonl chỉ GHI NỐI (fsync mỗi lần lưu); đọc lại: dòng sau cùng của mỗi `key` thắng."""

    def __init__(self, path):
        self.path, self.latest = path, {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self.latest[rec["key"]] = rec

    def put(self, rec):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.latest[rec["key"]] = rec


def clean_box(box, wh, min_side=2.0):
    """[x1, y1, x2, y2] (số bất kỳ thứ tự) -> kẹp trong ảnh, làm tròn 1 chữ số; cạnh < `min_side` px hoặc sai dạng -> None."""
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    W, H = wh
    x1, x2 = sorted((min(max(x1, 0), W), min(max(x2, 0), W)))
    y1, y2 = sorted((min(max(y1, 0), H), min(max(y2, 0), H)))
    if x2 - x1 < min_side or y2 - y1 < min_side:
        return None
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def is_done(rec):
    """Màn XONG khi và chỉ khi có kết luận success / fail VÀ box người chấm vẽ / chủ động nhận (người dùng chốt 2026-11-08)."""
    return bool(rec) and rec.get("success") in (True, False) and rec.get("box_src") in BOX_SRC and bool(rec.get("box"))


class Annotator:
    """Phiên chấm của một (tập, user): danh sách màn, vị trí, file kết quả. Thao tác gọi trong `lock` (Hub)."""

    def __init__(self, items, path, name, user):
        if not items:
            raise ValueError("không có màn nào")
        self.items, self.name, self.user = items, name, user
        self.index = {it["key"]: p for p, it in enumerate(items)}
        self.store = Store(path)
        self.lock = threading.RLock()
        self.views, self.filt = {}, None
        self.pos, self.t_show = -1, time.time()        # `pos`: vị trí trong `view` (= số màn − 1 trên giao diện)
        self.set_filter({})

    def set_filter(self, filt):
        """`filt`: {model / setting / task: giá trị | None} -> chỉ đi qua các màn khớp (`view` = vị trí trong `items`).
        Đổi bộ lọc: sang màn chưa chấm đầu tiên của bộ lọc mới. -> False khi bộ lọc không có màn nào."""
        f = tuple((k, filt[k]) for k in FILTER_KEYS if filt.get(k) not in (None, ""))
        if f != self.filt:
            if f not in self.views:
                view = [p for p, it in enumerate(self.items) if all(it[k] == v for k, v in f)]
                self.views[f] = (view, {p: i for i, p in enumerate(view)})
            self.filt = f
            self.view, self.vindex = self.views[f]
            self.pos = -1
            if self.view:
                self.first_unsaved()
        return bool(self.view)

    def filter_name(self):
        return " | ".join(v for _, v in self.filt) or "all runs"

    def cur(self):
        return self.items[self.view[self.pos]]

    def go(self, pos):
        pos = min(max(int(pos), 0), len(self.view) - 1)
        if pos != self.pos:
            self.pos, self.t_show = pos, time.time()

    def done(self, it):
        return is_done(self.store.latest.get(it["key"]))

    def first_unsaved(self):
        """Màn CHƯA XONG đầu tiên (chưa chấm, hoặc thiếu kết luận / box tự vẽ)."""
        self.go(next((i for i, p in enumerate(self.view) if not self.done(self.items[p])), len(self.view) - 1))

    def n_saved(self, all_items=False):
        """Số màn XONG (`is_done`)."""
        ps = range(len(self.items)) if all_items else self.view
        return sum(self.done(self.items[p]) for p in ps)

    def save(self, success, box=None, src=None):
        """Lưu kết luận + box tự vẽ (nếu có) của màn hiện tại. Đủ cả hai -> XONG, sang màn sau, trả ""; thiếu box -> vẫn lưu kết
        luận, ở lại màn, trả câu báo."""
        it = self.cur()
        if src == "picked":                       # box gợi ý số 1 lấy từ manifest, không tin toạ độ trình duyệt gửi
            box = it["cands"][0] if it["cands"] else None
        box = clean_box(box, it["wh"]) if src in BOX_SRC else None
        rec = {"key": it["key"], "dataset": it["ds"], "model": it["model"], "setting": it["setting"], "task": it["task"],
               "subset": it["subset"], "image_id": it["id"], "class": it["cls"], "success": bool(success), "box": box,
               "box_src": src if box else None, "wh": it["wh"], "user": self.user,
               "time": datetime.now().isoformat(timespec="milliseconds"), "sec": round(time.time() - self.t_show, 1)}
        self.store.put(rec)
        if is_done(rec):
            self.go(self.pos + 1)
            return ""
        return (f"{'SUCCESS' if success else 'FAIL'} saved, but this screen is NOT done: draw a box around the added / removed "
                "object (drag on the image) or press s to take suggestion 1, then press Enter / f again.")

    def eta(self):
        secs = [r["sec"] for r in list(self.store.latest.values())[-200:] if is_done(r) and 0 < r.get("sec", 0) <= 300]
        left = len(self.view) - self.n_saved()
        return _fmt(statistics.median(secs) * left) if secs and left else "-"

    def header(self, warn=""):
        n, done = len(self.view), self.n_saved()
        head = f"screen {self.pos + 1} / {n} | done {done} ({100 * done / n:.1f}%) | ETA {self.eta()}"
        if self.filt:
            head += f" | all runs of {self.name}: done {self.n_saved(True)} / {len(self.items)}"
        return head + (f"\n\n**Warning: {warn}**" if warn else "")

    def html(self, url, nonce=0):
        """`nonce`: số lần vẽ — trang luôn khác bản trước ⇒ Gradio luôn vẽ lại (trình duyệt bỏ cờ chờ `busy` kể cả khi màn không đổi)."""
        it = self.cur()
        rec = self.store.latest.get(it["key"])
        W, H = it["wh"]
        if rec and rec.get("box") and rec.get("box_src") in BOX_SRC:
            box, src = rec["box"], rec["box_src"]
        else:
            box, src = None, ""
        if rec is None:
            state = "not saved"
        else:
            state = (f'saved: <b class="{"cg_ok" if rec["success"] else "cg_bad"}">{"SUCCESS" if rec["success"] else "FAIL"}</b>'
                     + (" | box OK | <b>DONE</b>" if is_done(rec) else ' | <b class="cg_bad">NO BOX, not done</b>'))
        meta = [f"<b>{html.escape(it['model'])}</b> | <b>{it['setting']}</b>", f"id {html.escape(it['id'])}"]
        if it["subset"]:
            meta.append(html.escape(it["subset"]))
        if it["status"] != "processed":
            meta.append(f'<span class="cg_bad">diff: {html.escape(it["status"])}</span>')
        meta.append(state)
        nxt = [self.view[i] for i in range(max(self.pos - 1, 0), min(self.pos + 1 + PREFETCH, len(self.view))) if i != self.pos]
        pre = "|".join(url(self.items[p][k]) for p in nxt for k in ("in", "out", "hm") if self.items[p][k])

        def panel(cap, rel, cls):
            """Khung nhìn (`cg_vp`, tỉ lệ ảnh, cao tối đa theo màn hình); ảnh + SVG ở cỡ gốc W × H px, JS phóng / dịch bằng CSS
            transform — cả 3 khung chung một trạng thái zoom."""
            if not rel:
                return f'<div class="cg_panel {cls}"><div class="cg_cap">{cap}: none</div></div>'
            return (f'<div class="cg_panel {cls}"><div class="cg_cap">{cap}</div><div class="cg_vp" style="aspect-ratio: {W} / {H}">'
                    f'<div class="cg_wrap" style="width: {W}px; height: {H}px"><img src="{url(rel)}" draggable="false" alt="">'
                    f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg"></svg></div></div></div>')
        verb = "ADD" if it["task"] == "add" else "REMOVE"
        return (f'<div class="cg_root" data-n="{nonce}" data-key="{html.escape(it["key"])}" data-wh="{W},{H}" '
                f'data-img="{html.escape(it["ds"] + "|" + it["id"])}" '
                f'data-box="{",".join(str(v) for v in box) if box else ""}" data-src="{src}" '
                f'data-cands="{html.escape(json.dumps(it["cands"]))}" data-pre="{html.escape(pre)}">'
                f'<div><span class="cg_task cg_{it["task"]}">{verb} one {html.escape(it["cls"] or "?")}</span> '
                f'<span class="cg_meta">{" | ".join(meta)}</span></div>'
                f'<div class="cg_meta" id="cg_boxinfo"></div><div class="cg_row">'
                + panel("INPUT", it["in"], "cg_in") + panel("OUTPUT", it["out"], "cg_out") + panel("DIFF HEATMAP", it["hm"], "cg_hm")
                + "</div></div>")


class Hub:
    """Mọi phiên (tập × user) dựng sẵn từ manifest lúc mở app; trình duyệt gửi kèm Data + User ở mỗi thao tác ⇒ hai người dùng cùng
    lúc không lẫn trạng thái."""

    def __init__(self, manifest, out_dir, users=USERS):
        self.cache = manifest["cache"]
        self.sessions = {}
        for ds in DATASETS.values():
            items = [it for it in manifest["items"] if it["ds"] == ds]
            for k, u in enumerate(users):
                mine = split_items(items, k, len(users))
                if mine:
                    self.sessions[(LABELS[ds], u)] = Annotator(mine, os.path.join(out_dir, f"annot_{ds}_{u}.jsonl"),
                                                               f"{LABELS[ds]} {u}", u)
        self.users = list(users)
        self.models = sorted({it["model"] for it in manifest["items"]})
        self.datasets = [LABELS[ds] for ds in DATASETS.values() if any(k[0] == LABELS[ds] for k in self.sessions)]
        self.renders = itertools.count(1)

    def url(self, rel):
        return FILE_ROUTE + quote(os.path.join(os.path.abspath(self.cache), rel))

    def progress(self, data):
        """Bảng markdown: mỗi run (model × setting × task) của tập `data` — số màn đã chấm của từng user / tổng."""
        cnt = collections.defaultdict(lambda: collections.Counter())
        for (d, u), a in self.sessions.items():
            if d != data:
                continue
            for it in a.items:
                c = cnt[(it["model"], it["setting"], it["task"])]
                c[u + "_n"] += 1
                c[u] += a.done(it)
        if not cnt:
            return ""
        rows = ["| run | " + " | ".join(self.users) + " | all |", "|---|" + "---:|" * (len(self.users) + 1)]
        for (m, st, t), c in sorted(cnt.items(), key=lambda kv: (kv[0][0], SETTINGS.index(kv[0][1]), TASKS.index(kv[0][2]))):
            done, n = sum(c[u] for u in self.users), sum(c[u + "_n"] for u in self.users)
            rows.append(f"| {m} / {st} / {t} | " + " | ".join(f"{c[u]}/{c[u + '_n']}" for u in self.users)
                        + f" | {'**done**' if done == n else f'{100 * done / n:.0f}%'} |")
        return f"**Progress per run: {data}** (both users; done = verdict + your box)\n\n" + "\n".join(rows)

    def act(self, data, user, payload, action, filt=None):
        """-> (header markdown, html). `payload` (JSON từ trình duyệt): key màn đang hiện + box đang chọn; `filt`: bộ lọc
        {model, setting, task} trình duyệt đang chọn (None = không lọc; giao diện luôn gửi đủ cả ba = một run)."""
        a = self.sessions.get((data, user))
        if a is None:
            return CHOOSE, ""
        try:
            p = json.loads(payload or "{}")
        except ValueError:
            p = {}
        p = p if isinstance(p, dict) else {}
        warn = ""
        with a.lock:
            if not a.set_filter(filt or {}):
                return f"{a.name} | {a.filter_name()}: no screens, change Model / Setting / Task.", ""
            if action != "show" and p.get("key") in a.index:
                v = a.vindex.get(a.index[p["key"]])  # thao tác áp lên ĐÚNG màn đang hiện ở trình duyệt
                if v is not None:
                    a.go(v)
                elif action in ("ok", "fail"):       # màn đang hiện không thuộc bộ lọc (vừa đổi lọc): không lưu nhầm màn
                    return (a.header("Filter changed, not saved: check this screen and press again."),
                            a.html(self.url, next(self.renders)))
            if action == "ok":
                warn = a.save(True, p.get("box"), p.get("src"))
            elif action == "fail":
                warn = a.save(False, p.get("box"), p.get("src"))
            elif action == "prev":
                a.go(a.pos - 1)
            elif action == "next":
                a.go(a.pos + 1)
            elif action == "first":
                a.first_unsaved()
            return a.header(warn), a.html(self.url, next(self.renders))


KEYS_JS = """<script>
window.__cg = window.__cg || {box: null, src: '', key: null, img_key: null, busy: 0, img: new Map(), space: false,
                              view: {z: 1, cx: null, cy: null}};
function cgRoot() { return document.querySelector('.cg_root'); }
function cgWH() { const r = cgRoot(); return r ? r.dataset.wh.split(',').map(Number) : [1, 1]; }
function cgCands() { const r = cgRoot(); try { return JSON.parse(r.dataset.cands || '[]'); } catch (e) { return []; } }
function cgVps() { const r = cgRoot(); return r ? [...r.querySelectorAll('.cg_vp')].filter((v) => v.clientWidth > 0) : []; }
function cgScale(vp) {
  const [W, H] = cgWH();
  return Math.min(vp.clientWidth / W, vp.clientHeight / H) * window.__cg.view.z;
}
// một trạng thái zoom (z = bội so với vừa khung, cx / cy = tâm nhìn theo pixel ảnh) áp cho cả 3 khung
function cgApply() {
  const vps = cgVps();
  if (!vps.length) { cgDraw(); return; }
  const mh = `${Math.max(300, Math.floor(window.innerHeight - vps[0].getBoundingClientRect().top - window.scrollY - 12))}px`;
  vps.forEach((p) => { if (p.style.maxHeight !== mh) p.style.maxHeight = mh; });     // khung ảnh vừa phần màn hình còn lại
  const [W, H] = cgWH(), v = window.__cg.view, s = cgScale(vps[0]), vw = vps[0].clientWidth, vh = vps[0].clientHeight;
  if (v.cx === null) { v.cx = W / 2; v.cy = H / 2; }
  const cl = (c, n, len) => (n * s <= len ? n / 2 : Math.min(Math.max(c, len / (2 * s)), n - len / (2 * s)));
  v.cx = cl(v.cx, W, vw); v.cy = cl(v.cy, H, vh);
  vps.forEach((p) => {
    const sp = cgScale(p);
    p.querySelector('.cg_wrap').style.transform =
      `translate(${p.clientWidth / 2 - v.cx * sp}px, ${p.clientHeight / 2 - v.cy * sp}px) scale(${sp})`;
  });
  cgDraw();
}
function cgZoom(vp, f, mx, my) {        // mx, my: điểm giữ cố định, tính từ tâm khung (px màn hình)
  const v = window.__cg.view, s = cgScale(vp), ix = v.cx + mx / s, iy = v.cy + my / s;
  v.z = Math.min(Math.max(v.z * f, 1), 40);
  const s2 = cgScale(vp);
  v.cx = ix - mx / s2; v.cy = iy - my / s2;
  cgApply();
}
function cgDraw() {
  const r = cgRoot(); if (!r) return;
  const vps = cgVps(), sc = vps.length ? cgScale(vps[0]) : 1;
  // nét / chữ tính theo pixel MÀN HÌNH (SVG bị CSS transform phóng to, vector-effect không tính transform đó)
  const C = cgCands(), fs = 20 / sc, b = window.__cg.box, px = (v) => (v / sc).toFixed(3);
  let s = '';
  C.forEach((c, i) => {
    s += `<rect class="${i ? 'cg_cand' : 'cg_top'}" x="${c[0]}" y="${c[1]}" width="${c[2] - c[0]}" height="${c[3] - c[1]}" `
      + (i ? `stroke-width="${px(1.5)}" stroke-dasharray="${px(5)} ${px(3)}"/>` : `stroke-width="${px(2.5)}"/>`);
    s += `<text class="cg_lab" x="${c[0]}" y="${c[1] - fs * 0.15}" font-size="${fs}" stroke-width="${fs * 0.08}">${i + 1}</text>`;
  });
  if (b) s += `<rect class="cg_cur" x="${b[0]}" y="${b[1]}" width="${b[2] - b[0]}" height="${b[3] - b[1]}" stroke-width="${px(3)}"/>`;
  r.querySelectorAll('svg').forEach((svg) => { svg.innerHTML = s; });
  const info = document.getElementById('cg_boxinfo');
  if (info) info.textContent = (b ? `your box (${window.__cg.src === 'picked' ? 'suggestion 1' : 'drawn'}): `
      + b.map((v) => Math.round(v)).join(', ') : 'no box yet: press s for suggestion 1 or drag to draw')
      + ` | zoom ${window.__cg.view.z.toFixed(1)}x (wheel, + / -, 0 = fit; right drag or space + drag = move)`;
}
function cgReset() {
  const r = cgRoot(); if (!r) return;
  window.__cg.box = r.dataset.box ? r.dataset.box.split(',').map(Number) : null;
  window.__cg.src = r.dataset.src || '';
}
window.cgPayload = function () { return JSON.stringify({key: window.__cg.key, box: window.__cg.box, src: window.__cg.src}); };
function cgPt(e, svg) {
  const rc = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal, [W, H] = cgWH();
  const x = (e.clientX - rc.left) * vb.width / rc.width, y = (e.clientY - rc.top) * vb.height / rc.height;
  return [Math.min(Math.max(x, 0), W), Math.min(Math.max(y, 0), H)];
}
let cgDrag = null, cgPan = null;
document.addEventListener('mousedown', (e) => {
  const vp = e.target.closest && e.target.closest('.cg_vp');
  if (!vp) return;
  e.preventDefault();
  if (e.button === 1 || e.button === 2 || (e.button === 0 && window.__cg.space)) {
    cgPan = {x: e.clientX, y: e.clientY, s: cgScale(vp)};
    document.body.classList.add('cg_panning');
    return;
  }
  if (e.button !== 0) return;
  const svg = vp.querySelector('svg'), [x, y] = cgPt(e, svg);
  cgDrag = {svg, x0: x, y0: y, cx: e.clientX, cy: e.clientY, moved: false, prev: [window.__cg.box, window.__cg.src]};
});
document.addEventListener('contextmenu', (e) => { if (e.target.closest && e.target.closest('.cg_vp')) e.preventDefault(); });
document.addEventListener('wheel', (e) => {
  const vp = e.target.closest && e.target.closest('.cg_vp');
  if (!vp) return;
  e.preventDefault();
  const rc = vp.getBoundingClientRect(), d = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
  cgZoom(vp, Math.exp(-Math.max(Math.min(d, 100), -100) * 0.002),
         e.clientX - rc.left - vp.clientWidth / 2, e.clientY - rc.top - vp.clientHeight / 2);
}, {passive: false});
document.addEventListener('mousemove', (e) => {
  if (cgPan) {
    const v = window.__cg.view;
    v.cx -= (e.clientX - cgPan.x) / cgPan.s; v.cy -= (e.clientY - cgPan.y) / cgPan.s;
    cgPan.x = e.clientX; cgPan.y = e.clientY;
    cgApply();
    return;
  }
  if (!cgDrag) return;
  if (!cgDrag.moved && Math.hypot(e.clientX - cgDrag.cx, e.clientY - cgDrag.cy) < 4) return;
  cgDrag.moved = true;
  const [x, y] = cgPt(e, cgDrag.svg);
  window.__cg.box = [Math.min(cgDrag.x0, x), Math.min(cgDrag.y0, y), Math.max(cgDrag.x0, x), Math.max(cgDrag.y0, y)];
  window.__cg.src = 'drawn';
  cgDraw();
});
document.addEventListener('mouseup', () => {
  if (cgPan) { cgPan = null; document.body.classList.remove('cg_panning'); return; }
  if (!cgDrag) return;
  const d = cgDrag; cgDrag = null;
  if (!d.moved) return;
  const b = window.__cg.box;
  if (!b || b[2] - b[0] < 2 || b[3] - b[1] < 2) { [window.__cg.box, window.__cg.src] = d.prev; }
  else window.__cg.box = b.map((v) => Math.round(v));
  cgDraw();
});
function cgPrefetch(r) {
  (r.dataset.pre || '').split('|').filter(Boolean).forEach((u) => {
    const m = window.__cg.img;
    if (m.has(u)) { const im = m.get(u); m.delete(u); m.set(u, im); return; }
    const im = new Image(); im.src = u; if (im.decode) im.decode().catch(() => {});
    m.set(u, im);
    while (m.size > 80) m.delete(m.keys().next().value);
  });
}
function cgTyping(t) {
  return t && (t.tagName === 'TEXTAREA' || (t.tagName === 'INPUT' && t.type !== 'checkbox' && t.type !== 'radio'));
}
document.addEventListener('keyup', (e) => {
  if (e.key === ' ') { window.__cg.space = false; document.body.classList.remove('cg_space'); }
});
document.addEventListener('keydown', (e) => {
  if (cgTyping(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === ' ') {            // giữ Space + kéo = dịch ảnh (không để Space bấm nút đang focus)
    e.preventDefault();
    if (document.activeElement && document.activeElement.tagName === 'BUTTON') document.activeElement.blur();
    window.__cg.space = true; document.body.classList.add('cg_space');
    return;
  }
  if (e.repeat) return;
  const id = ({Enter: 'cg_ok', f: 'cg_fail', F: 'cg_fail', ArrowLeft: 'cg_prev', ArrowRight: 'cg_next'})[e.key];
  if (id) {
    e.preventDefault();
    if (window.__cg.busy && Date.now() - window.__cg.busy < 2000) return;      // chờ màn mới (giữ phím / bấm dồn không nhảy màn)
    const el = document.getElementById(id);
    if (el) { window.__cg.busy = Date.now(); el.click(); }
    return;
  }
  const vp = cgVps()[0];
  if (e.key === 's' || e.key === 'S') { const C = cgCands(); if (C.length) { window.__cg.box = C[0].slice(); window.__cg.src = 'picked'; cgDraw(); } }
  else if (e.key === 'r') { cgReset(); cgDraw(); }
  else if (e.key === 'x') { window.__cg.box = null; window.__cg.src = ''; cgDraw(); }
  else if (e.key === 'h') { document.body.classList.toggle('cg_nohm'); cgApply(); }
  else if ((e.key === '+' || e.key === '=') && vp) cgZoom(vp, 1.5, 0, 0);
  else if (e.key === '-' && vp) cgZoom(vp, 1 / 1.5, 0, 0);
  else if (e.key === '0') { window.__cg.view = {z: 1, cx: null, cy: null}; cgApply(); }
});
window.addEventListener('resize', () => cgApply());
new MutationObserver(() => {
  const r = cgRoot();
  if (!r || r.dataset.seen) return;
  r.dataset.seen = '1';
  window.__cg.busy = 0;
  if (r.dataset.key !== window.__cg.key) {
    window.__cg.key = r.dataset.key;
    cgReset();
    if (r.dataset.img !== window.__cg.img_key) {      // ảnh khác: về vừa khung; cùng ảnh (12 run liền nhau): giữ zoom
      window.__cg.img_key = r.dataset.img;
      window.__cg.view = {z: 1, cx: null, cy: null};
    }
  }
  cgApply();
  requestAnimationFrame(() => cgApply());
  cgPrefetch(r);
}).observe(document.documentElement, {childList: true, subtree: true});
</script>"""

CSS = (".gradio-container {max-width: 100% !important;} .cg_hidden {display: none !important;} "
       ".cg_task {font-size: 1.2em; font-weight: 700; padding: 3px 10px; border-radius: 6px; color: #fff; display: inline-block;} "
       ".cg_add {background: #1b7f3b;} .cg_remove {background: #b3261e;} .cg_ok {color: #1b7f3b;} .cg_bad {color: #b3261e;} "
       ".cg_meta {margin: 4px 0 4px 6px;} #cg_boxinfo {margin: 4px 0; font-family: monospace;} "
       ".cg_row {display: flex; gap: 8px; align-items: flex-start;} .cg_panel {flex: 1 1 0; min-width: 0;} "
       ".cg_cap {font-size: 0.8em; opacity: 0.7; margin-bottom: 2px;} "
       ".cg_vp {position: relative; width: 100%; max-height: 80vh; overflow: hidden; "
       "background: #000; cursor: crosshair; user-select: none; overscroll-behavior: contain;} "
       "body.cg_space .cg_vp {cursor: grab;} body.cg_panning .cg_vp {cursor: grabbing;} "
       ".cg_wrap {position: absolute; left: 0; top: 0; transform-origin: 0 0; line-height: 0;} "
       ".cg_wrap img {display: block; width: 100%; height: 100%; -webkit-user-drag: none;} "
       ".cg_wrap svg {position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; overflow: visible;} "
       "body.cg_nohm .cg_hm {display: none;} "
       ".cg_cand {fill: none; stroke: #ff3b30;} "
       ".cg_top {fill: none; stroke: #ff3b30;} "
       ".cg_cur {fill: rgba(255, 230, 0, 0.10); stroke: #ffe600;} "
       ".cg_lab {fill: #ff3b30; font-weight: 700; font-family: sans-serif; paint-order: stroke; stroke: #000;}")


def build_ui(hub, gr):
    """Thanh trên: 5 menu chọn (Data, User, Model, Setting, Task = một run) + 5 nút; dưới: 3 ảnh chia đều cả trang; cuối: bảng
    tiến độ mọi tập, hiện ngay từ đầu."""
    import inspect
    kw = {"title": "CE-Gen review", "css": CSS, "head": KEYS_JS, "fill_width": True}
    bparams = inspect.signature(gr.Blocks.__init__).parameters
    launch_kw = {k: v for k, v in kw.items() if k not in bparams and k in ("css", "head")}     # gradio >= 6: css / head ở launch
    hidden = {"elem_classes": "cg_hidden"}
    dd = {"value": None, "filterable": False, "min_width": 140}
    with gr.Blocks(**{k: v for k, v in kw.items() if k in bparams}) as demo:
        with gr.Row(equal_height=True):
            data = gr.Dropdown(hub.datasets, label="Data", **dd)
            user = gr.Dropdown(hub.users, label="User", **dd)
            model = gr.Dropdown(hub.models, label="Model", **dd)
            setting = gr.Dropdown(list(SETTINGS), label="Setting", **dd)
            task = gr.Dropdown(list(TASKS), label="Task", **dd)
        with gr.Row():
            ok = gr.Button("Success (Enter)", variant="primary", elem_id="cg_ok", min_width=100)
            fail = gr.Button("Fail (f)", variant="stop", elem_id="cg_fail", min_width=100)
            prev = gr.Button("<- Prev", elem_id="cg_prev", min_width=100)
            nxt = gr.Button("Next ->", elem_id="cg_next", min_width=100)
            first = gr.Button("Go to first not done", min_width=100)
        head = gr.Markdown(CHOOSE)
        img = gr.HTML()
        payload = gr.Textbox(**hidden)
        with gr.Row():
            progs = [gr.Markdown(hub.progress(d)) for d in hub.datasets]
        outs = [head, img, *progs]
        sel = [data, user, model, setting, task]
        q = {"queue": False, "show_progress": "hidden"}
        js = "(d, u, m, s, t, p) => [d, u, m, s, t, window.cgPayload ? window.cgPayload() : '']"

        def run(d, u, m, s, t, p, action):
            if not (d and u and m and s and t):
                return CHOOSE, "", *[hub.progress(x) for x in hub.datasets]
            head_md, page = hub.act(d, u, p, action, {"model": m, "setting": s, "task": t})
            return head_md, page, *[hub.progress(x) for x in hub.datasets]

        def mk(action):
            return lambda d, u, m, s, t, p: run(d, u, m, s, t, p, action)
        for b, action in ((ok, "ok"), (fail, "fail"), (prev, "prev"), (nxt, "next"), (first, "first")):
            b.click(mk(action), sel + [payload], outs, js=js, **q)
        for r in sel:
            r.input(lambda d, u, m, s, t: run(d, u, m, s, t, "", "show"), sel, outs, **q)
        demo.load(lambda: (CHOOSE, "", *[hub.progress(x) for x in hub.datasets]), None, outs, **q)
    return demo, launch_kw


def serve(out_dir, host="0.0.0.0", port=7860, share=False):
    import gradio as gr
    path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(path):
        sys.exit(f"thiếu {path} — chạy `cegen.py prepare` trước")
    t0 = time.time()
    with open(path) as f:
        man = json.load(f)
    hub = Hub(man, out_dir)
    for (d, u), a in hub.sessions.items():
        print(f"[{d} · {u}] {len(a.items)} màn, xong {a.n_saved()} (đủ kết luận + box tự vẽ) -> {a.store.path}", flush=True)
    print(f"[serve] manifest {len(man['items'])} màn ({man['time']}), dựng phiên {time.time() - t0:.1f}s", flush=True)
    demo, launch_kw = build_ui(hub, gr)
    demo.launch(server_name=host, server_port=port, share=share, allowed_paths=[man["cache"]], **launch_kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="dựng cache JPEG + manifest.json")
    p.add_argument("--src", default=SRC, help="thư mục chứa FSC_final/ và processed_dataset/")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--turn-index", default=os.path.join(OD, "data", "turn_index.json"))
    p.add_argument("--cocount-root", default=os.path.join(OD, "data", "cocount"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--top-k", type=int, default=5, help="số box gợi ý mỗi màn (theo điểm của records.jsonl)")
    p.add_argument("--force", action="store_true", help="làm lại cả ảnh cache đã có")
    p.add_argument("--allow-missing-v2", action="store_true", help="vẫn dựng khi thiếu results_diff_boxes_v2 (màn đó không có gợi ý)")
    p.add_argument("--expect-runs", type=int, default=12, help="số run mỗi tập (0 = không kiểm)")
    s = sub.add_parser("serve", help="mở app Gradio")
    s.add_argument("--out-dir", default=OUT_DIR)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--share", action="store_true", help="tạo link công khai *.gradio.live (hết hạn sau 72 giờ)")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.src, a.out_dir, a.turn_index, a.cocount_root, a.workers, a.top_k, a.force, a.allow_missing_v2, a.expect_runs,
                log=lambda m: print(m, flush=True))
    else:
        serve(a.out_dir, a.host, a.port, a.share)


if __name__ == "__main__":
    main()

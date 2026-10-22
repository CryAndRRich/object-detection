#!/usr/bin/env python3
"""Web rà box tay (người dùng, 2026-10-20): mỗi màn một ảnh + box (`sources.py`). Chạy LOCAL, một người rà, chỉ đọc ảnh + chú thích,
không model. Cần `pip install gradio`. Màn chỉ hiện ảnh + box (không chữ trên box, không gợi ý).

  Bấm vào box        loại box: box BIẾN MẤT khỏi ảnh. Box chồng nhau / trùng nhau: bấm trúng box nhỏ nhất (vẽ sau cùng), bấm tiếp
                     để loại box bên dưới
  Ctrl+Z / Backspace hoàn tác thao tác gần nhất (box hiện lại) — kể cả lần lưu (quay về màn đó, bỏ dòng đã lưu, giữ nguyên các box đã
                     loại để sửa rồi lưu lại); lịch sử hoàn tác chỉ sống trong phiên (tắt app là hết, file kết quả vẫn còn)
  Enter              lưu màn này rồi sang màn sau (lưu lại màn đã lưu = ghi đè)
  ← / →              màn trước / sau, KHÔNG lưu (box đã loại giữ làm nháp trong phiên)
  v / z              đổi ảnh (CE-130: original / turn t) / đổi mức phóng to (ảnh dày: 200–400 %, cuộn để xem)

Python cần torch + gradio — ở local dùng venv `ce_localization/.venv-cpu` (python3 của hệ thống không có):
  cd object-detection/ce_localization && .venv-cpu/bin/python box_review/app.py cocount

Lọc trùng tự động (CE-CoCount, người dùng 2026-10-21; `--no-dedup` để tắt): trước khi hiện, mỗi cụm box gần trùng (IoU > 0,9) chỉ
giữ một box — 2 box: box nhỏ hơn, >= 3: box diện tích trung vị (`sources.dedup_boxes`); box bị lọc không hiện, ghi riêng ở khoá `auto`.
File đã rà trước khi có bước này: lọc trên các box CÒN LẠI sau khi người rà loại (`bad` giữ nguyên), file kết quả được ghi thêm `auto`
(bản cũ chép ra `<file>.bak_<ngày giờ>` trước). Box bị loại cuối cùng = `bad` ∪ `auto`.

Hai cách dùng (hai app, chung code):
  1. Loại box vật lỗi CE-CoCount (sai loại vật, nhiều vật chung một box…) — mỗi màn một FILE (một lớp), box GỐC chưa lọc:
       python box_review/app.py cocount
  2. Soi box GT CE-130 — mỗi màn một NHÁNH: box vật (`all_bboxes`) + box lỗ (`inpainted_bboxes`, H1..H4), loại box nào lỗi:
       python box_review/app.py ce130 --split train          # val / test; --objects-only = chỉ rà box vật (như app 1)

Kết quả: `output/add/box_review/<nguồn>.jsonl` (cocount.jsonl, ce130_<split>.jsonl), MỘT dòng mỗi màn đã lưu, ghi đĩa ngay mỗi lần
lưu ⇒ tắt mở lại tiếp từ màn chưa lưu đầu tiên:
  {"id", "bad": {nhóm: [chỉ số box người rà loại]}, "auto": {nhóm: [chỉ số box lọc trùng tự động]} (chỉ khi lọc trùng),
   "n": {nhóm: số box}, "share": {nhóm: khoá}, "time", "sec"}
  CE-CoCount: id = nhóm = tên file (`..._positive` / `..._negative`), chỉ số = vị trí trong `exam_bbox` của `Anno_with_exam_bbox`;
  CE-130: `objects` (vị trí trong `all_bboxes`), `holes` (vị trí trong `inpainted_bboxes`). Box vật CE-130 dùng chung giữa các nhánh
  của cùng ảnh: nhánh chưa lưu được điền sẵn box đã loại của nhánh cùng ảnh lưu gần nhất.
"""

import argparse
import copy
import html
import inspect
import json
import os
import sys
import time
import warnings
from datetime import datetime
from urllib.parse import quote

from PIL import Image

OD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # object-detection/
sys.path.insert(0, OD)

from ce_localization.box_review.sources import SOURCES, dedup_boxes  # noqa: E402

__all__ = ["ReviewStore", "Session", "box_svg", "default_out", "build_ui"]

ZOOMS = ("fit", "100%", "150%", "200%", "300%", "400%")
PREFETCH = 3
FILE_ROUTE = "/gradio_api/file="
OUT_DIR = os.path.join(os.path.dirname(OD), "output", "add", "box_review")

KEYS_JS = """<script>
window.__brScroll = window.__brScroll || {};
document.addEventListener('keydown', (e) => {
  const t = e.target;
  if (t && (t.tagName === 'TEXTAREA' || (t.tagName === 'INPUT' && t.type !== 'checkbox' && t.type !== 'radio'))) return;
  let id = null;
  if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) id = 'br_undo';
  else if (e.ctrlKey || e.metaKey || e.altKey) return;
  else id = ({ArrowLeft: 'br_prev', ArrowRight: 'br_next', Enter: 'br_save', Backspace: 'br_undo', v: 'br_vnext', z: 'br_znext'})[e.key];
  const el = id && document.getElementById(id);
  if (el) { e.preventDefault(); el.click(); }
});
document.addEventListener('click', (e) => {            // bấm box: ghi mã box vào ô ẩn br_click rồi bấm nút ẩn br_apply
  const r = e.target.closest && e.target.closest('[data-b]');
  if (!r) return;
  const box = document.querySelector('#br_click textarea, #br_click input');
  const btn = document.getElementById('br_apply');
  if (!box || !btn) return;
  r.style.display = 'none';                            // ẩn ngay, không chờ server vẽ lại
  box.value = r.dataset.b + '|' + Date.now();
  box.dispatchEvent(new Event('input', {bubbles: true}));
  btn.click();
});
document.addEventListener('scroll', (e) => {           // nhớ vị trí cuộn theo (màn, zoom): vẽ lại sau mỗi lần bấm không nhảy về đầu
  const el = e.target;
  if (el && el.classList && el.classList.contains('br_scroll')) window.__brScroll[el.dataset.key] = [el.scrollLeft, el.scrollTop];
}, true);
new MutationObserver(() => {
  document.querySelectorAll('.br_scroll:not([data-restored])').forEach((el) => {
    el.dataset.restored = '1';
    const s = window.__brScroll[el.dataset.key];
    if (s) { el.scrollLeft = s[0]; el.scrollTop = s[1]; }
  });
}).observe(document.documentElement, {childList: true, subtree: true});
</script>"""

CSS = (".br_hidden {display: none !important;} #br_head p {margin: 0.15em 0;} "
       ".br_scroll {overflow: auto; max-height: 90vh;} "
       ".br_wrap {position: relative; line-height: 0;} .br_wrap img {display: block; width: 100%; height: auto;} "
       ".br_wrap svg {position: absolute; left: 0; top: 0; width: 100%; height: 100%;} "
       ".br_wrap rect[data-b] {cursor: pointer;}")


def default_out(source, split=None, objects_only=False):
    name = source + (f"_{split}" if source == "ce130" else "") + ("_objects" if objects_only else "")
    return os.path.join(OUT_DIR, name + ".jsonl")


class ReviewStore:
    """jsonl MỘT dòng mỗi `id`. Màn mới: ghi nối; ghi đè / xoá: ghi lại cả file ra `.tmp` rồi `os.replace` (tắt ngang vẫn còn bản cũ
    hoặc bản mới nguyên vẹn)."""

    def __init__(self, path):
        self.path, self.latest = path, {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self.latest[rec["id"]] = rec

    def get(self, sid):
        return self.latest.get(sid)

    def _rewrite(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            for rec in self.latest.values():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def put(self, rec):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        if rec["id"] in self.latest:
            self.latest[rec["id"]] = rec
            self._rewrite()
            return
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.latest[rec["id"]] = rec

    def delete(self, sid):
        if self.latest.pop(sid, None) is not None:
            self._rewrite()


def box_svg(sc, wh, marks, view=0, show=None):
    """SVG phủ ảnh (viewBox = pixel ảnh gốc). Box CHƯA loại: `<rect data-b="gi:i">` (bấm được), box to vẽ trước ⇒ chỗ chồng nhau bấm
    trúng box nhỏ nhất; box đã loại (`marks`) KHÔNG vẽ. Vật đã bị xoá ở view hiện tại (`gone`): mờ. `tags` (CE-130: H1..H4) là chữ
    duy nhất trên ảnh. `show`: tập "g<i>" các nhóm hiện (None = mọi nhóm)."""
    W, H = wh
    fs = max(W, H) / 45
    out = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">']
    items = [(gi, i, b) for gi, g in enumerate(sc["groups"]) if show is None or f"g{gi}" in show
             for i, b in enumerate(g["boxes"]) if i not in marks.get(g["key"], ())]
    items.sort(key=lambda x: -max(x[2][2] - x[2][0], 0) * max(x[2][3] - x[2][1], 0))
    labels = []
    for gi, i, (x1, y1, x2, y2) in items:
        g = sc["groups"][gi]
        faint = ' stroke-opacity="0.35"' if 0 < g["gone"].get(i, 0) <= view else ""
        out.append(f'<rect data-b="{gi}:{i}" x="{x1:.1f}" y="{y1:.1f}" width="{max(x2 - x1, 0):.1f}" height="{max(y2 - y1, 0):.1f}" '
                   f'fill="transparent" stroke="{g["color"]}" stroke-width="1.5"{faint} vector-effect="non-scaling-stroke"/>')
        if i in g["tags"]:
            ty = max(y1 - fs * 1.25, 0)
            labels.append(f'<rect x="{x1:.1f}" y="{ty:.1f}" width="{fs * 0.62 * len(g["tags"][i]) + fs * 0.4:.1f}" '
                          f'height="{fs * 1.25:.1f}" fill="{g["color"]}"/><text x="{x1 + fs * 0.2:.1f}" y="{ty + fs:.1f}" '
                          f'font-size="{fs:.1f}" fill="white" font-family="sans-serif">{html.escape(g["tags"][i])}</text>')
    out.append(f'<g pointer-events="none">{"".join(labels)}</g>')
    return "".join(out) + "</svg>"


class Session:
    """Trạng thái rà của một người: vị trí `pos`, view, zoom, box đã loại `marks` {nhóm: set chỉ số} của màn hiện tại, nháp các màn
    đã sửa mà chưa lưu (`drafts`), ngăn hoàn tác (`undo_stack`: ("mark", id, nhóm, i) | ("save", id, bản lưu trước đó | None))."""

    def __init__(self, screens, out_path, name="", dedup=False):
        if not screens:
            raise ValueError("không có màn nào")
        self.screens, self.name, self.dedup = screens, name, dedup
        self.index = {sc["id"]: p for p, sc in enumerate(screens)}
        self.store = ReviewStore(out_path)
        self.drafts, self.undo_stack, self.auto = {}, [], {}
        if dedup:
            self._dedup_saved()
        self.zoom, self.msg = "fit", ""
        self.show = {f"g{i}" for i in range(max(len(sc["groups"]) for sc in screens))}
        self._rebuild_share()
        self.first_unsaved()

    # ---- lọc trùng tự động
    def _auto_for(self, sc, removed):
        """Box lọc trùng của màn (tính MỘT lần mỗi màn, trên các box chưa bị loại `removed`)."""
        if not self.dedup:
            return {g["key"]: set() for g in sc["groups"]}
        if sc["id"] not in self.auto:
            rec = self.store.get(sc["id"])
            if rec and "auto" in rec:
                self.auto[sc["id"]] = {g["key"]: set(rec["auto"].get(g["key"], [])) for g in sc["groups"]}
            else:
                self.auto[sc["id"]] = {g["key"]: dedup_boxes(g["boxes"], set(range(len(g["boxes"]))) - removed.get(g["key"], set()))
                                       for g in sc["groups"]}
        return self.auto[sc["id"]]

    def _dedup_saved(self):
        """Màn đã lưu chưa có `auto` (rà trước khi có bước lọc trùng): lọc trên box còn lại sau `bad`, ghi `auto` vào file (chép bản cũ ra
        `.bak_<ngày giờ>` trước)."""
        todo = [r for r in self.store.latest.values() if "auto" not in r and r["id"] in self.index]
        if not todo:
            return
        bak = f"{self.store.path}.bak_{datetime.now():%m%d_%H%M%S}"
        with open(self.store.path) as f, open(bak, "w") as g:
            g.write(f.read())
        for rec in todo:
            sc = self.screens[self.index[rec["id"]]]
            auto = self._auto_for(sc, {k: set(v) for k, v in rec["bad"].items()})
            rec["auto"] = {k: sorted(v) for k, v in auto.items()}
        self.store._rewrite()
        print(f"[lọc trùng] ghi `auto` cho {len(todo)} màn đã lưu; bản cũ -> {bak}", flush=True)

    def hidden(self):
        """Box không vẽ của màn hiện tại = người rà loại ∪ lọc trùng."""
        return {k: self.marks[k] | self.auto_cur.get(k, set()) for k in self.marks}

    # ---- box dùng chung giữa các màn (CE-130: box vật mọi nhánh cùng ảnh)
    def _rebuild_share(self):
        best = {}
        for rec in self.store.latest.values():
            for gkey, sk in (rec.get("share") or {}).items():
                if sk is not None and (sk not in best or rec["time"] >= best[sk][0]):
                    best[sk] = (rec["time"], rec["id"], rec["bad"].get(gkey, []))
        self.share = best

    # ---- điều hướng
    def sc(self):
        return self.screens[self.pos]

    def saved(self, p):
        return self.screens[p]["id"] in self.store.latest

    def go(self, pos):
        """Mở màn `pos`: box đã loại = nháp trong phiên > bản đã lưu > điền sẵn từ màn cùng khoá `share`. View giữ nguyên nếu vẫn màn cũ."""
        old = getattr(self, "pos", None)
        self.pos = min(max(pos, 0), len(self.screens) - 1)
        sc = self.sc()
        rec, self.prefill = self.store.get(sc["id"]), ""
        if sc["id"] in self.drafts:
            self.marks = copy.deepcopy(self.drafts[sc["id"]])
        elif rec:
            self.marks = {g["key"]: set(rec["bad"].get(g["key"], [])) for g in sc["groups"]}
        else:
            self.marks = {g["key"]: set() for g in sc["groups"]}
            for g in sc["groups"]:
                if g["share"] in self.share and self.share[g["share"]][1] != sc["id"]:
                    self.marks[g["key"]] = set(self.share[g["share"]][2])
                    self.prefill = self.share[g["share"]][1]
        self.auto_cur = self._auto_for(sc, self.marks)
        if old != self.pos:                         # cùng màn (hoàn tác / bấm): giữ view + mốc giờ đo `sec`
            self.view, self.t_show = 0, time.time()
        else:
            self.view = min(self.view, len(sc["views"]) - 1)

    def first_unsaved(self):
        p = next((p for p in range(len(self.screens)) if not self.saved(p)), len(self.screens) - 1)
        self.go(p)

    def next(self):
        self.msg = ""
        self.go(self.pos + 1)

    def prev(self):
        self.msg = ""
        self.go(self.pos - 1)

    def next_view(self):
        self.view = (self.view + 1) % len(self.sc()["views"])

    def set_view(self, label):
        labels = [v[0] for v in self.sc()["views"]]
        if label in labels:
            self.view = labels.index(label)

    def next_zoom(self):
        self.zoom = ZOOMS[(ZOOMS.index(self.zoom) + 1) % len(ZOOMS)]

    # ---- sửa
    def mark(self, gi, i):
        """Loại box i của nhóm gi (đã loại / ngoài phạm vi: bỏ qua — box đã loại không còn trên ảnh, chỉ hoàn tác mới trả lại)."""
        sc = self.sc()
        if not (0 <= gi < len(sc["groups"]) and 0 <= i < len(sc["groups"][gi]["boxes"])):
            return
        key = sc["groups"][gi]["key"]
        if i in self.marks[key] or i in self.auto_cur.get(key, ()):
            return
        self.marks[key].add(i)
        self.drafts[sc["id"]] = copy.deepcopy(self.marks)
        self.undo_stack.append(("mark", sc["id"], key, i))
        self.msg = ""

    def click(self, code):
        """Mã từ trình duyệt: "gi:i|thời điểm"."""
        try:
            gi, i = (int(x) for x in str(code).split("|")[0].split(":"))
        except ValueError:
            return
        self.mark(gi, i)

    def save(self):
        """Lưu màn này rồi sang màn sau (màn cuối: đứng yên)."""
        sc = self.sc()
        rec = {"id": sc["id"], "bad": {g["key"]: sorted(self.marks[g["key"]]) for g in sc["groups"]},
               "n": {g["key"]: len(g["boxes"]) for g in sc["groups"]},
               "share": {g["key"]: g["share"] for g in sc["groups"] if g["share"] is not None},
               "time": datetime.now().isoformat(timespec="milliseconds"), "sec": round(time.time() - self.t_show, 1)}
        if self.dedup:
            rec["auto"] = {g["key"]: sorted(self.auto_cur[g["key"]]) for g in sc["groups"]}
        self.undo_stack.append(("save", sc["id"], self.store.get(sc["id"])))
        self.store.put(rec)
        self.drafts.pop(sc["id"], None)
        self._rebuild_share()
        self.msg = f"saved {sc['id']}"
        if self.pos < len(self.screens) - 1:
            self.go(self.pos + 1)

    def undo(self):
        if not self.undo_stack:
            self.msg = "nothing to undo"
            return
        kind, sid, *rest = self.undo_stack.pop()
        if kind == "mark":
            key, i = rest
            self.go(self.index[sid])
            self.marks[key].discard(i)
            self.drafts[sid] = copy.deepcopy(self.marks)
            self.msg = f"undid removal on {sid}"
        else:
            saved = self.store.get(sid)
            if rest[0] is None:
                self.store.delete(sid)
            else:
                self.store.put(rest[0])
            self._rebuild_share()
            self.go(self.index[sid])
            self.marks = {g["key"]: set(saved["bad"].get(g["key"], [])) for g in self.sc()["groups"]}
            self.drafts[sid] = copy.deepcopy(self.marks)
            self.msg = f"undid save of {sid}"

    # ---- hiển thị
    def wh(self, sc=None):
        sc = sc or self.sc()
        if "wh" not in sc:
            with Image.open(sc["views"][0][1]) as im:
                sc["wh"] = list(im.size)
        return sc["wh"]

    def status(self):
        sc = self.sc()
        if sc["id"] in self.drafts:
            return "modified, NOT saved"
        return "saved" if self.saved(self.pos) else "not saved"

    def header(self):
        """Một dòng tiến độ (+ tiêu đề của nguồn nếu có)."""
        n = len(self.screens)
        done = sum(self.saved(p) for p in range(n))
        sc = self.sc()
        return f"{self.pos + 1} / {n} · {self.status()} · saved {done}" + (f"\n\n{sc['title']}" if sc["title"] else "")

    def html(self):
        sc = self.sc()
        url = lambda p: FILE_ROUTE + html.escape(quote(os.path.abspath(p)))  # noqa: E731
        W, H = self.wh()
        width = f"min(100%, calc(88vh * {W / H:.4f}))" if self.zoom == "fit" else self.zoom      # fit: phóng / thu cho vừa khung
        nxt = [self.screens[p]["views"][0][1] for p in range(self.pos + 1, min(self.pos + 1 + PREFETCH, len(self.screens)))]
        pre = "".join(f'<img src="{url(p)}" alt="">' for p in nxt + [v[1] for v in sc["views"]])
        return (f'<div class="br_scroll" data-key="{html.escape(sc["id"])}|{self.zoom}">'
                f'<div class="br_wrap" style="width: {width}">'
                f'<img src="{url(sc["views"][self.view][1])}" style="aspect-ratio: {W} / {H}" alt="">'
                f'{box_svg(sc, (W, H), self.hidden(), self.view, self.show)}</div></div>'
                f'<div style="display:none">{pre}</div>')


def build_ui(sess, gr):
    kw = {"title": "Box review", "css": CSS, "head": KEYS_JS}
    bparams = inspect.signature(gr.Blocks.__init__).parameters
    launch_kw = {k: v for k, v in kw.items() if k not in bparams and k != "title"}       # gradio >= 6: css / head ở launch
    hidden = {"elem_classes": "br_hidden"}
    many_views = any(len(sc["views"]) > 1 for sc in sess.screens)
    many_groups = any(len(sc["groups"]) > 1 for sc in sess.screens)
    with gr.Blocks(**{k: v for k, v in kw.items() if k in bparams}) as demo:
        head = gr.Markdown(elem_id="br_head")
        with gr.Row():
            with gr.Column(scale=5):
                img = gr.HTML()
            with gr.Column(scale=1, min_width=220):
                with gr.Row():
                    prev = gr.Button("← (prev)", elem_id="br_prev", min_width=60)
                    nxt = gr.Button("→ (next)", elem_id="br_next", min_width=60)
                save = gr.Button("Save (Enter)", variant="primary", elem_id="br_save")
                undo = gr.Button("Undo (Ctrl+Z)", elem_id="br_undo")
                first = gr.Button("Go to first unsaved", size="sm")
                view = gr.Radio(label="Image (v)", choices=[], visible=many_views)
                zoom = gr.Radio(list(ZOOMS), value=sess.zoom, label="Zoom (z)")
                show = gr.CheckboxGroup(label="Show", choices=[], visible=many_groups)
                click = gr.Textbox(elem_id="br_click", **hidden)
                apply = gr.Button("apply", elem_id="br_apply", **hidden)
                vnext = gr.Button("view", elem_id="br_vnext", **hidden)
                znext = gr.Button("zoom", elem_id="br_znext", **hidden)

        def full():
            sc = sess.sc()
            labels = [v[0] for v in sc["views"]]
            shows = [(g["key"], f"g{i}") for i, g in enumerate(sc["groups"])]
            return [sess.header(), sess.html(), gr.update(choices=labels, value=labels[sess.view]),
                    gr.update(choices=shows, value=[v for _, v in shows if v in sess.show]), gr.update(value=sess.zoom)]

        def act(fn):
            def f(*args):
                fn(*args)
                return full()
            return f

        def set_show(v):
            sess.show = set(v or [])

        def set_zoom(z):
            sess.zoom = z if z in ZOOMS else "fit"

        outs = [head, img, view, show, zoom]
        q = {"queue": False}
        demo.load(full, None, outs, **q)
        apply.click(act(sess.click), [click], outs, **q)
        for b, fn in ((prev, sess.prev), (nxt, sess.next), (save, sess.save), (undo, sess.undo), (first, sess.first_unsaved),
                      (vnext, sess.next_view), (znext, sess.next_zoom)):
            b.click(act(fn), None, outs, **q)
        view.input(act(sess.set_view), [view], outs, **q)
        zoom.input(act(set_zoom), [zoom], outs, **q)
        show.input(act(set_show), [show], outs, **q)
    return demo, launch_kw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default="cocount", choices=list(SOURCES), help="mặc định cocount")
    ap.add_argument("--root", default=None, help="mặc định object-detection/data/cocount | object-detection/data/all_phase2_V2")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"], help="ce130")
    ap.add_argument("--objects-only", action="store_true", help="ce130: chỉ rà box vật (không hiện box lỗ)")
    ap.add_argument("--out", default=None, help="file kết quả .jsonl (mặc định output/add/box_review/<nguồn>.jsonl)")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--no-browser", action="store_true", help="không tự mở trình duyệt")
    ap.add_argument("--no-dedup", action="store_true", help="cocount: tắt lọc trùng tự động (mặc định bật; ce130 luôn tắt)")
    a = ap.parse_args()
    root = os.path.abspath(a.root or os.path.join(OD, "data", "cocount" if a.source == "cocount" else "all_phase2_V2"))
    t0 = time.time()
    screens = (SOURCES["cocount"](root) if a.source == "cocount" else SOURCES["ce130"](root, a.split, holes=not a.objects_only))
    out = a.out or default_out(a.source, a.split, a.objects_only)
    name = "CE-CoCount" if a.source == "cocount" else f"CE-130 {a.split}" + (" (objects only)" if a.objects_only else "")
    sess = Session(screens, out, name, dedup=a.source == "cocount" and not a.no_dedup)
    print(f"[{name}] {len(screens)} màn ({time.time() - t0:.1f}s), đã lưu {len(sess.store.latest)} -> {out}", flush=True)
    # gradio 6.17 + starlette 1.x: cảnh báo deprecated mỗi request (lỗi thư viện, request vẫn thành công) — bỏ đúng cảnh báo này
    warnings.filterwarnings("ignore", message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*")
    import gradio as gr
    demo, launch_kw = build_ui(sess, gr)
    demo.launch(server_name="127.0.0.1", server_port=a.port, inbrowser=not a.no_browser, allowed_paths=[root], **launch_kw)


if __name__ == "__main__":
    main()

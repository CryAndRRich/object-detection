#!/usr/bin/env python3
"""Web rà box tay (người dùng, 2026-10-20): mỗi màn một ảnh + box (`sources.py`). Chạy LOCAL, một người rà, chỉ đọc ảnh + chú thích,
không model. Cần `pip install gradio`. Màn chỉ hiện ảnh + box (không chữ trên box, không gợi ý).

  Bấm vào box        loại box: box BIẾN MẤT khỏi ảnh. Bấm trúng box có VIỀN gần con trỏ nhất (<= 10 px màn hình; gần bằng nhau:
                     box nhỏ hơn; không viền nào gần: box nhỏ nhất chứa con trỏ) — rê chuột: box sắp bấm trúng tô VÀNG (người
                     dùng 2026-10-30: chỗ dày box hay bấm nhầm)
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

Ba cách dùng (chung code):
  1. Loại box vật lỗi CE-CoCount (sai loại vật, nhiều vật chung một box…) — mỗi màn một FILE (một lớp), box GỐC chưa lọc:
       python box_review/app.py cocount
  2. Soi box GT CE-130 — mỗi màn một NHÁNH: box vật (`all_bboxes`) + box lỗ (`inpainted_bboxes`, H1..H4), loại box nào lỗi:
       python box_review/app.py ce130 --split train          # val / test; --objects-only = chỉ rà box vật (như app 1)
  3. CHỌN box đẹp trong K box model sinh (người dùng 2026-10-29 / 30) — MỘT app, chọn ngay trên giao diện (`PredHub`):
       python box_review/app.py pred          # dump mặc định DELTA8 ep 10; khác: --ce130-dump / --cocount-dump <out>_boxes.json
     - Data: CE-130 (mỗi màn một mẫu (nhánh, lượt): ảnh model nhận `samples/`) | CE-CoCount review5 (mỗi màn một file lớp: ẢNH GỐC,
       box vật sau bản rà `--review`, box model khoá `cocount` = box thô, không đổi cỡ).
     - User: user_1 | user_2 (`--n-users`) — chia việc theo ẢNH (`split_screens`; CE-130: mọi nhánh / lượt của một ảnh, CE-CoCount:
       hai lớp của một frame), xen kẽ theo thứ tự dump ⇒ hai người không trùng màn; mỗi người một file `…_user_k.jsonl`.
     Màn: ảnh + box vật (xanh, không bấm) + K box model (đỏ mảnh; lọc trùng IoU > 0,9, `--no-dedup` để tắt). Màn chưa lưu mở ra đã
     CHỌN SẴN 5 box (`sources.preselect`: top 5 C-NLL thấp nhất sau NMS IoU 0,5, trong box còn hiện) — ổn thì Enter luôn. Bấm box =
     chọn / bỏ chọn (viền đậm); Enter lưu (kể cả không chọn box nào = ảnh không có box đẹp). Đổi Data / User giữa chừng: nháp +
     hoàn tác của từng (Data, User) giữ nguyên trong phiên. Mở app: màn của cả hai Data đọc sẵn ở luồng nền.
     -> `output/add/box_review/pred_<run>_<epoch>_<tập>_user_k.jsonl`, mỗi dòng: "good" {"model": [chỉ số trong K box của dump]},
        "positive" [[x1, y1, x2, y2] pixel ẢNH GỐC của box đã chọn], "pre" {"model": [chỉ số chọn sẵn]}, "image_id", "class"
        (+ CE-130: "sample" tên file samples/, "t")

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
import threading
import time
import warnings
from datetime import datetime
from urllib.parse import quote

from PIL import Image

OD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # object-detection/
sys.path.insert(0, OD)

from ce_localization.box_review.sources import SOURCES, dedup_boxes, preselect  # noqa: E402

__all__ = ["ReviewStore", "Session", "PredHub", "box_svg", "default_out", "split_screens", "pred_out", "build_ui"]

ZOOMS = ("fit", "100%", "150%", "200%", "300%", "400%")
PREFETCH = 8          # số màn kế tải trước (ảnh giải mã sẵn ở trình duyệt, `brPrefetch`)
FILE_ROUTE = "/gradio_api/file="
OUT_DIR = os.path.join(os.path.dirname(OD), "output", "add", "box_review")

KEYS_JS = """<script>
window.__brScroll = window.__brScroll || {};
window.__brImg = window.__brImg || new Map();
const BR_HIT = 10;                                     // px màn hình: bấm cách viền box xa hơn -> không chọn box nào
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
// Chọn box theo VIỀN gần con trỏ nhất (người dùng 2026-10-30: chỗ nhiều box hay bấm nhầm): khoảng cách tới viền (px màn hình; trong
// box = tới cạnh gần nhất, ngoài box = tới hình chữ nhật), <= BR_HIT; gần bằng nhau (lệch <= 1 px): box nhỏ hơn. Không viền nào đủ
// gần: box NHỎ NHẤT chứa con trỏ. Box sắp chọn tô vàng khi rê chuột ⇒ thấy trước sẽ bấm trúng box nào.
function brPick(e) {
  const wrap = e.target.closest && e.target.closest('.br_wrap');
  const svg = wrap && wrap.querySelector('svg');
  if (!svg) return null;
  const rc = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
  const sx = rc.width / vb.width, sy = rc.height / vb.height;
  const px = (e.clientX - rc.left) / sx, py = (e.clientY - rc.top) / sy;
  let best = null, bd = Infinity, ba = Infinity, inner = null, ia = Infinity;
  svg.querySelectorAll('rect[data-b]').forEach((r) => {
    if (r.style.display === 'none') return;
    const x = +r.getAttribute('x'), y = +r.getAttribute('y'), w = +r.getAttribute('width'), h = +r.getAttribute('height');
    const dx = Math.max(x - px, 0, px - x - w) * sx, dy = Math.max(y - py, 0, py - y - h) * sy;
    const d = (dx || dy) ? Math.hypot(dx, dy) : Math.min((px - x) * sx, (x + w - px) * sx, (py - y) * sy, (y + h - py) * sy);
    if (d < bd - 1 || (d <= bd + 1 && w * h < ba)) { best = r; bd = Math.min(d, bd); ba = w * h; }
    if (!dx && !dy && w * h < ia) { inner = r; ia = w * h; }
  });
  return bd <= BR_HIT ? best : inner;
}
let brHot = null, brRaf = 0;
function brSetHot(r) {
  if (brHot === r) return;
  if (brHot) brHot.classList.remove('br_hot');
  brHot = r;
  if (r) r.classList.add('br_hot');
  document.querySelectorAll('.br_wrap').forEach((w) => w.classList.toggle('br_near', !!r));
}
document.addEventListener('mousemove', (e) => {
  if (brRaf) return;
  brRaf = requestAnimationFrame(() => { brRaf = 0; brSetHot(brPick(e)); });
});
document.addEventListener('click', (e) => {            // bấm box: ghi mã box vào ô ẩn br_click rồi bấm nút ẩn br_apply
  const r = brPick(e);
  if (!r) return;
  const box = document.querySelector('#br_click textarea, #br_click input');
  const btn = document.getElementById('br_apply');
  if (!box || !btn) return;
  if (r.dataset.w) {                                   // chế độ chọn: đổi viền mảnh <-> đậm ngay, không chờ server
    const [w0, w1] = r.dataset.w.split('|');
    const on = r.getAttribute('stroke-width') === w1;
    r.setAttribute('stroke-width', on ? w0 : w1);
    r.setAttribute('stroke-opacity', on ? r.dataset.o : '1');
  } else { r.style.display = 'none'; brSetHot(null); }  // chế độ loại: ẩn ngay, không chờ server vẽ lại
  box.value = r.dataset.b + '|' + Date.now();
  box.dispatchEvent(new Event('input', {bubbles: true}));
  btn.click();
});
document.addEventListener('scroll', (e) => {           // nhớ vị trí cuộn theo (màn, zoom): vẽ lại sau mỗi lần bấm không nhảy về đầu
  const el = e.target;
  if (el && el.classList && el.classList.contains('br_scroll')) window.__brScroll[el.dataset.key] = [el.scrollLeft, el.scrollTop];
}, true);
// Tải trước ảnh các màn kế (`data-pre`): giữ đối tượng Image ĐÃ GIẢI MÃ trong bộ nhớ (tối đa 40) ⇒ sang màn sau ảnh hiện ngay
function brPrefetch(el) {
  (el.dataset.pre || '').split('|').filter(Boolean).forEach((u) => {
    if (window.__brImg.has(u)) { const im = window.__brImg.get(u); window.__brImg.delete(u); window.__brImg.set(u, im); return; }
    const im = new Image();
    im.src = u;
    if (im.decode) im.decode().catch(() => {});
    window.__brImg.set(u, im);
    while (window.__brImg.size > 40) window.__brImg.delete(window.__brImg.keys().next().value);
  });
}
new MutationObserver(() => {
  document.querySelectorAll('.br_scroll:not([data-restored])').forEach((el) => {
    el.dataset.restored = '1';
    const s = window.__brScroll[el.dataset.key];
    if (s) { el.scrollLeft = s[0]; el.scrollTop = s[1]; }
    brHot = null;
    brPrefetch(el);
  });
}).observe(document.documentElement, {childList: true, subtree: true});
</script>"""

CSS = (".br_hidden {display: none !important;} #br_head p {margin: 0.15em 0;} "
       ".br_scroll {overflow: auto; max-height: 90vh;} "
       ".br_wrap {position: relative; line-height: 0;} .br_wrap img {display: block; width: 100%; height: auto;} "
       ".br_wrap svg {position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none;} "
       ".br_wrap.br_near {cursor: pointer;} .br_wrap rect.br_hot {stroke: #ffe600; stroke-opacity: 1;}")


def default_out(source, split=None, objects_only=False):
    name = source + (f"_{split}" if source == "ce130" else "") + ("_objects" if objects_only else "")
    return os.path.join(OUT_DIR, name + ".jsonl")


def split_screens(screens, user, n_users=2):
    """Phần việc của người `user` (1..n_users): nhóm màn theo `group` (mặc định `id`), đánh số nhóm theo thứ tự xuất hiện, người k
    nhận nhóm thứ k−1, k−1+n, k−1+2n… (xen kẽ ⇒ hai người cùng tiến độ phủ cùng một đoạn đầu danh sách, lớp trộn đều)."""
    if not 1 <= user <= n_users:
        raise ValueError(f"user {user} ngoài 1..{n_users}")
    order = {}
    for sc in screens:
        order.setdefault(sc.get("group", sc["id"]), len(order))
    return [sc for sc in screens if order[sc.get("group", sc["id"])] % n_users == user - 1]


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


def box_svg(sc, wh, marks, view=0, show=None, selected=None):
    """SVG phủ ảnh (viewBox = pixel ảnh gốc). Box CHƯA loại: `<rect data-b="gi:i">` (bấm được — trình duyệt chọn box theo viền gần
    nhất, `brPick`), box to vẽ trước; box đã loại (`marks`) KHÔNG vẽ. Vật đã bị xoá ở view hiện tại (`gone`): mờ. `tags` (CE-130: H1..H4) là chữ
    duy nhất trên ảnh. `show`: tập "g<i>" các nhóm hiện (None = mọi nhóm). `selected` ({nhóm: set}, chế độ chọn): box đã chọn viền
    đậm (`sel_width`), chưa chọn viền `width` / `opacity` của nhóm. Nhóm `click: False`: không bấm được (không `data-b`)."""
    W, H = wh
    fs = max(W, H) / 45
    out = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">']
    items = [(gi, i, b) for gi, g in enumerate(sc["groups"]) if show is None or f"g{gi}" in show
             for i, b in enumerate(g["boxes"]) if i not in marks.get(g["key"], ())]
    items.sort(key=lambda x: -max(x[2][2] - x[2][0], 0) * max(x[2][3] - x[2][1], 0))
    labels = []
    for gi, i, (x1, y1, x2, y2) in items:
        g = sc["groups"][gi]
        w, op = g.get("width", 1.5), g.get("opacity")
        if 0 < g["gone"].get(i, 0) <= view:
            op = 0.35
        attr = f'data-b="{gi}:{i}" ' if g.get("click", True) else 'pointer-events="none" '
        if selected is not None and g.get("click", True):
            sw = g.get("sel_width", 3.5)
            attr += f'data-w="{w}|{sw}" data-o="{op if op is not None else 1}" '
            if i in selected.get(g["key"], ()):
                w, op = sw, 1
        faint = f' stroke-opacity="{op}"' if op is not None else ""
        out.append(f'<rect {attr}x="{x1:.1f}" y="{y1:.1f}" width="{max(x2 - x1, 0):.1f}" height="{max(y2 - y1, 0):.1f}" '
                   f'fill="transparent" stroke="{g["color"]}" stroke-width="{w}"{faint} vector-effect="non-scaling-stroke"/>')
        if i in g["tags"]:
            ty = max(y1 - fs * 1.25, 0)
            labels.append(f'<rect x="{x1:.1f}" y="{ty:.1f}" width="{fs * 0.62 * len(g["tags"][i]) + fs * 0.4:.1f}" '
                          f'height="{fs * 1.25:.1f}" fill="{g["color"]}"/><text x="{x1 + fs * 0.2:.1f}" y="{ty + fs:.1f}" '
                          f'font-size="{fs:.1f}" fill="white" font-family="sans-serif">{html.escape(g["tags"][i])}</text>')
    out.append(f'<g pointer-events="none">{"".join(labels)}</g>')
    return "".join(out) + "</svg>"


class Session:
    """Trạng thái rà của một người: vị trí `pos`, view, zoom, box đã loại `marks` {nhóm: set chỉ số} của màn hiện tại, nháp các màn
    đã sửa mà chưa lưu (`drafts`), ngăn hoàn tác (`undo_stack`: ("mark", id, nhóm, i) | ("save", id, bản lưu trước đó | None)).
    `mode`: "remove" = bấm để LOẠI box (biến mất, ghi `bad`) | "select" = bấm để CHỌN / bỏ chọn box (viền đậm, ghi `good` + toạ độ
    box đã chọn ở `positive` + `meta` của màn)."""

    def __init__(self, screens, out_path, name="", dedup=False, mode="remove"):
        if not screens:
            raise ValueError("không có màn nào")
        if mode not in ("remove", "select"):
            raise ValueError(f"mode {mode!r}")
        self.screens, self.name, self.dedup, self.mode = screens, name, dedup, mode
        self.field = "bad" if mode == "remove" else "good"
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
        if self.mode == "select" and sc["id"] not in self.auto:          # chế độ chọn: lọc lại từ đầu (không phụ thuộc box đã chọn)
            self.auto[sc["id"]] = {g["key"]: dedup_boxes(g["boxes"]) if g.get("dedup", True) else set() for g in sc["groups"]}
        if sc["id"] not in self.auto:
            rec = self.store.get(sc["id"])
            if rec and "auto" in rec:
                self.auto[sc["id"]] = {g["key"]: set(rec["auto"].get(g["key"], [])) for g in sc["groups"]}
            else:
                self.auto[sc["id"]] = {g["key"]: dedup_boxes(g["boxes"], set(range(len(g["boxes"]))) - removed.get(g["key"], set()))
                                       if g.get("dedup", True) else set() for g in sc["groups"]}
        return self.auto[sc["id"]]

    def _dedup_saved(self):
        """Màn đã lưu chưa có `auto` (rà trước khi có bước lọc trùng): lọc trên box còn lại sau `bad`, ghi `auto` vào file (chép bản cũ ra
        `.bak_<ngày giờ>` trước)."""
        todo = [r for r in self.store.latest.values() if "auto" not in r and r["id"] in self.index and "bad" in r]
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
        """Box không vẽ của màn hiện tại = người rà loại ∪ lọc trùng (chế độ chọn: chỉ lọc trùng)."""
        if self.mode == "select":
            return {k: set(self.auto_cur.get(k, set())) for k in self.marks}
        return {k: self.marks[k] | self.auto_cur.get(k, set()) for k in self.marks}

    # ---- box dùng chung giữa các màn (CE-130: box vật mọi nhánh cùng ảnh)
    def _rebuild_share(self):
        best = {}
        for rec in self.store.latest.values():
            for gkey, sk in (rec.get("share") or {}).items():
                if sk is not None and (sk not in best or rec["time"] >= best[sk][0]):
                    best[sk] = (rec["time"], rec["id"], rec.get(self.field, {}).get(gkey, []))
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
            self.marks = {g["key"]: set(rec.get(self.field, {}).get(g["key"], [])) for g in sc["groups"]}
        else:
            self.marks = {g["key"]: set() for g in sc["groups"]}
            for g in sc["groups"]:
                if g["share"] in self.share and self.share[g["share"]][1] != sc["id"]:
                    self.marks[g["key"]] = set(self.share[g["share"]][2])
                    self.prefill = self.share[g["share"]][1]
        self.auto_cur = self._auto_for(sc, self.marks)
        self.pre_cur = ({g["key"]: preselect(g["boxes"], g.get("scores"), self.auto_cur.get(g["key"], set()))
                         for g in sc["groups"] if g.get("click", True)} if self.mode == "select" else {})
        if self.mode == "select" and not rec and sc["id"] not in self.drafts:      # màn chưa lưu, chưa sửa: chọn sẵn top-k
            for k, v in self.pre_cur.items():
                self.marks[k] = set(v)
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
        """Loại box i của nhóm gi (đã loại / ngoài phạm vi: bỏ qua — box đã loại không còn trên ảnh, chỉ hoàn tác mới trả lại). Chế độ
        chọn: bật / tắt chọn box i. Nhóm không bấm được / box đã lọc trùng: bỏ qua."""
        sc = self.sc()
        if not (0 <= gi < len(sc["groups"]) and 0 <= i < len(sc["groups"][gi]["boxes"])) or not sc["groups"][gi].get("click", True):
            return
        key = sc["groups"][gi]["key"]
        if i in self.auto_cur.get(key, ()) or (i in self.marks[key] and self.mode == "remove"):
            return
        self.marks[key] ^= {i}
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
        rec = {"id": sc["id"], self.field: {g["key"]: sorted(self.marks[g["key"]]) for g in sc["groups"]},
               "n": {g["key"]: len(g["boxes"]) for g in sc["groups"]},
               "share": {g["key"]: g["share"] for g in sc["groups"] if g["share"] is not None},
               "time": datetime.now().isoformat(timespec="milliseconds"), "sec": round(time.time() - self.t_show, 1)}
        if self.dedup:
            rec["auto"] = {g["key"]: sorted(self.auto_cur[g["key"]]) for g in sc["groups"]}
        if self.mode == "select":
            rec["positive"] = [[round(v, 1) for v in g["boxes"][i]] for g in sc["groups"] if g.get("click", True)
                               for i in sorted(self.marks[g["key"]])]
            rec["pre"] = {k: sorted(v) for k, v in self.pre_cur.items()}
            rec.update(sc.get("meta", {}))
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
            self.marks[key] ^= {i}                  # loại: trả box lại | chọn: đảo lại lần bấm
            self.drafts[sid] = copy.deepcopy(self.marks)
            self.msg = f"undid click on {sid}"
        else:
            saved = self.store.get(sid)
            if rest[0] is None:
                self.store.delete(sid)
            else:
                self.store.put(rest[0])
            self._rebuild_share()
            self.go(self.index[sid])
            self.marks = {g["key"]: set(saved[self.field].get(g["key"], [])) for g in self.sc()["groups"]}
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
        """Một dòng tiến độ (tên phiên — nguồn, người rà — nếu có; + tiêu đề của màn nếu có)."""
        n = len(self.screens)
        done = sum(self.saved(p) for p in range(n))
        sc = self.sc()
        sel = ""
        if self.mode == "select":
            same = any(self.pre_cur.values()) and all(self.marks.get(k) == v for k, v in self.pre_cur.items())
            sel = f" · selected {sum(len(v) for v in self.marks.values())}" + (" (= preselect: top-5 C-NLL after NMS)" if same else "")
        return ((f"{self.name} · " if self.name else "") + f"{self.pos + 1} / {n} · {self.status()} · saved {done}{sel}"
                + (f"\n\n{sc['title']}" if sc["title"] else ""))

    def html(self):
        sc = self.sc()
        url = lambda p: FILE_ROUTE + html.escape(quote(os.path.abspath(p)))  # noqa: E731
        W, H = self.wh()
        width = f"min(100%, calc(88vh * {W / H:.4f}))" if self.zoom == "fit" else self.zoom      # fit: phóng / thu cho vừa khung
        nxt = [self.screens[p]["views"][0][1] for p in range(max(self.pos - 1, 0), min(self.pos + 1 + PREFETCH, len(self.screens)))
               if p != self.pos]
        pre = "|".join(url(p) for p in [v[1] for v in sc["views"]] + nxt)
        return (f'<div class="br_scroll" data-key="{html.escape(sc["id"])}|{self.zoom}" data-pre="{pre}">'
                f'<div class="br_wrap" style="width: {width}">'
                f'<img src="{url(sc["views"][self.view][1])}" style="aspect-ratio: {W} / {H}" alt="">'
                f'{box_svg(sc, (W, H), self.hidden(), self.view, self.show, self.marks if self.mode == "select" else None)}</div></div>')


class PredHub:
    """Chế độ chọn box model (người dùng 2026-10-30): MỘT app, Data + User chọn trên giao diện. `datasets` {nhãn: (hàm -> list màn,
    file kết quả gốc .jsonl)}; mỗi (Data, User) một `Session` chế độ chọn trên phần việc `split_screens` của User, file
    `<gốc>_<user>.jsonl` — dựng khi chọn lần đầu (màn của Data đọc một lần), giữ lại khi đổi qua lại. Chưa chọn đủ cả hai: `sess` None.
    `preload`: đọc màn mọi Data ở luồng nền ngay khi mở app (CE-130 ~2 s, CE-CoCount ~6 s) ⇒ bấm chọn Data không phải chờ."""

    def __init__(self, datasets, users=("user_1", "user_2"), dedup=True, preload=False):
        self.datasets, self.users, self.dedup = datasets, list(users), dedup
        self.data = self.user = self.sess = None
        self._screens, self._sessions, self._threads = {}, {}, {}
        if preload:
            for d in datasets:
                self._threads[d] = threading.Thread(target=self._load, args=(d,), daemon=True)
                self._threads[d].start()

    def _load(self, data):
        self._screens[data] = self.datasets[data][0]()

    def screens(self, data):
        """Màn của Data (chờ luồng nền nếu đang đọc; luồng nền lỗi: đọc lại ở đây để lỗi hiện ra)."""
        if data in self._threads:
            self._threads[data].join()
        if data not in self._screens:
            self._load(data)
        return self._screens[data]

    def choose(self, data=None, user=None):
        if data is not None:
            self.data = data if data in self.datasets else None
        if user is not None:
            self.user = user if user in self.users else None
        if self.data is None or self.user is None:
            self.sess = None
            return
        key = (self.data, self.user)
        if key not in self._sessions:
            base = self.datasets[self.data][1]
            screens = split_screens(self.screens(self.data), self.users.index(self.user) + 1, len(self.users))
            out = base[:-len(".jsonl")] + f"_{self.user}.jsonl"
            self._sessions[key] = Session(screens, out, f"{self.data} · {self.user}", dedup=self.dedup, mode="select")
            print(f"[{self.data} · {self.user}] {len(screens)} màn / {len(self._screens[self.data])}, "
                  f"đã lưu {sum(self._sessions[key].saved(p) for p in range(len(screens)))} -> {out}", flush=True)
        self.sess = self._sessions[key]


class _One:
    """Một phiên cố định (cách dùng 1 / 2) — cùng giao diện `PredHub` cho `build_ui`."""

    def __init__(self, sess):
        self.sess = sess


def build_ui(sess, gr):
    """`sess`: `Session` (một phiên cố định) hoặc `PredHub` (thêm ô chọn Data + User; chưa chọn đủ thì màn trống, nút không làm gì)."""
    hub = sess if isinstance(sess, PredHub) else _One(sess)
    kw = {"title": "Box review", "css": CSS, "head": KEYS_JS}
    bparams = inspect.signature(gr.Blocks.__init__).parameters
    launch_kw = {k: v for k, v in kw.items() if k not in bparams and k != "title"}       # gradio >= 6: css / head ở launch
    hidden = {"elem_classes": "br_hidden"}
    with gr.Blocks(**{k: v for k, v in kw.items() if k in bparams}) as demo:
        head = gr.Markdown(elem_id="br_head")
        with gr.Row():
            with gr.Column(scale=5):
                img = gr.HTML()
            with gr.Column(scale=1, min_width=220):
                if isinstance(hub, PredHub):
                    data = gr.Radio(list(hub.datasets), value=hub.data, label="Data")
                    user = gr.Radio(hub.users, value=hub.user, label="User")
                with gr.Row():
                    prev = gr.Button("← (prev)", elem_id="br_prev", min_width=60)
                    nxt = gr.Button("→ (next)", elem_id="br_next", min_width=60)
                save = gr.Button("Save (Enter)", variant="primary", elem_id="br_save")
                undo = gr.Button("Undo (Ctrl+Z)", elem_id="br_undo")
                first = gr.Button("Go to first unsaved", size="sm")
                view = gr.Radio(label="Image (v)", choices=[], visible=False)
                zoom = gr.Radio(list(ZOOMS), value="fit", label="Zoom (z)")
                show = gr.CheckboxGroup(label="Show", choices=[], visible=False)
                click = gr.Textbox(elem_id="br_click", **hidden)
                apply = gr.Button("apply", elem_id="br_apply", **hidden)
                vnext = gr.Button("view", elem_id="br_vnext", **hidden)
                znext = gr.Button("zoom", elem_id="br_znext", **hidden)

        def full():
            s = hub.sess
            if s is None:
                return ["**Choose Data and User on the right to start.**", "", gr.update(choices=[], visible=False),
                        gr.update(choices=[], visible=False), gr.update(value="fit")]
            sc = s.sc()
            labels = [v[0] for v in sc["views"]]
            shows = [(g["key"], f"g{i}") for i, g in enumerate(sc["groups"])]
            return [s.header(), s.html(), gr.update(choices=labels, value=labels[s.view], visible=len(labels) > 1),
                    gr.update(choices=shows, value=[v for _, v in shows if v in s.show], visible=len(shows) > 1),
                    gr.update(value=s.zoom)]

        def act(fn, light=False):
            """Gọi `fn(phiên hiện tại, *args)` (chưa có phiên: bỏ qua) rồi vẽ lại. `light` (bấm box): trình duyệt đã đổi box ngay ⇒ chỉ
            cập nhật dòng tiến độ, KHÔNG gửi lại ảnh + SVG (không nháy ảnh, không mất box đang tô vàng)."""
            def f(*args):
                if hub.sess is not None:
                    fn(hub.sess, *args)
                if light and hub.sess is not None:
                    return [hub.sess.header()] + [skip() for _ in outs[1:]]
                return full()
            return f

        def set_show(s, v):
            s.show = set(v or [])

        def set_zoom(s, z):
            s.zoom = z if z in ZOOMS else "fit"

        outs = [head, img, view, show, zoom]
        skip = getattr(gr, "skip", gr.update)
        q = {"queue": False, "show_progress": "hidden"}             # không phủ lớp "đang tải" lên ảnh sau mỗi thao tác
        demo.load(full, None, outs, **q)
        apply.click(act(Session.click, light=True), [click], outs, **q)
        for b, fn in ((prev, Session.prev), (nxt, Session.next), (save, Session.save), (undo, Session.undo),
                      (first, Session.first_unsaved), (vnext, Session.next_view), (znext, Session.next_zoom)):
            b.click(act(fn), None, outs, **q)
        view.input(act(Session.set_view), [view], outs, **q)
        zoom.input(act(set_zoom), [zoom], outs, **q)
        show.input(act(set_show), [show], outs, **q)
        if isinstance(hub, PredHub):
            def pick(field):
                def f(v):
                    hub.choose(**{field: v})
                    return full()
                return f
            data.input(pick("data"), [data], outs, **q)
            user.input(pick("user"), [user], outs, **q)
    return demo, launch_kw


def pred_out(dump):
    """File kết quả gốc của chế độ chọn theo đường dẫn dump `…/snapshots/<run>/<tập>/<epoch>_boxes.json`."""
    parts = os.path.abspath(dump).split(os.sep)
    return os.path.join(OUT_DIR, f"pred_{parts[-3]}_{parts[-1].replace('_boxes.json', '')}_{parts[-2]}.jsonl")


def main():
    snap = os.path.join(os.path.dirname(OD), "output", "add", "snapshots", "delta8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", default="cocount", choices=["cocount", "ce130", "pred"], help="mặc định cocount")
    ap.add_argument("--root", default=None, help="mặc định object-detection/data/cocount | object-detection/data/all_phase2_V2")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"], help="ce130")
    ap.add_argument("--objects-only", action="store_true", help="ce130: chỉ rà box vật (không hiện box lỗ)")
    ap.add_argument("--out", default=None, help="cocount / ce130: file kết quả .jsonl (mặc định output/add/box_review/<nguồn>.jsonl)")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--no-browser", action="store_true", help="không tự mở trình duyệt")
    ap.add_argument("--no-dedup", action="store_true", help="cocount / pred: tắt lọc trùng tự động (mặc định bật; ce130 luôn tắt)")
    ap.add_argument("--ce130-dump", default=os.path.join(snap, "ce130_test_all", "ep010_boxes.json"), help="pred: dump CE-130")
    ap.add_argument("--cocount-dump", default=os.path.join(snap, "cocount_review5", "ep010_boxes.json"), help="pred: dump CE-CoCount")
    ap.add_argument("--turn-index", default=os.path.join(OD, "data", "turn_index.json"), help="pred: CE-130")
    ap.add_argument("--samples-root", default=os.path.join(OD, "data", "samples"), help="pred: CE-130")
    ap.add_argument("--review", default=os.path.join(OUT_DIR, "cocount.jsonl"), help="pred: bản rà box vật CE-CoCount (review5)")
    ap.add_argument("--n-users", type=int, default=2, help="pred: số người chia việc (user_1 … user_n)")
    a = ap.parse_args()
    t0 = time.time()
    if a.source == "pred":
        samples = os.path.abspath(a.samples_root)
        cc_root = os.path.abspath(a.root or os.path.join(OD, "data", "cocount"))
        for f in (a.ce130_dump, a.cocount_dump, a.turn_index, a.review):
            if not os.path.exists(f):
                sys.exit(f"thiếu {f}")
        datasets = {"CE-130": (lambda: SOURCES["ce130_pred"](a.ce130_dump, a.turn_index, samples), pred_out(a.ce130_dump)),
                    "CE-CoCount": (lambda: SOURCES["cocount_pred"](a.cocount_dump, cc_root, a.review), pred_out(a.cocount_dump))}
        sess = PredHub(datasets, [f"user_{k}" for k in range(1, a.n_users + 1)], dedup=not a.no_dedup, preload=True)
        roots = [samples, cc_root]
        print(f"[pred] CE-130 {a.ce130_dump}\n       CE-CoCount {a.cocount_dump}\n       chọn Data + User trên giao diện", flush=True)
    else:
        root = os.path.abspath(a.root or os.path.join(OD, "data", "cocount" if a.source == "cocount" else "all_phase2_V2"))
        screens = (SOURCES["cocount"](root) if a.source == "cocount" else SOURCES["ce130"](root, a.split, holes=not a.objects_only))
        out = a.out or default_out(a.source, a.split, a.objects_only)
        name = "CE-CoCount" if a.source == "cocount" else f"CE-130 {a.split}" + (" (objects only)" if a.objects_only else "")
        sess = Session(screens, out, name, dedup=a.source == "cocount" and not a.no_dedup)
        roots = [root]
        print(f"[{name}] {len(screens)} màn ({time.time() - t0:.1f}s), đã lưu {len(sess.store.latest)} -> {out}", flush=True)
    # gradio 6.17 + starlette 1.x: cảnh báo deprecated mỗi request (lỗi thư viện, request vẫn thành công) — bỏ đúng cảnh báo này
    warnings.filterwarnings("ignore", message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*")
    import gradio as gr
    demo, launch_kw = build_ui(sess, gr)
    demo.launch(server_name="127.0.0.1", server_port=a.port, inbrowser=not a.no_browser, allowed_paths=roots, **launch_kw)


if __name__ == "__main__":
    main()

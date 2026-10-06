#!/usr/bin/env python3
"""Web chấm user study (docs/EXPERIMENT_GAMMA.md mục 17). Chạy ở máy chấm (local), chỉ đọc `items.json` + ảnh `samples/`,
không model, không GPU. Cần `pip install gradio` (không có trong requirements.txt của server). Chữ trên web tiếng Anh.

Mỗi màn = một cặp (mẫu test, model). Ô "Models to rate" (hoặc `--models`) chọn model đang chấm: một model = dãy màn của riêng nó
(mẫu xáo ngẫu nhiên), nhiều model = các dãy xen ngẫu nhiên; đổi lựa chọn lúc nào cũng được, nhãn lưu theo mã màn nên không lẫn.
Đầu màn ghi rõ mẫu (nhánh, lượt, file) và tên hiển thị của model (docs/EXPERIMENT_GAMMA.md mục 13.0). Ảnh inpaint lượt t + box vật đang có (xanh dương, tắt được) + K box đề xuất đánh số
theo hạng (xanh lá = ổn, đỏ = không ổn). Lỗ GT không hiện. Nhãn mỗi box = DANH SÁCH lý do lỗi (rỗng = ổn; chọn được nhiều lý do
cùng lúc ở cột phải). Bấm vào box hoặc phím 1..K để đổi ổn <-> không ổn; box vừa chuyển đỏ mặc định lý do "On object".
  ← / → (Previous / Next)  sang màn trước / sau trong dãy đang chọn, KHÔNG lưu (Next = tạm bỏ qua màn)
  Enter (Save)             lưu màn này (box còn xanh = ổn) rồi sang màn sau; lưu lại màn đã lưu thì GHI ĐÈ dòng cũ
  Go to first unsaved      về màn chưa lưu đầu tiên của dãy đang chọn (lỡ Next bỏ qua)
`ratings.jsonl`: MỘT dòng mỗi màn (khoá = mã màn `<model>|<image_id>[|repeat]`; người dùng, 2026-10-06), lưu xuống đĩa ngay mỗi lần
Save ⇒ tắt mở lại vẫn tiếp từ màn chưa lưu đầu tiên; thêm model vào items.json (`build.py --add-to`) không ảnh hưởng nhãn cũ.

  cd object-detection/ce_localization
  python user_study/app.py --items ../../output/gamma/user_study/items.json --samples-root ../data/samples [--models gamma4]
Người khác chấm qua link (chạy trên server, docs/EXPERIMENT_GAMMA.md mục 17): thêm `--share` (in link *.gradio.live) và
`--ratings .../ratings_<tên>.jsonl` để file nhãn của họ không trùng tên `ratings.jsonl` của người chấm ở local khi tải về.

Hiển thị (2026-10-06, cho nhanh qua link share): ảnh nền = bản WebP cạnh dài MAX_SIDE dựng MỘT lần vào `--cache-dir`, trình duyệt
tải thẳng qua `/gradio_api/file=` (cache được); box vẽ bằng SVG đè lên ảnh (bấm box = phím i) ⇒ đổi trạng thái box chỉ gửi vài KB
HTML, không gửi lại ảnh; ảnh của PREFETCH màn kế tiếp được tải trước (thẻ <img> ẩn) ⇒ sang màn sau gần như tức thì.
"""

import argparse
import html
import inspect
import json
import os
import tempfile
import time
import warnings
from datetime import datetime
from urllib.parse import quote

from PIL import Image

__all__ = ["REASONS", "REASON_EN", "DEFAULT_BAD", "as_reasons", "toggle", "box_svg", "RatingStore", "Session"]

REASONS = ("on_object", "wrong_size", "implausible", "other")   # nhãn box = list con (thứ tự này); [] = ổn
REASON_EN = {"on_object": "On object", "wrong_size": "Wrong size", "implausible": "Implausible location", "other": "Other"}
DEFAULT_BAD = "on_object"
MAX_SIDE = 1100                                   # cạnh dài ảnh nền (px): ảnh test nhỏ (vd 576 × 384) cũng phóng lên
WEBP_QUALITY = 90
PREFETCH = 3                                      # số màn kế tiếp tải trước ảnh
COLOR = {"obj": "#1e90ff", "ok": "#00dc3c", "bad": "#eb1e1e"}
FILE_ROUTE = "/gradio_api/file="

HELP = """**A box is OK** if adding one more object of this class there looks plausible:
1. It does not cover an existing object
2. Its size and aspect ratio match the other objects of this class
3. Its location makes sense in the scene
4. It lies inside the image

Press key i (or click the box) to toggle box i. A box is not OK as soon as one reason is ticked; tick all that apply."""

KEYS_JS = """<script>
document.addEventListener('keydown', (e) => {
  if (e.target && (e.target.type === 'text' || e.target.tagName === 'TEXTAREA')) return;
  const id = /^[1-9]$/.test(e.key) ? 'us_tog' + e.key :
             ({ArrowLeft: 'us_prev', ArrowRight: 'us_next', Enter: 'us_save'})[e.key];
  const el = id && document.getElementById(id);
  if (el) { e.preventDefault(); el.click(); }
});
document.addEventListener('click', (e) => {          // bấm box trên SVG = bấm nút ẩn us_tog<i> (cùng đường với phím i)
  const g = e.target.closest && e.target.closest('[data-box]');
  const el = g && document.getElementById('us_tog' + g.dataset.box);
  if (el) el.click();
});
</script>"""
CSS = ("#us_head p {font-size: 1.1em; margin: 0.15em 0;} .us_hidden {display: none !important;} "
       ".us_wrap {position: relative; display: inline-block; line-height: 0; max-width: 100%;} "
       ".us_wrap img.us_bg {display: block; max-width: 100%; max-height: 86vh; width: auto; height: auto;} "
       ".us_wrap svg {position: absolute; left: 0; top: 0; width: 100%; height: 100%;} "
       ".us_wrap g[data-box] {cursor: pointer;}")


def as_reasons(label):
    """Nhãn một box -> list lý do theo thứ tự REASONS ([] = ổn). Nhận cả dạng cũ một chuỗi ("ok" / một lý do)."""
    if isinstance(label, str):
        label = [] if label == "ok" else [label]
    return [r for r in REASONS if r in label]


def toggle(reasons):
    return [] if reasons else [DEFAULT_BAD]


def box_svg(wh, objects, boxes, labels, show_objects=True):
    """SVG phủ lên ảnh, toạ độ = pixel ảnh gốc (viewBox W × H, co giãn theo ảnh). Box vật xanh dương (không bấm được); box đề xuất
    xanh lá / đỏ + nhãn số, mỗi box một <g data-box="i+1"> (bấm = đổi box i+1). Box to vẽ trước, box nhỏ đè lên ⇒ bấm chỗ chồng nhau
    trúng box nhỏ nhất."""
    W, H = wh
    fs = max(W, H) / 38                                                   # cỡ chữ nhãn theo đơn vị ảnh gốc
    out = [f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">']
    if show_objects:
        out += [f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{max(x2 - x1, 0):.1f}" height="{max(y2 - y1, 0):.1f}" fill="none" '
                f'stroke="{COLOR["obj"]}" stroke-width="1.5" vector-effect="non-scaling-stroke" pointer-events="none"/>'
                for x1, y1, x2, y2 in objects]
    order = sorted(range(len(boxes)), key=lambda i: -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
    for i in order:
        x1, y1, x2, y2 = boxes[i]
        c = COLOR["bad" if as_reasons(labels[i]) else "ok"]
        th, tw = fs * 1.25, fs * 0.9
        ty = y1 - th if y1 - th >= 0 else y1
        out.append(f'<g data-box="{i + 1}"><rect x="{x1:.1f}" y="{y1:.1f}" width="{max(x2 - x1, 0):.1f}" '
                   f'height="{max(y2 - y1, 0):.1f}" fill="transparent" stroke="{c}" stroke-width="3.5" '
                   f'vector-effect="non-scaling-stroke"/><rect x="{x1:.1f}" y="{ty:.1f}" width="{tw:.1f}" height="{th:.1f}" '
                   f'fill="{c}"/><text x="{x1 + tw / 2:.1f}" y="{ty + th * 0.8:.1f}" font-size="{fs:.1f}" font-weight="bold" '
                   f'fill="white" text-anchor="middle" font-family="sans-serif">{i + 1}</text></g>')
    return "".join(out) + "</svg>"


class RatingStore:
    """`ratings.jsonl`: MỘT dòng mỗi mã màn `id`, theo thứ tự lưu lần đầu. Màn mới: ghi nối một dòng. Lưu lại màn đã có: ghi đè dòng
    đó (ghi cả file ra `.tmp` rồi `os.replace` — tắt ngang lúc ghi vẫn còn nguyên bản cũ hoặc bản mới, không nửa vời). File cũ có nhiều
    dòng cùng `id` (bản trước 2026-10-06 ghi nối mọi lần lưu) được gộp ngay khi mở, giữ bản SAU CÙNG."""

    def __init__(self, path):
        self.path, self.latest = path, {}
        n = 0
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self.latest[rec["id"]] = rec          # dict giữ vị trí lần đầu, nội dung bản sau cùng
                        n += 1
        if n > len(self.latest):
            self._rewrite()

    def _rewrite(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            for rec in self.latest.values():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def save(self, rec):
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


class Session:
    """Trạng thái chấm (một người chấm): model đang chọn, dãy màn `view` (chỉ số vào `screens`, theo `order`), vị trí `pos`,
    nhãn K box đang sửa."""

    def __init__(self, items_path, samples_root, ratings_path=None, models=None, cache_dir=None):
        with open(items_path) as f:
            self.d = json.load(f)
        self.k, self.screens, self.items = self.d["k"], self.d["screens"], self.d["items"]
        self.samples_root = samples_root
        self.store = RatingStore(ratings_path or os.path.join(os.path.dirname(os.path.abspath(items_path)), "ratings.jsonl"))
        self.cache_dir = os.path.abspath(cache_dir or os.path.join(
            os.environ.get("GRADIO_TEMP_DIR") or tempfile.gettempdir(), "ce_loc_user_study"))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.set_models(models)

    def model_name(self, m):
        return self.d["models"][m].get("name") or m

    # ---- chọn model
    def set_models(self, models=None):
        """Chấm `models` (list mã; None / rỗng = mọi model) rồi về màn chưa lưu đầu tiên của dãy đó."""
        bad = [m for m in models or [] if m not in self.d["models"]]
        if bad:
            raise ValueError(f"model không có trong items.json: {bad} (có {list(self.d['models'])})")
        self.models = [m for m in self.d["models"] if not models or m in models]
        self.view = sorted((j for j, sc in enumerate(self.screens) if sc["model"] in self.models),
                           key=lambda j: (self.screens[j]["order"], self.screens[j]["id"]))
        self.first_unsaved()

    # ---- điều hướng
    def saved(self, pos):
        return self.screens[self.view[pos]]["id"] in self.store.latest

    def next_unrated(self, after):
        """Vị trí chưa lưu đầu tiên sau `after` trong dãy; hết thì quay vòng từ đầu; lưu hết -> None."""
        n = len(self.view)
        for p in list(range(after + 1, n)) + list(range(0, min(after + 1, n))):
            if not self.saved(p):
                return p
        return None

    def go(self, pos):
        self.pos = min(max(pos, 0), len(self.view) - 1)
        rec = self.store.latest.get(self.screen()[0]["id"])
        self.labels = [as_reasons(x) for x in rec["labels"]] if rec else [[] for _ in self.sel()["boxes"]]
        self.t_show = time.time()

    @property
    def done(self):
        return all(self.saved(p) for p in range(len(self.view)))

    def first_unsaved(self):
        p = self.next_unrated(-1)
        self.go(len(self.view) - 1 if p is None else p)

    def next(self):
        """Sang màn sau, KHÔNG lưu (tạm bỏ qua)."""
        self.go(self.pos + 1)

    def prev(self):
        self.go(self.pos - 1)

    def screen(self):
        sc = self.screens[self.view[self.pos]]
        return sc, self.items[sc["image_id"]]

    def sel(self):
        sc, it = self.screen()
        return it["models"][sc["model"]]

    # ---- sửa nhãn
    def set_reasons(self, i, reasons):
        """Đặt list lý do của box i ([] = ổn)."""
        if i < len(self.labels):
            self.labels[i] = as_reasons(list(reasons or []))

    def toggle(self, i):
        if i < len(self.labels):
            self.labels[i] = toggle(self.labels[i])

    def save(self):
        """Lưu màn này rồi sang màn sau (màn cuối: đứng yên)."""
        sc, it = self.screen()
        self.store.save({"id": sc["id"], "image_id": sc["image_id"], "model": sc["model"], "repeat_of": sc["repeat_of"],
                         "labels": [list(x) for x in self.labels], "sec": round(time.time() - self.t_show, 2),
                         "time": datetime.now().isoformat(timespec="seconds")})
        if self.pos < len(self.view) - 1:
            self.next()

    # ---- hiển thị
    def background(self, image):
        """Ảnh nền của file `image` (đường dẫn trong samples/): WebP cạnh dài MAX_SIDE, dựng một lần vào cache -> đường dẫn tuyệt đối."""
        out = os.path.join(self.cache_dir, image.replace("/", "__") + ".webp")
        if not os.path.exists(out):
            im = Image.open(os.path.join(self.samples_root, image)).convert("RGB")
            r = MAX_SIDE / max(im.size)
            im = im.resize((max(1, round(im.size[0] * r)), max(1, round(im.size[1] * r))), Image.BILINEAR)
            tmp = out + ".tmp.webp"
            im.save(tmp, "WEBP", quality=WEBP_QUALITY)
            os.replace(tmp, out)
        return out

    def html(self, show_objects=True):
        """Màn hiện tại: ảnh nền + SVG box + <img> ẩn tải trước ảnh của PREFETCH màn kế tiếp."""
        sc, it = self.screen()
        url = lambda p: FILE_ROUTE + html.escape(quote(p))  # noqa: E731
        nxt = {self.items[self.screens[self.view[p]]["image_id"]]["image"]
               for p in range(self.pos + 1, min(self.pos + 1 + PREFETCH, len(self.view)))} - {it["image"]}
        pre = "".join(f'<img src="{url(self.background(im))}" alt="">' for im in sorted(nxt))
        return (f'<div class="us_wrap"><img class="us_bg" src="{url(self.background(it["image"]))}" alt="">'
                f'{box_svg(it["wh"], it["objects"], self.sel()["boxes"], self.labels, show_objects)}</div>'
                f'<div style="display:none">{pre}</div>')

    def header(self):
        sc, it = self.screen()
        n = len(self.view)
        saved = sum(self.saved(p) for p in range(n))
        state = "saved" if self.saved(self.pos) else "not saved"
        if self.done:
            state += " | ALL SCREENS OF THE SELECTED MODELS SAVED"
        branch, turn = sc["image_id"].rsplit("_t", 1)
        return (f"Screen **{self.pos + 1} / {n}** ({state}) | Saved {saved} / {n} ({saved / n * 100:.1f}%)\n\n"
                f"Test sample: **{branch}**, turn {turn} ({it['image']}) | Model: **{self.model_name(sc['model'])}** | "
                f"{len(it['objects'])} existing objects\n\n"
                f"### Add one more: **{it['class']}**")


def build_ui(sess, gr):
    k = sess.k
    choices = [(REASON_EN[r], r) for r in REASONS]
    kw = {"title": "CE-Loc user study", "css": CSS, "head": KEYS_JS}
    bparams = inspect.signature(gr.Blocks.__init__).parameters
    launch_kw = {key: v for key, v in kw.items() if key not in bparams and key != "title"}   # gradio >= 6: css / head ở launch
    with gr.Blocks(**{key: v for key, v in kw.items() if key in bparams}) as demo:
        head = gr.Markdown(elem_id="us_head")
        with gr.Row():
            with gr.Column(scale=4):
                img = gr.HTML(elem_id="us_img")
            with gr.Column(scale=1, min_width=300):
                models = gr.Dropdown(choices=[(sess.model_name(m), m) for m in sess.d["models"]], value=list(sess.models),
                                     multiselect=True, label="Models to rate (empty = all)")
                with gr.Row():
                    prev = gr.Button("← Previous", elem_id="us_prev", min_width=60, scale=1)
                    nxt = gr.Button("Next →", elem_id="us_next", min_width=60, scale=1)
                save = gr.Button("Save (Enter)", variant="primary", elem_id="us_save")
                first = gr.Button("Go to first unsaved", size="sm")
                radios = [gr.CheckboxGroup(choices=choices, value=[], label=f"Box {i + 1}: OK") for i in range(k)]
                togs = [gr.Button(str(i + 1), elem_id=f"us_tog{i + 1}", elem_classes="us_hidden") for i in range(k)]
                show = gr.Checkbox(value=True, label="Show existing objects (blue)")
                gr.Markdown(HELP)

        def radio_updates():
            n = len(sess.labels)
            return [gr.update(value=sess.labels[i] if i < n else [], visible=i < n,
                              label=f"Box {i + 1}: " + ("not OK" if i < n and sess.labels[i] else "OK")) for i in range(k)]

        def full(show_objects):
            return [sess.header(), sess.html(show_objects)] + radio_updates()

        def nav(fn):
            def f(show_objects):
                fn()
                return full(show_objects)
            return f

        def on_models(ms, show_objects):
            sess.set_models(ms)
            return full(show_objects)

        def make_radio(i):
            def f(v, show_objects):
                sess.set_reasons(i, v)
                return [sess.html(show_objects)] + radio_updates()
            return f

        def make_tog(i):
            def f(show_objects):
                sess.toggle(i)
                return [sess.html(show_objects)] + radio_updates()
            return f

        outs_full = [head, img] + radios
        q = {"queue": False}                       # một người chấm: gọi thẳng, không qua hàng đợi (bớt một vòng mạng qua link share)
        demo.load(full, [show], outs_full, api_name="show", **q)
        models.input(on_models, [models, show], outs_full, api_name="models", **q)
        for i, r_ in enumerate(radios):
            r_.input(make_radio(i), [r_, show], [img] + radios, api_name=f"reasons{i + 1}", **q)
        for i, b in enumerate(togs):
            b.click(make_tog(i), [show], [img] + radios, api_name=f"toggle{i + 1}", **q)
        prev.click(nav(sess.prev), [show], outs_full, api_name="prev", **q)
        nxt.click(nav(sess.next), [show], outs_full, api_name="next", **q)
        save.click(nav(sess.save), [show], outs_full, api_name="save", **q)
        first.click(nav(sess.first_unsaved), [show], outs_full, api_name="first_unsaved", **q)
        show.change(lambda s_: sess.html(s_), [show], [img], api_name="objects", **q)
    return demo, launch_kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="items.json của user_study/build.py")
    ap.add_argument("--samples-root", default="../data/samples")
    ap.add_argument("--ratings", default=None, help="mặc định ratings.jsonl cạnh items.json")
    ap.add_argument("--models", nargs="*", default=None, help="mã model chấm lúc mở (mặc định mọi model; đổi được trên web)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--cache-dir", default=None, help="ảnh nền WebP dựng sẵn (mặc định $GRADIO_TEMP_DIR hoặc thư mục tạm "
                                                          "/ce_loc_user_study; ~25 KB / ảnh)")
    ap.add_argument("--share", action="store_true", help="tạo link công khai https://*.gradio.live (sống tối đa 1 tuần) để người "
                                                          "khác chấm qua trình duyệt, vd chạy trên server")
    a = ap.parse_args()
    # gradio 6.17 dựng bảng mã HTTP ở MỖI request, trong đó đọc `status.HTTP_422_UNPROCESSABLE_ENTITY` mà starlette 1.x đã đổi tên
    # ⇒ một cảnh báo deprecated mỗi lần bấm (request vẫn thành công). Lỗi của thư viện: chỉ bỏ ĐÚNG cảnh báo này.
    warnings.filterwarnings("ignore", message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*")
    import gradio as gr
    sess = Session(a.items, a.samples_root, a.ratings, a.models, a.cache_dir)
    print(f"{len(sess.screens)} màn ({len(sess.d['models'])} model), đã lưu {len(sess.store.latest)} -> {sess.store.path}",
          flush=True)
    demo, launch_kw = build_ui(sess, gr)
    demo.launch(server_name="127.0.0.1", server_port=a.port, inbrowser=not a.share, share=a.share,
                allowed_paths=[sess.cache_dir], **launch_kw)


if __name__ == "__main__":
    main()

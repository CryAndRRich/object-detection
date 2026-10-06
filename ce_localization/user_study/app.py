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
"""

import argparse
import inspect
import json
import os
import time
import warnings
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

__all__ = ["REASONS", "REASON_EN", "DEFAULT_BAD", "as_reasons", "box_at", "toggle", "render", "RatingStore", "Session"]

REASONS = ("on_object", "wrong_size", "implausible", "other")   # nhãn box = list con (thứ tự này); [] = ổn
REASON_EN = {"on_object": "On object", "wrong_size": "Wrong size", "implausible": "Implausible location", "other": "Other"}
DEFAULT_BAD = "on_object"
MAX_SIDE = 1100                                   # cạnh dài ảnh hiển thị (px): ảnh test nhỏ (vd 576 × 384) cũng phóng lên
COLOR = {"obj": (30, 144, 255), "ok": (0, 220, 60), "bad": (235, 30, 30)}

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
</script>"""
CSS = ("#us_img img {max-height: 86vh; object-fit: contain;} #us_head p {font-size: 1.1em; margin: 0.15em 0;} "
       ".us_hidden {display: none !important;}")


def box_at(boxes, x, y):
    """Chỉ số box chứa điểm (x, y) — nhiều box thì box NHỎ nhất (box lồng trong box khác vẫn bấm được); không box nào -> None."""
    best, area = None, None
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if x1 <= x <= x2 and y1 <= y <= y2:
            a = (x2 - x1) * (y2 - y1)
            if area is None or a < area:
                best, area = i, a
    return best


def as_reasons(label):
    """Nhãn một box -> list lý do theo thứ tự REASONS ([] = ổn). Nhận cả dạng cũ một chuỗi ("ok" / một lý do)."""
    if isinstance(label, str):
        label = [] if label == "ok" else [label]
    return [r for r in REASONS if r in label]


def toggle(reasons):
    return [] if reasons else [DEFAULT_BAD]


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                             # Pillow < 10.1
        return ImageFont.load_default()


def render(img, objects, boxes, labels, show_objects=True, max_side=MAX_SIDE):
    """Ảnh PIL gốc + box (pixel ảnh gốc) -> (ảnh đã vẽ, cạnh dài = max_side; hệ số r: toạ độ hiển thị = r × toạ độ gốc)."""
    W, H = img.size
    r = max_side / max(W, H)
    im = img.convert("RGB").resize((max(1, round(W * r)), max(1, round(H * r))), Image.BILINEAR)
    dr = ImageDraw.Draw(im)
    lw = max(2, round(max(im.size) / 300))
    if show_objects:
        for o in objects:
            dr.rectangle([v * r for v in o], outline=COLOR["obj"], width=max(1, lw // 2))
    font = _font(max(14, round(max(im.size) / 45)))
    for i, (b, lab) in enumerate(zip(boxes, labels)):
        c = COLOR["bad" if as_reasons(lab) else "ok"]
        x1, y1, x2, y2 = (v * r for v in b)
        dr.rectangle([x1, y1, x2, y2], outline=c, width=lw + 1)
        tag = str(i + 1)
        tb = dr.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0] + 6, tb[3] - tb[1] + 6
        tx, ty = x1, (y1 - th if y1 - th >= 0 else y1)
        dr.rectangle([tx, ty, tx + tw, ty + th], fill=c)
        dr.text((tx + 3 - tb[0], ty + 3 - tb[1]), tag, fill=(255, 255, 255), font=font)
    return im, r


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

    def __init__(self, items_path, samples_root, ratings_path=None, models=None):
        with open(items_path) as f:
            self.d = json.load(f)
        self.k, self.screens, self.items = self.d["k"], self.d["screens"], self.d["items"]
        self.samples_root = samples_root
        self.store = RatingStore(ratings_path or os.path.join(os.path.dirname(os.path.abspath(items_path)), "ratings.jsonl"))
        self._img = (None, None)
        self.r = 1.0
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

    def click(self, x, y):
        """(x, y) toạ độ trên ảnh HIỂN THỊ -> đổi box chứa điểm đó; -> chỉ số box hoặc None."""
        i = box_at(self.sel()["boxes"], x / self.r, y / self.r)
        if i is not None:
            self.toggle(i)
        return i

    def save(self):
        """Lưu màn này rồi sang màn sau (màn cuối: đứng yên)."""
        sc, it = self.screen()
        self.store.save({"id": sc["id"], "image_id": sc["image_id"], "model": sc["model"], "repeat_of": sc["repeat_of"],
                         "labels": [list(x) for x in self.labels], "sec": round(time.time() - self.t_show, 2),
                         "time": datetime.now().isoformat(timespec="seconds")})
        if self.pos < len(self.view) - 1:
            self.next()

    # ---- hiển thị
    def image(self, show_objects=True):
        sc, it = self.screen()
        if self._img[0] != it["image"]:
            self._img = (it["image"], Image.open(os.path.join(self.samples_root, it["image"])).convert("RGB"))
        im, self.r = render(self._img[1], it["objects"], self.sel()["boxes"], self.labels, show_objects)
        return im

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
                img = gr.Image(type="pil", interactive=False, show_label=False, elem_id="us_img")
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
            return [sess.header(), sess.image(show_objects)] + radio_updates()

        def nav(fn):
            def f(show_objects):
                fn()
                return full(show_objects)
            return f

        def on_models(ms, show_objects):
            sess.set_models(ms)
            return full(show_objects)

        def on_click(show_objects, evt: gr.SelectData):
            sess.click(*evt.index[:2])
            return [sess.image(show_objects)] + radio_updates()

        def make_radio(i):
            def f(v, show_objects):
                sess.set_reasons(i, v)
                return [sess.image(show_objects)] + radio_updates()
            return f

        def make_tog(i):
            def f(show_objects):
                sess.toggle(i)
                return [sess.image(show_objects)] + radio_updates()
            return f

        outs_full = [head, img] + radios
        demo.load(full, [show], outs_full, api_name="show")
        models.input(on_models, [models, show], outs_full, api_name="models")
        img.select(on_click, [show], [img] + radios, api_name="click")
        for i, r_ in enumerate(radios):
            r_.input(make_radio(i), [r_, show], [img] + radios, api_name=f"reasons{i + 1}")
        for i, b in enumerate(togs):
            b.click(make_tog(i), [show], [img] + radios, api_name=f"toggle{i + 1}")
        prev.click(nav(sess.prev), [show], outs_full, api_name="prev")
        nxt.click(nav(sess.next), [show], outs_full, api_name="next")
        save.click(nav(sess.save), [show], outs_full, api_name="save")
        first.click(nav(sess.first_unsaved), [show], outs_full, api_name="first_unsaved")
        show.change(lambda s_: sess.image(s_), [show], [img], api_name="objects")
    return demo, launch_kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="items.json của user_study/build.py")
    ap.add_argument("--samples-root", default="../data/samples")
    ap.add_argument("--ratings", default=None, help="mặc định ratings.jsonl cạnh items.json")
    ap.add_argument("--models", nargs="*", default=None, help="mã model chấm lúc mở (mặc định mọi model; đổi được trên web)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="tạo link công khai https://*.gradio.live (sống tối đa 1 tuần) để người "
                                                          "khác chấm qua trình duyệt, vd chạy trên server")
    a = ap.parse_args()
    # gradio 6.17 dựng bảng mã HTTP ở MỖI request, trong đó đọc `status.HTTP_422_UNPROCESSABLE_ENTITY` mà starlette 1.x đã đổi tên
    # ⇒ một cảnh báo deprecated mỗi lần bấm (request vẫn thành công). Lỗi của thư viện: chỉ bỏ ĐÚNG cảnh báo này.
    warnings.filterwarnings("ignore", message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*")
    import gradio as gr
    sess = Session(a.items, a.samples_root, a.ratings, a.models)
    print(f"{len(sess.screens)} màn ({len(sess.d['models'])} model), đã lưu {len(sess.store.latest)} -> {sess.store.path}",
          flush=True)
    demo, launch_kw = build_ui(sess, gr)
    demo.launch(server_name="127.0.0.1", server_port=a.port, inbrowser=not a.share, share=a.share, **launch_kw)


if __name__ == "__main__":
    main()

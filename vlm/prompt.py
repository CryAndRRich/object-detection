"""Prompt VLM = làm GIỐNG code GPT của tác giả bài Add One Take One nhất có thể (`refs/inference_layout.py` + `refs/example4.py`,
gpt-4.1-mini; người dùng chốt 2026-11-03):
- system = `SYSTEM` nguyên văn; user = `build_user_text_no_ref` nguyên văn (text rồi ảnh), canvas = cỡ ẢNH GỐC, ảnh gửi nguyên file;
- schema `insertion_layout` (`candidates`: name, box, score, …) gửi qua API (`schema: api`), strict False; xếp theo score;
- few-shot = 2 hội thoại của tác giả (`icl: author`, `vlm/icl/author.json`, sửa 1 khoá sai).
Chỉ khác tác giả: K = 30 thay 3 (giao thức ZETA; maxItems 30), thử lại khi 0 box. Backend không nhận schema qua API ⇒ `schema:
prompt` chép schema vào cuối system. `FORMATS` khác `json_pixel` = kiểu box gốc từng model, chỉ để dự phòng.

Message nội bộ: list lượt {"role": system | user | assistant, "text", "image": base64 | None, "mime"}; mỗi backend tự đổi sang API
của nó. `icl: N` = N mẫu CE-130 train (ảnh + lỗ GT); `icl: 0` = zero-shot."""

import base64
import hashlib
import io
import json
import os

from PIL import Image

__all__ = ["PROMPT_VERSION", "FORMATS", "SYSTEM", "USER", "AUTHOR_ICL", "box_schema", "schema_hint", "encode_image", "system_text",
           "uses_hint", "user_text", "answer_text", "author_turns", "train_turns", "build_messages", "prompt_hash"]

PROMPT_VERSION = 3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # object-detection/
AUTHOR_ICL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icl", "author.json")

# Nguyên văn SYSTEM của tác giả; {frame} = dòng 2, {output} = dòng cuối ("Output ONLY JSON matching the schema.")
SYSTEM = """You propose plausible placements for inserting NEW objects into a given background image.
{frame}

CRITICAL (NO reference boxes provided):
- First, visually identify existing instances of the SAME class in the image (if any).
- Infer a typical pixel size (width/height) for that class from those instances.
- Your proposed boxes for that class MUST closely match that typical size (target median; tolerance ±15%).
- If perspective varies, match the size of the instance at a similar vertical position (y-range).

Avoid covering people, faces, text, and main objects. Prefer empty/low-saliency regions.
Keep boxes inside the canvas and minimize overlap with existing objects.
{output}
"""

# Nguyên văn `build_user_text_no_ref`; {ask} = "as boxes [x1,y1,x2,y2]" (json_pixel) hay câu theo định dạng model
USER = ("Canvas size: {w}x{h}.\n"
        "Object(s) to insert: {cls}\n"
        "Return top {k} placement candidates {ask} within this canvas.\n"
        "No reference boxes are given.\n"
        "IMPORTANT: visually infer the typical pixel size (w,h) of existing instances of the SAME class in the image (if any), "
        "and match your proposed box sizes to that (target median; tolerance ±15%). "
        "If perspective varies, match an instance at similar y-position.")

AUTHOR_ASK = "as boxes [x1,y1,x2,y2]"

FORMATS = {
    "json_pixel": {   # MẶC ĐỊNH mọi model — đúng tác giả: pixel canvas, schema `candidates` (box_schema), xếp theo score
        "coord": "pixel",
        "frame": "Return boxes in the given canvas coordinate system as [x1,y1,x2,y2].",
        "ask": AUTHOR_ASK,
        "output": "Output ONLY JSON matching the schema.",
    },
    # Các định dạng dưới = kiểu box gốc từng model (không còn dùng mặc định, giữ để thử nếu pilot `json_pixel` của model đó hỏng)
    "json_norm": {    # InternVL3.5 (eval grounding: [[x1,y1,x2,y2]] / 1000), GLM (box / 1000 trong <|begin_of_box|>)
        "coord": "norm1000",
        "frame": "Return boxes as [x1,y1,x2,y2] with coordinates normalized to 0-1000 (0 = left / top edge, 1000 = right / bottom "
                 "edge of the canvas).",
        "ask": "as boxes [x1,y1,x2,y2] normalized to 0-1000",
        "output": "Output ONLY a JSON list of boxes, from most to least plausible: [[x1,y1,x2,y2], ...].",
    },
    "qwen": {         # Qwen3-VL cookbook 2d_grounding: bbox_2d tương đối 0–1000
        "coord": "norm1000",
        "frame": "Return boxes as bbox_2d [x1,y1,x2,y2] in relative coordinates ranging from 0 to 1000.",
        "ask": "as bbox_2d [x1,y1,x2,y2] in relative 0-1000 coordinates",
        "output": 'Output ONLY JSON, from most to least plausible: [{{"bbox_2d": [x1,y1,x2,y2], "label": "{cls}"}}, ...].',
    },
    "gemini": {       # docs Gemini Robotics-ER (robotics-spatial): khoá y / x / y2 / x2, 0–1000
        "coord": "norm1000",
        "frame": "Return 2D bounding boxes with coordinates normalized to 0-1000.",
        "ask": "as 2D bounding boxes normalized to 0-1000",
        "output": 'The answer should follow the JSON format, from most to least plausible: [{{"label": <label>, "y": <y_min>, '
                  '"x": <x_min>, "y2": <y_max>, "x2": <x_max>}}, ...] where coordinates are normalized to 0-1000.',
    },
    "perceptron": {   # docs Perceptron Mk1.5: thẻ <point_box> (x1,y1) (x2,y2) lưới 0–1000
        "coord": "norm1000",
        "frame": "Mark each proposed placement with a separate labeled bounding box on the 0-1000 image grid.",
        "ask": "as <point_box> boxes on the 0-1000 grid",
        "output": 'Output ONLY the boxes, from most to least plausible, each as <point_box mention="{cls}"> (x1,y1) (x2,y2) '
                  '</point_box>.',
    },
}


def box_schema(k):
    """Schema `insertion_layout` của tác giả (refs/inference_layout.py, gọi với strict False); chỉ đổi maxItems 5 -> k."""
    return {"name": "insertion_layout", "strict": False,
            "schema": {"type": "object", "required": ["candidates"], "additionalProperties": False,
                       "properties": {"candidates": {
                           "type": "array", "minItems": 1, "maxItems": int(k),
                           "items": {"type": "object", "required": ["name", "box", "score"], "additionalProperties": False,
                                     "properties": {"name": {"type": "string"},
                                                    "box": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
                                                    "score": {"type": "number"}, "placement_type": {"type": "string"},
                                                    "reason": {"type": "string"}}}}}}}


def schema_hint(k):
    """Chỉ khi `schema: prompt` (backend không nhận schema qua API): chép schema tác giả vào cuối system."""
    return "Schema: " + json.dumps(box_schema(k)["schema"])


MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp", "GIF": "image/gif"}


def encode_image(path, max_side=None, quality=95):
    """Ảnh -> {"b64", "mime", "sent_wh" (cỡ gửi), "orig_wh", "canvas_wh" (= gốc, như tác giả)}. Mặc định (`max_side` None) gửi NGUYÊN
    byte file như `to_data_url` của tác giả; `max_side` đặt và cạnh dài lớn hơn ⇒ thu giữ tỉ lệ, JPEG `quality`."""
    with Image.open(path) as im:
        W, H = im.size
        s = min(1.0, max_side / max(W, H)) if max_side else 1.0
        if s >= 1.0 and im.format in MIME:
            with open(path, "rb") as f:
                data, mime = f.read(), MIME[im.format]
        else:
            im = im.convert("RGB")
            if s < 1.0:
                im = im.resize((max(1, round(W * s)), max(1, round(H * s))), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality)
            data, mime = buf.getvalue(), "image/jpeg"
        sent = im.size
    return {"b64": base64.b64encode(data).decode(), "mime": mime, "sent_wh": tuple(sent), "orig_wh": (W, H), "canvas_wh": (W, H)}


def system_text(fmt, k=None, hint=False):
    """System chung mọi ảnh / ví dụ như tác giả (không chứa tên lớp: nhãn trong mẫu định dạng = "<object name>"). `hint` (json_pixel
    + `schema: prompt`): thêm dòng schema sau "Output ONLY JSON matching the schema."."""
    f = FORMATS[fmt]
    s = SYSTEM.format(frame=f["frame"], output=f["output"].format(cls="<object name>"))
    return s + schema_hint(k) + "\n" if hint and fmt == "json_pixel" else s


def uses_hint(cfg):
    return cfg["format"] == "json_pixel" and cfg.get("schema", "api") == "prompt"


def user_text(fmt, k, cls, canvas_wh):
    w, h = canvas_wh
    return USER.format(w=w, h=h, cls=cls, k=k, ask=FORMATS[fmt]["ask"])


def answer_text(boxes, fmt, cls, canvas_wh, scores=None, extra=None):
    """Box xyxy pixel canvas -> câu trả lời mẫu đúng định dạng `fmt` (lượt assistant few-shot). `extra`: các khoá khác của candidate
    tác giả (placement_type, reason) — chỉ dùng ở json_pixel."""
    W, H = canvas_wh
    if fmt == "json_pixel":
        scores = scores or [round(1.0 - i / (len(boxes) + 1), 2) for i in range(len(boxes))]
        cands = [{"name": cls, "box": [round(v) for v in b], "score": s, **((extra or [{}] * len(boxes))[i])}
                 for i, (b, s) in enumerate(zip(boxes, scores))]
        return json.dumps({"candidates": cands}, ensure_ascii=False)
    bs = [[round(b[0] * 1000 / W), round(b[1] * 1000 / H), round(b[2] * 1000 / W), round(b[3] * 1000 / H)] for b in boxes]
    if fmt == "json_norm":
        return json.dumps(bs)
    if fmt == "qwen":
        return json.dumps([{"bbox_2d": b, "label": cls} for b in bs])
    if fmt == "gemini":
        return json.dumps([{"label": cls, "y": b[1], "x": b[0], "y2": b[3], "x2": b[2]} for b in bs])
    return " ".join(f'<point_box mention="{cls}"> ({b[0]},{b[1]}) ({b[2]},{b[3]}) </point_box>' for b in bs)


def _load_author():
    with open(AUTHOR_ICL) as f:
        return json.load(f)


def author_turns(fmt, encode):
    """2 ví dụ hội thoại của tác giả (lượt user có ảnh, lượt sửa chỉ text, lượt assistant) theo định dạng `fmt`. Text user giữ nguyên
    văn; định dạng khác json_pixel thì thay "as boxes [x1,y1,x2,y2]" bằng câu `ask` của định dạng. `encode(path)` -> ảnh mã hoá."""
    turns = []
    for ex in _load_author()["examples"]:
        im = encode(os.path.join(ROOT, ex["image"]))
        if tuple(im["canvas_wh"]) != tuple(ex["canvas"]):
            raise ValueError(f"ảnh ví dụ {ex['image']} cỡ {im['canvas_wh']} khác canvas {ex['canvas']} trong {AUTHOR_ICL}")
        for i, t in enumerate(ex["turns"]):
            if t["role"] == "user":
                text = t["text"] if fmt == "json_pixel" else t["text"].replace(AUTHOR_ASK, FORMATS[fmt]["ask"])
                turns.append({"role": "user", "text": text, "image": im["b64"] if i == 0 else None, "mime": im["mime"]})
            else:
                c = t["candidates"]
                extra = [{k: v for k, v in x.items() if k not in ("name", "box", "score")} for x in c]
                turns.append({"role": "assistant", "image": None, "mime": None,
                              "text": answer_text([x["box"] for x in c], fmt, ex["class"], ex["canvas"], [x["score"] for x in c], extra)})
    return turns


def train_turns(fmt, k, examples):
    """`examples` = [(mẫu CE-130 train, ảnh mã hoá)]: lượt user theo `USER` + lượt assistant = lỗ GT (`holes`)."""
    turns = []
    for ex, im in examples:
        turns.append({"role": "user", "text": user_text(fmt, k, ex["class"], im["canvas_wh"]), "image": im["b64"], "mime": im["mime"]})
        turns.append({"role": "assistant", "text": answer_text(ex["holes"], fmt, ex["class"], im["canvas_wh"]), "image": None,
                      "mime": None})
    return turns


def build_messages(sample, image, cfg, examples=()):
    """`image` = `encode_image(sample["image"])`; `examples` = lượt few-shot đã dựng (`author_turns` / `train_turns`). Thứ tự như
    tác giả: system, ví dụ, lượt hiện tại (text rồi ảnh)."""
    return ([{"role": "system", "text": system_text(cfg["format"], cfg["k"], uses_hint(cfg)), "image": None, "mime": None}]
            + list(examples)
            + [{"role": "user", "text": user_text(cfg["format"], cfg["k"], sample["class"], image["canvas_wh"]), "image": image["b64"],
                "mime": image["mime"]}])


HASH_KEYS = ("backend", "model", "format", "coord", "order", "k", "icl", "image", "reasoning", "sampling", "max_tokens", "engine",
             "chat_template_kwargs", "seed", "schema")
HASH_SUB = {"openrouter": ("structured", "detail", "provider"), "gemini": ("thinking_level", "json")}


def prompt_hash(cfg):
    """Hash những gì đổi câu trả lời (model, prompt, ví dụ few-shot, sampling, ảnh…) — không gồm concurrency / giá / rpm / bộ nhớ.
    Đổi ⇒ resume hỏi lại."""
    key = {"v": PROMPT_VERSION, "system": SYSTEM, "user": USER, "formats": FORMATS[cfg["format"]],
           **{k: cfg.get(k) for k in HASH_KEYS},
           **{f"{s}.{k}": (cfg.get(s) or {}).get(k) for s, ks in HASH_SUB.items() for k in ks}}
    if cfg.get("icl") == "author":
        key["author_icl"] = _load_author()
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]

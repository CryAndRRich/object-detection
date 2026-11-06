"""Prompt VLM = làm GIỐNG code GPT của tác giả bài Add One Take One nhất có thể (`refs/inference_layout.py` + `refs/example4.py`,
gpt-4.1-mini; người dùng chốt 2026-11-03):
- system = `SYSTEM` nguyên văn; user = `build_user_text_no_ref` nguyên văn (text rồi ảnh), canvas = cỡ ẢNH GỐC, ảnh gửi nguyên file;
- schema `insertion_layout` (`candidates`: name, box, score, …) gửi qua API, strict False; xếp theo score;
- few-shot = 2 hội thoại của tác giả (`icl: author`, `vlm/icl/author.json`, sửa 1 khoá sai); `icl: 0` = zero-shot.
Chỉ khác tác giả: K = 30 thay 3 (giao thức ZETA; maxItems 30), thử lại khi 0 box.

ZETA4 RePlan (`prompt: replan`): prompt planner của RePlan (`refs/repos/RePlan` @ 6c9b12f, `RePlanPipeline.build_messages`) — MỘT
lượt user, ảnh trước rồi text = template `prompts/replan.txt` (nguyên byte) + "\n" + lệnh (`instruction`, mặc định "Add one more
{cls} to the image."); không system, không few-shot, không schema.

Message nội bộ: list lượt {"role": system | user | assistant, "text", "image": base64 | None, "mime", "image_first" (tuỳ chọn: ảnh
trước text)}; mỗi backend tự đổi sang API của nó."""

import base64
import hashlib
import io
import json
import os

from PIL import Image

__all__ = ["PROMPT_VERSION", "FORMAT", "SYSTEM", "USER", "AUTHOR_ICL", "box_schema", "encode_image", "system_text", "user_text",
           "author_turns", "build_messages", "prompt_hash", "REPLAN_INSTRUCTION", "replan_template", "replan_text"]

PROMPT_VERSION = 3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # object-detection/
AUTHOR_ICL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icl", "author.json")
REPLAN_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "replan.txt")
REPLAN_INSTRUCTION = "Add one more {cls} to the image."

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

# Nguyên văn `build_user_text_no_ref`; {ask} = "as boxes [x1,y1,x2,y2]"
USER = ("Canvas size: {w}x{h}.\n"
        "Object(s) to insert: {cls}\n"
        "Return top {k} placement candidates {ask} within this canvas.\n"
        "No reference boxes are given.\n"
        "IMPORTANT: visually infer the typical pixel size (w,h) of existing instances of the SAME class in the image (if any), "
        "and match your proposed box sizes to that (target median; tolerance ±15%). "
        "If perspective varies, match an instance at similar y-position.")

# Định dạng của tác giả (tên cũ `json_pixel`): pixel canvas, schema `candidates` (box_schema), xếp theo score. Nội dung (cả khoá
# "coord") giữ nguyên byte vì nằm trong `prompt_hash` — đổi là mất câu trả lời pilot đã có.
FORMAT = {
    "coord": "pixel",
    "frame": "Return boxes in the given canvas coordinate system as [x1,y1,x2,y2].",
    "ask": "as boxes [x1,y1,x2,y2]",
    "output": "Output ONLY JSON matching the schema.",
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


def system_text():
    """System chung mọi ảnh / ví dụ như tác giả (không chứa tên lớp)."""
    return SYSTEM.format(frame=FORMAT["frame"], output=FORMAT["output"])


def user_text(k, cls, canvas_wh):
    w, h = canvas_wh
    return USER.format(w=w, h=h, cls=cls, k=k, ask=FORMAT["ask"])


def _load_author():
    with open(AUTHOR_ICL) as f:
        return json.load(f)


def author_turns(encode):
    """2 ví dụ hội thoại của tác giả (lượt user có ảnh, lượt sửa chỉ text, lượt assistant = JSON `candidates`), text nguyên văn.
    `encode(path)` -> ảnh mã hoá."""
    turns = []
    for ex in _load_author()["examples"]:
        im = encode(os.path.join(ROOT, ex["image"]))
        if tuple(im["canvas_wh"]) != tuple(ex["canvas"]):
            raise ValueError(f"ảnh ví dụ {ex['image']} cỡ {im['canvas_wh']} khác canvas {ex['canvas']} trong {AUTHOR_ICL}")
        for i, t in enumerate(ex["turns"]):
            if t["role"] == "user":
                turns.append({"role": "user", "text": t["text"], "image": im["b64"] if i == 0 else None, "mime": im["mime"]})
            else:
                cands = [{"name": ex["class"], "box": [round(v) for v in x["box"]], "score": x["score"],
                          **{k: v for k, v in x.items() if k not in ("name", "box", "score")}} for x in t["candidates"]]
                turns.append({"role": "assistant", "text": json.dumps({"candidates": cands}, ensure_ascii=False), "image": None,
                              "mime": None})
    return turns


def replan_template():
    with open(REPLAN_TEMPLATE, encoding="utf-8") as f:
        return f.read()


def replan_text(cfg, cls):
    """Text lượt user của planner RePlan: template + "\n" + lệnh (như `RePlanPipeline.build_messages`)."""
    return replan_template() + "\n" + (cfg.get("instruction") or REPLAN_INSTRUCTION).format(cls=cls)


def build_messages(sample, image, cfg, examples=()):
    """`image` = `encode_image(sample["image"])`; `examples` = lượt few-shot (`author_turns`). Thứ tự như tác giả: system, ví dụ,
    lượt hiện tại (text rồi ảnh). `prompt: replan` -> một lượt user (ảnh rồi text) của planner RePlan."""
    if cfg.get("prompt") == "replan":
        return [{"role": "user", "text": replan_text(cfg, sample["class"]), "image": image["b64"], "mime": image["mime"],
                 "image_first": True}]
    return ([{"role": "system", "text": system_text(), "image": None, "mime": None}]
            + list(examples)
            + [{"role": "user", "text": user_text(cfg["k"], sample["class"], image["canvas_wh"]), "image": image["b64"],
                "mime": image["mime"]}])


HASH_KEYS = ("backend", "model", "k", "icl", "image", "reasoning", "sampling", "max_tokens", "engine", "chat_template_kwargs", "seed")
# Khoá config đã bỏ 2026-11-03 (format / coord / order / schema, openrouter.structured, gemini.json) vẫn vào hash với giá trị cố
# định như mọi config lúc còn dùng ⇒ hash không đổi, dùng lại câu trả lời đã có.
HASH_FIXED = {"format": "json_pixel", "coord": None, "order": None, "schema": "api", "openrouter.structured": None,
              "gemini.json": None}
# OpenAI trực tiếp và OpenRouter (chỉ chuyển tiếp tới OpenAI) chạy CÙNG model `openai/...` ⇒ cùng hash: dùng lại câu trả lời pilot,
# đổi / chuyển đường giữa chừng không hỏi lại (mỗi câu trả lời ghi `provider`).
HASH_ALIAS = {"backend": {"openai": "openrouter"}}
# "gemini.thinking_level" giữ dù backend Gemini đã bỏ (2026-11-06): mọi config có khoá này (= None) trong hash, bỏ là đổi hash mọi model.
HASH_SUB = {"openrouter": ("detail", "provider"), "gemini": ("thinking_level",)}
# Khoá `engine` vLLM chỉ đổi tốc độ / bộ nhớ, không đổi câu trả lời ⇒ ngoài hash (đổi được giữa các phiên, resume vẫn dùng lại).
ENGINE_FREE = ("max_num_seqs", "tensor_parallel_size", "enforce_eager", "limit_mm_per_prompt", "gpu_memory_utilization",
               "max_num_batched_tokens", "disable_log_stats")


def prompt_hash(cfg):
    """Hash những gì đổi câu trả lời (model, prompt, ví dụ few-shot, sampling, ảnh…) — không gồm concurrency / giá / rpm / bộ nhớ.
    Đổi ⇒ resume hỏi lại."""
    key = {"v": PROMPT_VERSION, "system": SYSTEM, "user": USER, "formats": FORMAT, **HASH_FIXED,
           **{k: cfg.get(k) for k in HASH_KEYS},
           **{k: m.get(cfg.get(k), cfg.get(k)) for k, m in HASH_ALIAS.items()},
           "engine": {k: v for k, v in cfg["engine"].items() if k not in ENGINE_FREE} if cfg.get("engine") else cfg.get("engine"),
           **{f"{s}.{k}": (cfg.get(s) or {}).get(k) for s, ks in HASH_SUB.items() for k in ks}}
    if cfg.get("icl") == "author":
        key["author_icl"] = _load_author()
    if cfg.get("prompt") == "replan":                      # chỉ config RePlan có các khoá này ⇒ hash model cũ không đổi
        key.update(prompt="replan", replan_template=replan_template(), instruction=cfg.get("instruction") or REPLAN_INSTRUCTION,
                   replan=cfg.get("replan"))
    return hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]

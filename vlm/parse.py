"""Câu trả lời VLM -> box xyxy PIXEL ẢNH GỐC (model trả pixel canvas đã khai trong prompt) + `score` model tự cho. Thứ tự thử: JSON
(schema `candidates` của tác giả / list 4 số / dict khoá box) -> regex nhóm 4 số. Box xếp theo `score` giảm dần (bằng nhau giữ thứ tự
model trả; box không có score xếp sau, giữ thứ tự) ⇒ box[0] = box confidence cao nhất = box chấm chính của ZETA5 / 6. Hợp lệ: 4 số
hữu hạn, diện tích > 0 (cạnh ngược thì đảo). KHÔNG kẹp vào ảnh — chấm tự đo `in_image`. `fill` đưa về đúng K box: thiếu thì lặp vòng
(box[0] và min C-NLL không đổi), 0 box thì box cả ảnh (bị phạt, đếm `n_fail`)."""

import json
import math
import re

__all__ = ["strip_think", "extract", "extract_scored", "to_pixel", "parse_candidates", "parse_scored", "parse_boxes", "fill"]

_NUM = r"(-?\d+(?:\.\d+)?)"
_QUAD = re.compile(rf"[\[(]\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*[\])]")
_BOX_KEYS = ("bbox_2d", "bbox", "box", "boxes")


def strip_think(text):
    """Bỏ phần suy nghĩ: khối <think>…</think>; chỉ có </think> (template tự mở thẻ) thì bỏ mọi thứ trước nó."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.split("</think>")[-1]


def _json_blocks(text):
    """Các khối JSON đọc được: khối ```json```, rồi đoạn từ dấu [ / { đầu tới ] / } cuối (dấu mở sớm hơn = khối ngoài, thử trước)."""
    out = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.S)]
    spans = [(text.find(o), text.rfind(c)) for o, c in (("[", "]"), ("{", "}"))]
    for i, j in sorted(s for s in spans if 0 <= s[0] < s[1]):
        out.append(text[i:j + 1])
    for s in out:
        try:
            yield json.loads(s)
        except ValueError:
            continue


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


_EXTRA = ("name", "placement_type", "reason")


def _score(c):
    v = c.get("score") if isinstance(c, dict) else None
    return float(v) if _num(v) and math.isfinite(v) else None


def _walk(obj, cand=None):
    """Gom (box, candidate) từ JSON: list 4 số; dict có x/y/x2/y2, x1..y2 / xmin..ymax, hay khoá bbox_2d / bbox / box / boxes …
    `{"candidates": [{name, box, score, …}]}` (schema tác giả): box mang candidate của nó, xếp theo `score` (số hữu hạn; khác ⇒ None)
    giảm dần như tác giả; candidate sai khoá vẫn lấy được box từ giá trị."""
    if isinstance(obj, (list, tuple)) and len(obj) == 4 and all(_num(x) for x in obj):
        return [([float(x) for x in obj], cand)]
    if isinstance(obj, dict) and isinstance(obj.get("candidates"), list):
        out = [p for c in obj["candidates"] for p in _walk(c, c if isinstance(c, dict) else None)]
        return sorted(out, key=lambda p: (_score(p[1]) is None, -(_score(p[1]) or 0.0)))     # sorted ổn định
    if isinstance(obj, dict):
        for keys in (("x", "y", "x2", "y2"), ("x1", "y1", "x2", "y2"), ("xmin", "ymin", "xmax", "ymax"),
                     ("x_min", "y_min", "x_max", "y_max")):
            if all(_num(obj.get(k)) for k in keys):
                return [([float(obj[k]) for k in keys], cand)]
        for k in _BOX_KEYS:
            if k in obj:
                return _walk(obj[k], cand)
        return [p for v in obj.values() for p in _walk(v, cand)]
    if isinstance(obj, (list, tuple)):
        return [p for v in obj for p in _walk(v, cand)]
    return []


def _extract(text):
    text = strip_think(text).replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    for obj in _json_blocks(text):
        pairs = _walk(obj)
        if pairs:
            return pairs
    return [([float(g) for g in m.groups()], None) for m in _QUAD.finditer(text)]


def extract_scored(text):
    """Text -> [(box xyxy pixel canvas, score | None)], chưa lọc, đã xếp theo score."""
    return [(b, _score(c)) for b, c in _extract(text)]


def extract(text):
    """Text -> box xyxy pixel canvas, chưa lọc."""
    return [b for b, _ in _extract(text)]


def to_pixel(boxes, canvas_wh, orig_wh):
    """Pixel canvas đã khai trong prompt (= ảnh gốc, như tác giả) -> pixel ảnh gốc."""
    sx, sy = orig_wh[0] / canvas_wh[0], orig_wh[1] / canvas_wh[1]
    return [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]


def _valid(b):
    if not all(math.isfinite(v) for v in b):
        return None
    x1, x2 = sorted((b[0], b[2]))
    y1, y2 = sorted((b[1], b[3]))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


COORDS = ("pixel", "norm1000")


def parse_candidates(text, cfg, canvas_wh, orig_wh):
    """-> [{"box": xyxy pixel ảnh gốc, "score": số | None, + name / placement_type / reason nếu model trả}] — box hợp lệ, tối đa K,
    xếp theo score giảm dần (phần tử 0 = confidence cao nhất). `cfg["coord"]`: "pixel" (mặc định, như prompt tác giả: pixel canvas) |
    "norm1000" (model trả thang 0–1000 theo cách nó được huấn luyện — Qwen3-VL / LLaVA-OV-2; chỉ đổi cách ĐỌC, prompt giữ nguyên)."""
    coord = cfg.get("coord") or "pixel"
    if coord not in COORDS:
        raise ValueError(f"coord {coord!r} không thuộc {COORDS}")
    if coord == "norm1000":
        canvas_wh = (1000, 1000)
    pairs = _extract(text)
    out = []
    for b, c in zip(to_pixel([b for b, _ in pairs], canvas_wh, orig_wh), (c for _, c in pairs)):
        v = _valid(b)
        if v is not None:
            out.append({"box": v, "score": _score(c), **{k: c[k] for k in _EXTRA if isinstance(c, dict) and isinstance(c.get(k), str)}})
    return out[:cfg["k"]]


def parse_scored(text, cfg, canvas_wh, orig_wh):
    """-> (box hợp lệ xyxy pixel ảnh gốc, score | None) — tối đa K, xếp theo score giảm dần (box[0] = confidence cao nhất)."""
    cands = parse_candidates(text, cfg, canvas_wh, orig_wh)
    return [c["box"] for c in cands], [c["score"] for c in cands]


def parse_boxes(text, cfg, canvas_wh, orig_wh):
    """-> box hợp lệ xyxy pixel ảnh gốc (tối đa K, xếp theo score)."""
    return parse_scored(text, cfg, canvas_wh, orig_wh)[0]


def fill(boxes, k, orig_wh):
    """-> đúng k box: lặp vòng box hợp lệ; rỗng -> box cả ảnh [0, 0, W, H] × k."""
    if not boxes:
        W, H = orig_wh
        return [[0.0, 0.0, float(W), float(H)]] * k
    return [list(boxes[i % len(boxes)]) for i in range(k)]

"""Câu trả lời VLM -> box xyxy PIXEL ẢNH GỐC. Thứ tự thử: parser theo `format` -> JSON chung (list 4 số / dict khoá box / schema
`candidates` của tác giả, xếp theo score) -> regex
nhóm 4 số. Hợp lệ: 4 số hữu hạn, diện tích > 0 (cạnh ngược thì đảo). KHÔNG kẹp vào ảnh — chấm tự đo `in_image`. `fill` đưa về đúng K
box: thiếu thì lặp vòng (min C-NLL không đổi), 0 box thì box cả ảnh (bị phạt, đếm `n_fail`)."""

import json
import math
import re

from vlm.prompt import FORMATS

__all__ = ["strip_think", "extract", "to_pixel", "parse_boxes", "fill"]

_NUM = r"(-?\d+(?:\.\d+)?)"
_QUAD = re.compile(rf"[\[(]\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*[\])]")
_PBOX = re.compile(rf"<point_box[^>]*>\s*\(\s*{_NUM}\s*,\s*{_NUM}\s*\)\s*\(\s*{_NUM}\s*,\s*{_NUM}\s*\)\s*</point_box>")
_BOX_KEYS = ("bbox_2d", "box_2d", "bbox", "box", "boxes")


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


def _is_quad(v):
    return isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _walk(obj, order):
    """Gom box từ JSON: list 4 số; dict có y/x/y2/x2 (Gemini), x1..y2 / xmin..ymax, hay khoá bbox_2d / box_2d (Gemini: ymin trước) …
    `{"candidates": [{name, box, score}]}` (schema tác giả): xếp theo score giảm dần như tác giả (mọi candidate có score; giữ thứ tự
    khi bằng nhau), candidate sai khoá vẫn lấy được box từ giá trị."""
    if _is_quad(obj):
        a = [float(x) for x in obj]
        return [[a[1], a[0], a[3], a[2]] if order == "yxyx" else a]
    if isinstance(obj, dict) and isinstance(obj.get("candidates"), list):
        c = obj["candidates"]
        if c and all(isinstance(x, dict) and _num(x.get("score")) for x in c):
            c = sorted(c, key=lambda x: -x["score"])
        return [b for v in c for b in _walk(v, order)]
    if isinstance(obj, dict):
        for keys in (("x", "y", "x2", "y2"), ("x1", "y1", "x2", "y2"), ("xmin", "ymin", "xmax", "ymax"),
                     ("x_min", "y_min", "x_max", "y_max")):
            if all(_num(obj.get(k)) for k in keys):
                return [[float(obj[k]) for k in keys]]
        for k in _BOX_KEYS:
            if k in obj:
                return _walk(obj[k], "yxyx" if k == "box_2d" else order)
        return [b for v in obj.values() for b in _walk(v, order)]
    if isinstance(obj, (list, tuple)):
        return [b for v in obj for b in _walk(v, order)]
    return []


def extract(text, fmt, order="xyxy"):
    """Text -> box xyxy trong hệ toạ độ của model (0–1000 hay pixel canvas), chưa lọc."""
    text = strip_think(text).replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    if fmt == "perceptron":
        boxes = [[float(g) for g in m.groups()] for m in _PBOX.finditer(text)]
        if boxes:
            return boxes
    for obj in _json_blocks(text):
        boxes = _walk(obj, order)
        if boxes:
            return boxes
    quads = [[float(g) for g in m.groups()] for m in _QUAD.finditer(text)]
    return [[q[1], q[0], q[3], q[2]] for q in quads] if order == "yxyx" else quads


def to_pixel(boxes, coord, canvas_wh, orig_wh):
    """0–1000 -> ảnh gốc; pixel -> theo canvas đã khai trong prompt (= ảnh gốc, như tác giả)."""
    W, H = orig_wh
    if coord == "norm1000":
        sx, sy = W / 1000.0, H / 1000.0
    elif coord == "pixel":
        sx, sy = W / canvas_wh[0], H / canvas_wh[1]
    else:
        raise ValueError(f"coord {coord!r} không thuộc norm1000 | pixel")
    return [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]


def _valid(b):
    if not all(math.isfinite(v) for v in b):
        return None
    x1, x2 = sorted((b[0], b[2]))
    y1, y2 = sorted((b[1], b[3]))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def parse_boxes(text, cfg, canvas_wh, orig_wh):
    """-> box hợp lệ xyxy pixel ảnh gốc (tối đa K, theo thứ tự model trả)."""
    coord = cfg.get("coord") or FORMATS[cfg["format"]]["coord"]
    raw = extract(text, cfg["format"], cfg.get("order", "xyxy"))
    boxes = [v for v in (_valid(b) for b in to_pixel(raw, coord, canvas_wh, orig_wh)) if v is not None]
    return boxes[:cfg["k"]]


def fill(boxes, k, orig_wh):
    """-> đúng k box: lặp vòng box hợp lệ; rỗng -> box cả ảnh [0, 0, W, H] × k."""
    if not boxes:
        W, H = orig_wh
        return [[0.0, 0.0, float(W), float(H)]] * k
    return [list(boxes[i % len(boxes)]) for i in range(k)]

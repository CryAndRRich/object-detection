"""Câu trả lời VLM -> box xyxy PIXEL ẢNH GỐC (model trả pixel canvas đã khai trong prompt) + `score` model tự cho. Thứ tự thử: JSON
(schema `candidates` của tác giả / list 4 số / dict khoá box) -> regex nhóm 4 số. Box xếp theo `score` giảm dần (bằng nhau giữ thứ tự
model trả; box không có score xếp sau, giữ thứ tự) ⇒ box[0] = box confidence cao nhất = box chấm chính của ZETA5 / 6. Hợp lệ: 4 số
hữu hạn, diện tích > 0 (cạnh ngược thì đảo). KHÔNG kẹp vào ảnh — chấm tự đo `in_image`. `fill` đưa về đúng K box: thiếu thì lặp vòng
(box[0] và min C-NLL không đổi), 0 box thì box cả ảnh (bị phạt, đếm `n_fail`).

ZETA4 RePlan (`prompt: replan`): `text` = các câu trả lời của planner nối bằng `REPLAN_SEP` (câu 0 = greedy, sau đó các câu sample).
Mỗi câu -> box `bbox_2d` của region "thêm" đầu tiên trong `<region>[...]</region>` (khối đầu tiên, `json.loads` như
`RePlanPipeline.extract_bbox_from_response`; bỏ region có hint bắt đầu "keep" — template mẫu của tác giả dùng chúng để khoanh vật GIỮ
nguyên). Box câu greedy có score 1 (= box chính), box câu sample không có score (xếp sau, theo thứ tự). ZETA8 (prompt tác giả + khối
`samples`): cùng cách, mỗi câu lấy candidate đầu (score cao nhất) của JSON `candidates`. Toạ độ `coord: qwen_resized`
= pixel ảnh sau `smart_resize` của Qwen2.5-VL (bội số 28, giới hạn số pixel) -> đổi về pixel ảnh gốc."""

import json
import math
import re

REPLAN_SEP = "\n<|replan_sample|>\n"

_NUM = r"(-?\d+(?:\.\d+)?)"
_QUAD = re.compile(rf"[\[(]\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*[\])]")
_BOX_KEYS = ("bbox_2d", "bbox", "box", "boxes")


def strip_think(text):
    """Bỏ phần suy nghĩ: khối <think>…</think>; chỉ có </think> (template tự mở thẻ) thì bỏ mọi thứ trước nó."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.split("</think>")[-1]


def _json_blocks(text, repair_tries=80):
    """Các khối JSON đọc được: khối ```json```; rồi với từng dấu mở [ / { đầu tiên (sớm hơn = khối ngoài, thử trước): đoạn tới dấu đóng
    cuối, khối JSON hoàn chỉnh đầu tiên (bỏ phần thừa phía sau), rồi SỬA khối bị cắt — cắt ở `}` đóng một candidate (từ cuối, tối đa
    `repair_tries` chỗ) rồi đóng `]}`. GLM 2026-11-04: 25 / 7.385 câu viết đủ 30 candidate rồi trôi sang khoảng trắng / chữ rác tới hết
    max_tokens, không đóng mảng."""
    out = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.S)]
    for s in out:
        try:
            yield json.loads(s)
        except ValueError:
            continue
    dec = json.JSONDecoder()
    spans = sorted(s for s in ((text.find(o), text.rfind(c)) for o, c in (("[", "]"), ("{", "}"))) if 0 <= s[0] < s[1])
    for i, j in spans:
        for parse in (lambda: json.loads(text[i:j + 1]), lambda: dec.raw_decode(text, i)[0]):
            try:
                yield parse()
            except ValueError:
                pass
        if text[i] == "{" and '"candidates"' in text[i:i + 40]:
            ends = [m.start() for m in re.finditer(r"\}", text[i:])][::-1][:repair_tries]
            for e in ends:
                try:
                    yield json.loads(text[i:i + e + 1] + "]}")
                    break
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


def replan_box(text):
    """1 câu trả lời planner RePlan -> bbox_2d (4 số) của region "thêm" đầu tiên (hint không bắt đầu "keep"), không có -> None."""
    m = re.search(r"<region>(.*?)</region>", text or "", re.S)
    if not m:
        return None
    try:
        regions = json.loads(m.group(1).strip())
    except ValueError:
        return None
    for r in regions if isinstance(regions, list) else []:
        if not isinstance(r, dict) or str(r.get("hint") or "").strip().lower().startswith("keep"):
            continue
        b = r.get("bbox_2d")
        if isinstance(b, list) and len(b) == 4 and all(_num(x) for x in b):
            return [float(x) for x in b]
    return None


def sample_cfg(cfg):
    """Khối "greedy + N câu sample" của config: `samples` (mọi prompt, ZETA8) | `replan` (tên cũ, ZETA4 / 4.1) | None."""
    return cfg.get("samples") or cfg.get("replan")


def _first_candidate(text):
    pairs = _extract(text)
    return pairs[0][0] if pairs else None


def _extract_samples(text, one):
    """`text` = câu greedy + các câu sample nối bằng REPLAN_SEP -> mỗi câu MỘT box (`one`); box câu greedy score 1, câu sample None."""
    out = []
    for i, t in enumerate((text or "").split(REPLAN_SEP)):
        b = one(t)
        if b is not None:
            out.append((b, {"score": 1.0} if i == 0 else None))
    return out


def qwen_resized_wh(orig_wh, factor=28, min_pixels=3136, max_pixels=12845056):
    """(w, h) ảnh mà Qwen2.5-VL thực sự thấy (`smart_resize`: cạnh làm tròn bội số `factor`, số pixel trong [min, max]); mặc định =
    `preprocessor_config.json` của RePlan-Qwen2.5-VL-7B (min 3136, max 12845056) — box `bbox_2d` model trả theo pixel ảnh này."""
    w, h = orig_wh
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb, wb = max(factor, math.floor(h / beta / factor) * factor), max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return wb, hb


COORDS = ("pixel", "norm1000", "qwen_resized")


def parse_candidates(text, cfg, canvas_wh, orig_wh):
    """-> [{"box": xyxy pixel ảnh gốc, "score": số | None, + name / placement_type / reason nếu model trả}] — box hợp lệ, tối đa K,
    xếp theo score giảm dần (phần tử 0 = confidence cao nhất). `cfg["coord"]`: "pixel" (mặc định, như prompt tác giả: pixel canvas) |
    "norm1000" (model trả thang 0–1000 theo cách nó được huấn luyện — Qwen3-VL / LLaVA-OV-2; chỉ đổi cách ĐỌC, prompt giữ nguyên)."""
    coord = cfg.get("coord") or "pixel"
    if coord not in COORDS:
        raise ValueError(f"coord {coord!r} không thuộc {COORDS}")
    if coord == "norm1000":
        canvas_wh = (1000, 1000)
    elif coord == "qwen_resized":
        canvas_wh = qwen_resized_wh(orig_wh, **(cfg.get("smart_resize") or {}))
    if cfg.get("prompt") == "replan":
        pairs = _extract_samples(text, replan_box)
    elif sample_cfg(cfg):
        pairs = _extract_samples(text, _first_candidate)
    else:
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


def fill(boxes, k, orig_wh):
    """-> đúng k box: lặp vòng box hợp lệ; rỗng -> box cả ảnh [0, 0, W, H] × k."""
    if not boxes:
        W, H = orig_wh
        return [[0.0, 0.0, float(W), float(H)]] * k
    return [list(boxes[i % len(boxes)]) for i in range(k)]

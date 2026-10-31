"""Nguồn dữ liệu của web rà box (`app.py`): mỗi hàm -> list MÀN, mỗi màn một dict

  id        mã màn (khoá dòng trong file kết quả)
  title     markdown đầu màn ("" = không hiện)
  views     [(nhãn, đường dẫn ảnh)] — cùng kích thước, toạ độ box = pixel ảnh gốc; ô "Image" / phím v đổi qua lại
  groups    [{key, color, boxes [[x1,y1,x2,y2]…], tags {i: nhãn luôn hiện}, gone {i: view đầu tiên mà vật đã bị xoá (vẽ mờ)},
              share: khoá dùng chung nhãn giữa các màn | None,
              tuỳ chọn: click (False = không bấm được), dedup (False = không lọc trùng), width / sel_width / opacity (viền),
              scores (C-NLL từng box: chế độ chọn CHỌN SẴN `preselect` trên màn chưa lưu)}]
            chỉ số i = vị trí trong danh sách box GỐC của file chú thích (kết quả ghi theo chỉ số này)
  meta      (tuỳ chọn) dict ghi kèm vào dòng kết quả (chế độ chọn)
  group     (tuỳ chọn) khoá chia việc giữa người rà (`app.split_screens`): mọi màn cùng group về MỘT người (mặc định = id)
"""

import glob
import json
import os
import re

import numpy as np
from PIL import Image

from ce_localization.data.cocount import clean_class, load_review, read_cocount
from ce_localization.data.turns import read_annotation
from ce_localization.data.turns import TurnIndex, assign_removed
from ce_localization.utils.box_ops_np import box_iou

__all__ = ["SOURCES", "COLORS", "DEDUP_IOU", "PRESELECT_K", "PRESELECT_IOU", "dedup_boxes", "preselect", "cocount_screens",
           "ce130_screens", "ce130_pred_screens", "cocount_pred_screens"]

DEDUP_IOU = 0.9          # lọc trùng trước khi rà (người dùng 2026-10-21): IoU > 0,9 ⇒ diện tích lệch < 10 % (IoU <= nhỏ / lớn)
PRESELECT_K, PRESELECT_IOU = 5, 0.5      # chọn sẵn (người dùng 2026-10-30): top 5 C-NLL sau NMS IoU 0,5
COLORS = {"positive": "#1e90ff", "negative": "#ff8c00", "objects": "#1e90ff", "holes": "#ff00c8", "model": "#ff1a1a"}


def _boxes(x):
    return np.asarray(x, dtype=np.float64).reshape(-1, 4)


def dedup_boxes(boxes, present=None, thr=DEDUP_IOU):
    """Lọc box gần trùng (người dùng 2026-10-21): trong các box `present` (None = mọi box), cụm = thành phần liên thông của đồ thị
    IoU > thr (IoU > 0,9 đã kéo theo diện tích xấp xỉ bằng nhau); mỗi cụm giữ MỘT box — box thứ ⌊(k−1)/2⌋ theo diện tích tăng dần
    (2 box: box nhỏ hơn; >= 3: trung vị, số chẵn lấy trung vị dưới; diện tích bằng nhau: chỉ số nhỏ trước). -> set chỉ số bị lọc."""
    b = _boxes(boxes)
    idx = np.array(sorted(range(len(b)) if present is None else present), dtype=int)
    if len(idx) < 2:
        return set()
    sub = b[idx]
    iou = np.nan_to_num(box_iou(sub, sub)[0])
    np.fill_diagonal(iou, 0)
    adj = iou > thr
    area = np.clip(sub[:, 2] - sub[:, 0], 0, None) * np.clip(sub[:, 3] - sub[:, 1], 0, None)
    seen, out = np.zeros(len(idx), dtype=bool), set()
    for s0 in range(len(idx)):
        if seen[s0] or not adj[s0].any():
            continue
        comp, stack = [], [s0]
        seen[s0] = True
        while stack:
            k = stack.pop()
            comp.append(k)
            for j in np.flatnonzero(adj[k] & ~seen):
                seen[j] = True
                stack.append(int(j))
        comp.sort(key=lambda k: (area[k], idx[k]))
        keep = comp[(len(comp) - 1) // 2]
        out |= {int(idx[k]) for k in comp if k != keep}
    return out


def preselect(boxes, scores, hidden=(), k=PRESELECT_K, iou=PRESELECT_IOU):
    """Box CHỌN SẴN của màn chọn (người dùng 2026-10-30): k box C-NLL thấp nhất sau NMS — duyệt theo C-NLL tăng dần, bỏ box IoU > `iou`
    với box đã giữ (đúng `engine.add_eval.select_nms`, viết lại để app không phải nạp model) — trong các box CÒN HIỆN (bỏ box lọc trùng
    `hidden`); box suy biến kẹp như eval (`clip_boxes`). C-NLL None (ảnh < 5 vật) -> không chọn sẵn. -> set chỉ số."""
    if scores is None:
        return set()
    vis = [i for i in range(len(boxes)) if i not in hidden and scores[i] is not None]
    b = _boxes(boxes)[vis]
    b[:, 2], b[:, 3] = np.maximum(b[:, 2], b[:, 0]), np.maximum(b[:, 3], b[:, 1])
    keep = []
    for j in np.argsort(np.asarray([scores[i] for i in vis], dtype=np.float64), kind="stable"):
        if len(keep) == k:
            break
        if not keep or box_iou(b[j:j + 1], b[keep])[0].max() <= iou:
            keep.append(int(j))
    return {vis[j] for j in keep}


def cocount_screens(root):
    """CE-CoCount: một màn = một FILE (một lớp của một frame; `_positive` / `_negative` là hai màn riêng). Box = `exam_bbox` của
    `Anno_with_exam_bbox`, CHƯA lọc, đúng thứ tự file. Tiêu đề = tên vật box đang bao (`class_name` của `Anno`, bỏ phần trong ngoặc)."""
    out = []
    for n in sorted(f[:-5] for f in os.listdir(os.path.join(root, "Anno_with_exam_bbox")) if f.endswith(".json")):
        with open(os.path.join(root, "Anno_with_exam_bbox", n + ".json")) as f:
            boxes = _boxes([x["bbox"] for x in json.load(f)["exam_bbox"]])
        with open(os.path.join(root, "Anno", n + ".json")) as f:
            cls = clean_class(json.load(f)["class_name"])
        pol = n.rsplit("_", 1)[1]
        out.append({"id": n, "title": f"## {cls}", "views": [("image", os.path.join(root, "Image", n + ".jpg"))],
                    "groups": [{"key": n, "color": COLORS.get(pol, COLORS["positive"]), "boxes": boxes.tolist(), "tags": {}, "gone": {},
                                "share": None}]})
    return out


def ce130_screens(root, split, holes=True):
    """CE-130 `all_phase2_V2/<split>/<iid>_b<k>/`: một màn = một NHÁNH. View `original` (`ground_truth.jpg`) + `turn t`
    (`inpainted_turn_t.png`, thiếu vật 1..t). Nhóm `objects` = `all_bboxes` (xyxy, đủ, cả box suy biến; nhãn dùng chung mọi nhánh của
    cùng ảnh qua `share` = iid — nhánh chưa lưu được điền sẵn từ nhánh đã lưu), nhóm `holes` = `inpainted_bboxes` (nhãn H1..H4 theo
    lượt; `holes=False`: chỉ rà box vật). Vật bị xoá ở lượt t (ghép bằng `assign_removed` như bài add) vẽ mờ từ view `turn t`."""
    out = []
    for br in sorted(glob.glob(os.path.join(root, split, "*_b*"))):
        ann = read_annotation(br)
        if ann is None or not os.path.exists(os.path.join(br, "ground_truth.jpg")):
            continue
        bname, iid = os.path.basename(br), os.path.basename(br).split("_b")[0]
        turns = sorted((int(m.group(1)), p) for p in glob.glob(os.path.join(br, "inpainted_turn_*.png"))
                       if (m := re.search(r"inpainted_turn_(\d+)\.png$", p)))
        views = [("original", os.path.join(br, "ground_truth.jpg"))] + [(f"turn {t}", p) for t, p in turns]
        objs, hls = _boxes(ann.get("all_bboxes", [])), _boxes(ann.get("inpainted_bboxes", []))
        removed = assign_removed(objs, hls)[0] if len(hls) else np.zeros(len(objs), dtype=int)
        with Image.open(views[0][1]) as im:
            wh = list(im.size)
        groups = [{"key": "objects", "color": COLORS["objects"], "boxes": objs.tolist(), "tags": {},
                   "gone": {i: int(t) for i, t in enumerate(removed) if t}, "share": iid}]
        if holes:
            groups.append({"key": "holes", "color": COLORS["holes"], "boxes": hls.tolist(),
                           "tags": {i: f"H{i + 1}" for i in range(len(hls))}, "gone": {}, "share": None})
        title = (f"Branch **{bname}** ({split}) | class **{ann.get('class_based_caption', '')}** | {len(objs)} objects"
                 + (f", {len(hls)} holes, {len(turns)} turns" if holes else ""))
        out.append({"id": bname, "title": title, "views": views, "groups": groups, "wh": wh})
    return out


def ce130_pred_screens(dump, turn_index, samples_root, key="inpainted", image_size=512):
    """Box model sinh trên CE-130 (người dùng 2026-10-29: chọn box đẹp) — `<out>_boxes.json` của `eval.py`: một màn = một MẪU (nhánh,
    lượt t), đúng thứ tự dump. View = ảnh model nhận (`samples/`, ảnh đã xoá vật 1..t). Nhóm `objects` = vật còn ở lượt t (xanh, KHÔNG
    bấm), nhóm `model` = K box của model (đỏ mảnh, bấm để chọn; lọc trùng IoU > 0,9 nếu bật). Box dump là pixel canvas `image_size`
    (letterbox góc trên trái) -> chia tỉ lệ về pixel ảnh gốc. `meta` = {sample (tên file samples), image_id, class, t}; `group` = ảnh
    (mọi nhánh / lượt của một ảnh về một người khi chia việc)."""
    index = TurnIndex(turn_index)
    out = []
    for x in _dump_rows(dump, key):
        e = index.turns[x["image_id"]]
        b = index.branches[e["branch"]]
        t, (w, h) = e["t"], b["wh"]
        removed = np.asarray(b["removed"], dtype=int)
        objs = _boxes(b["objects"])[(removed == 0) | (removed > t)]
        sample = os.path.splitext(os.path.basename(e["sample"]))[0]
        out.append(_pred_screen(x, (w, h), image_size, (f"turn {t}", os.path.join(samples_root, e["sample"])), objs,
                                f"**{x['image_id']}** ({sample}) | class **{b['class']}** | turn {t} | {len(objs)} objects",
                                {"sample": sample, "image_id": x["image_id"], "class": b["class"], "t": t},
                                x["image_id"].split("_b")[0]))
    return out


def cocount_pred_screens(dump, root, review, key="cocount", image_size=512):
    """Box model sinh trên CE-CoCount (người dùng 2026-10-30), cùng khuôn `ce130_pred_screens`: một màn = một FILE lớp của dump
    `<out>_boxes.json` (khoá `cocount`: box model thô, không đổi cỡ). View = ẢNH GỐC (`Image/`, không lỗ); nhóm `objects` = box vật
    cùng lớp sau bản rà tay `review` (= cái model nhận ở review5). `meta` = {image_id, class}; `group` = frame (hai lớp một người)."""
    rev = load_review(review)
    out = []
    for x in _dump_rows(dump, key):
        n = x["image_id"]
        path = os.path.join(root, "Image", n + ".jpg")
        with Image.open(path) as im:
            wh = im.size
        r = read_cocount(root, n, review=rev)
        out.append(_pred_screen(x, wh, image_size, ("image", path), r["objects"],
                                f"**{n}** | class **{r['class']}** | {len(r['objects'])} objects",
                                {"image_id": n, "class": r["class"]}, n.rsplit("_", 1)[0]))
    return out


def _dump_rows(dump, key):
    with open(dump) as f:
        res = json.load(f)["results"]
    return res[key] if key in res else res[key + "_mock"]


def _pred_screen(x, wh, image_size, view, objs, title, meta, group):
    """Màn chọn box của một dòng dump `x`: box canvas `image_size` -> pixel ảnh gốc (kẹp trong ảnh), vật xanh không bấm + K box đỏ
    (kèm C-NLL `cnll` của dump để chọn sẵn)."""
    w, h = wh
    s = min(image_size / w, image_size / h)
    if abs(int(w * s) - x["wh"][0]) > 1e-3 or abs(int(h * s) - x["wh"][1]) > 1e-3:
        raise ValueError(f"{x['image_id']}: vùng ảnh dump {x['wh']} khác ({int(w * s)}, {int(h * s)}) — sai --image-size?")
    model = _boxes(x["boxes"]) / s
    model[:, [0, 2]] = model[:, [0, 2]].clip(0, w)
    model[:, [1, 3]] = model[:, [1, 3]].clip(0, h)
    return {"id": x["image_id"], "wh": [w, h], "views": [view], "title": title, "meta": meta, "group": group,
            "groups": [{"key": "objects", "color": COLORS["objects"], "boxes": _boxes(objs).tolist(), "tags": {}, "gone": {},
                        "share": None, "click": False, "dedup": False, "width": 1.2},
                       {"key": "model", "color": COLORS["model"], "boxes": model.tolist(), "tags": {}, "gone": {},
                        "share": None, "width": 1, "sel_width": 3.5, "opacity": 0.6, "scores": x.get("cnll")}]}


SOURCES = {"cocount": cocount_screens, "ce130": ce130_screens, "ce130_pred": ce130_pred_screens, "cocount_pred": cocount_pred_screens}

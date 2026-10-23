"""Nguồn dữ liệu của web rà box (`app.py`): mỗi hàm -> list MÀN, mỗi màn một dict

  id        mã màn (khoá dòng trong file kết quả)
  title     markdown đầu màn ("" = không hiện)
  views     [(nhãn, đường dẫn ảnh)] — cùng kích thước, toạ độ box = pixel ảnh gốc; ô "Image" / phím v đổi qua lại
  groups    [{key, color, boxes [[x1,y1,x2,y2]…], tags {i: nhãn luôn hiện}, gone {i: view đầu tiên mà vật đã bị xoá (vẽ mờ)},
              share: khoá dùng chung nhãn giữa các màn | None}]
            chỉ số i = vị trí trong danh sách box GỐC của file chú thích (kết quả ghi theo chỉ số này)
"""

import glob
import json
import os
import re

import numpy as np
from PIL import Image

from ce_localization.data.cocount import clean_class
from ce_localization.data.turns import read_annotation
from ce_localization.data.turns import assign_removed
from ce_localization.utils.box_ops_np import box_iou

__all__ = ["SOURCES", "COLORS", "DEDUP_IOU", "dedup_boxes", "cocount_screens", "ce130_screens"]

DEDUP_IOU = 0.9          # lọc trùng trước khi rà (người dùng 2026-10-21): IoU > 0,9 ⇒ diện tích lệch < 10 % (IoU <= nhỏ / lớn)
COLORS = {"positive": "#1e90ff", "negative": "#ff8c00", "objects": "#1e90ff", "holes": "#ff00c8"}


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


SOURCES = {"cocount": cocount_screens, "ce130": ce130_screens}

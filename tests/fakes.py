"""Dữ liệu giả DÙNG CHUNG mọi package test (CE-Loc, baseline, VLM, latency) — CPU, không tải gì."""

import json
import os

import numpy as np
from PIL import Image

COCOUNT_NAMES = ("INTRA_FOO_TOM1_TOM2_00008_00006_0_200", "INTER_OFF_PEN0_PNC0_00007_00005_0_40")


def fake_cocount(root, seed=0):
    """CE-CoCount giả: mỗi frame hai file `_positive` / `_negative` (số vật theo tên: 8 / 6 và 7 / 5), vật hai lớp hai hàng, 10 loc_bbox
    ở chỗ trống. -> root."""
    rng = np.random.default_rng(seed)
    for d in ("Image", "Anno", "Anno_with_exam_bbox"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    for fi, stem in enumerate(COCOUNT_NAMES):
        W, H = (200, 120) if fi == 0 else (90, 160)
        for pol in ("positive", "negative"):
            Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(os.path.join(root, "Image", f"{stem}_{pol}.jpg"))
        na, nb = int(stem.split("_")[4]), int(stem.split("_")[5])
        for pol, n, cls, off in (("positive", na, "normal tomato(Big round tomato).", 0), ("negative", nb, "baby tomato.", 9)):
            objs = [[float(5 + 11 * k), float(5 + off), float(13 + 11 * k), float(13 + off)] for k in range(n)]
            loc = [[float(4 + 8 * k), float(40 + (k % 2) * 10), float(12 + 8 * k), float(48 + (k % 2) * 10)] for k in range(10)]
            anno = {"class_name": cls, "loc_bbox": loc, "counting_anno": {"count": n, "exemplars": [list(map(int, o)) for o in objs[:3]]}}
            with open(os.path.join(root, "Anno", f"{stem}_{pol}.json"), "w") as f:
                json.dump(anno, f)
            with open(os.path.join(root, "Anno_with_exam_bbox", f"{stem}_{pol}.json"), "w") as f:
                json.dump({"class_name": cls, "exam_bbox": [{"bbox": o, "score": 0.9} for o in objs]}, f)
    return root


def fake_review(root, path, drop_auto=None):
    """Bản rà giả (khuôn `box_review/app.py`): mỗi file loại tay box 0; `drop_auto` {tên: [chỉ số]} = lọc trùng tự động. -> path."""
    with open(path, "w") as f:
        for n in sorted(x[:-5] for x in os.listdir(os.path.join(root, "Anno_with_exam_bbox"))):
            m = len(json.load(open(os.path.join(root, "Anno_with_exam_bbox", n + ".json")))["exam_bbox"])
            f.write(json.dumps({"id": n, "bad": {n: [0]}, "auto": {n: (drop_auto or {}).get(n, [])}, "n": {n: m}}) + "\n")
    return path


def fake_cocount_review(root, seed=0):
    """CoCount giả + bản rà mặc định `root/review.jsonl` (tập review5 của eval). -> đường dẫn bản rà."""
    fake_cocount(root, seed)
    return fake_review(root, os.path.join(root, "review.jsonl"))

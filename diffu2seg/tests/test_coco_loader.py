#!/usr/bin/env python3
"""COCO val2017 loader: hình học pad hai chiều, trên dữ liệu THẬT.

Suite này SKIP sạch nếu chưa có data/coco/. Nó không cần GPU, không cần SD.

VÌ SAO ĐÁNG CÓ: loader CE-130 được phép giả định pad luôn ở ĐÁY (mọi ảnh CE-130
cao đúng 384 và rộng ít nhất bằng thế). COCO có ảnh dọc, pad ở PHẢI. Sai chỗ này
không crash — nó trồng prompt lên nền xám rồi trả về một mask khổng lồ, đúng
dạng "sai âm thầm" của cạm bẫy #1.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "..", "data")
JSON = os.path.join(ROOT, "coco", "annotations", "instances_val2017.json")
IMGS = os.path.join(ROOT, "coco", "val2017")

CANVAS = 1120          # cấu hình paper
N_CHECK = 200

# SKIP Ở TẦNG MODULE, không phải trong main().
#
# main() có nhánh skip riêng, nhưng pytest KHÔNG gọi main() — nó gọi thẳng từng
# hàm test_*, nên nhánh đó vô hiệu và cả 5 test fail với FileNotFoundError trên
# máy không có dữ liệu (server chỉ có PACO, COCO chỉ ở local). Đã xảy ra thật
# 2026-09-15.
#
# `pytest.importorskip` không dùng được (đây là dữ liệu, không phải module), và
# `pytest` chỉ import được khi đang chạy dưới pytest — nên bọc try/except để
# chạy trực tiếp `python tests/test_coco_loader.py` vẫn được.
_HAVE_DATA = os.path.isfile(JSON) and os.path.isdir(IMGS)
if not _HAVE_DATA:
    try:
        import pytest
        pytestmark = pytest.mark.skip(
            reason=f"chưa có COCO val2017 tại {os.path.normpath(ROOT)}/coco/ "
                   f"(cần annotations/instances_val2017.json + val2017/)")
    except ImportError:
        pass


def _ds(canvas=CANVAS):
    from data.coco_val import CocoVal
    return CocoVal(JSON, IMGS, canvas=canvas)


def test_gt_never_leaves_the_real_image_region():
    """Mọi GT phải nằm trong [0,valid_w] x [0,valid_h] — cả ảnh ngang lẫn dọc.

    Đây là test đã BẮT ĐƯỢC LỖI THẬT: nhân box bằng tỉ lệ float `s` trong khi
    ảnh resize về `int(W*s)` pixel làm box vượt valid_w 0,0004 trên ảnh dọc
    586x640. Nhỏ, nhưng nó đặt GT ra ngoài vùng mà bộ lọc pad gọi là ảnh thật.
    """
    ds = _ds()
    n_port = 0
    for i in range(min(N_CHECK, len(ds))):
        s = ds[i]
        if s["H"] > s["W"]:
            n_port += 1
        gt = s["gt_cxcywh"]
        if not len(gt):
            continue
        x1, y1 = gt[:, 0] - gt[:, 2] / 2, gt[:, 1] - gt[:, 3] / 2
        x2, y2 = gt[:, 0] + gt[:, 2] / 2, gt[:, 1] + gt[:, 3] / 2
        assert x1.min() >= -1e-9 and y1.min() >= -1e-9, s["file_name"]
        assert x2.max() <= s["valid_w"] + 1e-9, \
            f"{s['file_name']}: x2 {x2.max():.6f} > valid_w {s['valid_w']:.6f}"
        assert y2.max() <= s["valid_h"] + 1e-9, \
            f"{s['file_name']}: y2 {y2.max():.6f} > valid_h {s['valid_h']:.6f}"
    assert n_port > 0, "mẫu kiểm không có ảnh dọc nào -> chưa test được pad phải"


def test_exactly_one_side_is_padded():
    """Resize giữ tỉ lệ: đúng một chiều chạm biên canvas, chiều kia là pad."""
    ds = _ds()
    for i in range(min(50, len(ds))):
        s = ds[i]
        assert abs(max(s["valid_w"], s["valid_h"]) - 1.0) < 1e-9, \
            f"{s['file_name']}: không chiều nào lấp đầy canvas"
        if s["W"] > s["H"]:
            assert s["valid_w"] == 1.0 and s["valid_h"] < 1.0
        elif s["H"] > s["W"]:
            assert s["valid_h"] == 1.0 and s["valid_w"] < 1.0


def test_canvas_shape_and_padding_colour():
    """Canvas vuông đúng kích thước, vùng pad đúng màu CLIP-mean."""
    from data.coco_val import CLIP_MEAN
    ds = _ds()
    grey = (CLIP_MEAN * 255).round().astype(np.uint8)
    found = False
    for i in range(min(50, len(ds))):
        s = ds[i]
        assert s["image"].shape == (CANVAS, CANVAS, 3)
        assert s["image"].dtype == np.uint8
        if s["valid_h"] < 1.0:
            row = int(s["valid_h"] * CANVAS) + 2
            if row < CANVAS:
                assert np.array_equal(s["image"][row, 0], grey), "pad dưới sai màu"
                found = True
        if s["valid_w"] < 1.0:
            col = int(s["valid_w"] * CANVAS) + 2
            if col < CANVAS:
                assert np.array_equal(s["image"][0, col], grey), "pad phải sai màu"
                found = True
    assert found, "không ảnh nào có vùng pad để kiểm"


def test_crowd_annotations_are_excluded():
    """iscrowd=1 là vùng đám đông, không phải instance — phải bị loại."""
    import json
    with open(JSON) as f:
        blob = json.load(f)
    n_crowd = sum(1 for a in blob["annotations"] if a.get("iscrowd", 0))
    assert n_crowd > 0, "tập này không có crowd -> test vô nghĩa"

    ds = _ds()
    total = sum(len(v) for v in ds.boxes_xywh.values())
    assert total == len(blob["annotations"]) - n_crowd


def test_resolution_ceiling_is_high_on_coco():
    """Ở grid_r=140, vật COCO thừa to — trần độ phân giải gần như không cản.

    Đây là ĐỐI CHỨNG với CE-130, nơi p10 của box nhỏ nhất chỉ 0,89 ô. Nếu số
    này thấp thì canvas/grid đang bị cấu hình sai, không phải dữ liệu khó.
    """
    ds = _ds()
    ceils = [ds.resolution_ceiling(i, 140) for i in range(min(100, len(ds)))]
    ceils = [c for c in ceils if not np.isnan(c)]
    assert np.mean(ceils) > 0.95, \
        f"trần chỉ {np.mean(ceils):.3f} ở grid_r=140 — nghi cấu hình sai"


def main():
    if not (os.path.isfile(JSON) and os.path.isdir(IMGS)):
        print(f"SKIP  chưa có COCO val2017 tại {os.path.normpath(ROOT)}/coco/")
        print("      cần: coco/annotations/instances_val2017.json + coco/val2017/")
        return 0
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

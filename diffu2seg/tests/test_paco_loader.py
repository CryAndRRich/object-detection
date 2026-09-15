#!/usr/bin/env python3
"""PACO-LVIS val loader, trên dữ liệu THẬT. Tự SKIP nếu chưa có data/paco/.

Không cần GPU, không cần SD. Cần `pycocotools` (65,6 % annotation là RLE nén).

CHỖ ĐÃ SUÝT SAI: loader đầu tiên giả định mọi segmentation là polygon và chết
với `could not convert string to float: 'c'` — 'c' là ký tự đầu của khoá
`counts`. Hoá ra PACO tách chính xác theo loại: OBJECT là polygon, PART là RLE,
và CẢ HAI đều mang iscrowd=0 nên phép thử iscrowd quen thuộc không phân biệt
được. Các test dưới khoá đúng sự thật đó.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "..", "data", "paco")
JSON = os.path.join(ROOT, "paco_lvis_v1_val.json")
IMGS = os.path.join(ROOT, "images")

CANVAS = 1120
N_CHECK = 60

# SKIP Ở TẦNG MODULE — xem ghi chú dài trong test_coco_loader.py: pytest gọi
# thẳng từng test_*, không qua main(), nên nhánh skip trong main() vô hiệu.
_HAVE_DATA = os.path.isfile(JSON) and os.path.isdir(IMGS)
try:
    from pycocotools import mask as _mask_utils   # noqa: F401
    _HAVE_PYCOCO = True
except ImportError:
    _HAVE_PYCOCO = False

if not (_HAVE_DATA and _HAVE_PYCOCO):
    try:
        import pytest
        _why = (f"chưa có PACO tại {os.path.normpath(ROOT)}" if not _HAVE_DATA
                else "thiếu pycocotools (65,6 % annotation PACO là RLE nén)")
        pytestmark = pytest.mark.skip(reason=_why)
    except ImportError:
        pass


def _ds(canvas=CANVAS):
    from data.paco_val import PacoVal
    return PacoVal(JSON, IMGS, canvas=canvas)


def test_split_sizes_match_what_was_measured():
    ds = _ds()
    assert len(ds) == 2410, f"{len(ds)} ảnh, đo được 2410"
    assert ds.n_poly == 10974, f"{ds.n_poly} polygon, đo được 10974 (OBJECT)"
    assert ds.n_rle == 20945, f"{ds.n_rle} RLE, đo được 20945 (PART)"


def test_both_encodings_decode_to_nonempty_masks():
    """Polygon VÀ RLE đều phải ra mask thật — không cái nào âm thầm rỗng."""
    ds = _ds()
    n_part = n_obj = 0
    for i in range(N_CHECK):
        s = ds[i]
        m, ip = s["gt_masks"], s["is_part"]
        if not len(m):
            continue
        sums = m.reshape(len(m), -1).sum(1)
        assert (sums > 0).all(), f"{s['file_name']}: có mask rỗng"
        n_part += int(ip.sum())
        n_obj += int((~ip).sum())
    assert n_part > 0 and n_obj > 0, "mẫu kiểm phải có cả PART lẫn OBJECT"


def test_mask_agrees_with_its_own_bbox():
    """Mask phải nằm trong bbox của chính annotation đó.

    Bắt lỗi ghép nhầm mask với annotation khác — sai kiểu này không crash và
    IoU vẫn ra số.
    """
    ds = _ds()
    for i in range(20):
        s = ds[i]
        gt, m = s["gt_cxcywh"], s["gt_masks"]
        for k in range(len(m)):
            ys, xs = np.where(m[k])
            if not len(ys):
                continue
            x1 = (gt[k][0] - gt[k][2] / 2) / s["valid_w"] * s["W"]
            x2 = (gt[k][0] + gt[k][2] / 2) / s["valid_w"] * s["W"]
            y1 = (gt[k][1] - gt[k][3] / 2) / s["valid_h"] * s["H"]
            y2 = (gt[k][1] + gt[k][3] / 2) / s["valid_h"] * s["H"]
            assert xs.min() >= x1 - 2 and xs.max() <= x2 + 2, \
                f"{s['file_name']} mask {k} lệch khỏi bbox theo X"
            assert ys.min() >= y1 - 2 and ys.max() <= y2 + 2, \
                f"{s['file_name']} mask {k} lệch khỏi bbox theo Y"


def test_mask_area_matches_annotation_area():
    """Diện tích rasterise phải xấp xỉ trường `area` của annotation."""
    ds = _ds()
    ratios = []
    for i in range(30):
        s = ds[i]
        m, a = s["gt_masks"], s["gt_areas"]
        if not len(m):
            continue
        ratios.extend((m.reshape(len(m), -1).sum(1) / np.maximum(a, 1)).tolist())
    ratios = np.array(ratios)
    assert abs(np.median(ratios) - 1.0) < 0.10, \
        f"diện tích trung vị lệch {np.median(ratios):.3f}x so với annotation"


def test_gt_never_leaves_the_real_image_region():
    ds = _ds()
    n_port = 0
    for i in range(N_CHECK):
        s = ds[i]
        if s["H"] > s["W"]:
            n_port += 1
        gt = s["gt_cxcywh"]
        if not len(gt):
            continue
        assert (gt[:, 0] - gt[:, 2] / 2).min() >= -1e-9
        assert (gt[:, 1] - gt[:, 3] / 2).min() >= -1e-9
        assert (gt[:, 0] + gt[:, 2] / 2).max() <= s["valid_w"] + 1e-9
        assert (gt[:, 1] + gt[:, 3] / 2).max() <= s["valid_h"] + 1e-9
    assert n_port > 0, "mẫu kiểm không có ảnh dọc -> chưa test được pad phải"


def test_exactly_one_side_is_padded():
    ds = _ds()
    for i in range(30):
        s = ds[i]
        assert abs(max(s["valid_w"], s["valid_h"]) - 1.0) < 1e-9
        assert s["image"].shape == (CANVAS, CANVAS, 3)


def test_masks_are_at_original_resolution():
    """GT mask ở độ phân giải ẢNH GỐC, không phải canvas.

    AR được tính ở nơi GT sống. Hạ GT xuống lưới 140 sẽ xoá các vật nhỏ, mà
    68,2 % mục tiêu của PACO là small.
    """
    ds = _ds()
    for i in range(10):
        s = ds[i]
        if len(s["gt_masks"]):
            assert s["gt_masks"].shape[1:] == (s["H"], s["W"])


def test_part_fraction_is_two_thirds():
    """65,6 % mục tiêu là PART — con số định hình cách đọc mọi kết quả PACO."""
    ds = _ds()
    frac = ds.n_rle / float(ds.n_rle + ds.n_poly)
    assert 0.65 < frac < 0.67, f"tỉ lệ PART {frac:.3f}, đo được 0,656"


def main():
    if not (os.path.isfile(JSON) and os.path.isdir(IMGS)):
        print(f"SKIP  chưa có PACO tại {os.path.normpath(ROOT)}")
        print("      cần: paco_lvis_v1_val.json + images/ (2410 ảnh)")
        return 0
    try:
        from pycocotools import mask  # noqa: F401
    except ImportError:
        print("SKIP  thiếu pycocotools (65,6 % annotation PACO là RLE nén)")
        print("      pip install pycocotools")
        return 0
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

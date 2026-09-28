"""Loader CE-130 / COCO / PACO (bỏ qua khi thiếu dữ liệu)."""

import os
import sys

import numpy as np
import pytest
from PIL import Image

from diffuse2seg.data.ce130_coco import CE130Coco, resize_and_pad

PROJECT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "diffuse2seg")

# ============================================================================
# từ test_ce130_loader.py
# ============================================================================

ROOT = os.path.join(PROJECT, "..", "data")


JSON = os.path.join(ROOT, "ce130_coco", "ce130_agnostic_val.json")


IMAGES = os.path.join(ROOT, "all_phase2_V2")


needs_data = pytest.mark.skipif(
    not (os.path.isfile(JSON) and os.path.isdir(IMAGES)),
    reason=f"CE-130 data not present ({JSON})",
)


CANVAS = 512


def test_resize_and_pad_puts_padding_at_the_bottom():
    """Never on the right: CE-130 images are never taller than wide."""
    img = Image.new("RGB", (408, 384), (10, 20, 30))
    canvas, valid_h = resize_and_pad(img, CANVAS)

    assert canvas.shape == (CANVAS, CANVAS, 3)
    nh = int(384 * (CANVAS / 408.0))
    assert abs(valid_h - nh / CANVAS) < 1e-12

    assert tuple(canvas[0, -1]) == (10, 20, 30), "right edge must be real image"
    assert tuple(canvas[-1, -1]) != (10, 20, 30), "bottom edge must be padding"


def test_padding_is_clip_mean_not_black():
    """Black is -1.79 sigma after normalisation and fakes a hard edge."""
    canvas, _ = resize_and_pad(Image.new("RGB", (800, 384), (10, 20, 30)), CANVAS)
    assert tuple(canvas[-1, -1]) == (123, 117, 104)


def test_square_image_has_no_padding():
    _, valid_h = resize_and_pad(Image.new("RGB", (384, 384)), CANVAS)
    assert valid_h == 1.0


@needs_data
def test_val_split_matches_known_counts():
    """908 images / 38289 boxes -- the deduped counts recorded in docs/02."""
    ds = CE130Coco(JSON, IMAGES, CANVAS)
    assert len(ds) == 908
    total = sum(len(ds.boxes_xywh[im["id"]]) for im in ds.images)
    assert total == 38289


@needs_data
def test_first_sample_geometry():
    """val/1386_b2 is 408x384 with 35 boxes; first box is known exactly.

    Pins the whole COCO-xywh -> xyxy -> canvas-cxcywh chain to one hand-checked
    value, so a change in any link shows up here rather than as a quietly worse
    oracle_recall.
    """
    s = CE130Coco(JSON, IMAGES, CANVAS)[0]

    assert s["file_name"] == "val/1386_b2/ground_truth.jpg"
    assert (s["W"], s["H"]) == (408, 384)
    assert len(s["gt_cxcywh"]) == 35
    assert np.allclose(s["gt_cxcywh"][0], [0.2120, 0.4338, 0.0613, 0.0490], atol=1e-4)
    assert s["image"].shape == (CANVAS, CANVAS, 3)
    assert s["image"].dtype == np.uint8


@needs_data
def test_all_gt_inside_unit_square_and_above_padding():
    ds = CE130Coco(JSON, IMAGES, CANVAS)
    for i in range(0, 40):
        s = ds[i]
        gt = s["gt_cxcywh"]
        if not len(gt):
            continue
        assert (gt[:, 2] > 0).all() and (gt[:, 3] > 0).all(), "degenerate box survived"
        assert (gt[:, :2] >= 0).all() and (gt[:, :2] <= 1).all()
        assert gt[:, 1].max() <= s["valid_h"] + 1e-6, "GT centre inside the padding"


@needs_data
def test_every_image_is_384_tall():
    """The assumption the padding rule rests on."""
    ds = CE130Coco(JSON, IMAGES, CANVAS)
    assert {im["height"] for im in ds.images} == {384}
    assert min(im["width"] for im in ds.images) >= 384


@needs_data
def test_resolution_ceiling_is_a_fraction():
    ds = CE130Coco(JSON, IMAGES, CANVAS)
    for i in range(5):
        c = ds.resolution_ceiling(i, 64)
        assert np.isnan(c) or 0.0 <= c <= 1.0


def main():
    have = os.path.isfile(JSON) and os.path.isdir(IMAGES)
    ran = skipped = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        if not have and name not in (
            "test_resize_and_pad_puts_padding_at_the_bottom",
            "test_padding_is_clip_mean_not_black",
            "test_square_image_has_no_padding",
        ):
            print(f"SKIP {name} (no data)")
            skipped += 1
            continue
        fn()
        print(f"ok  {name}")
        ran += 1
    print(f"\n{ran} passed, {skipped} skipped")
    return 0


def test_out_name_unique_for_ce130_paths():
    """[NEGATIVE CONTROL] tên file ra phải RIÊNG BIỆT cho từng ảnh CE-130.

    CE-130 đặt file_name kiểu 'val/1386_b2/ground_truth.jpg' — cả 908 ảnh có
    ĐÚNG MỘT basename là 'ground_truth'. Đặt tên theo basename thì 50 ảnh vẽ ra
    ghi đè nhau còn 1 file, và không có gì báo lỗi: thư mục vẫn tồn tại, ảnh
    vẫn mở được. Mất dữ liệu âm thầm.
    """
    import ast
    tool = os.path.join(PROJECT,
                        "tools", "visualize_best.py")
    src = open(tool).read()
    fn = [n for n in ast.parse(src).body
          if isinstance(n, ast.FunctionDef) and n.name == "_out_name"][0]
    ns = {"os": os}
    exec(ast.get_source_segment(src, fn), ns)
    out_name = ns["_out_name"]

    names = {out_name(f"val/{i}_b1/ground_truth.jpg", float("nan"), i)
             for i in range(50)}
    assert len(names) == 50, f"chỉ có {len(names)} tên cho 50 ảnh -> ghi đè"

    # basename thuần sẽ gộp tất cả về 1 -- chứng minh bẫy là thật
    naive = {os.path.basename(f"val/{i}_b1/ground_truth.jpg") for i in range(50)}
    assert len(naive) == 1

    # PACO vẫn giữ tiền tố điểm để `ls` là bảng xếp hạng
    assert out_name("000000005142.jpg", 1.0, 5142) == "1.00_000000005142.png"


@needs_data
def test_ce130_returns_empty_masks_and_valid_w():
    """CE-130 chỉ có BOX. gt_masks rỗng đúng shape để đường chạy chung không vỡ."""
    ds = CE130Coco(JSON, IMAGES, canvas=512)
    s = ds[0]
    assert s["gt_masks"].shape == (0, s["H"], s["W"])
    assert s["gt_masks"].dtype == bool
    assert s["valid_w"] == 1.0, "CE-130 luôn rộng >= cao nên không pad ngang"
    assert len(s["gt_cxcywh"]) > 0, "nhưng box GT thì CÓ"



# ============================================================================
# từ test_coco_loader.py
# ============================================================================

ROOT_coco_loader = os.path.join(PROJECT,
                    "..", "data")


JSON_coco_loader = os.path.join(ROOT_coco_loader, "coco", "annotations", "instances_val2017.json")


IMGS = os.path.join(ROOT_coco_loader, "coco", "val2017")


CANVAS_coco_loader = 1120          # cấu hình paper


N_CHECK = 200


# SKIP Ở TẦNG MODULE, không phải trong main_coco_loader().
#
# main_coco_loader() có nhánh skip riêng, nhưng pytest KHÔNG gọi main_coco_loader() — nó gọi thẳng từng
# hàm test_*, nên nhánh đó vô hiệu và cả 5 test fail với FileNotFoundError trên
# máy không có dữ liệu (server chỉ có PACO, COCO chỉ ở local). Đã xảy ra thật
# 2026-09-15.
#
# `pytest.importorskip` không dùng được (đây là dữ liệu, không phải module), và
# `pytest` chỉ import được khi đang chạy dưới pytest — nên bọc try/except để
# chạy trực tiếp `python tests/test_coco_loader.py` vẫn được.
_HAVE_DATA = os.path.isfile(JSON_coco_loader) and os.path.isdir(IMGS)


# Marker RIÊNG cho đoạn COCO — không dùng `pytestmark` cấp module vì file gộp nhiều loader.
needs_coco = pytest.mark.skipif(
    not _HAVE_DATA,
    reason=f"chưa có COCO val2017 tại {os.path.normpath(ROOT_coco_loader)}/coco/")


def _ds(canvas=CANVAS_coco_loader):
    from diffuse2seg.data.coco_val import CocoVal
    return CocoVal(JSON_coco_loader, IMGS, canvas=canvas)


@needs_coco
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


@needs_coco
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


@needs_coco
def test_canvas_shape_and_padding_colour():
    """Canvas vuông đúng kích thước, vùng pad đúng màu CLIP-mean."""
    from diffuse2seg.data.coco_val import CLIP_MEAN
    ds = _ds()
    grey = (CLIP_MEAN * 255).round().astype(np.uint8)
    found = False
    for i in range(min(50, len(ds))):
        s = ds[i]
        assert s["image"].shape == (CANVAS_coco_loader, CANVAS_coco_loader, 3)
        assert s["image"].dtype == np.uint8
        if s["valid_h"] < 1.0:
            row = int(s["valid_h"] * CANVAS_coco_loader) + 2
            if row < CANVAS_coco_loader:
                assert np.array_equal(s["image"][row, 0], grey), "pad dưới sai màu"
                found = True
        if s["valid_w"] < 1.0:
            col = int(s["valid_w"] * CANVAS_coco_loader) + 2
            if col < CANVAS_coco_loader:
                assert np.array_equal(s["image"][0, col], grey), "pad phải sai màu"
                found = True
    assert found, "không ảnh nào có vùng pad để kiểm"


@needs_coco
def test_crowd_annotations_are_excluded():
    """iscrowd=1 là vùng đám đông, không phải instance — phải bị loại."""
    import json
    with open(JSON_coco_loader) as f:
        blob = json.load(f)
    n_crowd = sum(1 for a in blob["annotations"] if a.get("iscrowd", 0))
    assert n_crowd > 0, "tập này không có crowd -> test vô nghĩa"

    ds = _ds()
    total = sum(len(v) for v in ds.boxes_xywh.values())
    assert total == len(blob["annotations"]) - n_crowd


@needs_coco
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


def main_coco_loader():
    if not (os.path.isfile(JSON_coco_loader) and os.path.isdir(IMGS)):
        print(f"SKIP  chưa có COCO val2017 tại {os.path.normpath(ROOT_coco_loader)}/coco/")
        print("      cần: coco/annotations/instances_val2017.json + coco/val2017/")
        return 0
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0



# ============================================================================
# từ test_paco_loader.py
# ============================================================================

ROOT_paco_loader = os.path.join(PROJECT,
                    "..", "data", "paco")


JSON_paco_loader = os.path.join(ROOT_paco_loader, "paco_lvis_v1_val.json")


IMGS_paco_loader = os.path.join(ROOT_paco_loader, "images")


CANVAS_paco_loader = 1120


N_CHECK_paco_loader = 60


# SKIP Ở TẦNG MODULE — xem ghi chú dài trong test_coco_loader.py: pytest gọi
# thẳng từng test_*, không qua main_paco_loader(), nên nhánh skip trong main_paco_loader() vô hiệu.
_HAVE_DATA_paco_loader = os.path.isfile(JSON_paco_loader) and os.path.isdir(IMGS_paco_loader)


try:
    from pycocotools import mask as _mask_utils   # noqa: F401
    _HAVE_PYCOCO = True
except ImportError:
    _HAVE_PYCOCO = False


# Marker RIÊNG cho đoạn PACO.
needs_paco = pytest.mark.skipif(
    not (_HAVE_DATA_paco_loader and _HAVE_PYCOCO),
    reason=(f"chưa có PACO tại {os.path.normpath(ROOT_paco_loader)}" if not _HAVE_DATA_paco_loader
            else "thiếu pycocotools (65,6 % annotation PACO là RLE nén)"))


def _ds_paco_loader(canvas=CANVAS_paco_loader):
    from diffuse2seg.data.paco_val import PacoVal
    return PacoVal(JSON_paco_loader, IMGS_paco_loader, canvas=canvas)


@needs_paco
def test_split_sizes_match_what_was_measured():
    ds = _ds_paco_loader()
    assert len(ds) == 2410, f"{len(ds)} ảnh, đo được 2410"
    assert ds.n_poly == 10974, f"{ds.n_poly} polygon, đo được 10974 (OBJECT)"
    assert ds.n_rle == 20945, f"{ds.n_rle} RLE, đo được 20945 (PART)"


@needs_paco
def test_both_encodings_decode_to_nonempty_masks():
    """Polygon VÀ RLE đều phải ra mask thật — không cái nào âm thầm rỗng."""
    ds = _ds_paco_loader()
    n_part = n_obj = 0
    for i in range(N_CHECK_paco_loader):
        s = ds[i]
        m, ip = s["gt_masks"], s["is_part"]
        if not len(m):
            continue
        sums = m.reshape(len(m), -1).sum(1)
        assert (sums > 0).all(), f"{s['file_name']}: có mask rỗng"
        n_part += int(ip.sum())
        n_obj += int((~ip).sum())
    assert n_part > 0 and n_obj > 0, "mẫu kiểm phải có cả PART lẫn OBJECT"


@needs_paco
def test_mask_agrees_with_its_own_bbox():
    """Mask phải nằm trong bbox của chính annotation đó.

    Bắt lỗi ghép nhầm mask với annotation khác — sai kiểu này không crash và
    IoU vẫn ra số.
    """
    ds = _ds_paco_loader()
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


@needs_paco
def test_mask_area_matches_annotation_area():
    """Diện tích rasterise phải xấp xỉ trường `area` của annotation."""
    ds = _ds_paco_loader()
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


@needs_paco
def test_paco_loader_gt_never_leaves_the_real_image_region():
    ds = _ds_paco_loader()
    n_port = 0
    for i in range(N_CHECK_paco_loader):
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


@needs_paco
def test_paco_loader_exactly_one_side_is_padded():
    ds = _ds_paco_loader()
    for i in range(30):
        s = ds[i]
        assert abs(max(s["valid_w"], s["valid_h"]) - 1.0) < 1e-9
        assert s["image"].shape == (CANVAS_paco_loader, CANVAS_paco_loader, 3)


@needs_paco
def test_masks_are_at_original_resolution():
    """GT mask ở độ phân giải ẢNH GỐC, không phải canvas.

    AR được tính ở nơi GT sống. Hạ GT xuống lưới 140 sẽ xoá các vật nhỏ, mà
    68,2 % mục tiêu của PACO là small.
    """
    ds = _ds_paco_loader()
    for i in range(10):
        s = ds[i]
        if len(s["gt_masks"]):
            assert s["gt_masks"].shape[1:] == (s["H"], s["W"])


@needs_paco
def test_part_fraction_is_two_thirds():
    """65,6 % mục tiêu là PART — con số định hình cách đọc mọi kết quả PACO."""
    ds = _ds_paco_loader()
    frac = ds.n_rle / float(ds.n_rle + ds.n_poly)
    assert 0.65 < frac < 0.67, f"tỉ lệ PART {frac:.3f}, đo được 0,656"


def main_paco_loader():
    if not (os.path.isfile(JSON_paco_loader) and os.path.isdir(IMGS_paco_loader)):
        print(f"SKIP  chưa có PACO tại {os.path.normpath(ROOT_paco_loader)}")
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


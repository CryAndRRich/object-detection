"""CE-130 loader against the real json on disk.

Skips cleanly when data/ is absent (it is gitignored and lives only on the
machines that have it), so the suite still runs on a bare checkout.

Run:  python -m pytest tests/test_ce130_loader.py -q
      python tests/test_ce130_loader.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_coco import CE130Coco, resize_and_pad  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
JSON = os.path.join(ROOT, "ce130_coco", "ce130_agnostic_val.json")
IMAGES = os.path.join(ROOT, "all_phase2_V2")

needs_data = pytest.mark.skipif(
    not (os.path.isfile(JSON) and os.path.isdir(IMAGES)),
    reason=f"CE-130 data not present ({JSON})",
)

CANVAS = 512


# ------------------------------------------------- no data required

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


# ------------------------------------------------- real data

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


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Vẽ CE-130: bộ này KHÔNG có mask GT
# ---------------------------------------------------------------------------

def test_out_name_unique_for_ce130_paths():
    """[NEGATIVE CONTROL] tên file ra phải RIÊNG BIỆT cho từng ảnh CE-130.

    CE-130 đặt file_name kiểu 'val/1386_b2/ground_truth.jpg' — cả 908 ảnh có
    ĐÚNG MỘT basename là 'ground_truth'. Đặt tên theo basename thì 50 ảnh vẽ ra
    ghi đè nhau còn 1 file, và không có gì báo lỗi: thư mục vẫn tồn tại, ảnh
    vẫn mở được. Mất dữ liệu âm thầm.
    """
    import ast
    tool = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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

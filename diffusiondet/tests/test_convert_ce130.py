"""Self-test cho tools/convert_ce130.py — KHÔNG cần detectron2, chỉ numpy/PIL.

Dựng một cây thư mục CE-130 giả (2-3 branch/ảnh, có annotation.json/fixed_annotation.json,
box xyxy có case suy biến/tràn biên) rồi chạy converter, kiểm:

1. Dedupe đúng theo ảnh gốc (nhiều branch cùng id -> 1 ảnh trong json).
2. xyxy -> xywh COCO đúng công thức, không lệch trục.
3. KHÔNG trừ inpainted_bboxes (giữ nguyên toàn bộ all_bboxes).
4. fixed_annotation.json được ưu tiên khi có, fallback annotation.json khi không.
5. Box suy biến (w hoặc h <= 0) và box vượt biên bị cắt/loại đúng, không crash.
6. Mode class-agnostic -> đúng 1 category; mode closed-set -> nhiều category theo tên
   class thật, và tổng ảnh train72 + val72 == tổng ảnh trước khi chia.

Chạy: python tests/test_convert_ce130.py
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.convert_ce130 import (  # noqa: E402
    build_cat_id_map, build_coco, clip_box_xyxy, parse_branch_name, scan_dedup,
    split_train_72,
)


def _make_branch(root, split, iid, bnum, boxes_xyxy, category, w=200, h=150,
                  fixed=False, extra_ann_fields=None):
    br = os.path.join(root, split, f"{iid}_b{bnum}")
    os.makedirs(br, exist_ok=True)
    Image.new("RGB", (w, h), color=(120, 120, 120)).save(os.path.join(br, "ground_truth.jpg"))
    ann = {
        "all_bboxes": boxes_xyxy,
        "inpainted_bboxes": boxes_xyxy[:1] if boxes_xyxy else [],  # subset, phải KHÔNG bị trừ
        "class_based_caption": category,
    }
    if extra_ann_fields:
        ann.update(extra_ann_fields)
    with open(os.path.join(br, "annotation.json"), "w") as f:
        json.dump(ann, f)
    if fixed:
        # fixed_annotation.json khác nội dung để test dễ phân biệt được cái nào đã đọc
        ann_fixed = dict(ann)
        ann_fixed["class_based_caption"] = category + "_FIXED"
        with open(os.path.join(br, "fixed_annotation.json"), "w") as f:
            json.dump(ann_fixed, f)
    return br


def check(name, cond):
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {name}")
    assert cond, name


def main():
    print("test_convert_ce130:")
    tmp = tempfile.mkdtemp(prefix="ce130_test_")
    try:
        root = os.path.join(tmp, "all_phase2_V2")

        # --- ảnh 1001: 3 branch trùng nhau (dedupe phải gộp về 1) ---
        boxes_1001 = [[10, 20, 60, 80], [100, 30, 150, 120]]  # xyxy
        _make_branch(root, "train", "1001", 1, boxes_1001, "apple")
        _make_branch(root, "train", "1001", 2, boxes_1001, "apple")
        _make_branch(root, "train", "1001", 3, boxes_1001, "apple")

        # --- ảnh 1002: box suy biến (x1==x2) + box vượt biên phải bị cắt ---
        boxes_1002 = [
            [5, 5, 5, 50],       # suy biến: w=0 -> phải bị loại
            [150, 100, 250, 200],  # vượt biên phải/dưới (ảnh 200x150) -> phải bị cắt
        ]
        _make_branch(root, "train", "1002", 1, boxes_1002, "banana")

        # --- ảnh 1003 (val): có fixed_annotation.json -> phải ưu tiên đọc bản fixed ---
        boxes_1003 = [[0, 0, 20, 20]]
        _make_branch(root, "val", "1003", 1, boxes_1003, "sheep", fixed=True)

        # --- ảnh 1004 (val): chỉ có annotation.json (không fixed) -> fallback ---
        boxes_1004 = [[1, 1, 30, 30]]
        _make_branch(root, "val", "1004", 1, boxes_1004, "sheep")

        # ================= scan_dedup =================
        items_train = scan_dedup(os.path.join(root, "train"))
        check("dedupe: 2 ảnh train (1001 dedupe từ 3 branch, + 1002)",
              len(items_train) == 2)
        it_1001 = next(it for it in items_train if it["image_id"] == "1001")
        check("dedupe: giữ nguyên số box của 1001 (KHÔNG trừ inpainted_bboxes)",
              len(it_1001["boxes_xyxy"]) == 2)
        check("dedupe: box 1001 đúng giá trị gốc (không bị biến đổi)",
              it_1001["boxes_xyxy"] == boxes_1001)

        # ========== parse_branch_name ==========
        check("parse_branch_name: '1391_b2' -> ('1391', 2)",
              parse_branch_name("1391_b2") == ("1391", 2))
        check("parse_branch_name: chỉ số branch là SỐ (để _b10 > _b2, không sort chuỗi)",
              parse_branch_name("7_b10")[1] == 10 and parse_branch_name("7_b10")[1] > parse_branch_name("7_b2")[1])
        check("parse_branch_name: image-id chứa '_b' vẫn cắt đúng ở suffix cuối",
              parse_branch_name("ab_bc_b3") == ("ab_bc", 3))
        check("parse_branch_name: tên không đúng dạng -> (nguyên tên, 0)",
              parse_branch_name("khong_co_branch") == ("khong_co_branch", 0))

        items_val = scan_dedup(os.path.join(root, "val"))
        it_1003 = next(it for it in items_val if it["image_id"] == "1003")
        check("fixed_annotation.json được ưu tiên khi có",
              it_1003["category"] == "sheep_FIXED")
        it_1004 = next(it for it in items_val if it["image_id"] == "1004")
        check("fallback sang annotation.json khi không có bản fixed",
              it_1004["category"] == "sheep")

        # ===== branch BẤT ĐỒNG về GT: phải chọn branch chỉ số NHỎ NHẤT, tất định =====
        # Dữ liệu thật: fixed_annotation.json lệch giữa branch ở 86,5 % ảnh val / 79,7 %
        # ảnh test (mỗi branch chỉnh riêng box mình sắp inpaint). Quy tắc chọn phải tất
        # định, nếu không mỗi lần chạy ra GT khác nhau.
        amb_root = os.path.join(tmp, "amb")
        _make_branch(amb_root, "train", "500", 2, [[10, 10, 20, 20]], "x")   # tạo _b2 TRƯỚC
        _make_branch(amb_root, "train", "500", 1, [[11, 11, 22, 22]], "x")   # _b1 tạo SAU
        _make_branch(amb_root, "train", "500", 10, [[99, 99, 111, 111]], "x")  # _b10
        amb = scan_dedup(os.path.join(amb_root, "train"), verbose=False)
        check("branch bất đồng: chọn _b1 (chỉ số nhỏ nhất), không phụ thuộc thứ tự tạo file",
              len(amb) == 1 and amb[0]["boxes_xyxy"] == [[11, 11, 22, 22]])
        check("branch bất đồng: _b10 KHÔNG thắng _b2 (so bằng SỐ, không phải chuỗi)",
              amb[0]["boxes_xyxy"] != [[99, 99, 111, 111]])
        amb2 = scan_dedup(os.path.join(amb_root, "train"), verbose=False)
        check("scan_dedup tất định: chạy 2 lần cho kết quả giống hệt",
              [x["boxes_xyxy"] for x in amb] == [x["boxes_xyxy"] for x in amb2])
        check("scan_dedup không rò field nội bộ (_branch_idx/_variants) ra kết quả",
              all(k not in amb[0] for k in ("_branch_idx", "_variants")))

        # ================= clip_box_xyxy =================
        check("clip: box bình thường -> xywh đúng công thức",
              clip_box_xyxy([10, 20, 60, 80], 200, 150) == [10.0, 20.0, 50.0, 60.0])
        check("clip: box suy biến (w=0) -> None",
              clip_box_xyxy([5, 5, 5, 50], 200, 150) is None)
        clipped = clip_box_xyxy([150, 100, 250, 200], 200, 150)
        check("clip: box vượt biên -> cắt về trong ảnh (x2<=200, y2<=150)",
              clipped is not None and clipped[0] + clipped[2] <= 200.0 + 1e-6
              and clipped[1] + clipped[3] <= 150.0 + 1e-6)

        # ================= build_coco: class-agnostic =================
        coco, stats = build_coco(items_train, mode="class-agnostic")
        check("class-agnostic: đúng 1 category",
              len(coco["categories"]) == 1 and coco["categories"][0]["name"] == "object")
        check("class-agnostic: 2 ảnh trong images",
              len(coco["images"]) == 2)
        # 1001 có 2 box hợp lệ, 1002 có 1 box hợp lệ (1 bị loại vì suy biến) = 3
        check("class-agnostic: đúng số annotation sau khi loại box suy biến",
              len(coco["annotations"]) == 3)
        check("class-agnostic: stats khớp counts thật",
              stats["n_images"] == 2 and stats["n_annotations"] == 3
              and stats["n_dropped_degenerate_box"] == 1)
        # bbox COCO phải là list 4 số [x,y,w,h], w/h > 0
        for ann in coco["annotations"]:
            x, y, w, h = ann["bbox"]
            check(f"bbox {ann['id']} có w,h > 0", w > 0 and h > 0)
            check(f"area {ann['id']} khớp w*h", abs(ann["area"] - w * h) < 1e-6)

        # ===== cờ chất lượng: box khổng lồ (annotation hỏng) — ĐẾM chứ KHÔNG lọc =====
        # Dữ liệu thật: test có 855 box chiếm >50 % ảnh trên 16 ảnh (train/val = 0), 15
        # trong 16 ảnh đó có id 62xx liên tiếp = một lô annotation lỗi. Giữ nguyên dữ
        # liệu (để còn so được với A/B/C) nhưng phải đếm được, đừng để lẫn im lặng.
        huge_root = os.path.join(tmp, "huge")
        # ảnh 200x150 = 30.000 px^2; box 190x140 = 26.600 = 88,7 % ảnh
        _make_branch(huge_root, "train", "900", 1,
                     [[5, 5, 195, 145]] * 6 + [[10, 10, 20, 20]], "y")
        huge_items = scan_dedup(os.path.join(huge_root, "train"), verbose=False)
        _, st_huge = build_coco(huge_items, mode="class-agnostic")
        check("cờ chất lượng: đếm đúng số box chiếm >50% ảnh",
              st_huge["n_box_over_half_image"] == 6)
        check("cờ chất lượng: đánh dấu ảnh nghi annotation hỏng (>=5 box khổng lồ)",
              st_huge["n_images_suspect_annotation"] == 1)
        check("cờ chất lượng KHÔNG lọc dữ liệu (giữ nguyên đủ 7 box)",
              st_huge["n_annotations"] == 7)

        # ================= build_coco: closed-set =================
        coco_cs, stats_cs = build_coco(items_train, mode="closed-set")
        cat_names = {c["name"] for c in coco_cs["categories"]}
        check("closed-set: category theo đúng tên class thật",
              cat_names == {"apple", "banana"})

        # ================= split_train_72 =================
        all_items = items_train + items_val  # giả lập tập lớn hơn để chia có ý nghĩa
        tr, va = split_train_72(all_items, val_frac=0.5, seed=0)
        check("split_train_72: không mất/nhân đôi ảnh nào",
              len(tr) + len(va) == len(all_items))
        check("split_train_72: train/val không giao nhau (theo ảnh)",
              {x["image_id"] for x in tr} & {x["image_id"] for x in va} == set())

        # Stratified theo class: dựng dữ liệu giả nhiều ảnh/class, lệch đuôi dài (giống
        # thực tế CE-130), rồi bắt category(train) phải PHỦ category(val) — đây chính là
        # lỗi random-theo-ảnh đã đo được và phải sửa (71 vs 57 category, giao != đầy đủ).
        fake_dir = os.path.join(tmp, "fake_imgs")
        os.makedirs(fake_dir, exist_ok=True)
        fake_img = os.path.join(fake_dir, "img.jpg")
        Image.new("RGB", (100, 80), color=(50, 50, 50)).save(fake_img)
        fake_items = []
        for cat, n in (("cat_common", 20), ("cat_rare", 3), ("cat_singleton", 1)):
            for k in range(n):
                fake_items.append({"image_id": f"{cat}_{k}", "img_path": fake_img,
                                    "category": cat, "boxes_xyxy": [[1, 1, 20, 20]]})
        tr2, va2 = split_train_72(fake_items, val_frac=0.3, seed=0)
        cats_train = {x["category"] for x in tr2}
        cats_val = {x["category"] for x in va2}
        check("stratified: val chỉ chứa class đã có trong train (không zero-shot ngược)",
              cats_val <= cats_train)
        check("stratified: class đông (cat_common) xuất hiện ở CẢ train lẫn val",
              "cat_common" in cats_train and "cat_common" in cats_val)
        check("stratified: class 1 ảnh (cat_singleton) giữ nguyên ở train, val trống cho nó",
              "cat_singleton" in cats_train and "cat_singleton" not in cats_val)

        # ========== category_id PHẢI nhất quán giữa train72 và val72 ==========
        # Bug đã mắc: build_coco tự dựng bảng id cho TỪNG file -> train72 (đủ class) và
        # val72 (thiếu class hiếm) đánh số độc lập, cùng id trỏ sang tên class khác nhau
        # -> model học id 19 = A, bị chấm bằng id 19 = B, AP ≈ 0 dù model không sai.
        # Cả hai json vẫn hợp lệ nên KHÔNG assert nào bắt được -> phải test riêng.
        cat_id_of = build_cat_id_map(fake_items, mode="closed-set")
        coco_tr, st_tr = build_coco(tr2, mode="closed-set", cat_id_of=cat_id_of)
        coco_va, st_va = build_coco(va2, mode="closed-set", cat_id_of=cat_id_of)

        map_tr = {c["id"]: c["name"] for c in coco_tr["categories"]}
        map_va = {c["id"]: c["name"] for c in coco_va["categories"]}
        check("category_id -> tên class GIỐNG HỆT nhau giữa train72 và val72",
              map_tr == map_va)
        check("cả hai file liệt kê ĐỦ mọi class của bảng chung (kể cả class vắng ở file đó)",
              len(map_tr) == len(cat_id_of) == 3 and len(map_va) == len(cat_id_of))
        check("stats phân biệt được 'bảng chung' và 'class thật sự có mặt'",
              st_va["n_categories"] == 3 and st_va["n_categories_present"] < 3)

        # Kiểm chứng ngược: nếu để mỗi file tự dựng bảng (cat_id_of=None) thì bảng LỆCH —
        # xác nhận test này thật sự bắt được bug, không phải luôn pass.
        coco_tr_bad, _ = build_coco(tr2, mode="closed-set")
        coco_va_bad, _ = build_coco(va2, mode="closed-set")
        map_tr_bad = {c["id"]: c["name"] for c in coco_tr_bad["categories"]}
        map_va_bad = {c["id"]: c["name"] for c in coco_va_bad["categories"]}
        check("(kiểm chứng ngược) tự dựng bảng riêng từng file THÌ LỆCH -> test có hiệu lực",
              map_tr_bad != map_va_bad)

        # ================= relpath cho file_name =================
        coco_rel, _ = build_coco(items_train, mode="class-agnostic",
                                  image_root_for_relpath=root)
        for im in coco_rel["images"]:
            check(f"file_name tương đối, không phải đường dẫn tuyệt đối ({im['file_name']})",
                  not os.path.isabs(im["file_name"]))
            check(f"file_name trỏ đúng file thật ({im['file_name']})",
                  os.path.exists(os.path.join(root, im["file_name"])))

        print("ALL OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_convert_ce130():
    """Wrapper cho pytest. Không có hàm này thì ``pytest tests/`` báo "no tests ran"
    và **exit 0** — nhìn qua tưởng pass trong khi chưa chạy gì (đã bị nhầm một lần).
    Chạy trực tiếp bằng ``python3 tests/test_convert_ce130.py`` vẫn hoạt động như cũ."""
    main()


if __name__ == "__main__":
    main()

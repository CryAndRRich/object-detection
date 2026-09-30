"""DiffusionDet trên CE-130 (EXPERIMENT D.1 = baseline): đường dẫn, chuyển đổi dữ liệu, chỉ số box, MMR."""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

PROJECT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "diffusiondet")
# `objdet` và `tools.convert_ce130` nằm trong diffusiondet/ -> phải vào sys.path TRƯỚC khi import.
sys.path.insert(0, PROJECT)

from objdet.box_quality_metrics import (  # noqa: E402
    box_iou_xyxy, quality_one_image, roc_auc, summarise,
)
from objdet.mmr import compute_mmr_and_recall  # noqa: E402
from tools.convert_ce130 import (  # noqa: E402
    build_cat_id_map, build_coco, clip_box_xyxy, parse_branch_name, scan_dedup,
    split_train_72,
)

# ============================================================================
# từ test_ce130_paths.py
# ============================================================================

def check(name, cond, detail=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"\n        {detail}" if detail and not cond else ""))
    assert cond, f"{name} {detail}"


def main():
    print("test_ce130_paths:")

    # Đường dẫn converter ghi ra, đúng như lệnh trong README:
    #   python tools/convert_ce130.py --ce130-root ../data/all_phase2_V2
    # -> out_dir mặc định = <ce130-root>/../ce130_coco
    data_root = "../data"
    ce130_root = os.path.join(data_root, "all_phase2_V2")
    converter_out = os.path.abspath(os.path.join(ce130_root, "..", "ce130_coco"))

    # Đường dẫn datasets.py đọc vào (import muộn: module này không cần detectron2 để
    # định nghĩa 2 hàm đường dẫn, nhưng import trên cùng của datasets.py thì có -> đọc
    # source thay vì import, để test chạy được ở máy không cài detectron2).
    import re
    src_path = os.path.join(PROJECT,
                            "objdet", "datasets.py")
    with open(src_path) as f:
        src = f.read()

    def default_of(func_name, env_name):
        """Trích biểu thức os.path.join(...) trong lời gọi os.environ.get của hàm đó."""
        m = re.search(rf'def {func_name}\(.*?os\.environ\.get\(\s*"{env_name}",\s*'
                      rf'(os\.path\.join\([^)]*\))', src, re.S)
        assert m, f"không tìm thấy mặc định của {func_name} trong datasets.py"
        return eval(m.group(1), {"os": os, "root": data_root})  # noqa: S307

    registered_ann = os.path.abspath(default_of("ce130_ann_dir", "OBJDET_CE130_ANN_DIR"))
    registered_img = os.path.abspath(default_of("ce130_image_root", "OBJDET_CE130_IMAGE_ROOT"))
    real_img = os.path.abspath(ce130_root)

    check("json: converter ghi ra == datasets.py đọc vào",
          registered_ann == converter_out,
          f"converter={converter_out} != datasets.py={registered_ann}")
    check("ảnh: datasets.py trỏ đúng all_phase2_V2 thật",
          registered_img == real_img,
          f"datasets.py={registered_img} != thật={real_img}")

    # Cả hai phải nằm TRONG data root, không phải bên cạnh nó (bug cũ thừa một "..").
    data_abs = os.path.abspath(data_root)
    check("json nằm bên trong $OBJDET_DATA_ROOT",
          registered_ann.startswith(data_abs + os.sep), registered_ann)
    check("ảnh nằm bên trong $OBJDET_DATA_ROOT",
          registered_img.startswith(data_abs + os.sep), registered_img)

    # Tên 5 dataset đăng ký phải khớp tên file json converter sinh ra.
    for name, fname in [
        ("ce130_agnostic_train", "ce130_agnostic_train.json"),
        ("ce130_agnostic_val", "ce130_agnostic_val.json"),
        ("ce130_agnostic_test", "ce130_agnostic_test.json"),
        ("ce130_closedset_train72", "ce130_closedset_train72.json"),
        ("ce130_closedset_val72", "ce130_closedset_val72.json"),
    ]:
        check(f"dataset {name} khai báo trong _NUM_CLASSES", f'"{name}"' in src)
        check(f"tên file json {fname} khớp pattern trong datasets.py",
              ("ce130_agnostic_{split}.json" in src if "agnostic" in fname
               else "ce130_closedset_{split}.json" in src))

    print("ALL OK")


def test_ce130_paths():
    """Wrapper cho pytest — xem ghi chú trong test_convert_ce130.py."""
    main()



# ============================================================================
# từ test_convert_ce130.py
# ============================================================================

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


def check_convert_ce130(name, cond):
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {name}")
    assert cond, name


def main_convert_ce130():
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
        check_convert_ce130("dedupe: 2 ảnh train (1001 dedupe từ 3 branch, + 1002)",
              len(items_train) == 2)
        it_1001 = next(it for it in items_train if it["image_id"] == "1001")
        check_convert_ce130("dedupe: giữ nguyên số box của 1001 (KHÔNG trừ inpainted_bboxes)",
              len(it_1001["boxes_xyxy"]) == 2)
        check_convert_ce130("dedupe: box 1001 đúng giá trị gốc (không bị biến đổi)",
              it_1001["boxes_xyxy"] == boxes_1001)

        # ========== parse_branch_name ==========
        check_convert_ce130("parse_branch_name: '1391_b2' -> ('1391', 2)",
              parse_branch_name("1391_b2") == ("1391", 2))
        check_convert_ce130("parse_branch_name: chỉ số branch là SỐ (để _b10 > _b2, không sort chuỗi)",
              parse_branch_name("7_b10")[1] == 10 and parse_branch_name("7_b10")[1] > parse_branch_name("7_b2")[1])
        check_convert_ce130("parse_branch_name: image-id chứa '_b' vẫn cắt đúng ở suffix cuối",
              parse_branch_name("ab_bc_b3") == ("ab_bc", 3))
        check_convert_ce130("parse_branch_name: tên không đúng dạng -> (nguyên tên, 0)",
              parse_branch_name("khong_co_branch") == ("khong_co_branch", 0))

        items_val = scan_dedup(os.path.join(root, "val"))
        it_1003 = next(it for it in items_val if it["image_id"] == "1003")
        check_convert_ce130("fixed_annotation.json được ưu tiên khi có",
              it_1003["category"] == "sheep_FIXED")
        it_1004 = next(it for it in items_val if it["image_id"] == "1004")
        check_convert_ce130("fallback sang annotation.json khi không có bản fixed",
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
        check_convert_ce130("branch bất đồng: chọn _b1 (chỉ số nhỏ nhất), không phụ thuộc thứ tự tạo file",
              len(amb) == 1 and amb[0]["boxes_xyxy"] == [[11, 11, 22, 22]])
        check_convert_ce130("branch bất đồng: _b10 KHÔNG thắng _b2 (so bằng SỐ, không phải chuỗi)",
              amb[0]["boxes_xyxy"] != [[99, 99, 111, 111]])
        amb2 = scan_dedup(os.path.join(amb_root, "train"), verbose=False)
        check_convert_ce130("scan_dedup tất định: chạy 2 lần cho kết quả giống hệt",
              [x["boxes_xyxy"] for x in amb] == [x["boxes_xyxy"] for x in amb2])
        check_convert_ce130("scan_dedup không rò field nội bộ (_branch_idx/_variants) ra kết quả",
              all(k not in amb[0] for k in ("_branch_idx", "_variants")))

        # ================= clip_box_xyxy =================
        check_convert_ce130("clip: box bình thường -> xywh đúng công thức",
              clip_box_xyxy([10, 20, 60, 80], 200, 150) == [10.0, 20.0, 50.0, 60.0])
        check_convert_ce130("clip: box suy biến (w=0) -> None",
              clip_box_xyxy([5, 5, 5, 50], 200, 150) is None)
        clipped = clip_box_xyxy([150, 100, 250, 200], 200, 150)
        check_convert_ce130("clip: box vượt biên -> cắt về trong ảnh (x2<=200, y2<=150)",
              clipped is not None and clipped[0] + clipped[2] <= 200.0 + 1e-6
              and clipped[1] + clipped[3] <= 150.0 + 1e-6)

        # ================= build_coco: class-agnostic =================
        coco, stats = build_coco(items_train, mode="class-agnostic")
        check_convert_ce130("class-agnostic: đúng 1 category",
              len(coco["categories"]) == 1 and coco["categories"][0]["name"] == "object")
        check_convert_ce130("class-agnostic: 2 ảnh trong images",
              len(coco["images"]) == 2)
        # 1001 có 2 box hợp lệ, 1002 có 1 box hợp lệ (1 bị loại vì suy biến) = 3
        check_convert_ce130("class-agnostic: đúng số annotation sau khi loại box suy biến",
              len(coco["annotations"]) == 3)
        check_convert_ce130("class-agnostic: stats khớp counts thật",
              stats["n_images"] == 2 and stats["n_annotations"] == 3
              and stats["n_dropped_degenerate_box"] == 1)
        # bbox COCO phải là list 4 số [x,y,w,h], w/h > 0
        for ann in coco["annotations"]:
            x, y, w, h = ann["bbox"]
            check_convert_ce130(f"bbox {ann['id']} có w,h > 0", w > 0 and h > 0)
            check_convert_ce130(f"area {ann['id']} khớp w*h", abs(ann["area"] - w * h) < 1e-6)

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
        check_convert_ce130("cờ chất lượng: đếm đúng số box chiếm >50% ảnh",
              st_huge["n_box_over_half_image"] == 6)
        check_convert_ce130("cờ chất lượng: đánh dấu ảnh nghi annotation hỏng (>=5 box khổng lồ)",
              st_huge["n_images_suspect_annotation"] == 1)
        check_convert_ce130("cờ chất lượng KHÔNG lọc dữ liệu (giữ nguyên đủ 7 box)",
              st_huge["n_annotations"] == 7)

        # ================= build_coco: closed-set =================
        coco_cs, stats_cs = build_coco(items_train, mode="closed-set")
        cat_names = {c["name"] for c in coco_cs["categories"]}
        check_convert_ce130("closed-set: category theo đúng tên class thật",
              cat_names == {"apple", "banana"})

        # ================= split_train_72 =================
        all_items = items_train + items_val  # giả lập tập lớn hơn để chia có ý nghĩa
        tr, va = split_train_72(all_items, val_frac=0.5, seed=0)
        check_convert_ce130("split_train_72: không mất/nhân đôi ảnh nào",
              len(tr) + len(va) == len(all_items))
        check_convert_ce130("split_train_72: train/val không giao nhau (theo ảnh)",
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
        check_convert_ce130("stratified: val chỉ chứa class đã có trong train (không zero-shot ngược)",
              cats_val <= cats_train)
        check_convert_ce130("stratified: class đông (cat_common) xuất hiện ở CẢ train lẫn val",
              "cat_common" in cats_train and "cat_common" in cats_val)
        check_convert_ce130("stratified: class 1 ảnh (cat_singleton) giữ nguyên ở train, val trống cho nó",
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
        check_convert_ce130("category_id -> tên class GIỐNG HỆT nhau giữa train72 và val72",
              map_tr == map_va)
        check_convert_ce130("cả hai file liệt kê ĐỦ mọi class của bảng chung (kể cả class vắng ở file đó)",
              len(map_tr) == len(cat_id_of) == 3 and len(map_va) == len(cat_id_of))
        check_convert_ce130("stats phân biệt được 'bảng chung' và 'class thật sự có mặt'",
              st_va["n_categories"] == 3 and st_va["n_categories_present"] < 3)

        # Kiểm chứng ngược: nếu để mỗi file tự dựng bảng (cat_id_of=None) thì bảng LỆCH —
        # xác nhận test này thật sự bắt được bug, không phải luôn pass.
        coco_tr_bad, _ = build_coco(tr2, mode="closed-set")
        coco_va_bad, _ = build_coco(va2, mode="closed-set")
        map_tr_bad = {c["id"]: c["name"] for c in coco_tr_bad["categories"]}
        map_va_bad = {c["id"]: c["name"] for c in coco_va_bad["categories"]}
        check_convert_ce130("(kiểm chứng ngược) tự dựng bảng riêng từng file THÌ LỆCH -> test có hiệu lực",
              map_tr_bad != map_va_bad)

        # ================= relpath cho file_name =================
        coco_rel, _ = build_coco(items_train, mode="class-agnostic",
                                  image_root_for_relpath=root)
        for im in coco_rel["images"]:
            check_convert_ce130(f"file_name tương đối, không phải đường dẫn tuyệt đối ({im['file_name']})",
                  not os.path.isabs(im["file_name"]))
            check_convert_ce130(f"file_name trỏ đúng file thật ({im['file_name']})",
                  os.path.exists(os.path.join(root, im["file_name"])))

        print("ALL OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_convert_ce130():
    """Wrapper cho pytest. Không có hàm này thì ``pytest tests/`` báo "no tests ran"
    và **exit 0** — nhìn qua tưởng pass trong khi chưa chạy gì (đã bị nhầm một lần).
    Chạy trực tiếp bằng ``python3 tests/test_convert_ce130.py`` vẫn hoạt động như cũ."""
    main_convert_ce130()



# ============================================================================
# từ test_measure_box_quality_ce130.py
# ============================================================================

def check_measure_box_quality_ce130(name, cond):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")
    assert cond, name


def main_measure_box_quality_ce130():
    print("test_measure_box_quality_ce130:")

    # ---------------- box_iou_xyxy ----------------
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [100, 100, 110, 110]])
    iou = box_iou_xyxy(a, b)
    check_measure_box_quality_ce130("IoU box trùng nhau tuyệt đối = 1.0", abs(iou[0, 0] - 1.0) < 1e-9)
    # overlap [5,10]x[5,10] = 25, union = 100+100-25=175 -> 25/175
    check_measure_box_quality_ce130("IoU box chồng lấn một phần đúng công thức", abs(iou[0, 1] - 25 / 175) < 1e-9)
    check_measure_box_quality_ce130("IoU box không chạm nhau = 0", iou[0, 2] == 0.0)
    check_measure_box_quality_ce130("box_iou_xyxy rỗng không crash", box_iou_xyxy(np.zeros((0, 4)), b).shape == (0, 3))

    # ---------------- roc_auc ----------------
    check_measure_box_quality_ce130("AUC hoàn hảo (positive toàn điểm cao) = 1.0",
          abs(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8]) - 1.0) < 1e-9)
    check_measure_box_quality_ce130("AUC ngược hoàn toàn = 0.0",
          abs(roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.9, 0.8]) - 0.0) < 1e-9)
    check_measure_box_quality_ce130("AUC ngẫu nhiên/tie hoàn toàn ~ 0.5",
          abs(roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-9)
    check_measure_box_quality_ce130("AUC thiếu 1 class -> nan", np.isnan(roc_auc([0, 0, 0], [0.1, 0.2, 0.3])))

    # ---------------- quality_one_image ----------------
    gt = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
    pred_perfect = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
    scores_perfect = np.array([0.9, 0.9])
    best, hit, ngt, auc, scored_hit = quality_one_image(pred_perfect, scores_perfect, gt)
    check_measure_box_quality_ce130("perfect: hit cả 2 GT", hit == 2 and ngt == 2)
    check_measure_box_quality_ce130("perfect: mean best IoU = 1.0", abs(best.mean() - 1.0) < 1e-9)
    check_measure_box_quality_ce130("perfect: scored_hit == oracle hit khi score đều tốt", scored_hit == 2)

    # box tốt nhưng KHÔNG có GT nào -> n_gt=0, không crash
    best0, hit0, ngt0, auc0, sh0 = quality_one_image(pred_perfect, scores_perfect, np.zeros((0, 4)))
    check_measure_box_quality_ce130("không có GT -> (rỗng, 0, 0, nan, 0)",
          len(best0) == 0 and hit0 == 0 and ngt0 == 0 and np.isnan(auc0) and sh0 == 0)

    # không có prediction nào -> mọi GT best_iou = 0, hit = 0
    best1, hit1, ngt1, auc1, sh1 = quality_one_image(np.zeros((0, 4)), np.zeros(0), gt)
    check_measure_box_quality_ce130("không có prediction -> best toàn 0, hit=0, ngt=2",
          (best1 == 0).all() and hit1 == 0 and ngt1 == 2)

    # box TỐT nhưng SCORE ngược (đúng kịch bản "score head hỏng" mà tool này tồn tại để bắt):
    # 1 box đúng GT[0] điểm THẤP, 1 box rác điểm CAO -> oracle_recall vẫn cao nhưng
    # recall_scored (theo greedy score-order) thấp hơn hẳn.
    pred_mixed = np.array([[0, 0, 10, 10], [50, 50, 60, 60]])   # box 2 không trúng GT nào
    scores_mixed = np.array([0.1, 0.9])                          # box đúng điểm thấp hơn box rác
    best2, hit2, ngt2, auc2, sh2 = quality_one_image(pred_mixed, scores_mixed, gt)
    check_measure_box_quality_ce130("score hỏng: oracle vẫn thấy 1/2 GT trúng (box đúng có mặt, bất kể score)",
          hit2 == 1)
    check_measure_box_quality_ce130("score hỏng: AUC thấp vì box khớp bị xếp điểm thấp hơn box không khớp",
          auc2 < 0.5)

    # ---------------- summarise ----------------
    res = summarise([best], hit, ngt, [auc] if not np.isnan(auc) else [], scored_hit)
    check_measure_box_quality_ce130("summarise: oracle_recall == hit/n_gt", abs(res["oracle_recall"] - hit / ngt) < 1e-9)
    check_measure_box_quality_ce130("summarise: score_head_cost = oracle - scored",
          abs(res["score_head_cost"] - (res["oracle_recall"] - res["recall_scored"])) < 1e-9)
    res_empty = summarise([], 0, 0, [], 0)
    check_measure_box_quality_ce130("summarise: n_gt=0 không chia-cho-0 (oracle_recall=0.0, không NaN/crash)",
          res_empty["oracle_recall"] == 0.0 and res_empty["mean_bestIoU"] == 0.0)

    print("ALL OK")


def test_box_quality_metrics():
    """Wrapper cho pytest — xem ghi chú trong test_convert_ce130.py."""
    main_measure_box_quality_ce130()



# ============================================================================
# từ test_mmr.py
# ============================================================================

NO_IGNORE = np.zeros((0, 4))


GTS = np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=float)


DETS = np.array([[10, 10, 50, 50, 0.99], [100, 100, 150, 150, 0.98]], dtype=float)


def check_mmr(name, got, **want):
    line = "  ".join(f"{k}={got[k]:.2f}" for k in ("AP50", "mMR", "Recall"))
    print(f"  {name:34s} {line}")
    for k, v in want.items():
        assert abs(got[k] - v) < 0.01, f"{name}: {k} = {got[k]}, kỳ vọng {v}"


def main_mmr():
    print("test_mmr:")

    # detector hoàn hảo: trúng hết, không FP -> recall 100, mMR ~ 0
    check_mmr("hoàn hảo", compute_mmr_and_recall({1: (DETS, GTS, NO_IGNORE)}),
          Recall=100.0, mMR=0.0, AP50=100.0)

    # bỏ sót 1 trong 2 GT -> recall 50, MR = 50 ở mọi mốc FPPI nên mMR = 50
    check_mmr("bỏ sót 1/2", compute_mmr_and_recall({1: (DETS[:1], GTS, NO_IGNORE)}),
          Recall=50.0, mMR=50.0)

    # Detection thứ 3 nằm trong vùng ignore. Score của nó phải CAO hơn mọi true positive,
    # nếu không thì phép thử vô nghĩa: AP kiểu VOC bỏ qua false positive xếp sau khi
    # recall đã đạt 100%, nên một FP score thấp không làm AP giảm và hai nhánh có/không
    # khai ignore sẽ ra cùng kết quả.
    ign = np.array([[200, 200, 300, 300]], dtype=float)
    d3 = np.concatenate([[[210, 210, 290, 290, 0.995]], DETS])
    with_ign = compute_mmr_and_recall({1: (d3, GTS, ign)})
    check_mmr("có khai ignore", with_ign, Recall=100.0, mMR=0.0, AP50=100.0)

    # cùng detection đó nhưng không khai ignore -> thành FP xếp đầu, AP50 và mMR đều xấu đi
    without = compute_mmr_and_recall({1: (d3, GTS, NO_IGNORE)})
    check_mmr("không khai ignore", without, Recall=100.0)
    assert without["AP50"] < with_ign["AP50"], \
        f"vùng ignore phải loại được false positive: {without['AP50']} vs {with_ign['AP50']}"
    assert without["mMR"] > with_ign["mMR"], \
        f"FP score cao phải làm mMR xấu đi: {without['mMR']} vs {with_ign['mMR']}"

    # ảnh không có detection nào vẫn phải tính vào recall
    two_img = compute_mmr_and_recall({
        1: (DETS, GTS, NO_IGNORE),
        2: (np.zeros((0, 5)), GTS, NO_IGNORE),
    })
    check_mmr("1/2 ảnh không detect", two_img, Recall=50.0, mMR=50.0)

    # detector lệch hẳn (IoU < 0.5) -> không trúng gì
    off = np.array([[300, 300, 340, 340, 0.9]], dtype=float)
    check_mmr("detect sai chỗ", compute_mmr_and_recall({1: (off, GTS, NO_IGNORE)}),
          Recall=0.0, mMR=100.0, AP50=0.0)

    # không có GT -> nan chứ không crash
    nan_res = compute_mmr_and_recall({1: (DETS, NO_IGNORE, NO_IGNORE)})
    assert np.isnan(nan_res["mMR"]), nan_res
    print("  không có GT                        -> nan (không crash)")

    print("TẤT CẢ ĐẠT")


def test_mmr():
    """Wrapper cho pytest — xem ghi chú trong test_convert_ce130.py."""
    main_mmr()


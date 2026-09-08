#!/usr/bin/env python3
"""Chuyển CE-130 (`all_phase2_V2/`) sang COCO json cho DiffusionDet — EXPERIMENT D.

Vì sao cần: đo TRẦN THỰC TẾ của bài toán định vị CE-130 bằng một detector chuẩn đã kiểm
chứng (DiffusionDet), làm đối chứng cho AP50 0,0152 của CE-LocModel A/B/C. Đặc tả đầy đủ:
``docs/thiet-ke-experiment-d-diffusiondet-ce130.md``.

Nguồn dữ liệu và 3 bẫy đã có bằng chứng đo được (`object-detection/data/README.md` §8,
đối chiếu lại `count_editing/CE-LocModel/data/ce130_dataset.py` — bản đã verify):

1. **`all_bboxes` là `xyxy` tuyệt đối**, COCO json cần `[x, y, w, h]` góc trên-trái.
   Chuyển sai thì box vẫn nằm trong ảnh — không assert nào bắt được.
2. **DEDUPE THEO ẢNH GỐC.** Các branch ``{id}_b1/_b2/_b3`` dùng chung một
   ``ground_truth.jpg`` — không dedupe thì 8.829 branch thành 8.829 "ảnh" trùng nhau, tạo
   val leak và số đẹp giả. Sau dedupe đúng: train 1.911 / val 908 / test 779.

   ⚠️ Kiểm TOÀN BỘ (không phải lấy mẫu, 2026-09-08): ``ground_truth.jpg`` giống hệt nhau
   giữa mọi branch cùng id (md5 khớp **100 %**, 0/1.410 cặp val + 0/1.079 cặp test lệch),
   và ``annotation.json`` cũng **nhất quán tuyệt đối** (0 cặp lệch ở cả 3 split). NHƯNG
   ``fixed_annotation.json`` thì **KHÔNG**: lệch ở **1.410 cặp val / 1.079 cặp test** —
   mỗi branch chỉnh riêng box mà chính nó sắp inpaint. Hệ quả: **86,5 % ảnh val và
   79,7 % ảnh test có các branch bất đồng về GT**, chọn branch nào ảnh hưởng tới toạ độ
   (lệch tối đa **374 px** ở một box). Mức ảnh hưởng thực tế lên tổng số box thì rất nhỏ
   (chênh **11 box val / 10 box test**, ~0,03 %) — khác biệt gần như hoàn toàn là toạ độ,
   không phải thêm/bớt vật; vẽ ra nhìn thì hai bản chất lượng tương đương, không bản nào
   sai rõ ràng. Đây là **noise annotation của dữ liệu**, không phải bug converter.

   Cách xử lý ở đây: chọn **branch có chỉ số nhỏ nhất** (``_b1`` trước ``_b2``) — tất
   định, tái lập được, và khớp với hành vi của ``ce130_dataset.py`` gốc (nó cũng lấy
   branch đầu theo thứ tự sort) nên số liệu so được với A/B/C. Converter in ra số ảnh có
   branch bất đồng để con số này không bị quên.
3. **KHÔNG trừ `inpainted_bboxes`.** ``ground_truth.jpg`` là ảnh GỐC, CHƯA xoá vật nào
   (bằng chứng pixel: diff tại vùng ``inpainted_bboxes[0]`` = 51,96/255 so với 1,41/255
   toàn ảnh — vật thật rồi mới bị xoá ở turn sau). Trừ đi (như vòng 1 CE-Loc từng làm) là
   vứt bỏ 7-8% vật thật.

``fixed_annotation.json`` chỉ có ở val/test (0/4.653 branch train có bản fixed) — luôn
fallback sang ``annotation.json``.

Hai mode, tương ứng D.1 / D.2 trong đặc tả:

- ``--mode class-agnostic`` (D.1, MẶC ĐỊNH, LÀM TRƯỚC): 1 category duy nhất ``object``.
  Hợp lệ vì CE-130 mỗi ảnh chỉ có đúng 1 class (3.598/3.598, đã đo trong
  ``ce130_dataset.py`` gốc) — "mọi vật trong ảnh" và "vật thuộc category ảnh đó" là CÙNG
  một tập box trên bộ này.
- ``--mode closed-set`` (D.2, LÀM SAU): giữ nguyên tên class thật làm category, và chia
  lại **nội bộ split train** (72 class) thành ``train72``/``val72`` — một lần chạy sinh
  cả hai file, dùng CHUNG một bảng ``category_id`` (xem ``build_cat_id_map``). KHÔNG chạy
  closed-set trên train(72)/val(28)/test(28) gốc — 3 split đó có giao class = 0, AP sẽ
  ≈ 0 vì head phân loại theo index chưa từng thấy các class ở val/test.

Chạy (từ ``object-detection/diffusiondet/``):

    python tools/convert_ce130.py --ce130-root ../data/all_phase2_V2 --mode class-agnostic
    python tools/convert_ce130.py --ce130-root ../data/all_phase2_V2 --mode closed-set --split train

Sau khi sinh xong, BẮT BUỘC vẽ lại box từ json vừa sinh lên ảnh để kiểm mắt trước khi
train (bài học ``docs/bai-hoc-ce-loc-detection.md`` §5: visualize bắt được lỗi mà test +
review code bỏ sót) — dùng ``tools/visualize_ce130_coco.py``.
"""

import argparse
import glob
import json
import os
import re


def clip_box_xyxy(box, width, height):
    """``all_bboxes`` có box vượt biên và box suy biến (w hoặc h <= 0). Số hiện hành, đo
    lại 2026-09-08 trên đúng dữ liệu này: **85/71.852 box train**, 0 ở val/test — con số
    "14/37.110" trong docstring ``filter_degenerate`` của CE-LocModel là số cũ, đo TRƯỚC
    khi sửa "không trừ ``inpainted_bboxes``" nên tổng box khi đó ít hơn nhiều.

    Trả về ``[x, y, w, h]`` COCO hoặc None nếu rỗng sau khi cắt biên.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    x1c, y1c = max(0.0, x1), max(0.0, y1)
    x2c, y2c = min(float(width), x2), min(float(height), y2)
    if x2c <= x1c or y2c <= y1c:
        return None
    return [x1c, y1c, x2c - x1c, y2c - y1c]


def read_annotation(branch_dir):
    """``fixed_annotation.json`` chỉ có ở val/test — fallback sang ``annotation.json``."""
    for name in ("fixed_annotation.json", "annotation.json"):
        p = os.path.join(branch_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None


_BRANCH_RE = re.compile(r"^(?P<iid>.+)_b(?P<idx>\d+)$")


def parse_branch_name(name):
    """``"1391_b2"`` -> ``("1391", 2)``. Tên không đúng dạng -> ``(name, 0)``.

    Dùng regex thay vì ``name.split("_b")[0]``: cách cũ cắt sai nếu image-id chứa ``_b``
    (dữ liệu hiện tại 8.829/8.829 thư mục đều đúng dạng ``{số}_b{số}`` nên chưa xảy ra,
    nhưng cách cũ hỏng âm thầm nếu dữ liệu đổi). Trả chỉ số branch dạng **số nguyên** để
    sort đúng — sort chuỗi cho ``_b10 < _b2`` (hiện chỉ có _b1.._b3 nên chưa lộ).
    """
    m = _BRANCH_RE.match(name)
    if not m:
        return name, 0
    return m.group("iid"), int(m.group("idx"))


def scan_dedup(split_dir, verbose=True):
    """Dedupe theo ảnh gốc: mỗi ``{id}`` lấy **branch có chỉ số nhỏ nhất**.

    ``ground_truth.jpg`` giống hệt nhau giữa mọi branch cùng id (md5 khớp 100 %, kiểm
    toàn bộ). Nhưng GT thì KHÔNG luôn giống — ``fixed_annotation.json`` lệch giữa các
    branch ở 86,5 % ảnh val / 79,7 % ảnh test (xem docstring đầu file: noise annotation
    của dữ liệu, ảnh hưởng toạ độ chứ gần như không ảnh hưởng số lượng box). Chọn branch
    nhỏ nhất là quy tắc **tất định**, khớp hành vi ``ce130_dataset.py`` gốc nên số liệu
    vẫn so được với A/B/C.

    Trả về list dict: {image_id, img_path, boxes_xyxy, category}.
    """
    by_image = {}
    n_branch = 0
    for br in glob.glob(os.path.join(split_dir, "*")):
        if not os.path.isdir(br):
            continue
        ann = read_annotation(br)
        if ann is None:
            continue
        img_path = os.path.join(br, "ground_truth.jpg")
        if not os.path.exists(img_path):
            continue
        iid, bidx = parse_branch_name(os.path.basename(br))
        n_branch += 1
        boxes = ann.get("all_bboxes", [])   # KHÔNG trừ inpainted_bboxes — xem docstring
        rec = {
            "image_id": iid,
            "img_path": img_path,
            "boxes_xyxy": boxes,
            "category": ann.get("class_based_caption", "unknown"),
            "_branch_idx": bidx,
            "_variants": [boxes],
        }
        prev = by_image.get(iid)
        if prev is None:
            by_image[iid] = rec
        elif bidx < prev["_branch_idx"]:    # branch nhỏ hơn thắng (tất định)
            rec["_variants"] = prev["_variants"] + [boxes]
            by_image[iid] = rec
        else:
            prev["_variants"].append(boxes)

    items = [by_image[k] for k in sorted(by_image)]
    n_ambiguous = sum(1 for it in items
                      if any(v != it["_variants"][0] for v in it["_variants"][1:]))
    if verbose:
        print(f"  dedupe: {n_branch} branch -> {len(items)} ảnh | {n_ambiguous} ảnh có "
              f"branch BẤT ĐỒNG về GT "
              f"({100 * n_ambiguous / max(len(items), 1):.1f}% — noise annotation của dữ "
              f"liệu, đã chọn branch chỉ số nhỏ nhất; xem docstring đầu file)")
    for it in items:                        # dọn field nội bộ khỏi kết quả trả ra
        it.pop("_branch_idx", None)
        it.pop("_variants", None)
    return items


def build_cat_id_map(items, mode):
    """Bảng tên class -> category_id, dựng MỘT LẦN cho cả bộ dữ liệu.

    PHẢI dùng chung giữa mọi split. Nếu để ``build_coco`` tự dựng bảng cho từng file
    (bug đã mắc và đã sửa) thì train72 (72 class) và val72 (67 class — 5 class chỉ có
    1 ảnh nên không có mặt ở val) đánh số ĐỘC LẬP, khiến 54/72 id trỏ sang tên class
    khác nhau giữa hai file::

        id 19: train='cartridge'  val='cement bag'
        id 20: train='cassette'   val='cereal'

    Model học "id 19 = cartridge" rồi bị chấm bằng "id 19 = cement bag" -> AP ≈ 0 vì lý
    do không liên quan tới model. Cả hai json vẫn hợp lệ, train vẫn chạy bình thường,
    KHÔNG assert nào bắt được — nên bảng id phải dựng ở ngoài rồi truyền vào.

    Cơ chế bên dưới (đọc source detectron2 ``data/datasets/coco.py:102``):
    ``id_map = {v: i for i, v in enumerate(cat_ids)}`` được dựng từ **chính json đang
    load**, rồi thành ``thing_dataset_id_to_contiguous_id``. Hai file có tập
    ``categories`` khác nhau => hai ánh xạ ``category_id -> contiguous_id`` khác nhau,
    tức model train theo một ánh xạ và bị chấm theo ánh xạ kia. Vì vậy ``build_coco``
    liệt kê **đủ mọi class của bảng chung ở CẢ HAI file**, kể cả class không xuất hiện
    trong file đó. Đã kiểm bằng chính công thức trên: ``id_map`` và ``thing_classes``
    giống hệt nhau giữa train72/val72, ``cartridge`` cùng ở contiguous_id 18 hai phía,
    và ``cat_ids`` liên tục 1..72 nên detectron2 không phải remap.
    """
    if mode == "class-agnostic":
        cat_names = ["object"]
    elif mode == "closed-set":
        cat_names = sorted({it["category"] for it in items})
    else:
        raise ValueError(mode)
    return {name: i + 1 for i, name in enumerate(cat_names)}  # COCO id bắt đầu từ 1


def build_coco(items, mode, image_root_for_relpath=None, cat_id_of=None):
    """items -> dict COCO. Cần mở ảnh để lấy width/height (odgt không có, giống
    ``convert_crowdhuman.py``).

    ``cat_id_of``: bảng tên class -> id dùng CHUNG cho mọi split (xem
    ``build_cat_id_map``). Bỏ trống thì tự dựng từ ``items`` — chỉ an toàn khi đây là
    file duy nhất (D.1 class-agnostic có đúng 1 category nên luôn an toàn).
    """
    from PIL import Image

    if mode == "class-agnostic":
        cat_name_of = lambda _it: "object"
    elif mode == "closed-set":
        cat_name_of = lambda it: it["category"]
    else:
        raise ValueError(mode)

    if cat_id_of is None:
        cat_id_of = build_cat_id_map(items, mode)

    images, annotations = [], []
    ann_id = 1
    n_dropped_degenerate = 0
    n_dropped_outofimage = 0

    for img_id, it in enumerate(items, 1):
        with Image.open(it["img_path"]) as im:
            width, height = im.size

        file_name = it["img_path"]
        if image_root_for_relpath is not None:
            file_name = os.path.relpath(it["img_path"], image_root_for_relpath)

        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": width,
            "height": height,
            "ce130_image_id": it["image_id"],   # giữ id gốc để đối chiếu/debug
        })

        cat_id = cat_id_of[cat_name_of(it)]
        for box in it["boxes_xyxy"]:
            if len(box) != 4:
                n_dropped_degenerate += 1
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            if x2 <= x1 or y2 <= y1:            # suy biến TRƯỚC khi cắt biên (w/h <= 0)
                n_dropped_degenerate += 1
                continue
            xywh = clip_box_xyxy(box, width, height)
            if xywh is None:                     # hợp lệ nhưng nằm hoàn toàn ngoài ảnh
                n_dropped_outofimage += 1
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": xywh,
                "area": xywh[2] * xywh[3],
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": {"description": f"CE-130 ({mode}) chuyển sang COCO format",
                  "ce130_mode": mode},
        "images": images,
        "annotations": annotations,
        # Liệt kê ĐỦ mọi class của bảng chung, kể cả class không xuất hiện trong file
        # này — để category_id nhất quán giữa train/val, và để metadata detectron2
        # (thing_classes) giống nhau ở cả hai split.
        "categories": [{"id": cid, "name": name, "supercategory": "object"}
                        for name, cid in sorted(cat_id_of.items(), key=lambda kv: kv[1])],
    }
    # Cờ chất lượng annotation: box chiếm >50 % diện tích ảnh gần như chắc chắn không
    # phải một vật đơn lẻ trên CE-130 (vật điển hình chỉ ~0,4 % diện tích ảnh). Đo được
    # trên dữ liệu thật: train 0, val 0, **test 855 box / 16 ảnh** — 15 ảnh trong đó có
    # id dạng 62xx liên tiếp, tức MỘT LÔ annotation lỗi. Ảnh 6261 chẳng hạn: 325 box mà
    # 293 box bao gần trọn ảnh, chồng khít lên nhau, không box nào bao một quả táo.
    # KHÔNG tự lọc (giữ nguyên dữ liệu để số liệu còn so được với CE-LocModel A/B/C),
    # chỉ đếm và báo — xem README mục EXPERIMENT D.
    area_img = {im["id"]: im["width"] * im["height"] for im in images}
    n_huge = sum(1 for a in annotations if a["area"] > 0.5 * area_img[a["image_id"]])
    huge_per_image = {}
    for a in annotations:
        if a["area"] > 0.5 * area_img[a["image_id"]]:
            huge_per_image[a["image_id"]] = huge_per_image.get(a["image_id"], 0) + 1
    n_img_suspect = sum(1 for v in huge_per_image.values() if v >= 5)

    stats = {
        "n_images": len(images),
        "n_annotations": len(annotations),
        "n_categories": len(cat_id_of),                       # bảng chung
        "n_categories_present": len({a["category_id"] for a in annotations}),  # có mặt thật
        "n_dropped_degenerate_box": n_dropped_degenerate,
        "n_dropped_out_of_image": n_dropped_outofimage,
        "n_box_over_half_image": n_huge,                      # cờ chất lượng, KHÔNG lọc
        "n_images_suspect_annotation": n_img_suspect,         # >=5 box khổng lồ trong 1 ảnh
    }
    return coco, stats


def split_train_72(items, val_frac=0.15, seed=0):
    """D.2: chia LẠI split train (72 class) thành train_72/val_72 — mục đích là đo trần
    closed-set trong nội bộ 72 class, so với chính D.1 trên cùng split mới này, KHÔNG so
    với A/B/C (những cái chạy trên test 28-class zero-shot).

    BẮT BUỘC stratified THEO CLASS, không phải random thuần theo ảnh: converter đã đo
    được (chạy thật, ``--mode closed-set``) rằng 72 class trải rất lệch trên 1.911 ảnh
    (trung bình ~26 ảnh/class nhưng phân phối dài đuôi), nên random theo ảnh với
    ``val_frac=0.15`` để rơi vào tình huống train chỉ còn 71 category và val chỉ 57 —
    một số class hiếm bị đẩy hết sang 1 bên, category(train) != category(val). D.2 đo
    "trần closed-set" nên đòi hỏi val phải thấy đúng những class đã train — nếu không,
    số đo D.2 sụp vì zero-shot giống hệt vấn đề D.2 sinh ra để tách bạch khỏi D.1.

    Với mỗi class: nếu chỉ có 1 ảnh thì giữ cả ở train, không chia (val của class đó
    sẽ vắng — không thể tránh khi ảnh quá ít, nhưng không còn ảnh hưởng ngẫu nhiên diện
    rộng như random theo ảnh).
    """
    import random
    rng = random.Random(seed)
    by_class = {}
    for it in items:
        by_class.setdefault(it["category"], []).append(it)

    train_items, val_items = [], []
    for cat, group in by_class.items():
        group = list(group)
        rng.shuffle(group)
        n_val = int(len(group) * val_frac)
        if len(group) >= 2:
            n_val = max(1, min(n_val, len(group) - 1))   # luôn còn >=1 ảnh ở train
        else:
            n_val = 0
        val_items.extend(group[:n_val])
        train_items.extend(group[n_val:])
    return train_items, val_items


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ce130-root", required=True,
                   help="đường dẫn tới all_phase2_V2/ (chứa train/, val/, test/)")
    p.add_argument("--mode", default="class-agnostic",
                   choices=["class-agnostic", "closed-set"],
                   help="class-agnostic = D.1 (mặc định, làm trước); "
                        "closed-set = D.2 (chỉ áp dụng cho --split train, chia lại 72 class)")
    p.add_argument("--split", default=None, choices=["train", "val", "test"],
                   help="với --mode closed-set: BẮT BUỘC = train (D.2 chỉ dùng dữ liệu "
                        "train gốc, chia lại nội bộ). Với class-agnostic: bỏ trống = "
                        "chuyển cả 3 split.")
    p.add_argument("--out-dir", default=None,
                   help="mặc định: <ce130-root>/../ce130_coco/")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="chỉ dùng cho --mode closed-set: tỉ lệ ảnh train gốc dành cho "
                        "val_72")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.ce130_root, "..", "ce130_coco")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    image_root = os.path.abspath(args.ce130_root)

    if args.mode == "class-agnostic":
        splits = [args.split] if args.split else ["train", "val", "test"]
        for split in splits:
            split_dir = os.path.join(args.ce130_root, split)
            items = scan_dedup(split_dir)
            coco, stats = build_coco(items, mode="class-agnostic",
                                      image_root_for_relpath=image_root)
            out_path = os.path.join(out_dir, f"ce130_agnostic_{split}.json")
            with open(out_path, "w") as f:
                json.dump(coco, f)
            print(f"[{split}] {stats}  -> {out_path}")

    else:  # closed-set (D.2)
        split = args.split or "train"
        if split != "train":
            raise SystemExit(
                "--mode closed-set chỉ hợp lệ với --split train (D.2 chia lại NỘI BỘ "
                "72 class của split train gốc — xem docstring ở đầu file)."
            )
        split_dir = os.path.join(args.ce130_root, "train")
        items = scan_dedup(split_dir)
        train_items, val_items = split_train_72(items, args.val_frac, args.seed)

        # Bảng id dựng MỘT LẦN từ TOÀN BỘ 72 class (trước khi chia), dùng chung cho cả
        # hai file — xem docstring build_cat_id_map: để mỗi file tự dựng thì id trỏ sang
        # tên class khác nhau giữa train/val và AP về 0 vì lý do không liên quan model.
        cat_id_of = build_cat_id_map(items, mode="closed-set")
        print(f"bảng category dùng chung: {len(cat_id_of)} class "
              f"(dựng từ toàn bộ split train trước khi chia)")

        for name, subset in (("train72", train_items), ("val72", val_items)):
            coco, stats = build_coco(subset, mode="closed-set",
                                      image_root_for_relpath=image_root,
                                      cat_id_of=cat_id_of)
            out_path = os.path.join(out_dir, f"ce130_closedset_{name}.json")
            with open(out_path, "w") as f:
                json.dump(coco, f)
            print(f"[{name}] {stats}  -> {out_path}")

    print(f"\nẢnh (file_name trong json) là đường dẫn TƯƠNG ĐỐI so với "
          f"{image_root}\n(vd 'test/4499_b1/ground_truth.jpg') — objdet/datasets.py phải "
          f"trỏ image_root đúng chỗ này khi đăng ký dataset.")


if __name__ == "__main__":
    main()

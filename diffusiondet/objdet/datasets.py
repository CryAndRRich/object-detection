"""Đăng ký 3 dataset dùng trong repo này với detectron2.

Gốc dữ liệu lấy từ biến môi trường ``OBJDET_DATA_ROOT`` (mặc định ``./data``). Layout
mong đợi — đúng như thư mục sinh ra bởi ``data/_scripts`` và file ``objdet-data.zip``:

    $OBJDET_DATA_ROOT/
    ├── coco_minitrain/
    │   ├── annotations/instances_minitrain2017.json
    │   └── images/train2017/
    ├── coco/
    │   ├── annotations/instances_val2017.json
    │   └── val2017/
    ├── voc/VOCdevkit/{VOC2007,VOC2012}/
    └── crowdhuman/
        ├── images_train/  images_val/
        ├── annotation_train.odgt  annotation_val.odgt
        └── annotations/            <- json sinh bởi tools/convert_crowdhuman.py

Dataset đăng ký:

    coco_minitrain_train      25.000 ảnh, 80 class   (train)
    coco_2017_val_local       5.000 ảnh, 80 class    (eval cho minitrain)
    voc_2007_trainval / voc_2012_trainval / voc_2007_test        20 class
    crowdhuman_fbox_train / crowdhuman_fbox_val      1 class (full body)
    crowdhuman_vbox_train / crowdhuman_vbox_val      1 class (visible body)
    ce130_agnostic_{train,val,test}                  1 class  (EXPERIMENT D.1)
    ce130_closedset_{train72,val72}                  72 class (EXPERIMENT D.2)

CE-130 (D.1/D.2, xem docs/thiet-ke-experiment-d-diffusiondet-ce130.md) cần chạy
``tools/convert_ce130.py`` trước để sinh json — layout mong đợi::

    $OBJDET_DATA_ROOT/
    ├── all_phase2_V2/{train,val,test}/{id}_b{N}/    ảnh gốc + annotation CE-130
    └── ce130_coco/                                  json do convert_ce130.py sinh
        ├── ce130_agnostic_{train,val,test}.json     D.1
        └── ce130_closedset_{train72,val72}.json     D.2

``file_name`` trong json là đường dẫn TƯƠNG ĐỐI so với ``all_phase2_V2/`` (vd
``test/4499_b1/ground_truth.jpg``), nên image_root khi đăng ký phải là thư mục đó.
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances, register_pascal_voc

DATA_ROOT = os.environ.get("OBJDET_DATA_ROOT", "data")

# Số class của từng dataset — phải khớp MODEL.DiffusionDet.NUM_CLASSES trong config.
_NUM_CLASSES = {
    "coco_minitrain_train": 80,
    "coco_2017_val_local": 80,
    "voc_2007_trainval": 20,
    "voc_2012_trainval": 20,
    "voc_2007_test": 20,
    "crowdhuman_fbox_train": 1,
    "crowdhuman_fbox_val": 1,
    "crowdhuman_vbox_train": 1,
    "crowdhuman_vbox_val": 1,
    "ce130_agnostic_train": 1,
    "ce130_agnostic_val": 1,
    "ce130_agnostic_test": 1,
    "ce130_closedset_train72": 72,
    "ce130_closedset_val72": 72,
}


def dataset_num_classes(dataset_name):
    """Số class của dataset đã đăng ký (dùng để check config khớp dữ liệu)."""
    return _NUM_CLASSES.get(dataset_name)


def crowdhuman_ann_dir(root=None):
    """Nơi chứa json CrowdHuman do ``tools/convert_crowdhuman.py`` sinh ra.

    Mặc định nằm cạnh dữ liệu. Nhưng trên Kaggle ``/kaggle/input`` là read-only, nên nếu
    json chưa được đóng gói sẵn trong dataset thì phải trỏ chỗ khác ghi được:

        OBJDET_CROWDHUMAN_ANN_DIR=/kaggle/working/crowdhuman_ann
    """
    root = root or DATA_ROOT
    return os.environ.get(
        "OBJDET_CROWDHUMAN_ANN_DIR", os.path.join(root, "crowdhuman/annotations")
    )


def _register_coco_minitrain(root):
    """COCO-minitrain 25K để train, COCO val2017 để eval.

    Dùng ``register_coco_instances`` nên metadata (80 class, mapping id -> index) suy
    ra từ chính json, khớp với COCO chuẩn -> checkpoint DiffusionDet chính thức
    (80 class) vẫn dùng được nếu sau này muốn đối chiếu.
    """
    register_coco_instances(
        "coco_minitrain_train",
        {},
        os.path.join(root, "coco_minitrain/annotations/instances_minitrain2017.json"),
        os.path.join(root, "coco_minitrain/images/train2017"),
    )
    register_coco_instances(
        "coco_2017_val_local",
        {},
        os.path.join(root, "coco/annotations/instances_val2017.json"),
        os.path.join(root, "coco/val2017"),
    )


def _register_voc(root):
    """PASCAL VOC 07+12.

    ``register_pascal_voc`` gán evaluator_type = "pascal_voc"; với year=2007
    detectron2 dùng VOC07 11-point metric — đúng giao thức mà các baseline
    (Faster R-CNN 76,4 / detectron2 R50-C4 80,3 AP50) báo cáo.
    """
    # import detectron2.data.datasets (ở đầu file) đã tự đăng ký sẵn 3 cái tên này,
    # trỏ vào "datasets/VOC2007" mặc định (builtin.py) -> phải gỡ trước khi đăng ký
    # đè lại bằng đường dẫn thật. Phải gỡ CẢ HAI catalog: DatasetCatalog (loader) lẫn
    # MetadataCatalog (dirname, thing_classes, ...) — Metadata.__setattr__ tự assert
    # "không đổi giá trị attribute đã set", nên chỉ gỡ DatasetCatalog thôi vẫn vỡ khi
    # register_pascal_voc cố set lại "dirname" khác giá trị cũ.
    for name in ("voc_2007_trainval", "voc_2007_test", "voc_2012_trainval"):
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
        if name in MetadataCatalog.list():
            MetadataCatalog.remove(name)

    devkit = os.path.join(root, "voc/VOCdevkit")
    register_pascal_voc("voc_2007_trainval", os.path.join(devkit, "VOC2007"), "trainval", 2007)
    register_pascal_voc("voc_2007_test", os.path.join(devkit, "VOC2007"), "test", 2007)
    register_pascal_voc("voc_2012_trainval", os.path.join(devkit, "VOC2012"), "trainval", 2012)
    # register_pascal_voc() KHÔNG tự set evaluator_type — detectron2 chỉ set cờ này
    # ở builtin.py, một dòng riêng ngay sau lệnh gọi register_pascal_voc. Không set
    # tay thì build_evaluator() vỡ AttributeError lúc eval (không vỡ lúc train nên
    # dễ lọt qua smoke test ngắn không có eval).
    for name in ("voc_2007_trainval", "voc_2007_test", "voc_2012_trainval"):
        MetadataCatalog.get(name).set(evaluator_type="pascal_voc", objdet_root=root)


def _register_crowdhuman(root):
    """CrowdHuman, cả hai loại box.

    ``fbox`` (full body) là loại mà baseline Table 7 của paper DiffusionDet dùng —
    số Faster R-CNN 85,0 AP50 / 50,4 mMR / 90,2 Recall khớp chính xác baseline FPN
    full-body trong paper CrowdHuman gốc (84,95 / 50,42 / 90,24).
    ``vbox`` (visible) là loại dùng cho Table 1 (zero-shot COCO -> CrowdHuman).

    Json do ``tools/convert_crowdhuman.py`` sinh ra; nếu chưa chạy converter thì các
    dataset này vẫn đăng ký được nhưng sẽ lỗi khi thực sự load.
    """
    ch = os.path.join(root, "crowdhuman")
    ann_dir = crowdhuman_ann_dir(root)
    for box_type in ("fbox", "vbox"):
        for split, img_dir in (("train", "images_train"), ("val", "images_val")):
            register_coco_instances(
                f"crowdhuman_{box_type}_{split}",
                {"thing_classes": ["person"]},
                os.path.join(ann_dir, f"crowdhuman_{box_type}_{split}.json"),
                os.path.join(ch, img_dir),
            )
            MetadataCatalog.get(f"crowdhuman_{box_type}_{split}").set(
                crowdhuman_box_type=box_type,
                crowdhuman_odgt=os.path.join(ch, f"annotation_{split}.odgt"),
            )


def ce130_ann_dir(root=None):
    """Nơi chứa json CE-130 do ``tools/convert_ce130.py`` sinh ra.

    Mặc định ``$OBJDET_DATA_ROOT/ce130_coco`` — tức **bên trong** ``data/``, cạnh
    ``all_phase2_V2/`` (xem ``data/README.md`` §8), khớp đúng chỗ converter ghi ra khi
    chạy ``--ce130-root ../data/all_phase2_V2`` (mặc định của nó là
    ``<ce130-root>/../ce130_coco``). Trên Kaggle ``/kaggle/input`` read-only nên trỏ chỗ
    ghi được::

        OBJDET_CE130_ANN_DIR=/kaggle/working/ce130_ann
    """
    root = root or DATA_ROOT
    return os.environ.get("OBJDET_CE130_ANN_DIR", os.path.join(root, "ce130_coco"))


def ce130_image_root(root=None):
    """Thư mục ảnh gốc CE-130 — mặc định ``$OBJDET_DATA_ROOT/all_phase2_V2``.

    ``file_name`` trong json do ``convert_ce130.py`` sinh là đường dẫn TƯƠNG ĐỐI so với
    thư mục này (vd ``test/4499_b1/ground_truth.jpg``), KHÔNG phải so với ``ann_dir``.
    Ghi đè bằng ``OBJDET_CE130_IMAGE_ROOT`` nếu ảnh nằm chỗ khác json.
    """
    root = root or DATA_ROOT
    return os.environ.get("OBJDET_CE130_IMAGE_ROOT",
                          os.path.join(root, "all_phase2_V2"))


def _register_ce130(root):
    """CE-130 — EXPERIMENT D (đối chứng cho CE-LocModel A/B/C, không phải cải tiến).

    D.1 (class-agnostic, LÀM TRƯỚC): 3 split gốc, mỗi ảnh 1 class nên "mọi vật trong
    ảnh" == "vật thuộc category ảnh đó" (đã đo: 3.598/3.598) — hợp lệ làm đối chứng.
    D.2 (closed-set, LÀM SAU): chỉ 2 dataset train72/val72, chia LẠI nội bộ split train
    gốc (72 class) — KHÔNG so được với D.1/A/B/C chạy trên test 28-class zero-shot,
    chỉ so được với chính D.1 trên cùng split train72/val72 này.

    Json chưa sinh thì đăng ký vẫn không lỗi (giống crowdhuman) nhưng load sẽ báo lỗi
    file không tồn tại — chạy ``tools/convert_ce130.py`` trước.
    """
    ann_dir = ce130_ann_dir(root)
    img_root = ce130_image_root(root)

    for split in ("train", "val", "test"):
        register_coco_instances(
            f"ce130_agnostic_{split}",
            {},
            os.path.join(ann_dir, f"ce130_agnostic_{split}.json"),
            img_root,
        )

    for split in ("train72", "val72"):
        register_coco_instances(
            f"ce130_closedset_{split}",
            {},
            os.path.join(ann_dir, f"ce130_closedset_{split}.json"),
            img_root,
        )


def register_all(root=None):
    """Đăng ký tất cả. Gọi nhiều lần an toàn (bỏ qua dataset đã đăng ký).

    Kiểm tra bằng ``DatasetCatalog`` chứ không phải ``MetadataCatalog``: ``MetadataCatalog.get``
    tự tạo entry rỗng cho tên chưa tồn tại, nên dùng nó để kiểm tra sẽ cho kết quả sai.

    Riêng VOC là ngoại lệ: import detectron2.data.datasets tự đăng ký sẵn 3 cái tên
    ``voc_2007_*``/``voc_2012_*`` (builtin.py, trỏ sai đường dẫn), nên kiểm tra bằng
    ``DatasetCatalog`` luôn thấy "đã có" dù chưa phải bản của mình. Phải dùng marker
    ``objdet_root`` riêng để phân biệt "đã có do detectron2" và "đã có do ta đăng ký".
    """
    root = root or DATA_ROOT
    already = set(DatasetCatalog.list())
    if "coco_minitrain_train" not in already:
        _register_coco_minitrain(root)
    if MetadataCatalog.get("voc_2007_trainval").get("objdet_root") != root:
        _register_voc(root)
    if "crowdhuman_fbox_train" not in already:
        _register_crowdhuman(root)
    if "ce130_agnostic_train" not in already:
        _register_ce130(root)
    return root

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
    # đè lại bằng đường dẫn thật, nếu không register_pascal_voc bên dưới sẽ vỡ vì
    # trùng tên (DatasetCatalog không cho đăng ký 2 lần).
    for name in ("voc_2007_trainval", "voc_2007_test", "voc_2012_trainval"):
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)

    devkit = os.path.join(root, "voc/VOCdevkit")
    register_pascal_voc("voc_2007_trainval", os.path.join(devkit, "VOC2007"), "trainval", 2007)
    register_pascal_voc("voc_2007_test", os.path.join(devkit, "VOC2007"), "test", 2007)
    register_pascal_voc("voc_2012_trainval", os.path.join(devkit, "VOC2012"), "trainval", 2012)
    for name in ("voc_2007_trainval", "voc_2007_test", "voc_2012_trainval"):
        MetadataCatalog.get(name).set(objdet_root=root)


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
    return root

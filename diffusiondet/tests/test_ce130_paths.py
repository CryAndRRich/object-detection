"""Đường dẫn converter GHI RA phải khớp đường dẫn datasets.py ĐỌC VÀO — không cần
detectron2.

Vì sao cần test riêng: bug đã mắc thật. ``objdet/datasets.py`` từng dùng
``os.path.join(root, "..", "ce130_coco")`` trong khi converter ghi vào
``<ce130-root>/../ce130_coco``. Với ``OBJDET_DATA_ROOT=../data`` (đúng như README hướng
dẫn) thì converter ghi ra ``data/ce130_coco`` còn datasets.py đi tìm ở
``object-detection/ce130_coco`` — lệch đúng một cấp thư mục, cho CẢ json LẪN ảnh.

Không test này thì bug chỉ lộ khi thật sự train trên GPU (crash "file not found" ở bước
load) — tức sau khi đã đẩy code lên server và xếp hàng chờ GPU.

Chạy: python tests/test_ce130_paths.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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


if __name__ == "__main__":
    main()

"""CE-130 ảnh GỐC + prompt = tâm vật (density CountGD), cho segment map.

KHÁC `ce130_coco.py`: file đó đọc `ce130_coco/ce130_agnostic_{split}.json` do
`convert_ce130.py` sinh ra, nhưng json đó không còn sau khi repo tái cấu trúc.
Ở đây đọc thẳng `all_phase2_V2/` qua `scan_ce130` của ce_localization — cùng
một nguồn mà ALPHA/BETA đang dùng, nên ảnh và box khớp từng bit với chúng.

                        VÌ SAO PROMPT LÀ TÂM VẬT

Lưới đều của paper rải prompt mù. Đo trên 590 nhánh CE-130 val:

    nguồn prompt       prompt/ảnh   % rơi TRONG box GT   % box GT được chạm
    tâm density              20            100 %               92,3 %
    lưới stride 6            88             40,9 %             96,6 %

Tâm vật cho prompt ÍT HƠN 4,4× mà mọi prompt đều trúng vật. Vì ~95 % thời gian
chạy nằm ở lan truyền (tỉ lệ với số prompt), đây là 4× nhanh hơn. Đổi lại mất
~4 điểm coverage — density có số blob ≈ 0,9 × số vật và 1,4 % ảnh density trống.

⚠️ CHẤP NHẬN ĐƯỢC vì mục tiêu KHÔNG phải tách từng vật: self-attention của SD
"does not take object instances into account" (M2N2 §3.3), nên một prompt lan
sang MỌI vật giống nó. Vật thiếu prompt vẫn được phủ nếu có vật cùng loại được
chạm. Đây là lý do bỏ bước connected-components (xem `tools/run_ce130_points.py`).

                         NGUỒN ĐIỂM, VÀ MỘT CẢNH BÁO

`data/density_points.json` (ce_localization, `tools/build_density_points.py`):
đỉnh của density `full` = bản density diện tích lớn nhất của mỗi ảnh.
⚠️ Density được sinh trên ảnh **đã inpaint**, nên bản `full` vẫn THIẾU vật bị
xoá ở lượt 1. Ta chạy trên ảnh GỐC (`ground_truth.jpg`) nên luôn có ít nhất một
vật không có prompt. Đã tính tới ở đoạn trên.
"""

import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from ce_localization.data.dataset import scan_ce130  # noqa: E402

__all__ = ["CE130Points"]

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])


def resize_and_pad(img, target):
    """Giữ tỉ lệ, dán GÓC TRÊN-TRÁI, đệm bằng CLIP mean. Trả (canvas, valid_w, valid_h).

    Y hệt `letterbox` của ce_localization (và `resize_and_pad` của CE-Loc gốc),
    nên toạ độ của hai bên so được trực tiếp.
    """
    W, H = img.size
    s = min(target / float(W), target / float(H))
    nw, nh = int(W * s), int(H * s)
    canvas = np.empty((target, target, 3), dtype=np.uint8)
    canvas[:] = (CLIP_MEAN * 255).round().astype(np.uint8)
    canvas[:nh, :nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    return canvas, nw / float(target), nh / float(target)


class CE130Points:
    """Indexable: ảnh gốc CE-130 + điểm density + box GT (chỉ để chấm/vẽ)."""

    def __init__(self, root, split, points_json, canvas=512):
        self.root = root
        self.split = split
        self.items = scan_ce130(root, split)
        self.canvas = canvas
        with open(points_json) as f:
            blob = json.load(f)
        self.points = blob["points"]
        self.points_params = blob.get("params", {})
        self.n_no_points = sum(1 for it in self.items
                               if not self.points.get(str(it["image_id"])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        with Image.open(it["img_path"]) as raw:
            raw = raw.convert("RGB")
            W, H = raw.size
            canvas, valid_w, valid_h = resize_and_pad(raw, self.canvas)

        pts = np.asarray(self.points.get(str(it["image_id"]), []),
                         dtype=np.float64).reshape(-1, 2)
        return {
            "image": canvas,
            "valid_w": valid_w,
            "valid_h": valid_h,
            "points_xy": pts,              # pixel ẢNH GỐC
            "gt_boxes_xyxy": it["boxes_xyxy_px"],
            "text": it["text"],
            "image_id": it["image_id"],
            "file_name": os.path.relpath(it["img_path"], self.root),
            "W": W,
            "H": H,
        }

"""Cấu hình paper, NHƯNG ở độ phân giải gốc của SD 1.5 (512 thay vì 1120).

MỘT BIẾN DUY NHẤT ĐỔI: `canvas` 1120 -> 512, kéo theo `grid_r` 140 -> 64. Mọi
thứ khác giữ nguyên giá trị paper. Mục đích là trả lời đúng một câu:

    "AR thấp là vì SD 1.5 chạy ở 2,2x độ phân giải train của nó, hay vì thứ
     khác?"

VÌ SAO CÂU ĐÓ ĐÁNG HỎI. Paper chạy SD2, train ở 1024, dùng 1120 — tức 1,09x, và
họ nói rõ đó đã là "outside the native resolution of SD2 but still yields good
segmentation results". SD 1.5 train ở **512**, nên 1120 là **2,2x** — xa hơn
gấp đôi. Không ai đo Diffuse2Seg ở chế độ đó.

ĐÁNH ĐỔI, PHẢI IN RA CÙNG KẾT QUẢ: `grid_r=64` nghĩa là một ô latent = 8 px
canvas, và ở canvas 512 thì 1 ô ~ 10 px ảnh gốc (ảnh PACO trung vị 640x480).
Đo trên PACO val: area trung vị chỉ **362 px²** (cạnh ~19 px), nên vật trung vị
còn chưa tới 2 ô. `a_min = 100 px` vẫn tính trên ảnh gốc nên không đổi, nhưng
số mask nhỏ mất đi sẽ nhiều hơn hẳn. `run_paper.py` in trần độ phân giải thực
đo, đối chiếu được với 90,7 % ở grid_r=140.

CHẠY:
    python tools/run_on_free_gpu.py -- tools/run_paper.py --dataset paco \\
        --limit 50 --canvas 512 --out <log>/d2s_paper_paco_512.json

(`--canvas 512` trên dòng lệnh cho kết quả y hệt file này; file tồn tại để cấu
hình có tên gọi được, và để chỗ ghi lý do.)

⚠️ ĐỌC KẾT QUẢ: so 512 với 1120 chỉ nói về ĐỘ PHÂN GIẢI. Nếu cả hai đều thấp
so với 13,6 của paper thì nguyên nhân nằm ở chỗ khác — nhiều khả năng nhất là
`t=150` tinh chỉnh cho SD2 (xem ghi chú timestep trong paper.py: hướng quét
đúng là t NHỎ HƠN), hoặc dải `[h_min, h_max]` không khớp thang KL của SD1.5.
"""

from .base import Diffu2SegConfig
from .paper import cfg as _paper

cfg = Diffu2SegConfig(**{**_paper.__dict__, "canvas": 512}).validate()

# object-detection

Ba sub-project, phục vụ dự án `multi_condition` (xem [`../CLAUDE.md`](../CLAUDE.md)) theo hai
hướng khác nhau — hai cái đầu là **object detector dùng diffusion process** (khảo sát cơ chế đem
về CE-Loc), cái thứ ba là **chính CE-Loc**:

| | Trạng thái | Chi tiết |
|---|---|---|
| [`diffusiondet/`](diffusiondet/README.md) | **Đã train/eval xong trên Kaggle, không chạy lại** — chỉ còn giữ để đối chiếu import khi cấu trúc thư mục đổi | Số liệu đầy đủ ở [`../RESULTS.md`](../RESULTS.md) |
| [`diffu_grounding_dino/`](diffu_grounding_dino/README.md) | **Chưa train** — code đã verify đúng kiến trúc + checkpoint key-compat + 80 test (kể cả DDP 2-process), sắp finetune trên GPU server hoặc Kaggle 2×T4 | GroundingDINO + diffusion process trên reference point của decoder |
| [`count_editing/`](count_editing/README.md) | **Chưa train thật trên server** — code CE-LocModel đã sửa/ablation xong cục bộ, mới chuyển vào đây từ `multi_condition/count_editing/` để đi chung git repo | CE-Loc/CE-Gen (Add One, Take One, NeurIPS 2026) — bài của chính người dùng dự án |

Ba project **không phụ thuộc lẫn nhau** (không import chéo) — có thể đọc/sửa/chạy độc lập.
Phần dùng chung: `LICENSE`, `.gitignore`, `weights/` (tách sub-thư mục riêng cho mỗi project),
`data/` (dùng chung — COCO-minitrain cho 2 detector, `samples/`+`all_phase2_V2/` cho CE-Loc) —
xem dưới. **Trước đây `count_editing/` sửa xong phải zip đưa lên server; giờ đi cùng git repo này
nên chỉ cần `git pull`** — riêng `data/`/`weights/` vẫn không push git, vẫn zip/copy thủ công như
cũ.

## `weights/`

```
weights/
├── diffusiondet/           7 checkpoint .pth (3 tự train + 4 pretrained gốc), 5,8GB
└── diffu_grounding_dino/   groundingdino_swint_ogc.pth + bert-base-uncased/, 1,1GB
```

Nằm trong `.gitignore` — không push git. Copy/zip thủ công lên máy chạy (Kaggle trước đây,
GPU server bây giờ). Xem README của từng sub-project để biết checkpoint nào dùng cho việc gì.
(CE-LocModel checkpoint riêng do người dùng train, xem `count_editing/CE-LocModel/README.md`.)

## `data/`

3 dataset detector (COCO-minitrain 25K, VOC 07+12, CrowdHuman) + 2 thư mục CE-Loc
(`samples/` — dataset train gốc, `all_phase2_V2/` — dump CE-130 đầy đủ cho metric C-NLL), ~32GB —
xem [`data/README.md`](data/README.md) (provenance/checksum, layout, script tải). Dùng chung: cả
`diffusiondet/` lẫn `diffu_grounding_dino/` đều đọc COCO-minitrain từ đây; VOC/CrowdHuman hiện
chỉ `diffusiondet/` dùng; `samples/`/`all_phase2_V2/` chỉ `count_editing/` dùng. Cũng nằm trong
`.gitignore`, không push git — zip thủ công lên máy chạy giống `weights/`.

## Ghi công

DiffusionDet: Shoufa Chen, Peize Sun, Yibing Song, Ping Luo — [arXiv 2211.09788](https://arxiv.org/abs/2211.09788),
giấy phép CC-BY-NC 4.0 (xem [LICENSE](LICENSE)). GroundingDINO: Shilong Liu et al. — ECCV 2024.
DiffuGroundingDINO tự viết lại toàn bộ theo công thức, không import từ `repos/` — chi tiết ghi
công/tham khảo trong [`diffu_grounding_dino/README.md`](diffu_grounding_dino/README.md).
Count-Editing (CE-Loc/CE-Gen): bài NeurIPS 2026 của chính người dùng dự án — xem
[`count_editing/README_upstream.md`](count_editing/README_upstream.md) cho README gốc.

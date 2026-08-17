# object-detection

Hai sub-project độc lập, đều xoay quanh **object detector dùng diffusion process**, phục vụ
việc nâng cấp phần CE-Loc của dự án `multi_condition` (xem
[`../CLAUDE.md`](../CLAUDE.md)):

| | Trạng thái | Chi tiết |
|---|---|---|
| [`diffusiondet/`](diffusiondet/README.md) | **Đã train/eval xong trên Kaggle, không chạy lại** — chỉ còn giữ để đối chiếu import khi cấu trúc thư mục đổi | Số liệu đầy đủ ở [`../RESULTS.md`](../RESULTS.md) |
| [`diffu_grounding_dino/`](diffu_grounding_dino/README.md) | **Chưa train** — code đã verify đúng kiến trúc + checkpoint key-compat + 80 test (kể cả DDP 2-process), sắp finetune trên GPU server hoặc Kaggle 2×T4 | GroundingDINO + diffusion process trên reference point của decoder |

Hai project **không phụ thuộc lẫn nhau** (không import chéo) — có thể đọc/sửa/chạy độc lập.
Phần dùng chung: `LICENSE`, `.gitignore`, `weights/` (tách sub-thư mục riêng cho mỗi project),
`data/` (dùng chung, chủ yếu COCO-minitrain) — xem dưới.

## `weights/`

```
weights/
├── diffusiondet/           7 checkpoint .pth (3 tự train + 4 pretrained gốc), 5,8GB
└── diffu_grounding_dino/   groundingdino_swint_ogc.pth + bert-base-uncased/, 1,1GB
```

Nằm trong `.gitignore` — không push git. Copy/zip thủ công lên máy chạy (Kaggle trước đây,
GPU server bây giờ). Xem README của từng sub-project để biết checkpoint nào dùng cho việc gì.

## `data/`

3 dataset (COCO-minitrain 25K, VOC 07+12, CrowdHuman), 18GB — xem
[`data/README.md`](data/README.md) (provenance/checksum, layout, script tải). Dùng chung: cả
`diffusiondet/` lẫn `diffu_grounding_dino/` đều đọc COCO-minitrain từ đây; VOC/CrowdHuman hiện
chỉ `diffusiondet/` dùng. Cũng nằm trong `.gitignore`, không push git — zip thủ công lên máy
chạy giống `weights/`.

## Ghi công

DiffusionDet: Shoufa Chen, Peize Sun, Yibing Song, Ping Luo — [arXiv 2211.09788](https://arxiv.org/abs/2211.09788),
giấy phép CC-BY-NC 4.0 (xem [LICENSE](LICENSE)). GroundingDINO: Shilong Liu et al. — ECCV 2024.
DiffuGroundingDINO tự viết lại toàn bộ theo công thức, không import từ `repos/` — chi tiết ghi
công/tham khảo trong [`diffu_grounding_dino/README.md`](diffu_grounding_dino/README.md).

# object-detection

Hai sub-project độc lập, đều xoay quanh **object detector dùng diffusion process**, phục vụ
việc nâng cấp phần CE-Loc của dự án `multi_condition` (xem
[`../CLAUDE.md`](../CLAUDE.md)):

| | Trạng thái | Chi tiết |
|---|---|---|
| [`diffusiondet/`](diffusiondet/README.md) | **Đã train/eval xong trên Kaggle, không chạy lại** — chỉ còn giữ để đối chiếu import khi cấu trúc thư mục đổi | Số liệu đầy đủ ở [`../RESULTS.md`](../RESULTS.md) |
| [`diffu_grounding_dino/`](diffu_grounding_dino/README.md) | **Chưa train** — code đã verify đúng kiến trúc + checkpoint key-compat + 79 test, sắp finetune trên GPU server | GroundingDINO + diffusion process trên reference point của decoder |

Hai project **không phụ thuộc lẫn nhau** (không import chéo) — có thể đọc/sửa/chạy độc lập.
Phần dùng chung: `LICENSE`, `.gitignore`, `weights/` (tách sub-thư mục riêng cho mỗi project,
xem dưới).

## `weights/`

```
weights/
├── diffusiondet/           7 checkpoint .pth (3 tự train + 4 pretrained gốc), 5,8GB
└── diffu_grounding_dino/   groundingdino_swint_ogc.pth + bert-base-uncased/, 1,1GB
```

Nằm trong `.gitignore` — không push git. Copy/zip thủ công lên máy chạy (Kaggle trước đây,
GPU server bây giờ). Xem README của từng sub-project để biết checkpoint nào dùng cho việc gì.

## Ghi công

DiffusionDet: Shoufa Chen, Peize Sun, Yibing Song, Ping Luo — [arXiv 2211.09788](https://arxiv.org/abs/2211.09788),
giấy phép CC-BY-NC 4.0 (xem [LICENSE](LICENSE)). GroundingDINO: Shilong Liu et al. — ECCV 2024.
DiffuGroundingDINO tự viết lại toàn bộ theo công thức, không import từ `repos/` — chi tiết ghi
công/tham khảo trong [`diffu_grounding_dino/README.md`](diffu_grounding_dino/README.md).

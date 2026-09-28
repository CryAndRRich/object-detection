# diffusiondet — DiffusionDet + baseline D.1 trên CE-130

[DiffusionDet](https://arxiv.org/abs/2211.09788) (R50-FPN), code model lấy từ repo gốc
(CC-BY-NC 4.0, [../LICENSE](../LICENSE)). Hai việc đã XONG, **không chạy lại**:

1. **3 benchmark** (COCO-minitrain 25K, VOC 07+12, CrowdHuman) train trên Kaggle 2× T4 — số liệu
   ở `docs/old/ROUND_1_ARCHIVE.md` (phần 06), notebook ở `../../notebooks/`.
2. **D.1 = BASELINE của CE-Loc**: cùng code, chạy trên CE-130 class-agnostic (mục cuối).

## Dữ liệu

| Dataset | Train | Eval | class |
|---|---|---|---|
| COCO-minitrain 25K | 25.000 ảnh | COCO `val2017` (5.000) | 80 |
| VOC 07+12 | 16.551 ảnh | VOC2007 `test` (4.952) | 20 |
| CrowdHuman | 15.000 ảnh | `val` (4.370) | 1 |
| CE-130 (D.1) | 1.911 ảnh / 71.767 box | test 779 ảnh / 37.812 box | 1 |

```bash
export OBJDET_DATA_ROOT=../data     # tương đối với thư mục ĐANG ĐỨNG, không phải file code
```

Layout: `coco_minitrain/`, `coco/`, `voc/VOCdevkit/`, `crowdhuman/` (json CrowdHuman đã sinh;
sinh lại: `tools/convert_crowdhuman.py --box-type fbox`), `ce130_coco/` (sinh bằng
`tools/convert_ce130.py`).

## Cài đặt

`pip install -r ../requirements.txt`, rồi detectron2 **từ source, nhánh `main`** (tag v0.6 còn
`PIL.Image.LINEAR` đã bị xoá ở Pillow 10). ⚠️ Env `ce-locmodel` trên server hiện **không import
được detectron2** — lệnh dựng lại ở [`../README.md`](../README.md) mục "Môi trường".
Test (không cần GPU/detectron2): `python -m pytest tests/diffusiondet -q` từ `object-detection/`.

## Chạy

```bash
python tools/train_net.py --num-gpus 2 --config-file configs/diffdet.minitrain.res50.yaml [--resume]
python tools/train_net.py --num-gpus 2 --config-file configs/diffdet.minitrain.res50.yaml \
    --eval-only MODEL.WEIGHTS ../weights/diffusiondet/minicoco_model_final.pth
python tools/summarize.py output/minitrain_res50 --dataset coco_minitrain   # so baseline
```

**Dynamic boxes**: train một lần, eval nhiều cấu hình — đổi `MODEL.DiffusionDet.NUM_PROPOSALS`
và `SAMPLE_STEP` trên dòng lệnh, không train lại. Số box tối ưu bám mật độ vật (VOC 2,4 vật/ảnh
đỉnh ở 1000; CrowdHuman 22,8 vẫn tăng ở 3000).

**Kỳ vọng**: config ở đây ≈ **1/40 compute** của paper (45k iter × batch 4 vs 450k × 16) ⇒ số
thấp hơn baseline là do ngân sách, không do phương pháp. Báo cáo luôn ghi iteration + batch.

Baseline đã công bố: [`baselines/baselines.yaml`](baselines/baselines.yaml) (minitrain:
Faster R-CNN 27,7 AP; VOC AP50: Faster R-CNN R101 76,4; CrowdHuman fbox: Faster R-CNN
85,0 / 50,4 / 90,2, DiffusionDet 3@1000 91,4 / 45,7 / 98,4).

## Sửa gì so với repo gốc

9/13 file model y nguyên từng byte (kể cả `detector.py`, `head.py`, `loss.py`). Ba chỗ sửa đều
**ngoài** đường chạy R50: `timm.models.layers` → `timm.layers`; `indexing="ij"` cho
`torch.meshgrid`; bỏ nhánh torchvision < 0.7 trong `util/misc.py` (so version sai nên luôn
true). Phần thêm mới: `objdet/` (dataset, metric), `tools/`, `configs/`.

## Những chỗ dễ sai

- **Số class phải khớp dataset** (80/20/1) — sai thì vẫn train, kết quả vô nghĩa;
  `train_net.py` raise nếu lệch.
- **CrowdHuman dùng `fbox`** (full body) — baseline Table 7 là fbox. `tag == "mask"` và
  `extra.ignore == 1` thành `iscrowd=1` (vùng ignore).
- **Inference batch 1** — `ddim_sample` giả định batch 1 khi box renewal.
- ⛔ **KHÔNG bật AMP.** `scale_clamp ≈ 8,74` cho phép mỗi stage nhân box tới 6250×; qua 6 stage
  box trung gian lên 1e18 — fp16 tràn ở 65504 → `inf - inf = NaN` → `assert x2 >= x1`. Không vá
  được mà không đổi model. Log có dòng `with autocast(...)` của `AMPTrainer` = AMP đang bật.
- **COCO-minitrain có nhiều bản** — chỉ split của `giddyyupp/coco-minitrain` so được với 27,7 AP
  (bản HuggingFace `bryanbocao/...` chỉ trùng 5.281/25.000 file).
- **`z_T` là Gaussian** cả lúc train lẫn infer, không uniform; placeholder GT là `N(0.5, 1/6²)`.
- **Proposal nhìn nhau qua self-attention** ở mỗi trong 6 stage (`head.py:185`), như DETR.
- **`SAMPLE_STEP > 1`: bước cuối KHÔNG vào pool NMS** (`img = x_start; continue` chạy trước
  `ensemble_coord.append`) — `SAMPLE_STEP=4` chỉ dùng kết quả 3 bước đầu. Không đổi số đã đo,
  chỉ dễ hiểu lầm khi tự viết code visualize theo bước.

## D.1 — baseline trên CE-130 (2026-09-09)

Detector chuẩn đặt lên đúng dữ liệu CE-130 để biết trần thực tế của bài toán định vị.
**Hợp lệ class-agnostic** vì mỗi ảnh CE-130 chỉ có đúng 1 lớp (3.598/3.598). **Không có text**
(grep không ra đường text nào; json `categories: [{'id':1,'name':'object'}]`).

**Cấu hình**: `configs/diffdet.ce130.res50.yaml`, finetune ImageNet `R-50.pkl`, **12.000 iter**,
**batch 2**, LR 8,84e-6, STEPS (9000, 11000) = 12,6 epoch, **1× A30, ~1h07m**, max_mem 4,2 GB.

| N (test) | oracle_recall | mean_bestIoU | score_AUC | AP | AP50 |
|---|---|---|---|---|---|
| **300** | **0,6734** | 0,5974 | **0,9371** | 34,22 | **58,13** |
| 1000 | 0,7853 | 0,6763 | 0,9483 | 37,43 | 63,65 |
| 3000 | 0,8349 | 0,7062 | 0,9489 | 38,27 | 65,22 |

AP50 theo iteration: 40,6 → 51,5 → 54,6 → 56,4 → 57,9 → 58,1 (gần bão hoà).

```bash
export OBJDET_DATA_ROOT=../data
python tools/convert_ce130.py --ce130-root ../data/all_phase2_V2 --mode class-agnostic
python tools/visualize_ce130_coco.py --json ../data/ce130_coco/ce130_agnostic_train.json \
    --image-root ../data/all_phase2_V2 --out /mnt/disk1/aiotlab/haitn/output/ce130_viz --n 12
python tools/train_net.py --num-gpus 1 --config-file configs/diffdet.ce130.res50.yaml
for N in 300 1000 3000; do
  python tools/measure_box_quality_ce130.py --config-file configs/diffdet.ce130.res50.yaml \
      --num-proposals $N MODEL.WEIGHTS output/ce130_agnostic_res50/model_final.pth
done
```

`NUM_PROPOSALS: 300` chỉ cho lúc train; eval giữa chừng ở 300 chỉ để xem đường cong. CE-130 có
48,5 vật/ảnh (test, max 505) nên N=300 tự chặn trần recall — tool in cảnh báo khi xảy ra.

**Chất lượng dữ liệu CE-130** (không sửa, giữ để mọi thí nghiệm đọc cùng dữ liệu):
- **Lô annotation HỎNG ở test**: 16 ảnh (15 id dạng `62xx`) có 855 box > 50 % diện tích ảnh
  (ảnh `6261`: 293/325 box bao gần trọn ảnh) = **4,2 % GT test** không detector nào khớp được.
  Xem: `visualize_ce130_coco.py ... --suspect-only`.
- **Các branch cùng ảnh bất đồng GT** ở `fixed_annotation.json` (86,5 % ảnh val, 79,7 % test;
  lệch toạ độ tới 374 px, số box gần như không đổi). Converter chọn branch chỉ số nhỏ nhất
  (khớp `ce_localization/data/ce130_dataset.py`) và in số ảnh bất đồng.
- Converter khớp `CE130Detection.stats()` từng số: 1.911 / 908 / 779 ảnh, **71.767** / 38.289 /
  37.812 annotation (train có 85 box thoái hoá bị lọc từ 71.852 box thô).

**Bug đã bắt** (đều có test hoặc đã sửa): `datasets.py` từng đọc json/ảnh thừa một `..` so với
chỗ converter ghi; `measure_box_quality_ce130.py` phải dựng cfg y hệt `train_net.py` (import
thẳng `add_kaggle_configs`, `check_num_classes`); `DiffusionDetDatasetMapper(is_train=False)`
**xoá sạch GT** — đọc GT từ `DatasetCatalog` thay vì mapper.

D-coco (COCO pretrain) và D.2 (closed-set 72 lớp) chưa chạy, không còn ưu tiên. D.2 nếu chạy:
chia split **stratified theo lớp** và dựng bảng `category_id` MỘT lần từ đủ 72 lớp
(`build_cat_id_map`) — hai bẫy đã mắc (lớp val không ⊆ train; 54/72 id trỏ sai tên).

## Ghi công

DiffusionDet: Shoufa Chen, Peize Sun, Yibing Song, Ping Luo — arXiv 2211.09788. Dựa trên
detectron2 và Sparse R-CNN. Giấy phép CC-BY-NC 4.0 theo repo gốc.

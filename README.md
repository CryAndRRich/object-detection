# object-detection — chạy DiffusionDet trên 3 benchmark, so với baseline đã công bố

Train và eval [DiffusionDet](https://arxiv.org/abs/2211.09788) (R50-FPN) trên **COCO-minitrain
25K**, **PASCAL VOC 07+12** và **CrowdHuman**, rồi đối chiếu với baseline đã công bố trên
đúng ba bộ đó. Nhắm vào môi trường **Kaggle 2× Tesla T4** (30h GPU/tuần, ~9–12h mỗi
session) nên train nhiều session và resume từ checkpoint.

Code model lấy từ [repo gốc của DiffusionDet](https://github.com/ShoufaChen/DiffusionDet)
(CC-BY-NC 4.0, xem [LICENSE](LICENSE)) và đã sửa để chạy với thư viện phiên bản mới —
xem [§ Sửa gì so với repo gốc](#sửa-gì-so-với-repo-gốc).

## Mục lục

- [Dữ liệu](#dữ-liệu)
- [Cài đặt](#cài-đặt)
- [Chạy](#chạy)
- [Baseline để so sánh](#baseline-để-so-sánh)
- [Kỳ vọng thực tế về kết quả](#kỳ-vọng-thực-tế-về-kết-quả)
- [Sửa gì so với repo gốc](#sửa-gì-so-với-repo-gốc)
- [Cấu trúc repo](#cấu-trúc-repo)
- [Những chỗ dễ sai](#những-chỗ-dễ-sai)

## Dữ liệu

Ba dataset, tổng 17GB sau khi giải nén:

| Dataset | Train | Eval | Số class |
|---|---|---|---|
| COCO-minitrain 25K | 25.000 ảnh / 183.546 ann | COCO `val2017` (5.000 ảnh / 36.781 ann) | 80 |
| PASCAL VOC 07+12 | 16.551 ảnh (VOC07 5.011 + VOC12 11.540) | VOC2007 `test` (4.952 ảnh) | 20 |
| CrowdHuman | 15.000 ảnh / 438.783 ann | `val` (4.370 ảnh / 127.710 ann) | 1 |

Đặt gốc dữ liệu bằng biến môi trường (mặc định `./data`):

```bash
export OBJDET_DATA_ROOT=/đường/dẫn/tới/data
```

Layout mong đợi:

```
$OBJDET_DATA_ROOT/
├── coco_minitrain/{annotations/instances_minitrain2017.json, images/train2017/}
├── coco/{annotations/instances_val2017.json, val2017/}
├── voc/VOCdevkit/{VOC2007,VOC2012}/
└── crowdhuman/{images_train/, images_val/, annotation_*.odgt,
                annotations/crowdhuman_{fbox,vbox}_{train,val}.json}
```

Json CrowdHuman đã được sinh sẵn. Nếu cần sinh lại:

```bash
python tools/convert_crowdhuman.py --box-type fbox    # full body
python tools/convert_crowdhuman.py --box-type vbox    # visible
# dữ liệu ở /kaggle/input (read-only) thì ghi json ra chỗ khác:
python tools/convert_crowdhuman.py --box-type fbox --out-dir /kaggle/working/ch_ann
export OBJDET_CROWDHUMAN_ANN_DIR=/kaggle/working/ch_ann
```

## Cài đặt

```bash
pip install -r requirements.txt
# detectron2 phải build từ source cho khớp torch/CUDA đang có:
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

Dùng nhánh `main` chứ không phải tag `v0.6`: v0.6 (11/2021) còn `PIL.Image.LINEAR` đã bị
xoá ở Pillow 10, nhánh main đã sửa.

Kiểm tra nhanh phần không cần GPU:

```bash
python tests/test_mmr.py
```

## Chạy

```bash
# train (2 GPU)
python tools/train_net.py --num-gpus 2 --config-file configs/diffdet.minitrain.res50.yaml

# session sau: train tiếp từ checkpoint gần nhất trong OUTPUT_DIR
python tools/train_net.py --num-gpus 2 --config-file configs/diffdet.minitrain.res50.yaml --resume

# eval
python tools/train_net.py --num-gpus 2 --config-file configs/diffdet.minitrain.res50.yaml \
    --eval-only MODEL.WEIGHTS output/minitrain_res50/model_final.pth

# in bảng so sánh với baseline
python tools/summarize.py output/minitrain_res50 --dataset coco_minitrain
```

Ba config: `diffdet.minitrain.res50.yaml`, `diffdet.voc.res50.yaml`,
`diffdet.crowdhuman.res50.yaml`, đều kế thừa `Base-Kaggle-T4x2.yaml`.

### Tính chất dynamic boxes — đáng khai thác

DiffusionDet **train một lần, eval nhiều cấu hình** mà không cần train lại. Đổi số box và
số bước sampling ngay trên dòng lệnh:

```bash
python tools/train_net.py --num-gpus 2 --config-file configs/diffdet.crowdhuman.res50.yaml \
    --eval-only MODEL.WEIGHTS output/crowdhuman_fbox_res50/model_final.pth \
    MODEL.DiffusionDet.NUM_PROPOSALS 1000 \
    MODEL.DiffusionDet.SAMPLE_STEP 4
```

Đây là điểm bán chính của paper (Table 1, Figure 3): AP tăng theo số box và số bước, trong
khi DETR/Sparse R-CNN thì đứng yên hoặc tụt. Rẻ — chỉ tốn thời gian inference — nên nên
chạy quét `{1,4} step × {300,500,1000} boxes` sau khi train xong.

## Baseline để so sánh

Số đầy đủ ở [`baselines/baselines.yaml`](baselines/baselines.yaml). Tóm tắt:

**COCO-minitrain 25K → val2017** (baseline train trên **đúng 25K ảnh này**):
Faster R-CNN R50-FPN 27,7 AP · Mask R-CNN 28,5 · RetinaNet 25,7 · CornerNet 28,4 ·
ExtremeNet 27,3 · HoughNet 23,4

**VOC 07+12 → VOC07 test** (AP50): Faster R-CNN VGG16 73,2 · R101 76,4 · SSD512 76,8 ·
YOLOv2 78,6 · R-FCN 80,5 · detectron2 R50-C4 80,3

**CrowdHuman val, full body** (AP50 / mMR↓ / Recall — Table 7 của paper):
Faster R-CNN 85,0 / 50,4 / 90,2 · Sparse R-CNN 89,2 / 48,3 / 95,9 ·
DiffusionDet 3@1000 91,4 / 45,7 / 98,4

## Kỳ vọng thực tế về kết quả

Đọc phần này trước khi thất vọng vì số thấp.

Paper train DiffusionDet trên COCO bằng **450.000 iteration × batch 16 trên 8 GPU** ≈ 7,2
triệu ảnh đã xem. Config ở đây chạy **45.000 iteration × batch 4** ≈ 180.000 ảnh, tức
khoảng **1/40 lượng compute**. Baseline trên minitrain cũng dùng schedule đầy đủ của
detectron2, dài hơn ta nhiều.

Nên **kết quả tự train sẽ thấp hơn baseline một cách đáng kể, và đó là do ngân sách
compute chứ không phải do phương pháp**. `tools/summarize.py` luôn in kèm số iteration và
in cảnh báo này. Khi báo cáo, ghi rõ iteration + batch size; đừng viết "DiffusionDet kém
hơn Faster R-CNN" nếu chỉ chạy 1/40 schedule.

Muốn có số cao hơn trong cùng ngân sách thì có hai hướng, cả hai đều làm thay đổi ý nghĩa
so sánh nên phải nói rõ khi báo cáo:

1. **Khởi tạo từ checkpoint COCO của DiffusionDet** thay vì từ ResNet-50 ImageNet. Hội tụ
   nhanh hơn nhiều trên VOC/CrowdHuman. Với CrowdHuman thì đây thật ra đúng tinh thần
   "full tuning" của paper. Đặt `MODEL.WEIGHTS` trỏ tới checkpoint đã tải về.
2. **Giảm độ phân giải** (`INPUT.MIN_SIZE_TRAIN`) để chạy được nhiều iteration hơn trong
   cùng số giờ.

## Sửa gì so với repo gốc

Code model (`diffusiondet/`) copy từ repo gốc. **9/13 file y nguyên từng byte**, kể cả
`detector.py`, `head.py`, `loss.py` — tức toàn bộ đường chạy R50-FPN không bị sửa gì.

Ba chỗ sửa, đều **không nằm trên đường chạy R50**:

| Chỗ sửa | Vì sao | Có ảnh hưởng R50? |
|---|---|---|
| `swintransformer.py`: `timm.models.layers` → `timm.layers` (có fallback) | timm ≥ 0.9 đổi đường dẫn module | không — chỉ dùng cho backbone Swin |
| `swintransformer.py`, `util/box_ops.py`: thêm `indexing="ij"` cho `torch.meshgrid` | hết warning, không bị đổi hành vi ở torch mới (`"ij"` đúng là mặc định cũ) | không — `box_ops.masks_to_boxes` không được gọi |
| `util/misc.py`: bỏ nhánh torchvision < 0.7 trong `interpolate` | `float(torchvision.__version__[:3]) < 0.7` đọc `"0.21.0"` thành `0.2` nên **luôn true**, mà `torchvision.ops._new_empty_tensor` đã bị xoá từ torchvision 0.10 → crash nếu hàm đó được gọi | không — hàm `interpolate` không được gọi |

Kiểm lại bất cứ lúc nào:

```bash
diff -r <repo-gốc>/diffusiondet ./diffusiondet
```

Phần thêm mới (`objdet/`, `tools/`, `configs/`) là code của repo này, không có trong bản gốc.

### Đã thử và đã bỏ: AMP

Từng có 2 chỗ sửa nữa (buffer diffusion float32, `apply_deltas` tính fp32) để bật được AMP.
**Đã revert cả hai** vì AMP không dùng được với DiffusionDet — xem
[§ Những chỗ dễ sai](#những-chỗ-dễ-sai).

## Cấu trúc repo

```
diffusiondet/            model DiffusionDet (copy từ repo gốc + 4 chỗ sửa ở trên)
objdet/
├── datasets.py          đăng ký 3 dataset với detectron2
├── mmr.py               metric mMR/Recall/AP50 — numpy thuần, test được độc lập
└── crowdhuman_eval.py   evaluator CrowdHuman cho detectron2
configs/
├── Base-DiffusionDet.yaml    y nguyên bản gốc
├── Base-Kaggle-T4x2.yaml     batch/LR/AMP/checkpoint cho Kaggle
└── diffdet.{minitrain,voc,crowdhuman}.res50.yaml
tools/
├── train_net.py            train + eval
├── convert_crowdhuman.py   odgt → COCO json
└── summarize.py            bảng so sánh với baseline
baselines/baselines.yaml    số baseline đã công bố
tests/test_mmr.py           self-test metric (không cần GPU/detectron2)
```

## Những chỗ dễ sai

**Số class phải khớp dataset.** 80 cho minitrain, 20 cho VOC, 1 cho CrowdHuman. Sai chỗ này
thì train vẫn chạy bình thường mà kết quả vô nghĩa, nên `tools/train_net.py` chủ động
raise `ValueError` nếu `MODEL.DiffusionDet.NUM_CLASSES` không khớp dataset đang dùng.

**CrowdHuman: full body hay visible box.** Baseline Table 7 là **full body** (`fbox`) — số
Faster R-CNN 85,0 / 50,4 / 90,2 khớp chính xác baseline FPN full-body của paper CrowdHuman
gốc (84,95 / 50,42 / 90,24). Table 1 (zero-shot) mới là visible (`vbox`). Chọn sai loại box
thì so sánh với baseline mất ý nghĩa.

**Vùng ignore của CrowdHuman.** `tag == "mask"` (không phải người) và `extra.ignore == 1`
được chuyển thành `iscrowd=1`. Nhờ đó `DiffusionDetDatasetMapper` bỏ chúng khi train, và
`COCOEvaluator` coi chúng là vùng ignore khi eval. Nếu đưa vào làm positive thì model học
sai và AP tụt.

**Inference chạy batch size 1.** `ddim_sample` trong `detector.py` giả định batch 1 khi
box renewal (`outputs_class[-1][0]`, `torch.randn(1, ...)`). Đây là hành vi của repo gốc,
không đổi. Đừng tăng batch size ở lúc test.

**ĐỪNG bật AMP.** `SOLVER.AMP.ENABLED` phải để `False` (mặc định của repo này và của repo
gốc). Bật lên là train chết với `AssertionError` ở `generalized_box_iou`
(`assert x2 >= x1`), và đây **không phải bug vá được** mà là bất tương thích cấu trúc:

`RCNNHead` có `scale_clamp = log(100000/16) ≈ 8,74`, nên mỗi stage nhân kích thước box với
tối đa `exp(8,74) = 6250`. Head lại có **6 stage nối tiếp** (`bboxes = pred_bboxes.detach()`).
Trong giai đoạn đầu train, toạ độ box trung gian **hợp lệ về thuật toán** nhưng đạt cỡ:

| stage | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| độ lớn box | 1e6 | 1e10 | 1e14 | 1e18 |

fp32 chịu được (max 3,4e38), fp16 dừng ở **65504**. Tràn thành `±inf`, rồi stage sau tính
`ctr = -inf + inf = NaN`, và `NaN >= NaN` là `False`.

Đã thử vá bằng cách cho `apply_deltas` luôn tính fp32: kết quả là **đẩy lỗi từ iteration 0
sang iteration 12**, chứ không hết — vì `deltas` do lớp Linear fp16 sinh ra đã có thể tràn
từ trước khi vào `apply_deltas`. Cách duy nhất để nhét vào fp16 là hạ `scale_clamp`, mà làm
vậy là đổi hành vi model và mất tính so sánh với paper. Nên bỏ AMP.

Cách xác nhận AMP thực sự đã tắt: trong log **không được có** dòng nào của
`detectron2/engine/train_loop.py:490 ... with autocast(dtype=self.precision)` — dòng đó
thuộc `AMPTrainer.run_step`, nếu thấy nó nghĩa là AMP vẫn đang bật.

**COCO-minitrain có nhiều bản khác nhau.** Chỉ có split gốc của
[giddyyupp/coco-minitrain](https://github.com/giddyyupp/coco-minitrain) mới so được với
baseline 27,7 AP. Bản trên HuggingFace `bryanbocao/coco_minitrain` là **tập 25K khác** —
đối chiếu tên file thì chỉ 5.281/25.000 trùng. Đổi nguồn dữ liệu mà không kiểm là mất
luôn tính so sánh được.

## Ghi công

DiffusionDet: Shoufa Chen, Peize Sun, Yibing Song, Ping Luo — [arXiv 2211.09788](https://arxiv.org/abs/2211.09788).
Dựa trên [detectron2](https://github.com/facebookresearch/detectron2) và
[Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN). Giấy phép CC-BY-NC 4.0 theo repo gốc.

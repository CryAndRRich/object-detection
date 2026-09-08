# object-detection/diffusiondet — DiffusionDet trên 3 benchmark, so với baseline đã công bố

Train và eval [DiffusionDet](https://arxiv.org/abs/2211.09788) (R50-FPN) trên **COCO-minitrain
25K**, **PASCAL VOC 07+12** và **CrowdHuman**, rồi đối chiếu với baseline đã công bố trên
đúng ba bộ đó. Train trên **Kaggle 2× Tesla T4** (30h GPU/tuần, ~9–12h mỗi session) nên
train nhiều session và resume từ checkpoint.

Code model lấy từ [repo gốc của DiffusionDet](https://github.com/ShoufaChen/DiffusionDet)
(CC-BY-NC 4.0, xem [../LICENSE](../LICENSE)) và đã sửa để chạy với thư viện phiên bản mới —
xem [§ Sửa gì so với repo gốc](#sửa-gì-so-với-repo-gốc).

**3 dataset gốc (COCO-minitrain/VOC/CrowdHuman) đã train/eval xong, không chạy lại.** Toàn bộ
train/eval thật đã chạy qua notebook Kaggle — xem
[`../../notebooks/README.md`](../../notebooks/README.md), giữ lại làm bằng chứng tái lập số
liệu. Kết quả đã đo đầy đủ (data, config, số liệu, so baseline có venue) nằm ở
[`../../RESULTS.md`](../../RESULTS.md).

**Ngoại lệ: EXPERIMENT D dùng lại đúng code này cho CE-130 — xem
[§ EXPERIMENT D](#experiment-d--đối-chứng-trên-ce-130), chưa chạy, chỉ mới implement
converter/dataset/config/metric.** Tài liệu ở trên vẫn đúng cho việc hiểu code/cấu hình 3
dataset gốc, và cho việc kiểm import/mismatch nếu cấu trúc thư mục thay đổi.

## Mục lục

- [Dữ liệu](#dữ-liệu)
- [Cài đặt](#cài-đặt)
- [Chạy](#chạy)
- [Baseline để so sánh](#baseline-để-so-sánh)
- [Kỳ vọng thực tế về kết quả](#kỳ-vọng-thực-tế-về-kết-quả)
- [Sửa gì so với repo gốc](#sửa-gì-so-với-repo-gốc)
- [Cấu trúc repo](#cấu-trúc-repo)
- [Những chỗ dễ sai](#những-chỗ-dễ-sai)
- [EXPERIMENT D — đối chứng trên CE-130](#experiment-d--đối-chứng-trên-ce-130)

## Dữ liệu

Ba dataset, tổng 17GB sau khi giải nén:

| Dataset | Train | Eval | Số class |
|---|---|---|---|
| COCO-minitrain 25K | 25.000 ảnh / 183.546 ann | COCO `val2017` (5.000 ảnh / 36.781 ann) | 80 |
| PASCAL VOC 07+12 | 16.551 ảnh (VOC07 5.011 + VOC12 11.540) | VOC2007 `test` (4.952 ảnh) | 20 |
| CrowdHuman | 15.000 ảnh / 438.783 ann | `val` (4.370 ảnh / 127.710 ann) | 1 |

Đặt gốc dữ liệu bằng biến môi trường (mặc định `./data`, tức tương đối với **thư mục đang
đứng khi chạy lệnh**, không phải vị trí file code). Dữ liệu nằm ở
[`../data/`](../data/README.md) — nếu chạy lệnh từ trong `diffusiondet/` thì:

```bash
export OBJDET_DATA_ROOT=../data
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
# nếu dữ liệu ở đường dẫn read-only thì ghi json ra chỗ khác:
python tools/convert_crowdhuman.py --box-type fbox --out-dir /path/ghi/được/ch_ann
export OBJDET_CROWDHUMAN_ANN_DIR=/path/ghi/được/ch_ann
```

## Cài đặt

```bash
pip install -r requirements.txt
# detectron2 phải build từ source cho khớp torch/CUDA đang có:
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

Dùng nhánh `main` chứ không phải tag `v0.6`: v0.6 (11/2021) còn `PIL.Image.LINEAR` đã bị
xoá ở Pillow 10, nhánh main đã sửa.

Kiểm tra nhanh phần không cần GPU (chạy được ở máy không có torch/detectron2):

```bash
python -m pytest tests/ -q          # cả 4 bộ
# hoặc chạy từng file trực tiếp:
python tests/test_mmr.py
python tests/test_convert_ce130.py
python tests/test_ce130_paths.py
python tests/test_measure_box_quality_ce130.py
```

Mỗi file test vừa là script `main()` vừa có wrapper `test_*` cho pytest. Không có wrapper
đó thì `pytest tests/` báo *"no tests ran"* nhưng **exit 0** — nhìn qua tưởng pass trong
khi chưa chạy gì (đã bị nhầm một lần).

## Chạy

```bash
# train (2 GPU)
python tools/train_net.py --num-gpus 2 --config-file configs/diffdet.minitrain.res50.yaml

# session sau: train tiếp từ checkpoint gần nhất trong OUTPUT_DIR
python tools/train_net.py --num-gpus 2 --config-file configs/diffdet.minitrain.res50.yaml --resume

# eval
python tools/train_net.py --num-gpus 2 --config-file configs/diffdet.minitrain.res50.yaml \
    --eval-only MODEL.WEIGHTS ../weights/diffusiondet/minicoco_model_final.pth

# in bảng so sánh với baseline
python tools/summarize.py output/minitrain_res50 --dataset coco_minitrain
```

Ba config: `diffdet.minitrain.res50.yaml`, `diffdet.voc.res50.yaml`,
`diffdet.crowdhuman.res50.yaml`, đều kế thừa `Base-Kaggle-T4x2.yaml` (tên file giữ nguyên
theo đúng môi trường đã dùng để train — xem checkpoint tương ứng trong
[`../weights/diffusiondet/`](../weights/diffusiondet/)).

### Tính chất dynamic boxes — đáng khai thác

DiffusionDet **train một lần, eval nhiều cấu hình** mà không cần train lại. Đổi số box và
số bước sampling ngay trên dòng lệnh:

```bash
python tools/train_net.py --num-gpus 2 --config-file configs/diffdet.crowdhuman.res50.yaml \
    --eval-only MODEL.WEIGHTS ../weights/diffusiondet/crowdhuman_model_final.pth \
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
   "full tuning" của paper. Đặt `MODEL.WEIGHTS` trỏ tới checkpoint đã tải về
   (`../weights/diffusiondet/diffdet_coco_res50.pth`).
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
├── datasets.py              đăng ký dataset với detectron2 (3 dataset gốc + CE-130 EXPERIMENT D)
├── mmr.py                   metric mMR/Recall/AP50 — numpy thuần, test được độc lập
├── box_quality_metrics.py   oracle_recall/mean_bestIoU/score_AUC — numpy thuần (EXPERIMENT D)
└── crowdhuman_eval.py       evaluator CrowdHuman cho detectron2
configs/
├── Base-DiffusionDet.yaml    y nguyên bản gốc
├── Base-Kaggle-T4x2.yaml     batch/LR/AMP/checkpoint đã dùng khi train trên Kaggle
├── diffdet.{minitrain,voc,crowdhuman}.res50.yaml
└── diffdet.ce130{,_coco,_72cls}.res50.yaml    EXPERIMENT D (D.1 ImageNet/D.1 COCO/D.2)
tools/
├── train_net.py                    train + eval
├── convert_crowdhuman.py           odgt → COCO json
├── convert_ce130.py                all_phase2_V2/ → COCO json (EXPERIMENT D)
├── visualize_ce130_coco.py         vẽ box từ json CE-130 lên ảnh — CỬA CHẶN trước khi train
├── measure_box_quality_ce130.py    oracle_recall/mean_bestIoU trên CE-130 (EXPERIMENT D)
└── summarize.py                    bảng so sánh với baseline
baselines/baselines.yaml    số baseline đã công bố
tests/                                   (đều KHÔNG cần GPU/detectron2)
├── test_mmr.py                          metric mMR/Recall/AP50 của CrowdHuman
├── test_convert_ce130.py                converter CE-130: dedupe, xyxy→xywh, category_id
├── test_ce130_paths.py                  đường dẫn converter GHI RA == datasets.py ĐỌC VÀO
└── test_measure_box_quality_ce130.py    oracle_recall/mean_bestIoU/score_AUC
```

Checkpoint (`../weights/diffusiondet/`, không push git):

| File | Vai trò |
|---|---|
| `minicoco_model_final.pth`, `voc_model_final.pth`, `crowdhuman_model_final.pth` | 3 checkpoint tự train, số liệu trong `../../RESULTS.md` |
| `diffdet_coco_res50.pth`, `diffdet_coco_swinbase.pth`, `diffdet_lvis_res50.pth`, `diffdet_lvis_swinbase.pth` | 4 checkpoint pretrain gốc từ tác giả DiffusionDet, dùng làm điểm khởi tạo finetune |

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

**`z_T` là Gaussian, không phải uniform trong ảnh.** Cả lúc train (nhiễu cộng vào GT,
`q_sample`) lẫn lúc infer (khởi tạo `img = torch.randn(...)`, `detector.py:197`) đều dùng
N(0,1) — không có chỗ nào sample uniform trong không gian ảnh. Cảm giác "box phủ đều khắp
ảnh" chỉ đến từ phép decode sau đó (`clamp` → `/scale+1)/2` → `box_cxcywh_to_xyxy` →
`*images_whwh`), không phải vì nhiễu là uniform. Padding GT lúc train (`box_placeholder`,
dòng 384) cũng là Gaussian (`N(0.5, 1/6²)`), không phải uniform — đã ablate cả 2 loại trong
paper, Gaussian thắng.

**Proposal có tự nhìn nhau, qua self-attention — không chỉ qua backbone chung.** `head.py:185`:
`self.self_attn = nn.MultiheadAttention(...)` chạy trên `pro_features` (feature riêng mỗi
proposal) theo chiều "N proposal", ở **mỗi trong 6 stage cascade** — y hệt self-attention giữa
object query trong DETR. Nên 1 proposal "biết" proposal khác đang tập trung vùng nào, nhưng
gián tiếp qua feature đã học, không phải qua so trực tiếp toạ độ `(x,y,w,h)`.

**Với cấu hình nhiều bước (`SAMPLE_STEP>1`), bước cuối cùng bị loại khỏi kết quả cuối.**
`ddim_sample` (dòng 220-246): ở cặp `(time, time_next)` cuối (`time_next<0`), code chạy
`img = x_start; continue` **trước** khi tới đoạn `if self.use_ensemble: ensemble_coord.append(...)`
— nghĩa là lần gọi head cuối cùng (dù vẫn tốn compute) **không được** gộp vào pool NMS cuối
cùng. Với `SAMPLE_STEP=4`, kết quả hiển thị chỉ đến từ 3 bước đầu, không phải cả 4. Không ảnh
hưởng tới các số đã đo trong `RESULTS.md` (đó là hành vi thật của model, đã đo đúng) — chỉ dễ
gây hiểu lầm nếu tự viết code visualize/debug per-step mà không biết điều này (xem
[`../../notebooks/README.md`](../../notebooks/README.md), mục
`diffusiondet_diffusion_trace.ipynb`).

## EXPERIMENT D — đối chứng trên CE-130

Đặc tả đầy đủ: [`../../docs/thiet-ke-experiment-d-diffusiondet-ce130.md`](../../docs/thiet-ke-experiment-d-diffusiondet-ce130.md).
**KHÔNG phải cải tiến của CE-LocModel A/B/C** — là đối chứng: đặt một detector chuẩn đã
kiểm chứng (COCO/VOC/CrowdHuman) lên đúng dữ liệu CE-130 để biết TRẦN THỰC TẾ của bài
toán định vị. Không có D thì `AP50 0,0152` của A/B/C lơ lửng: không biết thấp vì kiến
trúc hay vì dữ liệu.

**Trạng thái: code đã implement, CHƯA CHẠY THẬT trên GPU.** Converter đã chạy thử và
kiểm mắt (xem dưới), nhưng chưa train.

### Chạy

```bash
export OBJDET_DATA_ROOT=../data     # cùng biến môi trường như 3 dataset gốc

# 1. Sinh json COCO — D.1 (class-agnostic, LÀM TRƯỚC)
python tools/convert_ce130.py --ce130-root ../data/all_phase2_V2 --mode class-agnostic

# 2. CỬA CHẶN bắt buộc — nhìn bằng mắt trước khi train (bài học docs/bai-hoc-ce-loc-detection.md
#    §5: visualize bắt được lỗi mà test + review code bỏ sót)
python tools/visualize_ce130_coco.py \
    --json ../data/ce130_coco/ce130_agnostic_train.json \
    --image-root ../data/all_phase2_V2 --out /tmp/ce130_viz --n 12

# 3. Train D.1 (ImageNet pretrain — so công bằng với A/B/C)
python tools/train_net.py --num-gpus 1 --config-file configs/diffdet.ce130.res50.yaml

# 4. Train D.1 (COCO pretrain — trần trên; chênh lệch với bước 3 = "cần bao nhiêu pretrain")
python tools/train_net.py --num-gpus 1 --config-file configs/diffdet.ce130_coco.res50.yaml

# 5. Đo — KHÔNG dùng AP làm kết luận chính, dùng oracle_recall/mean_bestIoU.
#    PHẢI quét số box: CE-130 có 48,5 vật/ảnh nên 300 chặn trần recall (xem mục dưới).
for N in 300 1000 2000 3000; do
  python tools/measure_box_quality_ce130.py --config-file configs/diffdet.ce130.res50.yaml \
      --num-proposals $N MODEL.WEIGHTS output/ce130_agnostic_res50/model_final.pth
done

# 6. (CHỈ khi bước 5 để lại câu hỏi chưa trả lời được) D.2 closed-set
python tools/convert_ce130.py --ce130-root ../data/all_phase2_V2 --mode closed-set --split train
python tools/train_net.py --num-gpus 1 --config-file configs/diffdet.ce130_72cls.res50.yaml
```

### D.1 vs D.2

**D.1 (class-agnostic, `NUM_CLASSES=1`, LÀM TRƯỚC, phép đo chính).** Hợp lệ vì CE-130 mỗi
ảnh chỉ có đúng 1 class (3.598/3.598, đã đo) — "detect mọi vật trong ảnh" và "detect vật
thuộc category ảnh đó" là CÙNG một tập box trên bộ này. D.1 không có đường text (khác
A/B/C dùng CLIP text) nên nếu D.1 > A/B/C thì kết luận là "điều kiện hoá text đang cản
trở", KHÔNG PHẢI "DiffusionDet giỏi hơn CE-Loc".

**D.2 (closed-set, `NUM_CLASSES=72`, LÀM SAU, chỉ khi D.1 để lại câu hỏi).** Chia LẠI nội
bộ split train gốc (72 class) thành `train72`/`val72` — KHÔNG dùng split test/val gốc (28
class, giao = 0 với train, chạy thẳng sẽ cho AP ≈ 0 vì head phân loại theo index chưa
từng thấy). D.2 chỉ so được với chính D.1 đo lại trên cùng `train72`/`val72`, KHÔNG so
trực tiếp với D.1/A/B/C trên test 28-class zero-shot.

**Hai bẫy của D.2 đã mắc rồi sửa — đều là loại "json vẫn hợp lệ, train vẫn chạy, không
assert nào bắt được, AP về 0 vì lý do không liên quan model":**

1. **Chia split theo random thuần trên ảnh** làm category(train) ≠ category(val) — đo thử
   với `val_frac=0.15`: train còn 71 category, val 57, **không phải tập con của nhau**.
   Nguyên nhân: 72 class trải rất lệch trên 1.911 ảnh, class hiếm bị ngẫu nhiên đẩy hết
   sang một bên. Sửa: **stratified theo class** (`split_train_72`) — mỗi class tự góp tỉ
   lệ `val_frac` ảnh của chính nó, giữ ≥1 ảnh ở train. Đo lại: 72/72 train, 67/72 val,
   val ⊆ train (5 class chỉ có 1 ảnh nên đúng là không thể có ở val, không phải bug).

2. **`category_id` đánh số độc lập cho từng file.** `build_coco` từng tự dựng bảng id từ
   chính `items` được truyền vào, nên train72 (72 class) và val72 (67 class) đánh số lệch
   nhau — **54/72 id trỏ sang tên class khác**:

   ```
   id 19:  train='cartridge'  val='cement bag'
   id 20:  train='cassette'   val='cereal'
   ```

   Model học "id 19 = cartridge" rồi bị chấm bằng "id 19 = cement bag". Sửa:
   `build_cat_id_map()` dựng bảng **một lần từ toàn bộ 72 class trước khi chia**, truyền
   vào cả hai lần gọi; `categories` trong json liệt kê **đủ 72** ở cả hai file (kể cả
   class vắng mặt) để metadata detectron2 khớp nhau. Verify lại trên dữ liệu thật: bảng
   id giống hệt nhau, 0 id lệch, `n_categories_present` 72 (train) / 67 (val).
   `tests/test_convert_ce130.py` có test riêng cho việc này, **kèm kiểm chứng ngược**
   (dựng bảng riêng từng file thì test phải FAIL) để chắc test có hiệu lực thật.
   D.1 không dính lỗi này (chỉ 1 category).

### ⚠️ Số box lúc eval — 300 CHẶN TRẦN recall trên CE-130

CE-130 dày hơn mọi dataset đã chạy trước đó:

| split | box/ảnh TB | max | số ảnh > 300 box |
|---|---|---|---|
| test | 48,5 | 505 | 7 |
| val | 42,2 | 1.229 | 9 |

`RESULTS.md` §4 đã đo trên 3 dataset: **số box tối ưu bám mật độ vật thể** — VOC (2,43
vật/ảnh) đỉnh ở 1000; COCO (7,36) bão hoà 2000; CrowdHuman (22,76) **vẫn còn tăng ở
3000** (300→3000 cho `Recall` +12,33, `AP50` +9,23, không train thêm gì). CE-130 có 48,5
vật/ảnh — **đông gấp đôi CrowdHuman** — nên để 300 là tự chặn trần recall bằng cấu trúc,
kể cả khi model hoàn hảo.

`NUM_PROPOSALS: 300` trong config **chỉ dành cho lúc TRAIN** (đúng config paper;
DiffusionDet không có tham số nào phụ thuộc số box nên train 300 rồi eval 3000 là dùng
đúng thiết kế *dynamic boxes*, xem `RESULTS.md` §2.1). `EVAL_PERIOD` giữa chừng cũng chạy
ở 300 — **chỉ để theo dõi đường cong, đừng đọc như kết quả**. Kết quả cuối phải quét:

```bash
for N in 300 1000 2000 3000; do
  python tools/measure_box_quality_ce130.py --config-file configs/diffdet.ce130.res50.yaml \
      --num-proposals $N MODEL.WEIGHTS output/ce130_agnostic_res50/model_final.pth
done
```

`measure_box_quality_ce130.py` tự đếm số ảnh có nhiều GT hơn số box và **in cảnh báo**
nếu số box đang chặn trần — để không ai đọc nhầm "recall thấp = model kém".

### ⚠️ Hai vấn đề CHẤT LƯỢNG DỮ LIỆU phát hiện khi rà (không sửa, phải biết khi đọc số)

Cả hai đều là tính chất của dữ liệu CE-130, không phải bug converter. **Không tự lọc** —
giữ nguyên để số liệu còn so được với CE-LocModel A/B/C (chúng đọc cùng dữ liệu này).

**1. Các branch cùng một ảnh BẤT ĐỒNG về GT.** Kiểm toàn bộ (không lấy mẫu):
`ground_truth.jpg` giống hệt nhau giữa mọi branch (md5 khớp **100 %**) và
`annotation.json` cũng nhất quán tuyệt đối (0 cặp lệch ở cả 3 split) — nhưng
`fixed_annotation.json` **lệch ở 1.410 cặp val / 1.079 cặp test**, vì mỗi branch chỉnh
riêng box mà chính nó sắp inpaint. Hệ quả: **86,5 % ảnh val và 79,7 % ảnh test** có các
branch bất đồng, chọn branch nào ảnh hưởng tới toạ độ (lệch tối đa **374 px** ở một box).
Ảnh hưởng lên *số lượng* box thì rất nhỏ (chênh 11 box val / 10 box test, ~0,03 %) — khác
biệt gần như hoàn toàn là toạ độ, và vẽ ra thì hai bản chất lượng tương đương, không bản
nào sai rõ ràng.

Converter chọn **branch có chỉ số nhỏ nhất** (tất định, khớp hành vi `ce130_dataset.py`
gốc nên vẫn so được với A/B/C) và **in ra số ảnh bất đồng** mỗi lần chạy để con số này
không bị quên.

**2. Split test có một lô annotation HỎNG.** Đếm box chiếm > 50 % diện tích ảnh (vật điển
hình CE-130 chỉ ~0,4 %):

| split | box > 50 % ảnh | ảnh có ≥ 5 box như vậy |
|---|---|---|
| train | 0 | 0 |
| val | 0 | 0 |
| **test** | **855** | **16** |

15/16 ảnh đó có id dạng `62xx` liên tiếp — một lô lỗi. Ảnh `6261`: **325 box mà 293 box
bao gần trọn ảnh**, chồng khít lên nhau, không box nào bao một quả táo. Tổng GT của 16
ảnh này là **1.594/37.812 = 4,2 % GT của split test** — tức 4,2 % "GT" mà không detector
nào có thể khớp đúng, kéo mọi chỉ số trên test xuống, cho **cả D lẫn A/B/C**.

Xem tận mắt: `python tools/visualize_ce130_coco.py --json .../ce130_agnostic_test.json
--image-root ../data/all_phase2_V2 --out /tmp/viz --suspect-only`. Converter cũng in
`n_box_over_half_image` / `n_images_suspect_annotation` trong stats mỗi lần chạy.

### Đối chiếu số liệu converter với `ce130_dataset.py` (bản đã verify của CE-LocModel)

`tools/convert_ce130.py` port lại đúng 3 quy tắc của `count_editing/CE-LocModel/data/ce130_dataset.py`
(KHÔNG import — hai stack khác nhau hoàn toàn: detectron2/COCO-json ở đây so với
numpy-dict/CLIP-cache bên kia): dedupe theo ảnh gốc, giữ nguyên `all_bboxes` (không trừ
`inpainted_bboxes`), fallback `fixed_annotation.json` → `annotation.json`. Chạy thật và so
trực tiếp với `CE130Detection.stats()`, khớp **chính xác từng số**:

| | train | val | test |
|---|---|---|---|
| n_images | 1.911 | 908 | 779 |
| box thô trong `all_bboxes` | 71.852 | 38.289 | 37.812 |
| degenerate bị lọc (w hoặc h ≤ 0) | 85 | 0 | 0 |
| **`n_annotations` trong json** (= thô − degenerate) | **71.767** | **38.289** | **37.812** |

⚠️ Hai con số dễ lẫn ở split train: **71.852** là box thô, **71.767** là số annotation
thật trong json. Val/test bằng nhau vì không có box degenerate. Khi đối chiếu trên server
thì so **71.767 / 38.289 / 37.812** (đó là cái converter in ra ở `n_annotations`).

Lưu ý: con số "14/37.110 box degenerate" trong docstring của `filter_degenerate`
(`CE-LocModel/utils/box_ops_np.py`) là số đo ở MỘT THỜI ĐIỂM DỮ LIỆU KHÁC (trước khi sửa
"không trừ `inpainted_bboxes`" — khi đó tổng box ít hơn nhiều). Số hiện hành, đo lại
2026-09-08, là **85 degenerate / 71.852 box thô**, khớp cả 2 cách tính độc lập (converter
riêng và gọi thẳng `filter_degenerate`). `data/README.md` §8 **đã cập nhật**.

### Cửa chặn kiểm mắt — ĐÃ CHẠY, ĐẠT

`tools/visualize_ce130_coco.py` chạy trên `ce130_agnostic_train.json` thật (không phải dữ
liệu giả), 6 ảnh ngẫu nhiên bao gồm 1 ảnh mật độ cao (220 box, chim trên dây điện, đĩa xếp
chồng, cà chua trong rổ) — box khớp chính xác vật thể thật, không lệch trục, không dịch
chuyển, kể cả ở mật độ 220 box/ảnh.

### Bug đường dẫn: converter ghi một chỗ, `datasets.py` đọc một chỗ khác

`objdet/datasets.py` từng dùng `os.path.join(root, "..", "ce130_coco")` trong khi
converter ghi vào `<ce130-root>/../ce130_coco`. Với `OBJDET_DATA_ROOT=../data` (đúng như
README hướng dẫn ở § Dữ liệu) thì:

```
converter ghi json vào :  object-detection/data/ce130_coco     ← đúng
datasets.py tìm json ở :  object-detection/ce130_coco          ← thừa một ".."
datasets.py tìm ảnh ở  :  object-detection/all_phase2_V2       ← cũng thừa một ".."
```

Lệch đúng một cấp thư mục, cho **cả json lẫn ảnh**. `all_phase2_V2/` và `ce130_coco/` nằm
**bên trong** `data/`, không phải cạnh nó. Bug này chỉ lộ khi thật sự chạy trên GPU
(crash *file not found* lúc load) — tức sau khi đã đẩy code lên server và xếp hàng chờ
GPU. `tests/test_ce130_paths.py` giờ so trực tiếp "đường dẫn converter ghi ra" với
"đường dẫn `datasets.py` đọc vào" nên bắt được ở máy local, không cần detectron2.

### `measure_box_quality_ce130.py` phải dựng config y hệt `train_net.py`

Tool tự dựng `cfg` bằng `get_cfg() + add_diffusiondet_config()` là **chưa đủ**:
`train_net.py:277-281` còn gọi `add_model_ema_configs()` và `add_kaggle_configs()`. Cái
sau định nghĩa `SOLVER.CHECKPOINT_MAX_TO_KEEP`, mà `Base-Kaggle-T4x2.yaml` — config gốc
của **cả 3** config CE-130 — có set key đó, nên thiếu nó thì `merge_from_file` vỡ ngay
với *"Non-existent config key"*. Tool giờ **import thẳng `add_kaggle_configs` và
`check_num_classes` từ `train_net.py`** thay vì chép lại, để hai bên không lệch nhau; và
gọi luôn `check_num_classes` vì số class sai không crash mà chỉ cho kết quả rác.

### Vì sao `DiffusionDetDatasetMapper(is_train=False)` không dùng được để lấy GT

Bug đã bắt được khi viết `measure_box_quality_ce130.py` (trước khi chạy GPU, không phải
sau): `dataset_mapper.py` có `if not self.is_train: dataset_dict.pop("annotations",
None); return dataset_dict` — **mapper lúc eval xoá sạch GT**, không tạo `Instances`. Đọc
`inp["instances"]` ở mapper eval sẽ luôn `None`, khiến mọi `n_gt=0` một cách câm lặng.
Phải đọc GT trực tiếp từ `DatasetCatalog.get(dataset_name)` (COCO json gốc), khớp theo
`image_id`, độc lập hoàn toàn với mapper.

### Việc còn lại trước khi có số

1. Chạy converter D.1 thật trên server (đã chạy local để kiểm — số liệu ở trên), rồi
   `visualize_ce130_coco.py` lại lần nữa trên đúng máy sẽ train (phòng trường hợp đường
   dẫn ảnh khác).
2. Train D.1 (ImageNet), ~6.000 iter, < 1 giờ trên A30 theo ước tính trong đặc tả.
3. `measure_box_quality_ce130.py` trên test split, so `oracle_recall`/`mean_bestIoU` trực
   tiếp với số đã có của A/B/C.
4. Train D.1 (COCO pretrain) nếu D.1 (ImageNet) cho số cần so sánh thêm.
5. D.2 chỉ nếu bước 3 để lại câu hỏi chưa trả lời được.

## Ghi công

DiffusionDet: Shoufa Chen, Peize Sun, Yibing Song, Ping Luo — [arXiv 2211.09788](https://arxiv.org/abs/2211.09788).
Dựa trên [detectron2](https://github.com/facebookresearch/detectron2) và
[Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN). Giấy phép CC-BY-NC 4.0 theo repo gốc.

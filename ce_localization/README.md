# ce_localization — CE-Loc

Chọn box cho count editing trên CE-130: cho ảnh + tên lớp, sinh N box của lớp đó bằng
khuếch tán trên toạ độ box. Mô hình nền hiện tại = EXPERIMENT A vòng 2
(lịch sử + số đo: `docs/old/ROUND_2_EXPERIMENT_A.md` ở gốc `multi_condition/`).

**Kết quả hiện tại** (test, N=300): AP50 **10,36** — baseline D.1 (DiffusionDet) **58,13**.

## Kiến trúc

```
ảnh  -> CLIP ViT-B/16 frozen -+-> patch_raw [B,4096,768] -------> RoI 3x3 ở MỖI tầng
                              +-> Linear -+
text -> CLIP text frozen   -----> Linear -+--> memory [B,1+4096,256]

x_T ~ N(0,I) [B,N,4] --DDIM 4 bước--> mỗi bước 6 DiTBlock:
    (1) r <- gate(r, RoI(patch_raw, x))      ảnh tại toạ độ HIỆN TẠI
    (2) seq = [h ; r + mark(x)]              h: token hình học, r: token ảnh
    (3) self-attn 2N, mask chặn r->h, adaLN-Zero theo t
    (4) chỉ r cross-attend memory; FFN riêng
    (5) x <- update_box(x, box_delta(h))     toạ độ CỘNG DỒN qua tầng
    -> N box + N score (score đọc r)
```

Loss như DiffusionDet (5·L1 + 2·GIoU + 2·Focal, SimOTA), đặt ở mọi tầng và **cộng** lại.

## Cấu trúc

| | |
|---|---|
| `train.py` / `eval.py` | điểm vào |
| `config/default.yaml` | cấu hình duy nhất (@1024px) |
| `models/` | `clip_encoder`, `dit_blocks` (DiTBlock, update_box, mask), `roi_sampler`, `detector` (BoxDiT, CELocDetector, DDIM), `criterion` |
| `data/` | `ce130_dataset` (đọc CE-130, cache), `loader` (DataLoader) |
| `utils/` | hình học box + khuếch tán (bản torch và bản numpy tham chiếu), matcher, `metrics_np` (chấm điểm), `checkpoint`, `grad_monitor`, `log` |
| `tools/` | `build_cache`, `check_data_facts`, `visualize_data` |
| `checkpoints/` | không vào git |

Test ở `object-detection/tests/ce_localization/`.

## Chạy

Mọi lệnh chạy trong `object-detection/ce_localization/` trên server, sau
`export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache`.

| việc | thời lượng | dạng |
|---|---|---|
| test (`cd .. && python -m pytest tests -q`) | ~2 phút | trực tiếp |
| build cache 1 split @1024 | 5–40 phút | **nền** |
| train 300 epoch | ~21 giờ | **nền** |
| eval test N=300 | 1–3 phút | trực tiếp |

**Cache** (cần trước khi train; đã có trên server ở `../data/cache_clip_1024`):
```bash
LOG=/mnt/disk1/aiotlab/haitn/log/cache_train_$(date +%m%d_%H%M).log
nohup python ../tools/run_on_free_gpu.py -- tools/build_cache.py --split train \
    --image-size 1024 --batch-size 2 --out ../data/cache_clip_1024 > $LOG 2>&1 &
echo "PID $! -> $LOG"
```
Split eval thêm `--no-flip` (eval không lật ảnh, tốn nửa dung lượng).

**Train** (`--save-dir` bắt buộc; `last.pt` lưu mỗi epoch, ghi nguyên tử):
```bash
LOG=/mnt/disk1/aiotlab/haitn/log/train_<tên>_$(date +%m%d_%H%M).log
nohup python ../tools/run_on_free_gpu.py -- train.py --save-dir checkpoints/<tên> \
    > $LOG 2>&1 &
echo "PID $! -> $LOG"
```
Bị ngắt thì chạy lại **đúng lệnh đó + `--resume`** (mất tối đa một epoch). Không có `--resume`
mà `last.pt` đã có thì train DỪNG ngay, không ghi đè.

**Eval** — số so với baseline D.1 (cùng giao thức bảng vòng 1):
```bash
python ../tools/run_on_free_gpu.py -- eval.py --ckpt checkpoints/<tên>/best.pt \
    --split test --num-proposals 300 --top-k 100 --nms --oracle-score \
    --out /mnt/disk1/aiotlab/haitn/output/<tên>_test_N300.json
```

## Đọc số

| chỉ số | đo gì |
|---|---|
| `oracle_recall` | GT được ít nhất một box phủ (IoU ≥ 0,5) — chất lượng BOX, không dùng score. Chọn checkpoint bằng nó |
| `score_AUC` | box khớp GT có score cao hơn box còn lại không — chất lượng XẾP HẠNG |
| `--oracle-score` | trần AP khi score = IoU thật, cùng box. Trần ≫ thật: sửa xếp hạng; trần cũng thấp: sửa box |
| `recall/tầng` | phẳng = cộng dồn toạ độ không mang lại gì |
| `iou_matched` | chỉ để đọc — **không** chọn checkpoint (mù với GT không box nào chạm tới) |

⚠️ Chỉ số val trong log train là **khử nhiễu 1 bước từ GT đã thêm nhiễu** — cao gần 2× suy
luận thật. Số báo cáo chỉ lấy từ `eval.py`. N=30 có trần `oracle_recall` 83,8/82,4/76,1 %
(train/val/test); N=300 ~97 %.

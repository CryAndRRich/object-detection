# CE-Loc — EXPERIMENT A (2026-09-05)

**Nhánh CE-Loc gốc (ResNet18 + SpatialSoftmax + Conv1D U-Net, single-box) đã XOÁ khỏi thư mục
này.** Bản read-only đầy đủ vẫn ở `refs/repos/Count-Editing/CE-LocModel/` nếu cần đọc lại.

Thiết kế đầy đủ + toàn bộ số đo: [`../../../docs/thiet-ke-ce-loc-vong-2.md`](../../../docs/thiet-ke-ce-loc-vong-2.md).
Lỗi vòng 1 (đọc TRƯỚC khi sửa gì): [`../../../docs/bai-hoc-ce-loc-detection.md`](../../../docs/bai-hoc-ce-loc-detection.md).

**Ràng buộc**: thân model là **Diffusion Policy transformer-based**, chỉ **mượn cơ chế sinh N box**
của DiffusionDet.

## Thay đổi cốt lõi

Memory của decoder từ **2 token** → **1026 token có vị trí**. Vòng 1 đo được: với 2 token thì cả N
box nhận **cùng** một vector 256-d, và gradient trên box unmatched có hướng ngẫu nhiên (cosine
−0,0074 ≈ tung đồng xu). Đây là "RoIAlign của người nghèo" — box đọc ảnh tại vị trí của chính nó.

| | CE-Loc gốc | EXPERIMENT A |
|---|---|---|
| Vision | ResNet18(4ch) + SpatialSoftmax → 1 vector 128-d | **CLIP ViT-B/16 FROZEN** → 1024 patch token |
| Text | CLIP B/32, pooled | CLIP B/16 **FROZEN**, 1 token |
| Density map | kênh thứ 4 | **BỎ** (bài detection) |
| Bơm điều kiện | FiLM cộng bias | **cross-attention** |
| Box token | `Linear(4→D)` | **sinusoidal PE trên (cx,cy,w,h)** |
| Số box | 1 | **N=100** train / **300** eval |
| Loss | MSE trên epsilon | **5,0·L1 + 2,0·GIoU + 2,0·Focal** (giống DiffusionDet) |
| Schedule | linear, T=100 | **cosine**, T=1000 (đo được 3,70× AP) |

Tham số học được: **~8,3M** / tổng 158M (CLIP frozen).

## Cấu trúc

```
utils/box_ops_np.py  diffusion_np.py  matcher_np.py   <- NUMPY THUẦN, nguồn chân lý
utils/box_ops.py     diffusion_math.py  matcher.py    <- port cơ học sang torch
data/ce130_dataset.py                                 <- dedupe, pad CLIP mean, flip
models/clip_encoder.py  box_transformer.py            <- CLIP frozen + decoder
models/detector.py  criterion.py                      <- ghép + loss
train.py  eval.py  config/experiment_a.yaml
tools/visualize_data.py  profile_and_memory.py  build_cache.py  overfit_one.py
tests/  (6 file, 78 test — .gitignore, chỉ có ở máy dev)
```

**Vì sao tách numpy/torch**: vòng 1 chôn logic toán trong module torch nên chỉ verify được trên
GPU → không ai verify. Giờ phần toán test được ở local bằng numpy có đáp án giải tích, rồi
`tests/test_torch_vs_numpy.py` đảm bảo bản torch không lệch — chạy **không cần GPU, không cần train**.

**`tests/` nằm trong `.gitignore`** nên KHÔNG lên server. Hệ quả: sau `git pull` không chạy được
`pytest` ở đó, nên mọi thay đổi code phải test ở máy dev TRƯỚC khi push. Trên server thì cửa chặn
là `tools/overfit_one.py` (§4 dưới) — nếu loss không về ~0 thì dừng, đừng train dài.

## Chạy

```bash
# 1. Test — CHỈ CHẠY ĐƯỢC Ở MÁY DEV (tests/ không push git), ~15s
python3 -m pytest tests/ -q

# 2. Nhìn ảnh TRƯỚC khi train — vòng 1 visualize bắt được 3 lỗi mà test bỏ sót
python3 tools/visualize_data.py --n 20 --out /tmp/viz
python3 tools/visualize_data.py --n 8 --out /tmp/viz_ph --placeholder --t 50

# 3. Đo memory + có cần cache không (trên server)
python3 tools/profile_and_memory.py --batch-size 8 --steps 10

# 4. CỬA CHẶN: overfit 1 ảnh. Không đạt thì DỪNG, đừng train dài.
python3 tools/overfit_one.py --steps 300

# 5. Train — TỰ CHỌN GPU TRỐNG NHẤT (server dùng chung, không mặc định GPU 0)
nohup python3 tools/run_on_free_gpu.py -- train.py --config config/experiment_a.yaml \
    > /mnt/disk1/aiotlab/haitn/log/v2_train.log 2>&1 & echo $!
#    Script cần chạy đặt SAU `--`. Không có ngưỡng free-memory (đã thử 2 lần đều
#    hỏng — xem docstring). Có --retries: job chết thì đọc lại nvidia-smi và thử
#    GPU khác; nhưng job bị `kill` (rc âm) thì KHÔNG thử lại.
#    Ép GPU cụ thể: --gpu 2 (đặt TRƯỚC --)
#    Checkpoint chỉ lưu ~8,3M tham số HỌC ĐƯỢC (95 MB). Lưu cả CLIP frozen thì
#    nặng 698 MB mà 98 % là trọng số tải lại được từ HuggingFace.

# 6. Eval — train N=100 nhưng eval N=300
python3 tools/run_on_free_gpu.py -- eval.py --ckpt checkpoints/experiment_a/best.pth --split test
```

## Ba chỉ số phải nhìn khi train

Quan trọng ngang loss — vòng 1 thiếu nên mù suốt 5 vòng sửa:

| chỉ số | ngưỡng cảnh báo |
|---|---|
| **% cặp match giữ nguyên giữa 2 epoch** | vòng 1 chỉ ~55 % → hơn nửa nhãn đổi mỗi epoch |
| **std của `sigmoid(score)`** | < 0,05 → head kẹt ở hằng số (focal hội tụ về hằng số khi không phân biệt được) |
| **IoU trung bình cặp matched** | tách khỏi loss, dễ đọc |
| **val_loss** | best chọn theo val (không phải train) — 1.911 ảnh + class rời nhau thì overfit rất nhanh |

## Log và số liệu

Không dùng tqdm — in thẳng ra stdout để đọc được trong file log.

**Train**: mỗi epoch in **3 dòng cố định**:

```
[ep   12/300] train   5.6230 (l1 0.4136 giou 1.5768 ce 0.2008)   val   5.6716 (l1 ... )
           IoU train 0.0272 / val 0.0266 | matched 84.5/100 (84%) | GT/ảnh 64.8 |
           ổn_định_nhãn 0.025 | lr 1.00e-04 | grad 19.360
           score μ 0.3364 σ 0.0364 [0.251, 0.439] p50 0.3347 | 26s+17s (5602ms/batch) |
           đã chạy 2m07s | ETA 41m
           ⚠ std_score < 0,05 — score head có thể kẹt ở hằng số
```

Cảnh báo tự động khi: `std_score < 0,05` (head kẹt hằng số), `ổn_định_nhãn < 0,40`
(nhãn đổi quá nhiều), `grad norm > 100`.

**Eval**: in tiến độ 5 %/lần kèm ms/ảnh và ETA; cuối in AP ở **10 ngưỡng IoU**
(AP50…AP95 + AP trung bình kiểu COCO), P/R/F1, **trần precision** và `precision/trần`,
phân bố score, thời gian.

**`checkpoints/experiment_a/history.json`** — ghi lại sau **MỖI epoch** (không đợi train xong, để job
chết giữa chừng vẫn đọc được). Chứa:

| khoá | nội dung |
|---|---|
| `tom_tat` | best epoch/val_loss, tổng thời gian, epoch nào có cảnh báo, val có tăng liên tiếp không |
| `moi_truong` | hostname, GPU, `CUDA_VISIBLE_DEVICES`, torch/python version, lệnh chạy, cwd, thời điểm |
| `config` | toàn bộ config đã dùng |
| `dataset` | thống kê train + val |
| `epochs[]` | mỗi epoch: loss 4 thành phần (train+val), IoU, n_matched, `lr`, thời gian, ETA, cảnh báo, và **phân bố đầy đủ** (mean/std/min/max/p1/p25/p50/p75/p99) của **score**, **grad norm**, **GT/ảnh**, **ms/batch** |

Eval ghi `<ckpt>_eval_<split>_N<N>.json` — kèm **số liệu per-ảnh** (image_id, class, n_gt,
số box sau top-k và sau NMS, score min/max, thời gian) để tìm ảnh nào hỏng.

## Cạm bẫy đã khoá bằng test

- **Hai nguồn annotation, hai định dạng box**: `all_bboxes` là **xyxy**, `target_bbox` là **cxcywh**.
- **KHÔNG trừ `inpainted_bboxes`**: `ground_truth.jpg` là ảnh **gốc chưa xoá gì** (diff pixel 51,96
  vs 1,41). Vòng 1 trừ đi → vứt 7–8 % vật thật.
- **Class 3 split RỜI NHAU hoàn toàn** (72/28/28, giao = 0) → bài toán là **zero-shot**. Text
  encoder **phải** freeze; không so số với detector closed-set.
- **Giải mã toạ độ**: extent là `(norm+1)/2`, **không phải** `(norm+1)/4` (IoU giữa hai cách: 0,25).
- **Placeholder**: gốc to gấp **7,3×** vật CE-130 và 13,7 % rơi vào vùng pad → đã sửa cả hai.
- **Top-k, không ngưỡng 0,5**: focal hội tụ về hằng số → mọi box bị lọc → argmax giữ đúng 1 box.
- **Annotation bỏ sót vật** (ảnh có ~8 con trâu, chỉ 6 box) → precision đo được **thấp hơn** thật.
- **Trần precision cấu trúc** `min(M,N)/N` = 0,376 với N=100 → so P/R thô giữa các N là vô nghĩa.

## Ghi chú

- **CLIP attention**: code tự dò SDPA, lùi về `eager` nếu transformers < 4.45 (server đang 4.42).
  Với `eager` thì attention matrix được materialize, nhưng vì CLIP chạy trong `no_grad` nên chỉ
  giữ ~2 tensor cùng lúc chứ không phải 12 → batch 8 tốn ~0,8 GB, không đáng lo trên A30 24GB.
- **`.venv-cpu/`** là venv Python 3.11 để test ở local (máy dev không có torch cho Python 3.14).
  Đã `.gitignore`. Trên server dùng env riêng.
- **Loss dùng GIoU** (trọng số 2,0, giống DiffusionDet); **metric báo cáo dùng IoU** — GIoU âm được
  khi hai box rời nhau nên không đọc được như một chỉ số theo dõi.

## Chưa làm

Adapter transformer trên patch token (hoãn có chủ đích — can thiệp một-biến nếu số kém);
`box_renewal` / `use_ensemble` (cần score head chứng minh phân biệt được); SimOTA center prior;
score head kiểu `box_feature · text_feature`; kiểm feature CLIP frozen @512 (nội suy 2,3×).

# CE-Loc — EXPERIMENT A, B, A.1, A.2

## EXPERIMENT A (2026-09-05) — ĐÃ CHẠY XONG, KẾT QUẢ THẤP

Train 300 epoch / 2h16m, best epoch 266 (val_loss 2,5096). **Đã bão hoà** — 100
epoch cuối chỉ cải thiện val_loss 0,032, lần đầu tiên trong dự án chạm trần.

| | test | val |
|---|---|---|
| AP50 | 0,0152 | 0,0073 |
| AP (COCO) | 0,0028 | 0,0013 |
| precision / recall | 0,106 / 0,118 | 0,067 / 0,090 |

**Thắng lợi thật**: score head hết kẹt (sd 0,175, khoảng [0,14–0,98]) — vòng 1 kẹt
cứng ở 0,263. Việc đưa memory từ 2 lên 1026 token có tác dụng.

**Nhưng định vị vẫn yếu.** Nhìn ảnh dự đoán (`tools/visualize_predictions.py`)
thấy rõ: box bám ĐÚNG VÙNG có vật (ảnh viên bi ở góc → box ở góc) nhưng KÍCH
THƯỚC gần như cố định — to hơn hạt đậu, nhỏ hơn con voi. Recall theo cỡ vật:
0,217 (vật trung bình) → 0,041 (vật rất nhỏ).

Đo thêm: recall@0,10 = 0,328, tức 67 % vật không có box nào chạm vào.

## EXPERIMENT B — THIẾT KẾ (đổi đúng một biến)

`model.roi_k: 3`. Mỗi box đọc thêm feature CLIP lấy mẫu trên lưới 3×3 **bên
trong chính nó** (`models/roi_sampler.py`), cộng vào box token.

**Vì sao**: A bắt mạng TỰ HỌC ánh xạ từ sinusoidal PE của `(cx,cy)` sang "phải
attend vào patch nào trong 1024" — hai hệ toạ độ không liên quan, `cond_pos_emb`
khởi tạo ngẫu nhiên. 1.911 ảnh không đủ. DiffusionDet không học thứ này bao giờ:
RoIAlign cắt feature ngay tại toạ độ box, quan hệ là CỨNG.

**Đo trước khi implement** (CLIP frozen, ảnh test thật):

| lưới | AUC vật/nền | AUC "đúng cỡ" vs "to gấp đôi" |
|---|---|---|
| 1×1 (tâm) | 0,990 | **0,000** |
| 3×3 | 0,989 | **0,896** |

Lấy 1 điểm ở tâm cho CÙNG một vector dù box to hay nhỏ → AUC 0,000 không phải
nhiễu mà là chứng minh nó vô cảm với kích thước. Lưới 3×3 lấy lại tín hiệu đó mà
không mất gì.

k=3 chứ không phải 7×7 của DiffusionDet: vật CE-130 chỉ ~2,0×1,7 patch trên lưới
32×32, đặt 49 điểm vào đó là lấy mẫu thừa với chi phí 5,4×.

**Zero-init**: `roi.out` khởi tạo 0 nên **step 0 thì B giống hệt A từng bit**
(khoá bằng `tests/test_roi.py::test_B_equals_A_at_step_zero`). So sánh A vs B là
một biến duy nhất. Lớp vẫn train bình thường — với `y=Wx+b, W=0` thì
`∂L/∂W = δxᵀ ≠ 0`.

**Cửa chặn rẻ**: `roi_branch_norm` in mỗi epoch. Nếu sau ~30 epoch vẫn ~0 thì
chính mạng đang nói RoI feature vô dụng → dừng sớm thay vì đốt 300 epoch. Smoke
test 1 epoch đã cho 0,1185, tức nhánh đang được dùng.

Cache dùng chung với A (`image_size`, `clip_name` không đổi) — RoI lấy mẫu trên
chính `patch_raw` đã cache, không phải build lại.



**Nhánh CE-Loc gốc (ResNet18 + SpatialSoftmax + Conv1D U-Net, single-box) đã XOÁ khỏi thư mục
này.** Bản read-only đầy đủ vẫn ở `refs/repos/Count-Editing/CE-LocModel/` nếu cần đọc lại.

Thiết kế đầy đủ + toàn bộ số đo: [`../../../docs/thiet-ke-ce-loc-vong-2.md`](../../../docs/thiet-ke-ce-loc-vong-2.md).
Lỗi vòng 1 (đọc TRƯỚC khi sửa gì): [`../../../docs/bai-hoc-ce-loc-detection.md`](../../../docs/bai-hoc-ce-loc-detection.md).

**Ràng buộc**: thân model là **Diffusion Policy transformer-based**, chỉ **mượn cơ chế sinh N box**
của DiffusionDet.

## EXPERIMENT B — ĐÃ CHẠY XONG (2026-09-06), KHÔNG CẢI THIỆN

Train 300 epoch / 2h07m trên 1 GPU A30 (batch 8), best epoch **87**.

| test | A | B |
|---|---|---|
| AP50 | 0,0152 | **0,0144** |
| AP (COCO) | 0,0028 | 0,0025 |
| val IoU tốt nhất | 0,345 | 0,347 |

`roi_branch_norm` 0 → **15,84**, tăng đều suốt 300 epoch → nhánh RoI **có** được
dùng, không chết. Nhưng val IoU +0,002 (trong nhiễu).

**Đáng chú ý hơn: B overfit sớm.** Best epoch tụt từ 266 (A) xuống 87, và
`val_rising_streak = 142` — val loss tăng liên tục 142 epoch cuối, trong khi train
IoU vẫn tăng 0,339 → 0,372. Thêm 0,787M tham số trên 1.911 ảnh với split class rời
nhau → nhánh RoI học đặc trưng bám class thay vì quy luật hình học tổng quát.

### Đo tiếp: nút thắt là TÂM, không phải kích thước

`tools/measure_size_regression.py` trên checkpoint B (`recall@0.50`, thay từng nửa
box bằng GT):

| | test | val |
|---|---|---|
| thật | 0,029 | 0,020 |
| thay **w,h** bằng GT | 0,060 | 0,047 |
| thay **cx,cy** bằng GT | **0,284** | **0,232** |

Sửa kích thước cho +0,031; sửa tâm cho **+0,254** — gấp **8 lần**, nhất quán cả 2
split. Giả thuyết "kích thước hằng số" chỉ **PARTIAL** (`size_ratio_within_image`
0,264/0,312, IQR ratio 0,72 — model có biến thiên size, hẹp hơn GT). Và
`l1_share_wh` 53 %/47 % — **cân bằng**, nên giả thuyết "L1 bỏ quên w,h" là **sai**.

→ **Không rebalance loss về w,h.** Model không định vị được từng vật, chứ không
phải định vị được rồi vẽ hộp lệch.

## EXPERIMENT A.1 / A.2 — CHƯA TRAIN

Sau A và B, AP thấp còn **4 nguyên nhân chồng lên nhau, không tách được**:

1. code/thiết kế sai?
2. zero-shot (train 72 class / test 28, **giao = 0**)
3. chỉ 1.911 ảnh
4. vật cực nhỏ, cực đông (37,6 vật/ảnh, 0,41 % diện tích)

COCO-minitrain **gỡ đồng thời (2) và (3)**: 80 class dùng chung train/eval,
73.531 mẫu.

### A.1 — model của A, y nguyên, chạy trên COCO

**Chỉ đổi DỮ LIỆU.** `models/` không sửa dòng nào; score head vẫn **1 chiều**.

Một ảnh COCO có trung bình 2,94 class (chỉ 20,7 % ảnh có đúng 1 class), mà model
nhận **1 text/forward** → tách mỗi ảnh thành nhiều mẫu theo class:

```
ảnh_42 + "person" → 4 box người   (bỏ qua xe, chó)
ảnh_42 + "car"    → 2 box xe
ảnh_42 + "dog"    → 1 box chó
```

→ **73.531 cặp (ảnh, class)** từ 25.000 ảnh.

**Ngưỡng đọc kết quả, chốt TRƯỚC khi chạy** (nếu không thì số nào cũng biện minh
được):

| AP50 trên val2017 | kết luận |
|---|---|
| **< 0,05** | stack hỏng → dừng CE-Loc, sửa code trước |
| 0,05–0,15 | stack chạy đúng, kiến trúc nhỏ |
| **> 0,15** | stack ổn → số thấp trên CE-130 là do **bài toán**, không do bug |

Tham chiếu: DiffusionDet R50 đạt ~30 AP trên đúng 25K ảnh này — nhưng có FPN,
RoIAlign, 6 stage deep supervision, backbone train được. Ghi để thấy khoảng cách
kiến trúc, **không phải mục tiêu**.

**CHỈ ĐỌC AP50, BỎ QUA PRECISION THÔ.** COCO có 2,47 box/cặp so với CE-130 37,6 →
với N=100 trần precision cấu trúc `min(M,N)/N` ở đây là ~0,01 còn CE-130 là 0,376.

### A.2 — bỏ text, score head 80 chiều

**A.1 và A.2 khác đúng MỘT thứ**: class đến với model qua đâu.

| | class đi qua | memory | head | mẫu |
|---|---|---|---|---|
| A.1 | **text đầu vào** | 1025 token (patch + text) | 1 chiều | 73.531 cặp |
| A.2 | **head đầu ra** | 1024 token (chỉ patch) | 80 chiều | 25.000 ảnh |

Cùng ảnh, cùng annotation (181.475 box), cùng loss/matcher/diffusion/N.

Đọc kết quả:
- **A.2 ≈ A.1** → đường text hoạt động tốt ngang head trực tiếp. Tốt nhất cho CE-Loc.
- **A.2 ≫ A.1** → đường text **là nút thắt** → khớp với mismatch không gian đã đo
  ([docs/y-tuong-khong-gian-box-vs-anh.md](../../../docs/y-tuong-khong-gian-box-vs-anh.md)).
- **cả hai đều kém** → lỗi ở phần chung (diffusion/matcher/decoder).

**A.2 đang giải bài DỄ HƠN** — 1 forward xong cả ảnh (7,26 box) thay vì ~2,94
forward; và nó tự đặt tên class nên không bị phạt vì "đặt nhầm class". Nên A.2 cao
hơn là **dự kiến**; chỉ khoảng cách **lớn** mới là bằng chứng. Eval đã làm
class-aware (`eval.py` tách theo class) nên ít nhất box đúng vị trí sai class
không được tính.

**Ngân sách khớp theo lượt-ảnh, không theo epoch**: A.1 20 epoch × 73.531 =
1,47M; A.2 **59** epoch × 25.000 = 1,475M (lệch 0,3 %). Bằng epoch thì A.1 được
gấp 3 lần compute và phép so đo ngân sách chứ không đo điều kiện hoá.

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
# 0. CỬA CHẶN Ở MÁY DEV — chạy TRƯỚC KHI push. Không cần GPU, ~5 phút.
#    5 bước: pytest -> 1 bước train THẬT mỗi config (data -> collate -> loss ->
#    backward) -> import mọi tool -> CHẠY THẬT preflight trên dữ liệu nhỏ -> đối
#    chiếu chéo 4 config.
python3 tools/check_before_train.py
#    Hai bước "chạy thật" là bắt buộc, mỗi bước ứng với một bug ĐÃ LỌT:
#    - `KeyError: 'labels'` trong collate: dataset trả key, TorchWrap không
#      chuyển tiếp. Mọi test dataset pass, train chết ngay bước đầu.
#    - `NameError: tl` trong 4 callback của preflight.py: py_compile chỉ bắt
#      SyntaxError, tên trong hàm chỉ resolve khi hàm CHẠY. Lọt tới server.
#    Cả hai đều có test âm bản: tái tạo lỗi -> check phải đỏ.

# 1. Test riêng (nếu chỉ muốn phần này), ~2 phút
python3 -m pytest tests/ -q

# 2. Nhìn ảnh TRƯỚC khi train — vòng 1 visualize bắt được 3 lỗi mà test bỏ sót
python3 tools/visualize_data.py --n 20 --out /tmp/viz
python3 tools/visualize_data.py --n 8 --out /tmp/viz_ph --placeholder --t 50

# 3. Đo memory + có cần cache không (trên server)
python3 tools/run_on_free_gpu.py -- tools/profile_and_memory.py --batch-size 8 --steps 10
#    ĐO ĐƯỢC trên A30: CLIP chiếm 76,8 % thời gian (252/328 ms) -> NÊN cache.
#    Build (~20s, 6,0 GB cho train), rồi train thêm --cache:
python3 tools/run_on_free_gpu.py -- tools/build_cache.py --split train --out ../../data/cache_clip
python3 tools/run_on_free_gpu.py -- tools/build_cache.py --split val   --out ../../data/cache_clip
#    -> train nhanh ~4,3x (328 -> 76 ms/batch): 300 epoch từ 6,5h xuống 1,5h

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

### A.1 / A.2 trên COCO — không dùng cache

Cache cho A.1 sẽ là **78,6 GB** (25.000 ảnh × 2 bản flip × 1024 × 768 fp16) nên
**không build** — CLIP chạy thật mỗi bước. Bù lại batch 32 thay vì 8.

```bash
# preflight (có riêng mục "experiment wiring" kiểm A.1/A.2)
python3 tools/run_on_free_gpu.py -- tools/preflight.py --config config/experiment_a1.yaml

# A.1 — 20 epoch, ~2-5h
nohup python3 tools/run_on_free_gpu.py -- train.py --config config/experiment_a1.yaml \
    > /mnt/disk1/aiotlab/haitn/log/experiment_a1.log 2>&1 & echo "PID=$!"

# A.2 — 59 epoch (khớp lượt-ảnh với A.1, KHÔNG khớp epoch), ~2-5h
nohup python3 tools/run_on_free_gpu.py -- train.py --config config/experiment_a2.yaml \
    > /mnt/disk1/aiotlab/haitn/log/experiment_a2.log 2>&1 & echo "PID=$!"

# eval (val2017; COCO không có split thứ ba, --split test trỏ về cùng file)
python3 tools/run_on_free_gpu.py -- eval.py --config config/experiment_a1.yaml \
    --ckpt checkpoints/experiment_a1/best.pth --split val --num-proposals 300
```

**`loss_ce` của A.2 sẽ rất lớn ở epoch đầu — đừng "sửa".** Với C=80 và logit=0 thì
nó đúng bằng **80×** giá trị 1 chiều (đo được 520 vs 6,5). Focal dập rất nhanh
(520 → 3,16 ở p=0,1 → 0,02 ở p=0,02), hết ngay trong epoch đầu. Hệ quả thật: grad
đầu run lớn nên clipping quan trọng hơn, và **`loss_ce` không so được** giữa A.1
và A.2 — so AP và IoU.

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

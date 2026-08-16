# DiffuGroundingDINO — diffusion trên reference point cho GroundingDINO

Ghép ý tưởng **DiffuDINO** (biến thể diffusion của DINO, trong paper
[DiffuDETR, ICLR 2026](../../refs/DiffuDETR.md)) vào **GroundingDINO**
([ECCV 2024](../../refs/GroudingDINO.md)): thay vì khởi tạo reference point của decoder
bằng top-k proposal của encoder, coi reference point là **biến latent của một DDPM** —
train thì noise box GT tới timestep `t` ngẫu nhiên rồi bắt decoder khử nhiễu, eval thì
xuất phát từ nhiễu thuần và chạy DDIM 3 bước.

Mục tiêu bài toán giữ nguyên **OD-style open-vocabulary**: cho ảnh + prompt
`"person . car . dog ."` → detect **tất cả** instance của các category đó (không phải chọn
1 instance kiểu referring expression).

Đây là bước khảo sát tiếp theo của dự án sau DiffusionDet (xem [`../../RESULTS.md`](../../RESULTS.md)).

## Trạng thái

| | |
|---|---|
| Code | **Đủ để train** — 9.100 dòng Python, không import gì từ `repos/` |
| Test | **79/79 pass** trên CPU, không cần weights hay dataset (`python tests/run_all.py`) |
| Kiến trúc | **Đã đối chiếu 2 lần** với Open-GroundingDino/GroundingDINO/DiffuDETR gốc (encoder/fusion/decoder/ContrastiveEmbed/matcher/criterion đúng tuyệt đối; 4 điểm lệch default đã sửa, xem [§ Audit kiến trúc](#audit-kiến-trúc-lần-2)) |
| Checkpoint | **Đã tải + verify khớp key** — `tools/check_checkpoint.py` báo `RESULT: OK` (938/940 tensor load, phần còn lại là module diffusion mới) |
| Train/eval thật | **Chưa chạy** — bước tiếp theo, trên GPU server (xem [§ Chạy](#chạy)) |

## Ý tưởng, ngắn gọn

Decoder của GroundingDINO vốn đã tinh chỉnh box qua từng layer:
`box_{n} = sigmoid(delta_n + logit(box_{n-1}))`. Diffusion **không đụng vào cơ chế đó** —
chỉ đổi thứ nằm ở `box_0`:

| | Baseline | Diffusion |
|---|---|---|
| `box_0` lúc train | top-k proposal của encoder | box GT đã noise tới `t ~ U(0,T)` |
| `box_0` lúc eval | top-k proposal của encoder | nhiễu thuần, khử dần qua 3 bước DDIM |
| Điều kiện thêm | — | timestep embedding chèn vào query mỗi decoder layer (3 điểm/layer, xem [§ Audit kiến trúc](#audit-kiến-trúc-lần-2)) |
| Loss | set-prediction chuẩn | như trên, nhân thêm trọng số `w(t)` |

Ba thứ **cố ý không** diffuse:

- **Content query** (`tgt_embed`) — paper mô tả là "static learnable content queries"; code
  DiffuDINO có tính `diff_labels` nhưng **không dùng** tới nó.
- **Nhánh encoder proposal** (`interm_outputs`) — vẫn tính và vẫn train. Nó không còn quyết
  định box khởi tạo nữa, nhưng chính nó dạy encoder sinh feature có tính "object", mà cả
  pipeline đọc từ feature đó.
- **Matcher** — vẫn Hungarian. (Code DiffuDINO có truyền `indices` precompute vào criterion
  nhưng bị ghi đè ngay bởi matcher ở
  [`diffu_criterion.py:363`](../../repos/DiffuDETR/projects/diffu_dino/modeling/diffu_criterion.py#L363) —
  tức là dead code.)

Nhờ vậy `use_diffusion=False` cho lại **đúng** baseline, dùng để A/B.

## Cấu trúc

```
diffu_grounding_dino/
├── models/
│   ├── diffusion/{schedule,timestep}.py   # DDPM trên reference point + timestep embedding
│   ├── backbone/{swin,position_encoding}.py
│   ├── ops/ms_deform_attn.py              # deformable attention thuần PyTorch
│   ├── text/bert.py                       # tokenizer, BERT, sub-sentence mask
│   ├── transformer.py                     # encoder/decoder, TÁCH encode()/decode()
│   ├── diffu_groundingdino.py             # model chính + ddim_sample()
│   ├── criterion.py, matcher.py, postprocess.py, layers.py, fusion.py
├── datasets/{odvg,coco,coco_eval,transforms}.py
├── util/{config,misc,box_ops,param_dicts,vl_utils,logger}.py
├── config/{cfg_odvg,cfg_odvg_diffusion}.py + datasets_*.json
├── tools/{coco2odvg,download_weights,check_checkpoint,run_train}.py
├── tests/                                 # 5 suite, chạy được offline
└── engine.py, main.py
```

Vì sao tách `encode()` / `decode()`: DDIM cần chạy decoder nhiều lần cho **cùng một ảnh**.
Nếu để nguyên `forward()` như upstream thì backbone + BERT + encoder cũng chạy lại mỗi bước,
tức là ×3 chi phí inference mà không được gì (ảnh và text có thay đổi đâu). Sau khi tách:
3 bước sampling = **1 lần** backbone/BERT/encoder + **3 lần** decoder, đúng con số +17% GFLOPs
mà DiffuDETR báo cáo (Bảng 12). Có test khoá lại điều này: `test_ddim_sample_encodes_once_per_image`.

## Cài đặt

```bash
pip install -r requirements.txt
python tools/download_weights.py --dest ../weights/diffu_grounding_dino   # 1,1GB, hoặc tải tay
```

Cần 2 thứ trong `../weights/diffu_grounding_dino/`:

| File | Dung lượng | Vai trò |
|---|---|---|
| `groundingdino_swint_ogc.pth` | 694MB | pretrain Swin-T (O365+GoldG+Cap4M) — **không có thì mất hết khả năng open-vocab** |
| `bert-base-uncased/` | 440MB | text encoder (lấy `model.safetensors`, đừng lấy `pytorch_model.bin`) |

Sau khi tải, trỏ config vào bản local để không phụ thuộc mạng:

```python
text_encoder_type = "../weights/diffu_grounding_dino/bert-base-uncased"
```

Không cần compile CUDA extension. `MultiScaleDeformableAttention` bản compiled nếu có thì
dùng (nhanh hơn), không có thì chạy bản `grid_sample` thuần PyTorch — cùng phép toán, có test
đối chiếu với một bản cài đặt bilinear viết tay (`test_deformable_attn_core_matches_naive_reference`).

## Chuẩn bị dữ liệu

Train đọc **ODVG jsonl**, eval đọc **COCO json**:

```bash
python tools/coco2odvg.py \
  --input ../../data/coco_minitrain/annotations/instances_minitrain2017.json \
  --output-jsonl ../../data/coco_minitrain/annotations/minitrain_odvg.jsonl \
  --output-label-map ../../data/coco_minitrain/annotations/label_map.json
```

`config/datasets_coco_minitrain.json` đã trỏ sẵn đúng đường dẫn. VOC và CrowdHuman: convert
sang COCO json trước (CrowdHuman đã có
[`../diffusiondet/tools/convert_crowdhuman.py`](../diffusiondet/tools/convert_crowdhuman.py)),
rồi chạy cùng `coco2odvg.py` — nó đọc category id từ file chứ không giả định.

## Chạy

Chạy trên GPU server riêng (không còn Kaggle) — **1 GPU mỗi lần** (server dùng chung với
job khác, không có nhiều GPU rảnh cùng lúc như Kaggle 2×T4 trước đây).
[`tools/run_train.py`](tools/run_train.py) tự chọn GPU trống nhất qua `nvidia-smi` (nhiều free
memory nhất trong số GPU không bận tính toán) rồi gọi `main.py` với mọi tham số truyền vào:

```bash
# nhánh diffusion
python tools/run_train.py -c config/cfg_odvg_diffusion.py \
  --datasets config/datasets_coco_minitrain.json \
  --output_dir output/diffu_run1 \
  --pretrain_model_path ../weights/diffu_grounding_dino/groundingdino_swint_ogc.pth \
  --finetune_ignore time_ diffusion

# baseline để A/B (cùng data, cùng seed, chỉ khác 1 flag)
python tools/run_train.py -c config/cfg_odvg.py \
  --datasets config/datasets_coco_minitrain.json \
  --output_dir output/baseline_run1 \
  --pretrain_model_path ../weights/diffu_grounding_dino/groundingdino_swint_ogc.pth

# eval, quét số bước sampling
for S in 1 3 5 10; do
  python tools/run_train.py -c config/cfg_odvg_diffusion.py \
    --datasets config/datasets_coco_minitrain.json \
    --output_dir output/eval_s$S --eval \
    --resume output/diffu_run1/checkpoint_best_regular.pth \
    --options diff_sampling_timesteps=$S
done
```

Ép chạy trên 1 GPU cụ thể (bỏ qua auto-detect): thêm `--gpu <index>` ngay sau
`run_train.py`. Muốn gọi thẳng `main.py` không qua wrapper (ví dụ để debug) thì bỏ
`tools/run_train.py` và tự `export CUDA_VISIBLE_DEVICES=<index>` trước.

`--finetune_ignore time_ diffusion` là bắt buộc ở lần đầu: checkpoint pretrain không thể có
module timestep. Có test khoá lại rằng 2 keyword đó phủ hết mọi key mới
(`test_diffusion_extra_keys_are_covered_by_finetune_ignore`).

## Verification

```bash
python tests/run_all.py                       # 79 test, ~3 phút CPU, không cần weights
python tools/check_checkpoint.py -c config/cfg_odvg_diffusion.py \
  --checkpoint ../weights/diffu_grounding_dino/groundingdino_swint_ogc.pth
```

Ánh xạ sang 7 mục trong [plan](../../docs/diffu-grounding-dino-plan.md):

| Mục plan | Ở đâu | Trạng thái |
|---|---|---|
| 1. `betas`/`alphas_cumprod` khớp công thức cosine | `test_cosine_schedule_matches_reference` | ✅ khớp tuyệt đối (atol=0) |
| 2. `prepare_diffusion_refpoints`, kể cả `num_gt=0` | `test_refpoints_*` (7 test) | ✅ |
| 3. Forward 1 batch, shape + không NaN | `test_diffusion_training_forward` | ✅ |
| 4. `ddim_sample` với 1/3/10 bước, encode 1 lần | `test_ddim_sample_encodes_once_per_image` | ✅ đếm bằng counter |
| 5. Overfit + gradient của `time_embed` | `test_loss_decreases_when_overfitting_one_batch`, `test_time_embed_learns_after_the_first_step` | ✅ ở quy mô nhỏ; **overfit 20-50 ảnh COCO thật thì chưa chạy** |
| 6. `use_diffusion=False` khớp baseline | `test_baseline_forward_matches_encode_decode_split`, `test_baseline_and_diffusion_share_head_weights` | ✅ |
| 7. Load pretrain, kiểm `missing_keys` | `tools/check_checkpoint.py` | ✅ đã chạy với checkpoint thật — `RESULT: OK` |

Mục 5 bản đầy đủ (sau khi có data) — sanity check chuẩn trước khi finetune thật: overfit một
tập nhỏ 20-50 ảnh để bắt lỗi wiring (gradient không chảy, loss không giảm...) rẻ và nhanh hơn
nhiều so với phát hiện ra sau một lần train đầy đủ:

```bash
head -50 ../../data/coco_minitrain/annotations/minitrain_odvg.jsonl > /tmp/overfit50.jsonl
# sửa "anno" trong datasets json trỏ vào /tmp/overfit50.jsonl, rồi:
python tools/run_train.py -c config/cfg_odvg_diffusion.py --datasets config/datasets_overfit.json \
  --output_dir output/overfit --options epochs=20 batch_size=2 diff_warmup_iters=0 --save_log
```
Kỳ vọng: `loss_bbox`/`loss_giou`/`loss_ce` giảm đơn điệu, không NaN.

## Audit kiến trúc lần 2

Sau khi có checkpoint thật, đã đọc lại kỹ code gốc của **Open-GroundingDino**,
**GroundingDINO**, và **DiffuDETR** (`repos/`) và đối chiếu từng phần với rewrite ở đây (không
chỉ đọc paper). Encoder/fusion, decoder box-refinement, `ContrastiveEmbed`, matcher, criterion,
Swin backbone, BERT wrapper: **đúng tuyệt đối**, không bug. Cơ chế diffusion đúng phần lõi
(noising, `q_sample`, loss target `x0`, cấu trúc DDIM) nhưng lộ ra 4 điểm lệch so với DiffuDETR
gốc, đã sửa:

| Điểm lệch | Trước | Sau (khớp DiffuDETR gốc) | Vì sao |
|---|---|---|---|
| **FiLM timestep injection** | 1 điểm/layer (sau self-attn), docstring nói "khớp eq.3" nhưng thực ra DiffuDETR chèn tận 3 điểm | 3 block FiLM **độc lập tham số** mỗi layer: trước self-attn, giữa self-attn/cross-attn, trước FFN (`diff_time_inject_point="triple"`) | Đây là finding nghiêm trọng nhất — docstring cũ overclaim, hành vi thực tế yếu hơn bản gốc |
| `diff_ddim_eta` mặc định | `1.0` (sampler ngẫu nhiên) | `0.0` | DiffuDETR hardcode `ddim_sampling_eta=0.0`, phần `+sigma·noise` là dead code trong bản gốc |
| `diff_pad_mode` mặc định | `"normal"` (jitter kiểu DiffusionDet) | `"center"` (box hằng số `[0.5,0.5,0.5,0.5]`) | Đúng mặc định `noisy_gt=False` của DiffuDETR |
| `inverse_sigmoid` eps | `1e-5` | `1e-3` | Không phải lựa chọn cố ý — khớp lại đúng convention GroundingDINO/Open-GroundingDino gốc |

Cả 4 đều có test bao phủ (`tests/run_all.py`, 79/79 pass) và `tools/check_checkpoint.py` vẫn
báo `RESULT: OK` sau khi sửa. Ba điểm còn **cố ý khác** DiffuDETR gốc (bảng dưới) không đổi.

## Khác biệt cố ý so với DiffuDINO gốc

Đọc code thật ở [`repos/DiffuDETR/projects/diffu_dino/`](../../repos/DiffuDETR/projects/diffu_dino/)
chứ không chỉ paper. Các chỗ chọn khác dưới đây **vẫn giữ nguyên** (khác với 4 điểm ở
[§ Audit kiến trúc lần 2](#audit-kiến-trúc-lần-2), vốn là lệch không chủ đích và đã sửa khớp
lại). Đều có cờ config để quay lại bản gốc:

| Chỗ | DiffuDINO gốc | Ở đây | Vì sao |
|---|---|---|---|
| FiLM residual | `return x + h` → ở init là `2x` | `return h` → ở init là identity | Họ train from scratch nên không sao; ta **finetune** từ checkpoint pretrain, nhân đôi query ở bước 0 sẽ phá decoder. Cờ `diff_film_residual=True` |
| Trọng số `w(t)` | `0.5·√ᾱ/(2−ᾱ)`, mean ≈ 0.2 | chuẩn hoá về mean = 1 | Bản gốc âm thầm chia `loss_bbox` cho ~5, khiến `bbox_loss_coef=5.0` không còn cùng nghĩa với baseline → A/B mất ý nghĩa. Cờ `diff_normalize_loss_weight=False` |
| Bước DDIM | trộn không gian: dùng `bbox_start` ∈ [0,1] với hệ số của không gian latent ([dòng 1032](../../repos/DiffuDETR/projects/diffu_dino/modeling/dino_diffu_det_noise.py#L1032)) — bản thân code gốc đã tự mâu thuẫn với schedule họ train | map về latent rồi mới bước, nhất quán trong suốt | Trộn 2 không gian làm chuỗi sampling lệch khỏi schedule đã train; nghĩa là ở đây sẽ không tái lập bit-for-bit số của DiffuDETR, nhưng đúng toán hơn |
| Khởi tạo lúc eval | noise **top-k proposal của encoder** ở `t=T−1` | `randn` thuần | Với cosine `ᾱ_{T−1} ≈ 2e−9` nên gần như tương đương; `randn` thì đúng định nghĩa DDPM và không tạo phụ thuộc ngầm vào nhánh two-stage |

Ngoài ra **chưa có CDN** (300 contrastive denoising query mà DiffuDINO dùng): nền
Open-GroundingDino hardcode `dn_number=0` và **không có code CDN nào cả**. Đây là khác biệt
lớn nhất so với DiffuDINO, cần ghi rõ khi báo cáo số. Cũng chưa có **box renewal + ensemble
multi-step** ở `ddim_sample` (DiffuDETR loại bỏ box tin cậy thấp giữa chuỗi DDIM và thay bằng
nhiễu mới, rồi NMS-ensemble qua các bước) — không ảnh hưởng tính đúng của training, chỉ ảnh
hưởng khi so sánh chất lượng inference bit-for-bit với số đã công bố.

Các mặc định khác bám sát: `snr_scale=2.0` (khớp cả DiffusionDet lẫn `self.scale` của
DiffuDINO), cosine schedule, `sampling_timesteps=3` (tối ưu theo ablation Bảng 6/7 — **không
phải** `SAMPLE_STEP=1` của DiffusionDet), `w(t)` kiểu Improved-DDPM `lvlb_weights`.

## Ba bug của Open-GroundingDino đã sửa

Phát hiện khi viết lại và test, đều tồn tại trong repo gốc:

1. **`create_positive_map` dùng `caption.find(category)`** — sai khi tên category này là
   substring của category kia. Prompt `"carrot . car ."` → `find("car")` trả về 0 → **mọi box
   `car` bị supervise vào token của `carrot`**. Ảnh hưởng COCO thật. Ở đây offset được tính xác
   định từ cách dựng caption (`util/vl_utils.py`), test `test_positive_map_is_substring_safe`.
2. **`num_pos_feats=256` hardcode** cho positional embedding của text (`TransformerEncoder`).
3. **`gen_sineembed_for_position` hardcode 128**.

(2) và (3) chỉ vỡ khi `hidden_dim ≠ 256` nên không ảnh hưởng số liệu published; đã đổi thành
`self.d_model` và `d_model // 2`.

Còn một chỗ **giữ nguyên có chủ ý**: `loss_ce` chuẩn hoá theo số positive **của riêng rank
đó**, không all-reduce (khác `num_boxes` thì có). Đây là hành vi upstream; sửa sẽ làm lệch cán
cân classification/box và mất khả năng so với con số 57.3 mAP đã công bố.

## Những chỗ dễ sai

- **Không bật `--amp`.** Mặc định tắt theo [CLAUDE.md](../../CLAUDE.md). Toán schedule đã bị
  ép fp32 bằng `force_fp32()` bất kể caller (buffer `sqrt_recipm1_alphas_cumprod` lên tới
  2e4 ở `t=999`, fp16 mất sạch đuôi), nhưng phần multi-stage box refinement thì chưa kiểm
  chứng. Muốn bật thì đo A/B trước.
- **Caption bắt buộc kết thúc bằng `" ."`.** Không có dấu chấm cuối thì category cuối cùng
  **không được cấp block attention** — `[SEP]` ở cột cuối rơi vào nhánh "special token đứng
  một mình". Xem `test_trailing_separator_is_required`. Luôn dựng caption bằng
  `util.vl_utils.build_caption`.
- **Thứ tự trong `main.py`: `get_param_dict` → freeze, không được đảo.** `get_param_dict`
  lọc theo tham số nó nhìn thấy; freeze trước thì param biến mất khỏi optimizer vĩnh viễn và
  unfreeze sau vô nghĩa. Ở đây gọi với `include_frozen=True` rồi freeze sau.
- **Warmup freeze tính từ `global_step`**, không dùng flag → resume ở step 5000 với warmup
  2000 là unfreeze ngay, không cần lưu state (`test_warmup_freeze_is_resume_safe`).
- **`transformers` mặc định dùng SDPA cho BERT, mà SDPA không nhận mask 3D.** Đã ép eager
  bằng `force_eager_attention()` (set cả `config._attn_implementation` lẫn attribute trên
  module, vì `BertModel` cache lại lúc `__init__`). Test `test_bert_rejects_3d_mask_under_sdpa`
  sẽ báo khi nào bỏ được workaround này.
- **Tên module phải khớp checkpoint.** Mọi key của `groundingdino_swint_ogc.pth` phải load
  được; đổi tên module = âm thầm train lại phần đó từ đầu. Chạy `tools/check_checkpoint.py`
  trước mỗi lần train.

## Việc chưa làm

- Train/eval thật — checkpoint đã sẵn sàng, chưa chạy finetune (bước tiếp theo trên GPU server).
- Overfit 20-50 ảnh COCO thật — mục 5 bản đầy đủ, nên chạy trước khi finetune full.
- CDN denoising query (khác biệt lớn nhất còn lại so với DiffuDINO).
- Box renewal + ensemble multi-step ở `ddim_sample` (xem [§ Khác biệt cố ý](#khác-biệt-cố-ý-so-với-diffudino-gốc)).
- EMA — DiffuDETR §A.1.4 nói EMA giúp ổn định training diffusion; config đã có
  `use_ema`/`ema_decay` nhưng engine chưa dùng.
- Ablation `T=100` (paper) vs `T=1000` (code), noise Gaussian/Beta/Sigmoid.
- VOC / CrowdHuman (COCO-minitrain trước, theo plan).

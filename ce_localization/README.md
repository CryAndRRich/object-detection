# ce_localization — CE-Loc

Chọn box cho count editing trên CE-130: cho ảnh + tên lớp, sinh N box của lớp đó bằng khuếch tán
trên toạ độ box. Mục tiêu cuối là CE-Loc; detect trên CE-130 là phương tiện học cơ chế "điều kiện
hoá box theo ảnh + text" (xem `CLAUDE.md` ở gốc `multi_condition/`).

**Một thí nghiệm = một config** (`config/<nhóm>/<tên>.yaml`), cùng `train.py` / `eval.py`. Idea mới
thì thêm config, và nếu cần thì thêm module vào đúng package (`data/`, `models/`, `engine/`) kèm khoá
config bật nó — **không** tạo thư mục hay script riêng theo tên thí nghiệm.

## Kiến trúc

```
ảnh [B,3|4,T,T] (letterbox góc trên-trái, T = 1024; kênh 4 = density nếu có)
  -> R-50 (mọi conv train, BN đóng băng) + FPN P2..P5
box nhiễu [B,N,4] (N = 200) --6 stage, mỗi stage:
    RoIAlign(P2..P5, box) -> 1 token / box
    tầng decoder Diffusion Policy: self-attn giữa N token -> cross-attn memory [t ; text ; (ảnh)] -> FFN
    head DiffusionDet: score + delta -> box mới (detach trước stage sau)
Loss DiffusionDet (SimOTA, focal + L1 + GIoU) ở cả 6 stage; suy luận DDIM 1 hoặc nhiều bước.
```

Bài **ADD** (`task: add`, GAMMA / DELTA — `model.arch: box_policy`): CE-Loc gốc, đầu vào kiểu bài (`to_tensor` + density `.convert("L")`),
box chia canvas. Ảnh qua ResNet18 của bài (`vision: r18_paper`) hoặc R-50 ("CE-Loc gốc + R-50", GAMMA0).
```
ảnh inpaint lượt t [B,3|4,512,512] -> ResNet18 / R-50 -> C5 -> SpatialSoftmax kiểu bài -> Linear -> 128 ──┐
tên lớp -> CLIP text frozen -> Linear -> Mish -> 128 ──────────────────────────────────────────────────────┤ cond
box nhiễu [B,1,4] + t -> U-Net 1D (FiLM theo cond) -> ε̂ ; ε-MSE, β tuyến tính ; suy luận vòng mock 100 bước của bài / DDPM, K mẫu độc lập
```
DELTA (`model.refiner`, `models/clip_refiner.py`): BoxRefiner của tác giả gốc TRONG mạng khử nhiễu (CLIP ViT + text frozen, RoIAlign,
4 block RoPE 2D + AdaLN-Zero), `refiner.unet_input`:
- `concat`: refiner trước U-Net, `refined = x_t + Δ`, U-Net nhận `[x_t ; refined]`;
- `none`: không có U-Net, refiner là cả mạng khử nhiễu (DELTA3);
- `post`: U-Net trước, refiner nhìn x̂0 của U-Net rồi sửa ε (DELTA3.1 / 3.2).
- `mid`: refiner chèn giữa U-Net ở vị trí mid: Linear h → box b (giám sát về x0, `mid_aux_weight`) → refiner → Linear về h (DELTA4.1).
- `model.box_coder: log` (DELTA5.x, `utils/box_coder.py`, bản log-scale của tác giả): w, h trên thang log; `refiner.aux_giou_weight`: GIoU trong loss phụ; sampler `ddim` (`eval.py --add-samplers ddim --ddim-steps 100 --eta 1`).

Kết hợp được với `obj_attn` (DELTA4): luồng vật vào U-Net 8 kênh dưới dạng `[o ; o]`.
GAMMA1 (`model.arch: box_refiner`, `models/box_refiner.py`): cùng điều kiện nhưng là memory `[t ; text ; SpatialSoftmax C5]`;
1 box nhiễu -> 6 tầng (RoIAlign P2..P5 -> token -> cross-attn memory -> FFN -> delta box, KHÔNG self-attn), dự đoán x0,
L1 + GIoU ở mọi tầng; suy luận DDIM `--steps`.

## Cấu trúc

| | |
|---|---|
| `train.py` / `eval.py` | điểm vào duy nhất (1 GPU hoặc `torchrun`; `--resume`, `--max-hours`, `--bench`, `--nan-debug`) |
| `config/alpha/`, `config/beta/`, `config/gamma/`, `config/delta/` | config từng thí nghiệm (bảng dưới) |
| `data/` | `dataset` (quét CE-130, letterbox, đích box / điểm), `density` (giải mã jet, chỉ mục, chọn bản), `points` (đỉnh density, cỡ giả kNN), `turns` (bài add: chỉ mục (nhánh, lượt) ↔ `samples/`, dataset) |
| `models/` | `backbone`, `roi`, `memory`, `head`, `detector`, `text` (CLIP ViT-B/32 frozen; DELTA: + token chữ CLIP text); bài add: `box_policy` + `unet1d` (GAMMA0 / pha 1 / GAMMA4 / DELTA), `box_refiner` (GAMMA1), `propose_refine` + `dynamic_head` (GAMMA2 / 3 / 4.1), `clip_refiner` (DELTA: BoxRefiner của tác giả gốc, port từ `refs/CE-Loc-update/`) |
| `engine/` | `diffusion`, `criterion` (SimOTA + loss, chế độ box / điểm), `evaluate` (suy luận + chỉ số), `add_eval` (bài add: K mẫu, IoU với lỗ, C-NLL, on_object), `train_utils`, `nan_debug` |
| `utils/` | hình học box (torch / numpy), toán khuếch tán, chấm điểm numpy, checkpoint ghi nguyên tử, grad theo nhóm, log |
| `tools/` | `build_density_index`, `build_density_points`, `build_turn_index` (bài add, cửa G0), `visualize_data` (xem đầu vào bằng mắt), `check_data_facts`, `plot_denoise_trajectory` (bài add: box qua từng bước khử nhiễu, checkpoint của bài hoặc GAMMA0), `plot_refiner_steps` (quỹ đạo khử nhiễu + SpatialSoftmax cho GAMMA1 / CE-Loc kiểu bài / DELTA; DELTA vẽ thêm box refiner + box U-Net mỗi bước; `--compact` = 1 hình mỗi ca, mỗi hàng theo vết một loại box) |
| `user_study/` | user study bài add: `selection` (chọn 4 box từ 30 mẫu), `build` (gộp dump → `items.json`), `app` (web Gradio chấm), `score` (chỉ số) |
| `box_review/` | web Gradio rà box dữ liệu bằng tay: `sources` (CE-CoCount theo file, CE-130 theo nhánh: box vật + box lỗ), `app` (bấm box = loại, box biến mất; Enter lưu, Ctrl+Z hoàn tác) |
| `notebooks/` | `train_kaggle.ipynb` (một config trên T4×2: ALPHA / BETA / GAMMA0–1), `add_kaggle.ipynb` (bài add từ GAMMA2: `gamma2_celoc` / `gamma2` / `gamma2_1` / `gamma3` / `gamma3_1` / `gamma4` / `gamma4_0_1` / `gamma4_0_2` / `gamma4_1`, DELTA) — gitignore, chỉ ở local |

Test ở `object-detection/tests/ce_localization/` theo module: `test_data`, `test_models`,
`test_engine`, `test_train_eval` (trọn luồng mọi loại config); dữ liệu giả dùng chung ở `helpers.py`.
Code soi SpatialSoftmax (`celoc_paper/`, `tools/inspect_*spatial_softmax.py`, TN1–TN3 của
`docs/SPATIAL_SOFTMAX.md`) đã xoá 2026-10-01 — còn trong lịch sử git.

## Thí nghiệm (config)

| config | khác `alpha/alpha0.yaml` ở | kế hoạch + kết quả |
|---|---|---|
| `alpha/alpha0.yaml` | — (memory `[t ; text]`) | `docs/EXPERIMENT_ALPHA.md` |
| `alpha/alpha1.yaml` | `model.memory: spatial_softmax` (+ 1 token SpatialSoftmax(P5)) | 〃 |
| `alpha/alpha2.yaml` | `model.memory: grid` (lưới ô P5 16×16 + PE 2D) | 〃 |
| `alpha/alpha3_1.yaml` | `model.in_channels: 4`, `data.density: full` (density kênh 4, cố định bản đầy đủ nhất) | 〃 mục 5 |
| `alpha/alpha3_2.yaml` | như trên, `data.density: mix` (1/3 đầy đủ · 1/3 thiếu vật · 1/3 trống) | 〃 mục 5 |
| `alpha/alpha3_2_36k.yaml` | như `alpha3_2`, train 36k iter (steps 27k / 33k) | 〃 mục 12.5 |
| `beta/beta0.yaml` | `data.targets: point` (đích = box giả từ điểm density, box GT chỉ để chấm) | `docs/EXPERIMENT_BETA.md` |
| `gamma/gamma0.yaml` | **bài add**: `task: add`, `model.arch: box_policy`, canvas 512, batch 16, density của chính mẫu (kênh 4) | `docs/EXPERIMENT_GAMMA.md` |
| `gamma/gamma0_1.yaml` | như `gamma0`, CHỈ RGB | 〃 |
| `gamma/gamma1.yaml` | bài add, `model.arch: box_refiner` (6 tầng RoI + cross-attn, loss mọi tầng), RGB + density | 〃 mục 11 |
| `gamma/gamma1_1.yaml` | như `gamma1`, `model.box_token: coords` (bỏ RoI: token = Linear(4 toạ độ box)) | 〃 mục 11.1 |
| `gamma/gamma2_celoc.yaml` | GAMMA2 pha 1: CE-Loc gốc (ResNet18) + SpatialSoftmax mask phần đệm, `data.split_source: samples` | 〃 mục 13 |
| `gamma/gamma2.yaml` | GAMMA2: `model.arch: propose_refine` — CE-Loc pha 1 đóng băng -> refine 6 stage kiểu DiffusionDet | 〃 mục 13 |
| `gamma/gamma2_1.yaml` | như `gamma2`, `freeze_proposer: false` (train chung 2 loss) | 〃 mục 13 |
| `gamma/gamma3.yaml` | như `gamma2` + `model.geo`: FiLM theo hình học tương đối với box vật đang có (`models/geo.py`) ở đầu mỗi stage refine | 〃 mục 14 |
| `gamma/gamma3_1.yaml` | như `gamma2` + `model.relation`: attention kiểu Relation-DETR từ box tới feature RoI các vật gần nhất, điểm cộng Rel hình học (`models/relation.py`) | 〃 mục 15 |
| `gamma/gamma4.yaml` | như `gamma2_celoc` (CE-Loc đứng một mình, train từ đầu) + `model.obj_attn`: cross-attn box nhiễu → box vật ở mọi tầng U-Net, token vật mã hoá gương (`models/unet1d.py`) | 〃 mục 16 |
| `gamma/gamma4_0_1.yaml` | như `gamma4` + `model.obj_k: 3`, `obj_select: nearest`: cross-attn chỉ nhìn 3 box vật có tâm gần tâm x_t nhất, chọn lại mỗi lần gọi U-Net | 〃 mục 16.8 |
| `gamma/gamma4_0_2.yaml` | như `gamma4` + `model.obj_k: 3`, `obj_select: random`: 3 box vật ngẫu nhiên (train rút lại mỗi iter, eval một bộ mỗi ảnh) | 〃 mục 16.8 |
| `gamma/gamma4_1.yaml` | như `gamma2` nhưng CE-Loc đề xuất = `gamma4` (đóng băng, nhận box vật) → refine 6 stage | 〃 mục 16.7 |
| `delta/delta1.yaml` | như `gamma2_celoc` + `model.refiner` (BoxRefiner của tác giả gốc trong mạng khử nhiễu: CLIP ViT-B/16 + text frozen, RoIAlign 3×3, 4 block RoPE 2D + AdaLN-Zero; U-Net nhận `[x_t ; refined]`; + 0,1·MSE(refined, x0)) | `docs/EXPERIMENT_DELTA.md` |
| `delta/delta1_1.yaml` | như `delta1`, `model.use_condition: false` (không điều kiện ảnh / text, refiner bỏ qua — mốc học được của prior) | 〃 |
| `delta/delta2_1.yaml` / `delta2_2.yaml` / `delta2_3.yaml` | như `delta1`, refiner dùng CLIP ViT-B/32 / L/14 / L/14-336 (`refiner.clip_model_name`), token chữ vẫn CLIP text B/16 (`refiner.text_model_name`) | 〃 mục 9.1 |
| `delta/delta3.yaml` | CHỈ BoxRefiner (`refiner.unet_input: none`, `aux_loss_weight: 1.0`): không ResNet18 / FiLM / U-Net, loss = MSE(refined, x0), sampler nhận ε̂ suy từ refined | 〃 mục 9.2 |
| `delta/delta4.yaml` | `gamma4` (box attention tới mọi vật) + refiner của `delta2_1` (CLIP ViT-B/32, text B/16) trước U-Net; luồng vật `[o ; o]` | 〃 mục 10.1 |
| `delta/delta3_1.yaml` / `delta3_2.yaml` | `delta2_1` / `delta4` với `refiner.unet_input: post`: U-Net trước, refiner sửa ε, loss = MSE(ε̂, ε) + 1·MSE(ε̂₁, ε) | 〃 mục 10.2 |
| `delta/delta4_1.yaml` | `delta4` với `refiner.unet_input: mid` + `mid_aux_weight: 0.1`: refiner chèn giữa U-Net, loss = ε + 0,1·MSE(refined, x0) + 0,1·MSE(b, x0) | 〃 mục 10.6 |
| `delta/delta5.yaml` / `delta5_1.yaml` / `delta5_2.yaml` / `delta5_3.yaml` / `delta5_4.yaml` / `delta5_5.yaml` | `delta4` + `model.box_coder: log` + val bằng sampler ddim của tác giả; `aux_giou_weight` 0 / 0,1 / 0,5 / 0,25 / 0,75 / 1,0 | 〃 mục 11 |

## Chạy

Mọi lệnh chạy trong `object-detection/ce_localization/` trên server, sau
`export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache TORCH_HOME=/mnt/disk1/aiotlab/haitn/torch_cache`.

| việc | thời lượng (A30) | dạng |
|---|---|---|
| test (`cd .. && python -m pytest tests/ce_localization -q`) | ~15 phút CPU server | **nền** |
| train 12k iter + 6 lần eval val | ~1h15–1h45 | **nền** |
| eval test (1 + 4 bước, chẩn đoán) | 5–10 phút mỗi lượt | **nền** |
| chỉ mục density (một lần) | ~13 phút | **nền** |

**Train** (`--save-dir` bắt buộc; `last.pth` mỗi 1000 iter, ghi nguyên tử):
```bash
LOG=/mnt/disk1/aiotlab/haitn/log/<nhóm>/<tên>_$(date +%m%d_%H%M).log
nohup python ../tools/run_on_free_gpu.py -- train.py --config config/alpha/alpha0.yaml \
    --save-dir ../weights/detection/<tên> > $LOG 2>&1 &
echo "PID $! -> $LOG"
```
Bị ngắt thì chạy lại **đúng lệnh đó + `--resume`**. Không có `--resume` mà `last.pth` đã có thì
train DỪNG ngay, không ghi đè. Kaggle: `torchrun --standalone --nproc_per_node=2 train.py ...`.

**Eval** (config lấy từ checkpoint; model 4 kênh thêm `--density full|partial|empty`):
```bash
LOG=/mnt/disk1/aiotlab/haitn/log/<nhóm>/<tên>_eval_$(date +%m%d_%H%M).log
nohup python ../tools/run_on_free_gpu.py -- eval.py --ckpt ../weights/detection/<tên>/best.pth \
    --split test --num-proposals 200 --top-k 100 --nms --oracle-score --steps 1 4 --attn-diag 20 \
    --out /mnt/disk1/aiotlab/haitn/output/<nhóm>/<tên>_test_N200.json > $LOG 2>&1 &
echo "PID $! -> $LOG"
```
Có NMS thì JSON báo cả hai thứ tự hậu xử lý: `steps{k}` (top-k trước) và `steps{k}_nmsfirst`.

**Density** (config có `data.density`): dựng chỉ mục một lần
```bash
python tools/build_density_index.py --samples ../data/samples --ce130 ../data/all_phase2_V2 \
    --out ../data/density_index.json --workers 8
```

## Bài add (GAMMA)

Chỉ mục (nhánh, lượt) ↔ `samples/` dựng một lần: `tools/build_turn_index.py` (lệnh trong docstring). Train /
eval như mọi config (`--save-dir ../weights/add/<tên>`, log `log/add/`, kết quả `output/add/`); `eval.py`
tự nhận `task: add`: `--image inpainted original`, `--n-samples`, `--add-density`, `--steps` (GAMMA1). Lệnh đầy đủ:
`docs/EXPERIMENT_GAMMA.md`.

CE-CoCount (tập test phụ của bài, `data/cocount/`): `eval.py --dataset cocount --cocount-root ../data/cocount` (ảnh gốc, 10 chỗ trống GT
làm lỗ, density trống; `data/cocount.py`); C-NLL của box GT trên CE-130 + CE-CoCount: `tools/gt_cnll.py`; `--obj-size` thêm khoá `_objsize` (giữ tâm, cỡ = TB box vật cùng lớp). Mục 18 của EXPERIMENT_GAMMA. C-NLL `F1all` (chiều quan hệ trên box vật cả hai lớp, Gaussian trên lớp cần thêm) + chấm lại dump cũ không cần model: `tools/rescore_cocount.py` — mục 10.9 của EXPERIMENT_DELTA.

DELTA (`config/delta/`, `docs/EXPERIMENT_DELTA.md`): train trên Kaggle (`add_kaggle.ipynb`, chỉ train), eval trên server (khuôn lệnh mục 7):
vòng mock 100 bước + `_norefine` (bỏ refiner, khi có U-Net) + `_noobj` (khi có box vật); cần
`HF_HOME` (CLIP tải lúc dựng model — không nằm trong checkpoint, vân tay `clip_sha` kiểm khớp). Cửa G1-HF (CLIP thật):
`CE_HF_TESTS=1 [CE_HF_CLIP=<clip_model_name>] python -m pytest tests/ce_localization/test_delta.py -k hf -s`.

User study (người chấm box đề xuất, `user_study/`): `eval.py --dump-boxes` (server) → `user_study/build.py` (chọn 4 box / model,
xáo màn) → `user_study/app.py` (web Gradio chấm ở local, `pip install gradio`) → `user_study/score.py`. Lệnh + chỉ số:
`docs/EXPERIMENT_GAMMA.md` mục 17.

Rà box dữ liệu bằng tay (`box_review/`, local, chạy bằng `.venv-cpu/bin/python`; 2026-10-20): `python box_review/app.py cocount` (loại box vật lỗi
CE-CoCount, mỗi màn một file = một lớp, box gốc chưa lọc, chỉ hiện ảnh + box) | `python box_review/app.py ce130 --split train|val|test` (soi box vật +
box lỗ CE-130, mỗi màn một nhánh; `--objects-only` chỉ box vật). Kết quả `output/add/box_review/<nguồn>.jsonl` (một dòng mỗi màn: chỉ
số box bị loại theo thứ tự file chú thích gốc). CE-CoCount lọc trùng tự động trước khi hiện (2026-10-21: cụm IoU > 0,9 giữ box
nhỏ hơn / trung vị diện tích, khoá `auto`; box bị loại cuối cùng = `bad` ∪ `auto`). Phím + định dạng: docstring `box_review/app.py`.

## Đọc số

| chỉ số | đo gì |
|---|---|
| `oracle_recall` | GT được ít nhất một box phủ (IoU ≥ 0,5) — chất lượng BOX, không dùng score. Chọn checkpoint bằng nó |
| `score_AUC` | box khớp GT có score cao hơn box còn lại không — chất lượng XẾP HẠNG |
| `--oracle-score` | trần AP khi score = IoU thật, cùng box. Trần ≫ thật: sửa xếp hạng; trần cũng thấp: sửa box |
| `oracle_recall_per_stage` | recall theo stage; phẳng = các stage sau không tinh chỉnh thêm |
| `density_recall`, `size_recall` | recall tách theo số vật / ảnh và theo cỡ vật (quy về canvas 512) |
| `iou_matched` (log train) | chỉ để đọc — **không** chọn checkpoint (mù với GT không box nào chạm tới) |
| bài add: `mean_iou_any` | IoU TB của MỘT box bất kỳ với lỗ gần nhất (không oracle) — chọn checkpoint; `best_iou@K` / `hit50@K` = oracle best-of-K (giao thức của bài) |

⚠️ Chỉ số trong log train là **khử nhiễu 1 bước từ GT đã thêm nhiễu**, cao hơn suy luận thật. Số
báo cáo chỉ lấy từ `eval.py` (DDIM từ nhiễu thuần).

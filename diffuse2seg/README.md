# diffuse2seg — Diffuse2Seg training-free

Port phần **training-free** của Diffuse2Seg (arXiv 2609.06491): self-attention SD frozen →
affinity → p-Laplacian (Algorithm 1) → gộp nhiều mức (Algorithm 2) → mask → box. **Không** port
bước train Mask2Former. **Khảo sát cơ chế, đứng ngoài luồng CE-Loc.**

Paper, pipeline, sai lệch khi reproduce, kết quả, cạm bẫy: **`docs/DIFFUSE2SEG.md`** (gốc
`multi_condition/`). Kết quả chính: AR₁₀₀₀ **11,80** / 150 ảnh PACO val (paper 13,6).

## Cấu trúc

| | |
|---|---|
| `d2s/` | `attention` (hook SD, SDPA viết tay), `affinity`, `plaplacian` (Alg.1), `merging` (Alg.2), `masks`, `boxes`, `prompts`, `markov` (chỉ minh hoạ), `pipeline` |
| `config/` | `paper.py` (đúng paper), `stage1.py` (đường CE-130), `paper_sd15_native.py` |
| `data/` | loader `paco_val`, `coco_val`, `ce130_coco` |
| `utils/` | `ar_metrics` (AR₁₀₀₀ đúng định nghĩa paper), `metrics`, `mask_ops`, `box_ops_np` |
| `tools/` | chạy (`run_paper`, `run_paco`, `run_coco`, `run_stage1`), quét (`sweep_t_w1`), cửa chặn (`check_attention_separates`, `check_plaplacian_vs_p2`), vẽ (`visualize_best`, `visualize_masks`, `visualize_markov`) |

Test ở `object-detection/tests/diffuse2seg/` (`python -m pytest tests/diffuse2seg -q` từ
`object-detection/`). Bộ test chạy trên affinity giả lập — xanh chỉ chứng minh phần toán và
ghép nối đúng, không chứng minh phương pháp chạy được trên dữ liệu thật.

## Ba đường chạy — KHÔNG so với nhau

| | dữ liệu | cấu hình | metric | tool |
|---|---|---|---|---|
| **A. reproduce** | PACO-LVIS val (2410 ảnh) | `config/paper.py` | **AR₁₀₀₀** | `run_paper.py` |
| A'. phụ | COCO val2017 | như trên | oracle_recall | `run_coco.py` |
| B. CE-130 | CE-130 | `config/stage1.py` | oracle_recall | `run_stage1.py` |

Chỉ AR₁₀₀₀ của đường A so được với 13,6. AR₁₀₀₀ ≠ `oracle_recall` (AR ghép một-một trên mask,
trung bình 10 ngưỡng IoU; `oracle_recall` luôn cao hơn ~2,4×).

## Checkpoint: SD 1.5 (SD2 đã bị khoá trên HuggingFace)

```bash
# ở LOCAL, rồi zip + tự upload lên server cùng đường dẫn (không scp/rsync)
cd object-detection/weights
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --local-dir diffuse2seg/stable-diffusion-v1-5 \
    --exclude "*.ckpt" "*.bin" "*.safetensors.index.json"
# phải thấy: model_index.json unet/ vae/ text_encoder/ tokenizer/ scheduler/
```

`config.local_model_dir` trỏ vào `../weights/diffuse2seg/stable-diffusion-v1-5` (tính theo vị trí
package, không theo cwd); có `model_index.json` thì dùng, không thì rơi về tên repo HF.
⚠️ Server đổi tên thư mục cũ nếu còn: `mv weights/diffu2seg weights/diffuse2seg`.

## Chạy

Trong `object-detection/diffuse2seg/` trên server, sau
`export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache`. Log in mỗi ảnh một dòng kèm ETA, cuối log
có bảng phân rã thời gian và ngoại suy cả tập.

| việc | thời lượng |
|---|---|
| reproduce PACO 150 ảnh | ~42 phút (16,7 s/ảnh — 95 % là lan truyền + Alg.2) |
| PACO đủ 2410 ảnh | **~11 giờ** — có `--checkpoint-every` / `--resume` |
| quét `t × w1` 16 cấu hình, 20 ảnh | ~1h40m |

**A. Reproduce** (việc đáng làm tiếp: chạy đủ 2410 ảnh với `t=150, w1=0,85` của paper):
```bash
LOG=/mnt/disk1/aiotlab/haitn/log/diffuse2seg/d2s_paper_$(date +%m%d_%H%M).log
nohup python ../tools/run_on_free_gpu.py -- tools/run_paper.py --dataset paco \
    --out /mnt/disk1/aiotlab/haitn/output/diffuse2seg/d2s_paper_paco.json > $LOG 2>&1 &
echo "PID $! -> $LOG"
```
Mặc định `--limit 50`; cả tập: `--limit 2410 --checkpoint-every 50` (ngắt thì thêm
`--resume`). Mốc in sẵn trong log: Diffuse2Seg 13,6 | CutLER 10,7 | DiffSeg 9,8 | M2N2 9,6 |
UnSAM 9,3.

**Quét `t × w1`** (20 ảnh để CHỌN, không để báo số — xác nhận bằng `--start` khác):
```bash
LOG=/mnt/disk1/aiotlab/haitn/log/diffuse2seg/d2s_sweep_$(date +%m%d_%H%M).log
nohup python ../tools/run_on_free_gpu.py -- tools/sweep_t_w1.py --limit 20 \
    --timesteps 50 100 150 300 --w1 1.0 0.85 0.5 0.15 \
    --out /mnt/disk1/aiotlab/haitn/output/diffuse2seg/d2s_sweep.json > $LOG 2>&1 &
echo "PID $! -> $LOG"
```

**Nhìn ảnh trước khi tin số** (`--pick spread` mặc định; `best` là mẫu thiên vị):
```bash
python ../tools/run_on_free_gpu.py -- tools/visualize_best.py \
    --from-json /mnt/disk1/aiotlab/haitn/output/diffuse2seg/d2s_paper_paco.json \
    --limit 30 --pick spread --out-dir /mnt/disk1/aiotlab/haitn/output/diffuse2seg/d2s_viz
```
Mỗi mức granularity là một phân hoạch riêng — vẽ **từng mức một**, không chồng lên nhau.

**Đường CE-130** (B): cửa chặn 0 `check_attention_separates.py` (~5 phút, có mốc lưới đều mù
ảnh), cửa chặn 1 `check_plaplacian_vs_p2.py` (`p<2` có thật hơn `p=2`? ~40–70 phút), rồi
`run_stage1.py`. CE-130 **không có mask GT** nên chỉ xem mask/box, không tính được AR.

## Ghi công

Diffuse2Seg — Hümmer, Sicking, Hüger, Gottschalk (CARIAD SE / TU Berlin), arXiv 2609.06491; không
có repo chính thức → `d2s/plaplacian.py` viết lại từ Algorithm 1. M2N2 — Karmann & Urfalioglu,
CVPR 2025: `d2s/attention.py` chép từ `refs/repos/m2n2/` rồi sửa (né `cv2`, thêm `prompt_text`,
nhận list timestep, `exit()` → `raise`); không dùng `matrix_ipf`, flood-fill numba (thay bằng
`scipy.ndimage.label`), JBU, chấm điểm theo nhãn người dùng.

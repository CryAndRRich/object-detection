# object-detection

Code của dự án `multi_condition` (bối cảnh: [`../CLAUDE.md`](../CLAUDE.md)).

| | nội dung |
|---|---|
| [`ce_localization/`](ce_localization/README.md) | **CE-Loc — trọng tâm.** Chọn box cho count editing trên CE-130 bằng khuếch tán trên toạ độ box |
| [`baseline/`](baseline/README.md) | Baseline detector trên CE-130: BASELINE0 DiffusionDet (gồm code DiffusionDet + D.1 cũ), 1 Sparse R-CNN, 2 Faster R-CNN, 3.1/3.2 Grounding DINO — kế hoạch `../docs/BASELINES.md` |
| [`diffuse2seg/`](diffuse2seg/README.md) | Diffuse2Seg training-free (arXiv 2609.06491), đứng ngoài luồng CE-Loc |
| `tests/` | **toàn bộ test**, chia thư mục con theo project |
| `tools/run_on_free_gpu.py` | chạy một script trên GPU có nhiều bộ nhớ trống nhất — dùng chung |
| `data/`, `weights/` | dữ liệu + weight, **không vào git** (zip thủ công lên server) — [`data/README.md`](data/README.md) |

Mỗi project là một **package**, import theo tên đầy đủ (`ce_localization.models...`,
`diffuse2seg.d2s...`) vì các project trùng tên package con (`data`, `utils`, `config`). Script
tự thêm `object-detection/` vào `sys.path`, nên vẫn chạy `python train.py` trong thư mục project.

```bash
python -m pytest tests -q                      # từ object-detection/, ~15–20 phút CPU server
cd ce_localization && python ../tools/run_on_free_gpu.py -- train.py --config config/alpha/alpha0.yaml --save-dir ../weights/detection/x
```

## `weights/` và `data/`

**Mọi weight nằm trong `weights/`** — cả weight tải về lẫn checkpoint tự train (`--save-dir` /
`OUTPUT_DIR` trỏ vào đây). **Không còn thư mục `checkpoints/`** ở bất cứ project nào.

```
weights/detection/<run>/                     checkpoint tự train cho bài DETECT CE-130: alpha*, beta*, baseline*
                                             (last.pth, best.pth, history.json; baseline thêm config.yaml, log.txt)
weights/add/<run>/                           checkpoint tự train cho bài ADD (GAMMA); weight CE-Loc gốc của bài + TN2 đã xoá 2026-10-01
weights/diffusion_policy/                    checkpoint Push-T công bố của Diffusion Policy (TN3)
weights/diffusiondet/                        7 checkpoint .pth, 5,8 GB (3 benchmark DiffusionDet cũ)
weights/gdino/groundingdino_swint_ogc.pth    Grounding DINO Swin-T chính thức (BASELINE3)
weights/diffuse2seg/stable-diffusion-v1-5/   SD cho diffuse2seg (hoặc tải qua $HF_HOME)
data/all_phase2_V2/, samples/                CE-130 (CE-Loc)
data/cache_clip_1024/                        cache patch token CLIP @1024 (sinh bằng tools/build_cache.py)
data/ce130_coco/                             CE-130 dạng COCO (baseline, diffuse2seg)
data/coco*, voc, crowdhuman, paco            dataset detector / PACO
```

## Môi trường

Một `requirements.txt` dùng chung. Server: env `ce-locmodel`
(`/mnt/disk1/aiotlab/envs/ce-locmodel/bin/python`), Python 3.10.20, torch 2.3.0+cu121,
driver 535, **3× A30 24 GB dùng chung**. `torch` chỉ để sàn — dựng env mới thì cài `torch`
khớp driver trước. `diffuse2seg` cần thêm `diffusers` + `accelerate`; PACO cần `pycocotools`.

- ⚠️ **`huggingface_hub` phải `< 1.0`** (server: 0.36.2). Bản 1.x làm mọi `import diffusers`
  chết với traceback trỏ vào diffusers, nhưng nguyên nhân ở `transformers`. Sửa:
  `pip install -U "huggingface_hub>=0.34,<1.0"`. Bản 0.x dùng `huggingface-cli`, không có `hf`.
- ⚠️ **`baseline/` (detectron2) hiện không import được detectron2** trên env này (D.1 từng chạy với
  `/mnt/disk1/aiotlab/haitn/d2src/detectron2`, đường nối nay đã mất). Cần cho BASELINE0–2:
  ```bash
  pip install fvcore iopath pycocotools omegaconf hydra-core cloudpickle tabulate termcolor yacs opencv-python timm
  pip install --no-build-isolation -e /mnt/disk1/aiotlab/haitn/d2src/detectron2   # nhánh main, KHÔNG v0.6
  ```
  `detectron2._C` không build được (không có `nvcc`) — D.1 vẫn chạy xong 12k iteration với
  cảnh báo đó, đừng cố sửa.

## Ghi công

DiffusionDet: Chen, Sun, Song, Luo — arXiv 2211.09788, CC-BY-NC 4.0 ([LICENSE](LICENSE)).
Diffuse2Seg: Hümmer, Sicking, Hüger, Gottschalk — arXiv 2609.06491; không có repo chính thức,
`diffuse2seg/d2s/plaplacian.py` viết lại từ Algorithm 1, `d2s/attention.py` dựa trên M2N2
(Karmann & Urfalioglu, CVPR 2025). CE-Loc: bài NeurIPS 2026 của chính người dùng; repo gốc
vendored ở `../refs/repos/Count-Editing/`.

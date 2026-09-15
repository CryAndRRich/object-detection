# object-detection

Bốn sub-project, phục vụ dự án `multi_condition` (xem [`../CLAUDE.md`](../CLAUDE.md)) theo ba
hướng khác nhau — hai cái đầu là **object detector dùng diffusion process** (khảo sát cơ chế đem
về CE-Loc), cái thứ ba là **chính CE-Loc**, cái thứ tư là **công cụ khảo sát cơ chế
training-free** (chưa gắn vào pipeline):

| | Trạng thái | Chi tiết |
|---|---|---|
| [`diffusiondet/`](diffusiondet/README.md) | **Đã train/eval xong trên Kaggle, không chạy lại** — chỉ còn giữ để đối chiếu import khi cấu trúc thư mục đổi | Số liệu đầy đủ ở [`../docs/06-benchmark-detector.md`](../docs/06-benchmark-detector.md) |
| [`diffu_grounding_dino/`](diffu_grounding_dino/README.md) | **Chưa train** — code đã verify đúng kiến trúc + checkpoint key-compat + 80 test (kể cả DDP 2-process), sắp finetune trên GPU server hoặc Kaggle 2×T4 | GroundingDINO + diffusion process trên reference point của decoder |
| [`count_editing/`](count_editing/README.md) | **Chưa train thật trên server** — code CE-LocModel đã sửa/ablation xong cục bộ, mới chuyển vào đây từ `multi_condition/count_editing/` để đi chung git repo | CE-Loc/CE-Gen (Add One, Take One, NeurIPS 2026) — bài của chính người dùng dự án |
| [`diffu2seg/`](diffu2seg/README.md) | **Chưa chạy cửa chặn** — code + 68 test pass ở local, sẵn sàng chạy cửa chặn trên server | Diffuse2Seg training-free (arXiv 2609.06491) trên CE-130: self-attention SD2 frozen → affinity → p-Laplacian → mask → box. **Không train gì** |

Bốn project **không phụ thuộc lẫn nhau** (không import chéo) — có thể đọc/sửa/chạy độc lập.
Phần dùng chung: `LICENSE`, `.gitignore`, `weights/` (tách sub-thư mục riêng cho mỗi project),
`data/` (dùng chung — COCO-minitrain cho 2 detector, `samples/`+`all_phase2_V2/` cho CE-Loc) —
xem dưới. **Trước đây `count_editing/` sửa xong phải zip đưa lên server; giờ đi cùng git repo này
nên chỉ cần `git pull`** — riêng `data/`/`weights/` vẫn không push git, vẫn zip/copy thủ công như
cũ.

## `weights/`

```
weights/
├── diffusiondet/           7 checkpoint .pth (3 tự train + 4 pretrained gốc), 5,8GB
└── diffu_grounding_dino/   groundingdino_swint_ogc.pth + bert-base-uncased/, 1,1GB
```

Nằm trong `.gitignore` — không push git. Copy/zip thủ công lên máy chạy (Kaggle trước đây,
GPU server bây giờ). **`diffu2seg/` KHÔNG dùng `weights/`**: nó tải Stable Diffusion 2 thẳng từ
HuggingFace vào `$HF_HOME` (đặt `HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache` trên server — cache
mặc định `~/.cache` không ghi được ở đó). Xem README của từng sub-project để biết checkpoint nào dùng cho việc gì.
(CE-LocModel checkpoint riêng do người dùng train, xem `count_editing/CE-LocModel/README.md`.)

## `data/`

3 dataset detector (COCO-minitrain 25K, VOC 07+12, CrowdHuman) + 2 thư mục CE-Loc
(`samples/` — dataset train gốc, `all_phase2_V2/` — dump CE-130 đầy đủ cho metric C-NLL), ~32GB —
xem [`data/README.md`](data/README.md) (provenance/checksum, layout, script tải). Dùng chung: cả
`diffusiondet/` lẫn `diffu_grounding_dino/` đều đọc COCO-minitrain từ đây; VOC/CrowdHuman hiện
chỉ `diffusiondet/` dùng; `samples/`/`all_phase2_V2/` thì `count_editing/` và `diffu2seg/` dùng (`diffu2seg/` đọc qua
`data/ce130_coco/*.json` mà `diffusiondet/tools/convert_ce130.py` đã sinh sẵn). Cũng nằm trong
`.gitignore`, không push git — zip thủ công lên máy chạy giống `weights/`.

## Môi trường

Một `requirements.txt` ở gốc repo này, dùng chung cho **`count_editing/CE-LocModel/`**,
**`diffusiondet/`** và **`diffu2seg/`** (gộp 2026-09-15 từ 2 file rời rạc). Hai project còn
lại giữ requirements riêng và **không gộp**: `count_editing/CE-GenModel/` (env `unicombine`)
và `diffu_grounding_dino/` (ghim `torch +cu121` cho driver 535 — gộp vào sẽ phá môi trường
của nó).

```bash
conda activate ce-locmodel
pip install -r requirements.txt
```

### Env thật trên server (đo 2026-09-15)

`gpus-Super-Server`, env `ce-locmodel` → `/mnt/disk1/aiotlab/envs/ce-locmodel/bin/python`

| | |
|---|---|
| Python | 3.10.20 |
| torch / torchvision | 2.3.0+cu121 / 0.18.0+cu121 |
| CUDA runtime / driver | 12.1 / 535.309.01 |
| GPU | 3× NVIDIA A30 24 GB |

`torch`/`torchvision` chỉ để **sàn**, không ghim cứng: env đã có bản khớp CUDA sẵn nên
`pip install -r` không đụng vào chúng. Dựng env **mới** thì cài `torch` trước cho khớp
driver, rồi mới chạy file requirements.

**CE-LocModel** đã có đủ mọi thứ. **`diffu2seg/`** chỉ cần thêm đúng 2 gói: `diffusers` +
`accelerate` (nó **không** cần `opencv` — `d2s/attention.py` đã thay `cv2.resize` bằng PIL;
cũng **không** cần `numba` — thay flood-fill JIT của M2N2 bằng `scipy.ndimage.label`).

#### ⚠️ `huggingface_hub` phải `< 1.0` — đã vỡ env một lần

`pip install -U "huggingface_hub[cli]"` (để lấy lệnh `hf download`) kéo lên **1.31.0**, và
`transformers` bản đang cài đòi `huggingface-hub>=0.23.2,<1.0`. Hậu quả: **mọi** `import
diffusers` chết với `RuntimeError: Failed to import ... pipeline_stable_diffusion_img2img`.
Traceback trỏ vào `diffusers`, nhưng nguyên nhân nằm ở `transformers` — dễ đi sai hướng.

Sửa: `pip install -U "huggingface_hub>=0.34,<1.0"` (server đang dùng **0.36.2**). Không nâng
`transformers` — CE-LocModel đã train xong với bản hiện tại, đổi là mất tính so sánh.

⚠️ Bản `0.x` dùng lệnh `huggingface-cli download`, **không** có lệnh `hf`.

### ⚠️ `diffusiondet/` hiện KHÔNG chạy được trên env này

Đo 2026-09-15: `import detectron2` → `ModuleNotFoundError`; `import fvcore` → cũng vậy;
`PYTHONPATH` rỗng; `site-packages` không có `.pth`/`egg-link` nào trỏ tới detectron2.

Nhưng log **EXPERIMENT D.1** (2026-09-08, chạy xong 12.000 iteration) thì có:

```
detectron2   0.6 @ /mnt/disk1/aiotlab/haitn/d2src/detectron2
PyTorch      2.3.0+cu121 @ .../envs/ce-locmodel/lib/python3.10/site-packages/torch
fvcore 0.1.5.post20221221 · iopath 0.1.9 · cv2 5.0.0
```

Tức D.1 chạy bằng **chính env này**, nhưng có thêm một đường nối tới `d2src/` mà nay đã mất
(đã loại trừ cả editable-install lẫn `PYTHONPATH`). Thư mục `d2src/detectron2` **vẫn còn**
trên đĩa. `CE-LocModel` và `diffu2seg` không đụng detectron2 nên **không bị ảnh hưởng**;
chỉ khi nào chạy lại `diffusiondet` mới cần dựng lại:

```bash
pip install fvcore iopath pycocotools omegaconf hydra-core cloudpickle \
            tabulate termcolor yacs opencv-python timm
pip install --no-build-isolation -e /mnt/disk1/aiotlab/haitn/d2src/detectron2
```

Dùng nhánh `main`, **không** dùng tag `v0.6`: v0.6 (11/2021) còn `PIL.Image.LINEAR` đã bị xoá
ở Pillow 10 (env đang có Pillow 12).

⚠️ **`detectron2._C` không build được trên máy này** — không có `nvcc`, `CUDA_HOME` invalid.
Log D.1 ghi rõ `not built correctly: No module named 'detectron2._C'` mà **vẫn chạy xong
12.000 iteration**. Đừng mất công cài `nvcc` để "sửa" dòng cảnh báo đó.

### `.gitignore`

Cũng đã gộp về một file ở gốc repo (2026-09-15), từ 3 file rời rạc. Hai thay đổi hành vi:

1. **Bỏ `tests/`.** `CE-LocModel/.gitignore` cũ ignore nó ("chỉ chạy ở local") nên server
   không có bộ test. Nay test **đi lên server**: `diffu2seg/` cần chạy `tests/run_all.py`
   trên server **trước** cửa chặn, để bắt lệch môi trường khi còn rẻ (vài giây CPU) thay vì
   sau khi đã tốn GPU.
2. **Bỏ `*output*`/`*/output`** của `count_editing/.gitignore` — quá rộng, nuốt cả file code
   tên `output_utils.py`. Thay bằng `output/` + `outputs/` (có dấu `/`, chỉ khớp thư mục).

## Ghi công

DiffusionDet: Shoufa Chen, Peize Sun, Yibing Song, Ping Luo — [arXiv 2211.09788](https://arxiv.org/abs/2211.09788),
giấy phép CC-BY-NC 4.0 (xem [LICENSE](LICENSE)). GroundingDINO: Shilong Liu et al. — ECCV 2024.
DiffuGroundingDINO tự viết lại toàn bộ theo công thức, không import từ `repos/` — chi tiết ghi
công/tham khảo trong [`diffu_grounding_dino/README.md`](diffu_grounding_dino/README.md).
Diffuse2Seg: Hümmer, Sicking, Hüger, Gottschalk (CARIAD SE / TU Berlin) — arXiv 2609.06491.
Repo chính thức không tồn tại, nên `diffu2seg/d2s/plaplacian.py` tự viết lại từ Algorithm 1;
`d2s/attention.py` copy từ `refs/repos/m2n2/` (M2N2, Karmann & Urfalioglu, CVPR 2025) rồi sửa,
không import — chi tiết trong [`diffu2seg/README.md`](diffu2seg/README.md).
Count-Editing (CE-Loc/CE-Gen): bài NeurIPS 2026 của chính người dùng dự án — xem
[`count_editing/CE-LocModel/README.md`](count_editing/CE-LocModel/README.md) cho README gốc
(thư mục `count_editing/` hiện là bản copy nguyên vẹn của `refs/repos/Count-Editing/`, chưa có
sửa đổi nào của dự án — xem `../docs/02-du-lieu-ce130.md` cho bài học vòng trước).

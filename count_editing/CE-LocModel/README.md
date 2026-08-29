# CE-LocModel

A diffusion-based object localization model that predicts bounding boxes conditioned on an RGB image, a density map, and a text class label. It uses a DDPM-style reverse diffusion process with a ResNet18 vision encoder, a frozen CLIP text encoder, and a conditional 1D U-Net noise prediction network — the original CE-Loc setup. This fork also supports 4 vision backbones × 2 noise-net architectures (8-way ablation, `config/variants/`) and, as the newest addition, denoising N boxes jointly instead of one (`config/variants/main/c_multibox.yaml`) — see [§ Three Main Variants](#three-main-variants).

---

## Environment Setup

### Requirements
- Python 3.10
- CUDA 12.1 (for GPU acceleration)
- Conda (recommended)

### Create and activate environment

```bash
conda create -n ce-locmodel python=3.10 -y
conda activate ce-locmodel
```

### Install PyTorch with CUDA 12.1

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

### Install remaining dependencies

```bash
pip install -r requirements.txt
```

> Note: the `torch` and `torchvision` lines in `requirements.txt` serve as version documentation. Install them via the command above to get the correct CUDA variant.

---

## Download Model and Data

All required files are hosted in this shared folder:

**https://adelaideuniversity.box.com/s/cxrzvy0s33llavj7n5nfzzp4gspz8os1**

The folder contains two files:
- `best_model.pth` — pretrained model weights (~435 MB)
- `data.zip` — dataset (train and test splits)

### Step 1 — Download both files

Open the link in a browser and download `best_model.pth` and `data.zip` manually, or use the Box direct-download URLs:

```bash
curl -L "https://adelaideuniversity.box.com/shared/static/best_model.pth" -o best_model.pth
curl -L "https://adelaideuniversity.box.com/shared/static/data.zip"       -o data.zip
```

### Step 2 — Place the model checkpoint

```bash
mkdir -p checkpoints/best_ckpt
mv best_model.pth checkpoints/best_ckpt/
```

Expected layout:

```
checkpoints/best_ckpt/
├── best_model.pth          # Model weights (~435 MB)
├── model_config_final.yaml # Model architecture config (already in repo)
└── train_config_final.yaml # Training hyperparameters (already in repo)
```

### Step 3 — Extract and place the dataset

`data/` sits alongside `count_editing/` (this repo's parent directory), not inside it —
`config/default.yaml` and `test_mul_box.py --all_phase2_dir` both resolve paths relative to that
layout:

```
<root>/
├── data/
│   ├── samples/{train,test}/
│   └── all_phase2_V2/        # original CE-130 dump, used for the C-NLL metric
└── count_editing/
    └── CE-LocModel/          # this repo
```

```bash
mkdir -p ../../data
unzip data.zip -d /tmp/ce130_extracted
mv /tmp/ce130_extracted/train ../../data/samples/train
mv /tmp/ce130_extracted/test  ../../data/samples/test
```

Expected layout after extraction:

```
../../data/samples/
├── train/
│   ├── images/       # RGB images (*.jpg, *.png)
│   ├── density/      # Grayscale density maps (*.png, same stem as images)
│   └── annotation/   # JSON annotations (*.json, same stem as images)
└── test/
    ├── images/
    ├── density/
    └── annotation/
```

`all_phase2_V2/` (used by `test_mul_box.py` to compute C-NLL) should be extracted the same way,
into `../../data/all_phase2_V2/`.

### Annotation format

Each JSON file in `annotation/` corresponds to one image:

```json
{
  "class": "object class name",
  "target_bbox": [center_x, center_y, width, height]
}
```

Coordinates are in absolute pixels of the original image.

---

## Training

### Build the sample cache first (one time)

Profiling the training loop on the server showed it spends **92.6% of wall-clock waiting on the
DataLoader** (28.3 of 30.6 min/epoch), and PNG decode is ~83% of that per-sample cost. Since
resize+pad is deterministic, decoding the same 20k PNGs on all 200 epochs is wasted work — do it
once instead:

```bash
python tools/build_cache.py --roots ../../data/samples/train ../../data/samples/test --size 512
```

This writes `cache_512.u8` (a flat uint8 memmap, ~21 GB for train + ~6 GB for test) and
`cache_512.json` next to each split. Verified bit-exact against the PNG path (max abs diff
**0.0** on every field) and ~5.7× faster per sample. Then pass `--use_cache` to training and eval.
Without the flag both fall back to decoding PNGs, so the cache is entirely optional.

### Run

Edit `config/default.yaml` to set hyperparameters (batch size, learning rate, epochs, data paths),
pick a variant config, then run:

```bash
python tools/run_on_free_gpu.py -- train_w_args.py --variant resnet18_cnn --epochs 200 --batch_size 32 --lr 5e-5 --use_cache --num_workers 8
```

`tools/run_on_free_gpu.py` auto-picks the least-busy GPU via `nvidia-smi` before each run (the
server is shared with other jobs) and sets `CUDA_VISIBLE_DEVICES` accordingly instead of a
hard-coded device index.

Checkpoints are saved per-variant to `checkpoints/{variant}/` after each epoch. The best model
(lowest training loss) is saved to `checkpoints/{variant}/best_model.pth`.

There are two independent sets of variant configs under `config/variants/`:

- **8 architecture-ablation configs** at the top level (`resnet18_cnn.yaml`, `resnet18_transformer.yaml`,
  `resnet34_cnn.yaml`, ..., `vit_clip_transformer.yaml`) — `{ResNet18, ResNet34, ResNet50, CLIP ViT-B/32}
  × {CNN/FiLM U-Net, Transformer/cross-attention U-Net}`. A quick survey of encoder/architecture choice.
- **3 main variants** under `config/variants/main/` — `a_repro`, `b_transformer`, `c_multibox` — each
  changing exactly ONE thing relative to `a_repro` (the reproduction baseline). See
  [§ Three Main Variants](#three-main-variants) below. Pass `--variant main/a_repro` (the `main/`
  prefix is part of the value; `train_w_args.py`/`test_mul_box.py` join it under `config/variants/`
  and `checkpoints/` transparently, so `checkpoints/main/a_repro/` etc. is created automatically).

```bash
python tools/run_on_free_gpu.py -- train_w_args.py --variant main/a_repro     --epochs 200 --batch_size 32 --lr 5e-5 --use_cache --num_workers 8
python tools/run_on_free_gpu.py -- train_w_args.py --variant main/b_transformer --epochs 200 --batch_size 32 --lr 5e-5 --use_cache --num_workers 8
python tools/run_on_free_gpu.py -- train_w_args.py --variant main/c_multibox   --epochs 200 --batch_size 8  --lr 5e-5 --use_cache --num_workers 8
```

(`c_multibox` uses a smaller `--batch_size` example above — it does `num_proposals=100` boxes ×
batch through the same U-Net/decoder per step, so it needs more memory per sample than (a)/(b);
tune to whatever the GPU allows.)

### Key training config (`config/default.yaml`)

| Parameter | Default | Final trained value |
|---|---|---|
| `batch_size` | 32 | 32 |
| `learning_rate` | 1e-4 | 5e-5 |
| `num_epochs` | 100 | 300 |
| `num_timesteps` | 1000 | 1000 |
| `train_path` | `../../data/samples/train/` | `../../data/samples/train/` |

---

## Three Main Variants

Three variants asking three separate questions, each changing exactly ONE thing relative to
`main/a_repro` (which itself reproduces the original CE-Loc: ResNet18 + Conv1D/FiLM U-Net):

| | Config | Changes vs. (a) | Question |
|---|---|---|---|
| **(a)** reproduction | `main/a_repro.yaml` | — (baseline) | Does re-deriving CE-Loc from the paper + fixing the known sampling bug (see [`../../CLAUDE.md`](../../CLAUDE.md) "Đường về CE-Loc" step 1) reproduce the original numbers? |
| **(b)** transformer noise net | `main/b_transformer.yaml` | `noise_net.type: cnn` → `transformer` | Does swapping the FiLM-conditioned Conv1D U-Net for a Transformer decoder (box token cross-attends to a `[time, cond]` memory) change anything, holding the encoder and the "condition is one flat vector" limitation fixed? |
| **(c)** multi-box | `main/c_multibox.yaml` | `noise_net.num_proposals: 1` → `100` | Does denoising **N boxes jointly** (DiffusionDet's `q_sample`/Hungarian-matching/DDIM machinery, applied to CE-Loc's `[cx,cy,w,h]` space instead of pixel-space boxes) let CE-Loc's inherently one-to-many "where to add" task use its multimodality, instead of relying on 30 independent single-box samples? |

**(a) and (b) are exactly the `resnet18_cnn`/`resnet18_transformer` architecture-ablation configs**,
just with their own `checkpoints/main/{a_repro,b_transformer}/` result directory so the 3-way main
comparison and the 8-way ablation survey don't overwrite each other's checkpoints. No new code.

**(c) is new**: the box-diffusion target changes from "the single box this sample is about" to
"every same-category box in the source image" (`all_bboxes`, looked up from `all_phase2_V2` by
matching this sample's `target_bbox` against every branch's `inpainted_bboxes` — see
`utils/cnll.py`'s docstring and `data/dataset.py`'s `_build_multi_box_target`), padded/cropped to
`num_proposals=100` exactly like DiffusionDet's `prepare_diffusion_concat`
(`object-detection/diffusiondet/diffusiondet/detector.py:370`). Training does Hungarian matching
(`utils/matcher.py`) between the N predictions and the real (non-padding) boxes before computing
loss — plain 1-to-1 matching on box cost only (L1 + GIoU), not DiffusionDet's own SimOTA/dynamic-K
matcher, because CE-Loc has no classification head to gate on (every box in one image is the same
category already). Sampling (`test_mul_box.py`'s `sample_boxes_multibox`) is DDIM over one
`[1, N, 4]` latent — all N boxes denoised in the same chain, seeing each other via the
Transformer/CNN backbone's own mixing across the horizon axis (for `type: cnn`, via the U-Net's
convolutions along the box axis; for a would-be `type: transformer` + multi-box combination — not
one of the 3 main variants — via the decoder's self-attention among box tokens).

**Known limitation of (c)**: DiffusionDet's `box_renewal` (drops low-confidence proposals mid-chain,
replenishes with fresh noise) and `use_ensemble` (accumulates predictions across DDIM steps + NMS)
are both gated on a per-box classification score. CE-Loc has no score head, so neither is ported —
every one of the N output boxes comes from the single final denoising step, unfiltered. This is
listed as still-missing work in [`../../CLAUDE.md`](../../CLAUDE.md) ("Còn thiếu để DiffuGroundingDINO
dùng được cho CE-Loc" — a score head is needed there too, for the same underlying reason).

### A coordinate-space trap worth knowing about

CE-Loc normalizes **sizes with the same affine map as centers** (`data/dataset.py::_normalize_bbox`,
inherited verbatim from the original repo):

```
norm_cx = (cx / target_w) * 2 - 1        norm_w = (w / target_w) * 2 - 1
```

For a center that is the ordinary `[0,size] → [-1,1]` mapping. For a *size* it means a box covering
a fraction `f` of the image gets `norm_w = 2f - 1`, so **every box smaller than half the image has a
negative `norm_w`** — measured: 100% of real boxes in `samples/train`. Anything that treats `norm_w`
as a literal width builds an inverted box (`x2 < x1`) and gets nonsense out; a box's GIoU with
*itself* comes back as `5e5` instead of `1.0`.

`utils/matcher.py` therefore decodes sizes back with `(norm + 1) / 2` before forming corners, and
clamps away negative extents. Two consequences to keep in mind:

- The original repo's own `calculate_iou` in `test_mul_box.py` does **not** do this. It is left
  untouched so `mean_iou_k10`/`mean_iou_k30` stay comparable with the paper's published numbers —
  meaning those two metrics carry the original's quirk by design, for all of (a)/(b)/(c) equally.
- The new `multibox_precision`/`multibox_recall` have no such back-compatibility constraint and use
  the corrected geometry (`utils.matcher.box_iou_normalized`), so they are **not** on the same
  footing as IoU@K. Don't compare the two families of number directly.

---

## Testing

```bash
python tools/run_on_free_gpu.py -- test_mul_box.py --checkpoint checkpoints/main/a_repro/best_model.pth --variant main/a_repro --use_cache
```

Same GPU auto-selection as training (see above). This runs multiple-sampling inference (30
bounding box samples per image for (a)/(b); for (c), one DDIM chain over the model's own
`num_proposals`) and reports three metrics, comparable across all variants: Mean IoU@10, Mean
IoU@30 (best-of-K against the ground truth box), and C-NLL (spatial coherence against the other
same-category objects in the source image, no ground truth needed). For variant (c) only, two more
fields are added — `multibox_precision`/`multibox_recall` (greedy IoU matching, threshold via
`--pr_iou_threshold`, default 0.5) against the full same-category box set — meaningful only for
(c) (a set-vs-set comparison), not a fair (a)/(b) baseline, so they're reported separately from
the 3 shared metrics. Results are written to:

```
samples_cocount/processed_dataset/output_multiple_sampling_density1class_inf100_{variant}/
checkpoints/{variant}/eval_results.json
```

Each output file contains the predicted bounding boxes. The metrics are printed to stdout and
saved to `eval_results.json` when evaluation completes.

---

## Inference on a single image

Use `inference.py` for a quick single-image test:

```python
python inference.py
```

Edit the file to set your image path, density map path, and text prompt. Output images with drawn bounding boxes are saved to `examples/outputs/`.

---

## Model Architecture

```
ObjectPlacementPolicy
├── SpatialVisualEncoder       resnet18 | resnet34 | resnet50 | vit_b32_clip
│                              (4-channel input: RGB + density, pretrained)
│                              → SpatialSoftmax → Linear → 128D
├── CLIPTextEncoder            openai/clip-vit-base-patch32 (frozen)
│                              → Linear + Mish → 128D
└── noise_net                  type: "cnn" | "transformer"; horizon = num_proposals (1 for (a)/(b), 100 for (c))
    ├── ConditionalUnet1D      1D U-Net, input [B,N,4] (cx,cy,w,h) x N boxes
    │                          conditioned on 256D by FiLM-style addition;
    │                          the box axis is halved twice and rebuilt through
    │                          skip connections, so valid N is 1, or >=8 and
    │                          divisible by 4 (N=4 is NOT valid; enforced in
    │                          diffusion_module.py)
    └── TransformerNoisePredNet  N box tokens cross-attend to a [time, cond]
                               memory AND self-attend to each other (Diffusion
                               Policy's transformer variant, extended to
                               horizon=N; n_cond_layers=0 → MLP cond encoder,
                               as in every released Diffusion Policy config)
```

Only the text encoder is frozen; the vision encoder and noise net are fully
trained — matching the paper ("The text encoder is frozen, while other modules
are optimized", §3.2) and the official code.

Diffusion schedule: linear DDPM, β from 0.0001 to 0.02 over 1000 steps (read from
`diffusion.beta_start`/`beta_end`/`num_timesteps` in the variant yaml).
Bounding boxes are normalized to `[-1, 1]` during training and denormalized at inference.
For `num_proposals > 1` (variant (c)), boxes are additionally scaled by `diffusion.snr_scale`
before `q_sample` (DiffusionDet's SNR_SCALE trick) and a Hungarian match (`utils/matcher.py`)
is computed between the N predictions and the real (non-padding) targets before the loss.

### Verification

Confirmed with a local CPU venv (torch 2.2.2, no GPU) before handing this to the server:
- All 11 configs (8 architecture-ablation + 3 main variants) run forward+backward without
  error, with gradients reaching every `noise_net`.
- `utils/matcher.hungarian_match` recovers the correct permutation on a synthetic known-shuffle
  test, handles N≠M and the M=0 edge case, and — regression test — decodes CE-Loc's negative
  `norm_w` correctly (IoU of a 5%-of-image box nested in a 25% box = 0.0400, matching the
  analytic `(0.05/0.25)² = 0.04`; a degenerate `norm_w < -1` stays inside GIoU's valid range).
- `data/dataset.py`'s multi-box target matched **0/19998 train and 0/6122 test** samples
  unmatched against `all_phase2_V2` (full dataset, not a sample) — every sample's own
  `target_bbox` was recovered inside its multi-box target's real (non-padding) boxes.
- `test_mul_box.py`'s `sample_boxes_multibox` DDIM loop converges to the true `x_start` to
  machine precision (0.0 full-schedule, 6e-8 with 10-step striding) under an oracle denoiser
  that returns the exact noise used to construct its input — same style of check already used
  to verify the single-box `sample_boxes`.
- Numerical-sanity checks that a "does it crash" test would miss: with an *untrained* network the
  x0 reconstruction stays inside the box range (`|pred_box| ≤ 1.00` clamped, vs ~240–330 without
  the clamp, since `1/√ᾱ` reaches 157× at t=999); `x_T` is drawn with unit variance rather than
  `snr_scale`×; and the DDIM update's noise is re-derived from the clamped `x_start` so the two
  stay mutually consistent (residual 2e-07, vs 5.4 when the raw noise is reused).
- The `num_proposals` guard was validated by sweeping N=1..140 through `ConditionalUnet1D`
  directly: only N=1 and N≥8 divisible by 4 both run *and* return N boxes. N=4 raises, and
  N=2/7/103 silently return a different count — the guard rejects all of them.
- A full `test_mul_box.py` run (untrained checkpoint, small split) exercises the whole multi-box
  eval path end to end — checkpoint load, multi-box dataset, DDIM sampling, all metrics.

**C-NLL cost**: `compute_cnll` re-fitted the Gaussian and re-derived the calibration term
`min_z(-log q(z))` for *every* candidate box, though both depend only on `{B_j}`. Hoisting them
out (`prepare_cnll` + `compute_cnll_prepared`) is **63× faster with bit-identical results**
(max difference exactly 0.0 over 100 candidates × 40 exemplars). This mattered most for (c),
which scores `num_proposals=100` candidates instead of 30, but it speeds up the 8-variant
ablation eval too. `compute_cnll` is kept as a wrapper so no existing caller changes behavior.

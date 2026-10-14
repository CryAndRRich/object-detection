#!/usr/bin/env python3
"""Soi quỹ đạo khử nhiễu của CE-Loc bài add (GAMMA1 `box_refiner`, CE-Loc kiểu bài `box_policy` r18_paper, DELTA) — mặc định MỘT hình
4 hàng × 6 cột mỗi ca (nhánh, lượt t):

  hàng 1  ảnh inpaint lượt t (đã xoá t vật) + density của CHÍNH ảnh đó (file `samples/` khớp bằng hash pixel)
  hàng 2  ảnh inpaint như hàng 1 + density TRỐNG
  hàng 3  ảnh gốc chưa xoá (`ground_truth.jpg`) + density của ảnh gốc (bản `full` của chỉ mục ALPHA3 — như eval.py)
  hàng 4  ảnh gốc như hàng 3 + density TRỐNG
  (density trống = PNG toàn màu nền jet (0, 0, 127) cùng cỡ ảnh qua cùng bước đọc — như `CE130AddDataset(density="empty")`)

  cột 1     Density Map — kênh density model nhận (thang [0, 1] cố định), lỗ (xanh lá) để so hai bản density
  cột 2–5   quỹ đạo khử nhiễu, mỗi ô = trạng thái SAU một bước: box x_t THẬT (đỏ đậm, không kẹp — có thể ra ngoài canvas) +
            tâm (chấm đỏ); ô cuối = đầu ra (t = 0), vẽ y như các ô khác. Box các trạng thái trước mờ dần (~20 box trên cả
            quỹ đạo); đường đen = tâm qua MỌI trạng thái, từ nhiễu thuần ban đầu (t = 999 với DDIM) tới trạng thái hiện tại.
            DDIM (`--steps` 4, `--eta` 1 như eval / 0 tất định): x_t_next = √ᾱ·x̂0 + nhiễu theo lịch (√ᾱ_t trên tiêu đề).
            `--sampler mock` (vòng của CE-Loc gốc): nhiễu thuần nhưng báo model t = 99..0, `x -= ε̂/100` (ε̂ suy từ x̂0
            tầng 6), 100 bước, vẽ 4 trạng thái cách đều; `--t-start` = DDIM bắt đầu ở t nhỏ hơn 999. x̂0 (box tầng 6 model
            đoán ở bước đó) chỉ vẽ khi `--show-x0`. Lỗ (xanh lá, CHỈ ảnh inpaint): lỗ mới nhất = đích (đậm), lỗ cũ nét đứt.
            Một box mỗi lần sinh; cùng `--seed` ⇒ cùng nhiễu ban đầu ở cả 4 hàng, khác biệt chỉ do ảnh / density.
  cột 6     SpatialSoftmax Output trên C5 (token `vis`): mỗi chấm = một kênh trong 2048, vị trí = toạ độ kỳ vọng (đổi trục
            như `tools/plot_spatial_softmax.py`), màu = số ô hiệu dụng (log), cỡ ∝ ‖W ss_proj‖ của kênh.

CE-Loc kiểu bài (`model.arch: box_policy`, `vision: r18_paper` — GAMMA2-pha 1; hoặc checkpoint CE-Loc gốc của bài
`weights/add/paper/best_model.pth`, nạp bằng `BoxPolicy.load_celoc_paper`, SpatialSoftmax không mask): cùng bố cục; vòng `--sampler mock` (mặc định,
100 bước như bài: nhiễu thuần, t = 99..0, x -= ε̂/100) hoặc `ddpm` (1000 bước), vẽ 4 trạng thái SAU bước N/4, N/2, 3N/4, N (ô cuối = đầu
ra); x̂0 = box suy từ ε̂ theo ᾱ_t; box chuẩn hoá theo canvas; SpatialSoftmax C5 512 kênh của ResNet18, CÓ mask phần đệm nếu
checkpoint `ss_mask` (đúng phép tính model dùng); cỡ chấm ∝ ‖W projection‖. Không có `--mode stages`.
  python tools/plot_refiner_steps.py --ckpt ../weights/add/gamma2_celoc/best.pth --quota 5 5 10 \
      --files test/images/3342_3.png ... --out ../../output/add/viz/gamma2_celoc/4_step

`--mode stages`: thay 4 cột quỹ đạo bằng 6 cột = box ra của 6 tầng trong MỘT lượt khử nhiễu ở t = `--stage-t`: trạng
thái 0 = box nhiễu đã kẹp mà tầng 1 nhận (mặc định t = T − 1 = 999), rồi tầng 1..6 (box tầng k là đầu vào RoI của tầng k + 1); hình 4 × 8 cột.

Chọn ca:
  `--files`  file `samples/` (vd test/images/3342_3.png — ảnh đã soi checkpoint của bài) -> dò ngược (split CE-130, nhánh,
             lượt) bằng hash pixel; tên thư mục `samples/` KHÔNG phải split CE-130 (samples/train = train + val).
  `--cases`  `<split>/<nhánh>[:lượt]` (mặc định lượt 1).
  `--quota`  số ca mỗi split train / val / test (mặc định 0 0 4); phần thiếu sau `--files` / `--cases` tự chọn lượt 1 từ
             `--pool` nhánh ngẫu nhiên mỗi split: ưu tiên ca density ảnh gốc khác bản inpaint ở lỗ (`gap` = density TB trong
             lỗ của ảnh gốc − của ảnh inpaint ≥ `--min-gap`), rồi lấy cách đều theo cỡ lỗ √(wh)/canvas ⇒ đủ dải kích cỡ vật.
Text = tên lớp thật. Chữ trên hình tiếng Anh. Ra `--out`: `<split>_<nhánh>_t<lượt>.png` + `refiner_steps.json`.
DELTA (`box_policy` r18_paper + `model.refiner`, vd `weights/add/delta1/best.pth`; CLIP ViT tải từ HF ⇒ `export HF_HOME`): mỗi ô
quỹ đạo vẽ thêm HAI box của bước dẫn tới trạng thái đó — **box refiner đề xuất** `refined = x_t + Δ` (tím hồng) và **box U-Net nhắm
tới** x̂0 = (x_t − √(1−ᾱ_t)·ε̂) / √ᾱ_t (xanh dương; vòng mock kéo x về phía √ᾱ_t·x̂0); JSON ghi IoU giữa chúng / với lỗ ở từng ô và
`per_step` (cả 100 bước). Model train AMP + `--device cuda` ⇒ lấy mẫu dưới autocast fp16 như eval.py. DELTA1.1 (`use_condition:
false`) bỏ qua refiner như lúc train ⇒ vẽ như CE-Loc thường; DELTA3 (không U-Net, không ResNet18) không nhận.

`--compact` (box_policy): ảnh inpaint lượt t + density của chính nó, mỗi hàng 6 cột (ảnh test + lỗ, SpatialSoftmax, 4 ô sau bước 25 / 50 /
75 / 100); mỗi hàng theo vết MỘT loại box (x_t đỏ / x̂0 U-Net xanh / refiner tím hồng nếu có ⇒ 2 hoặc 3 hàng): vết = box các bước trước
viền mảnh mờ dần + đường tâm, hai loại kia chỉ box ở bước đó; tiêu đề ô chỉ ghi số bước. Số đo đầy đủ vẫn ở JSON.
  python tools/plot_refiner_steps.py --ckpt ../weights/add/delta1/best.pth --name "CE-Loc Update (DELTA1)" --compact \
      --cases test/3342_b1:3 ... --device cuda --out ../../output/add/viz/delta1/compact

Chỉ nạp model + CLIP + vài trăm ảnh để chọn, ~3–8 phút, CPU được:
  python tools/plot_refiner_steps.py --ckpt ../weights/add/gamma1/best.pth --quota 5 5 10 \\
      --files test/images/3342_3.png ... --out ../../output/add/viz/gamma1/4_step
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.dataset import _read_annotation  # noqa: E402
from ce_localization.data.density import DensityIndex  # noqa: E402
from ce_localization.data.turns import image_inputs, pixel_hash  # noqa: E402
from ce_localization.models.box_policy import paper_spatial_softmax, unit_to_boxes  # noqa: E402
from ce_localization.models.clip_refiner import refiner_config, text_model_name  # noqa: E402
from ce_localization.models.detector import load_checkpoint_model  # noqa: E402
from ce_localization.models.text import TextTable, encode_class_names, encode_class_tokens  # noqa: E402
from ce_localization.tools.viz_common import (SHARP, center, dot_sizes, draw_holes, effective_cells, iou_to, rect,  # noqa: E402
                                              sorted_box, ss_scatter, tidy)

SPLITS = ("train", "val", "test")
C_UNET, C_REFINER = "#1E90FF", "#FF00FF"           # box U-Net (x̂0 suy từ ε̂) xanh dương / box refiner tím hồng — tách rõ khỏi đỏ / xanh lá
ROWS = (("inpainted", "own"), ("inpainted", "blank"), ("original", "own"), ("original", "blank"))


def row_label(kind, dens, turn):
    if kind == "inpainted":
        img = f"Inpainted ({turn} object{'s' if turn > 1 else ''} removed)"
        return f"{img}\ndensity: {'own (inpainted image)' if dens == 'own' else 'blank'}"
    return f"Original (nothing removed)\ndensity: {'own (original image)' if dens == 'own' else 'blank'}"


PAPER_CFG = {"data": {"image_size": 512},
             "model": {"arch": "box_policy", "vision": "r18_paper", "clip_text": "openai/clip-vit-base-patch32"}}


def load_model(path, device):
    """-> (model, cfg, nhãn iter, state CLIP text | None). Checkpoint CE-Loc gốc của bài (`model_state_dict`) nạp bằng
    `BoxPolicy.load_celoc_paper` (canvas 512, đầu vào kiểu bài, SpatialSoftmax KHÔNG mask, text bằng CLIP của chính checkpoint)."""
    try:
        model, cfg, info = load_checkpoint_model(path, archs={"box_refiner", "box_policy"})
    except ValueError as e:
        sys.exit(f"{path}: {e}")
    if cfg is not None and cfg["model"].get("arch") == "box_policy" and cfg["model"].get("vision") != "r18_paper":
        sys.exit(f"{path}: box_policy chỉ nhận vision r18_paper (GAMMA2-pha 1, DELTA1 / 2.x) — vision {cfg['model'].get('vision')}")
    return model.to(device).eval(), cfg or PAPER_CFG, info["label"], info["clip"]


def match_sample(samples, image_path, iid):
    """ảnh inpaint của (nhánh, lượt) -> 'train/<stem>' | 'test/<stem>' của file samples/ khớp pixel, None nếu không thấy."""
    h = pixel_hash(image_path)
    for sdir in ("train", "test"):
        for p in sorted(glob.glob(os.path.join(samples, sdir, "images", f"{iid}_*.png"))):
            if pixel_hash(p) == h:
                return f"{sdir}/{os.path.splitext(os.path.basename(p))[0]}"
    return None


def find_turn(a, rel):
    """file samples/ (vd test/images/3342_3.png) -> (split CE-130, nhánh, lượt) có ảnh inpaint trùng pixel, None nếu không."""
    iid = os.path.basename(rel).split("_")[0]
    h = pixel_hash(os.path.join(a.samples, rel))
    for split in SPLITS:
        for bdir in sorted(glob.glob(os.path.join(a.ce130, split, f"{iid}_b*"))):
            for p in sorted(glob.glob(os.path.join(bdir, "inpainted_turn_*.png"))):
                if pixel_hash(p) == h:
                    return split, os.path.basename(bdir), int(os.path.splitext(p)[0].rsplit("_", 1)[1])
    return None


def load_case(a, split, br, turn, dindex, T):
    """(split, nhánh, lượt) -> dict ca (ảnh, đường dẫn density, lỗ, gap, cỡ lỗ) hoặc None nếu thiếu dữ liệu."""
    bdir = os.path.join(a.ce130, split, br)
    iid = br.split("_b")[0]
    ann = _read_annotation(bdir)
    ipath = os.path.join(bdir, f"inpainted_turn_{turn}.png")
    if ann is None or iid not in dindex or not os.path.exists(ipath) or len(ann["inpainted_bboxes"]) < turn:
        return None
    sample = match_sample(a.samples, ipath, iid)
    if sample is None:
        return None
    sdir, stem = sample.split("/")
    imgs = {"inpainted": Image.open(ipath).convert("RGB"),
            "original": Image.open(os.path.join(bdir, "ground_truth.jpg")).convert("RGB")}
    own = {"inpainted": os.path.join(a.samples, sdir, "density", stem + ".png"),
           "original": dindex.path(dindex.pick(iid, "full")[0])}
    den, scale = {}, None
    for kind in imgs:
        x, scale, _, _ = image_inputs(imgs[kind], own[kind], T)
        den[kind] = x[3].numpy()
    holes = np.asarray(ann["inpainted_bboxes"][:turn], dtype=np.float64).reshape(-1, 4) * scale
    x1, y1, x2, y2 = [int(round(v)) for v in holes[-1]]
    cut_o, cut_i = den["original"][y1:y2, x1:x2], den["inpainted"][y1:y2, x1:x2]       # lỗ rất nhỏ / sát mép: có thể rỗng
    gap = float(cut_o.mean() - cut_i.mean()) if cut_o.size and cut_i.size else 0.0
    w, h = holes[-1, 2] - holes[-1, 0], holes[-1, 3] - holes[-1, 1]
    return {"split": split, "branch": br, "turn": turn, "class": ann["class_based_caption"], "sample": sample,
            "imgs": imgs, "own": own, "holes": holes, "gap": gap, "size": float(np.sqrt(max(w * h, 0)) / T)}


def pick_cases(a, dindex, T, rng):
    cases = []
    for rel in a.files or []:
        f = find_turn(a, rel)
        c = None if f is None else load_case(a, *f, dindex, T)
        print(f"--files {rel} -> " + ("KHÔNG khớp nhánh / lượt nào" if c is None else
                                       f"{c['split']}/{c['branch']} lượt {c['turn']}"), flush=True)
        if c is not None:
            c["source"] = rel
            cases.append(c)
    for spec in a.cases or []:
        sb, _, t = spec.partition(":")
        split, br = sb.split("/")
        c = load_case(a, split, br, int(t or 1), dindex, T)
        if c is None:
            print(f"--cases {spec}: thiếu dữ liệu, bỏ", flush=True)
        else:
            cases.append(c)
    for split, quota in zip(SPLITS, a.quota):
        need = quota - sum(c["split"] == split for c in cases)
        if need <= 0:
            continue
        taken = {c["branch"].split("_b")[0] for c in cases}
        names = sorted(os.listdir(os.path.join(a.ce130, split))) if os.path.isdir(os.path.join(a.ce130, split)) else []
        pool = []
        for i in rng.permutation(len(names)):
            if len(pool) >= a.pool:
                break
            if names[i].split("_b")[0] in taken:
                continue
            c = load_case(a, split, names[i], 1, dindex, T)
            if c is not None:
                pool.append(c)
        good = [c for c in pool if c["gap"] >= a.min_gap]
        cand = sorted(good if len(good) >= need else pool, key=lambda c: c["size"])
        idx = sorted(set(np.linspace(0, len(cand) - 1, need).round().astype(int).tolist())) if cand else []
        cases += [cand[i] for i in idx]
        print(f"{split}: pool {len(pool)} (gap >= {a.min_gap}: {len(good)}), chọn {len(idx)} theo cỡ lỗ: "
              + ", ".join(f"{cand[i]['branch']} {cand[i]['size']:.3f}" for i in idx), flush=True)
    return sorted(cases, key=lambda c: (SPLITS.index(c["split"]), c["size"]))


@torch.no_grad()
def spatial_softmax(model, x, valid_hw=None):
    """-> (toạ độ [C,2] (dọc, ngang) trong [−1, 1], attention [C,H,W]) trên C5, đúng phép tính của token `vis` (GAMMA1) /
    của CE-Loc (box_policy r18_paper: ResNet18, mask phần đệm nếu `ss_mask`)."""
    if is_policy(model):
        xy, att = model.vision.keypoints(x, valid_hw)
        return xy[0].cpu(), att[0].cpu()
    c5 = model.backbone.forward_c5(x)
    N, C, H, W = c5.shape
    kp, att = paper_spatial_softmax(c5, return_att=True)
    return kp.reshape(N, C, 2)[0].cpu(), att.reshape(N, C, H, W)[0].cpu()


def is_policy(model):
    return getattr(model, "vision_kind", None) == "r18_paper"


def box_iou_any(a, b):
    """IoU hai box xyxy có thể w, h âm (box nhiễu không kẹp) — kẹp về (min, max) trước, toạ độ âm về 0 như IoU với lỗ."""
    return float(iou_to(sorted_box(a), sorted_box(b).clip(min=0))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint box_refiner (../weights/add/gamma1/best.pth) hoặc box_policy "
                                                  "r18_paper (../weights/add/gamma2_celoc/best.pth)")
    ap.add_argument("--ce130", default="../data/all_phase2_V2")
    ap.add_argument("--samples", default="../data/samples")
    ap.add_argument("--density-index", default="../data/density_index.json")
    ap.add_argument("--files", nargs="*", default=None, help="file samples/ phải có trong các ca, vd test/images/3342_3.png")
    ap.add_argument("--cases", nargs="*", default=None, help="<split>/<nhánh>[:lượt], vd test/1050_b1:2")
    ap.add_argument("--quota", type=int, nargs=3, default=[0, 0, 4], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--pool", type=int, default=40, help="số nhánh ngẫu nhiên xét mỗi split khi tự chọn")
    ap.add_argument("--min-gap", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eta", type=float, default=1.0, help="DDIM eta: 1 = như eval (thêm nhiễu MỚI mỗi bước), 0 = tất định")
    ap.add_argument("--sampler", default=None, choices=["ddim", "mock", "ddpm"],
                    help="mock = vòng của CE-Loc gốc: nhiễu thuần nhưng t = mock_steps−1..0, x -= ε̂/mock_steps; mặc định ddim "
                         "(box_refiner) / mock (box_policy); ddpm chỉ box_policy")
    ap.add_argument("--t-start", type=int, default=None, help="DDIM bắt đầu ở t này (mặc định T − 1 = 999)")
    ap.add_argument("--mock-steps", type=int, default=100, help="số bước vòng mock (bài: 100); vẽ 4 trạng thái cách đều")
    ap.add_argument("--show-x0", action="store_true", help="vẽ thêm x̂0 (box tầng 6 model đoán ở bước đó), xanh chấm chấm")
    ap.add_argument("--mode", default="steps", choices=["steps", "stages"],
                    help="steps = quỹ đạo khử nhiễu (4 trạng thái); stages = MỘT lượt ở t = --stage-t, cột = box ra của 6 tầng")
    ap.add_argument("--stage-t", type=int, default=None, help="--mode stages: mức nhiễu của lượt (mặc định T − 1, cao nhất)")
    ap.add_argument("--name", default=None, help="tên hiển thị trên tiêu đề hình (mặc định <thư mục>/<file> của --ckpt), "
                                                 "vd 'CE-Loc (masked SpatialSoftmax)' — docs/EXPERIMENT_GAMMA.md mục 13.0")
    ap.add_argument("--compact", action="store_true",
                    help="box_policy: ảnh inpaint + density của nó; mỗi hàng 6 cột (ảnh, SpatialSoftmax, 4 ô quỹ đạo) theo vết một loại "
                         "box (x_t / U-Net / refiner); tiêu đề ngắn")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    dev = torch.device(a.device)
    model, cfg, it, clip_sd = load_model(a.ckpt, dev)
    policy = is_policy(model)
    if a.sampler is None:
        a.sampler = "mock" if policy else "ddim"
    if policy and (a.sampler == "ddim" or a.mode == "stages"):
        sys.exit("box_policy (CE-Loc): chỉ --sampler mock | ddpm, --mode steps")
    if not policy and a.sampler == "ddpm":
        sys.exit("box_refiner: --sampler ddim | mock")
    if policy:
        a.mock_steps = 100                                                        # mock_sample của BoxPolicy: 100 bước như bài
    if a.compact and not policy:
        sys.exit("--compact: chỉ box_policy (CE-Loc kiểu bài / DELTA)")
    rows = ROWS[:1] if a.compact else ROWS                                        # compact: ảnh inpaint + density của chính nó
    # DELTA: refiner trong mạng khử nhiễu (cần token chữ CLIP; `use_condition: false` thì refiner bị bỏ qua như lúc train)
    refiner = policy and getattr(model, "refiner", None) is not None
    amp = dev.type == "cuda" and bool(cfg.get("training", {}).get("amp")) and clip_sd is None     # như eval.py
    T = cfg["data"]["image_size"]
    ab = model.alphas_cumprod.cpu()
    if a.stage_t is None:
        a.stage_t = model.num_timesteps - 1
    if not 0 <= a.stage_t < model.num_timesteps:
        sys.exit(f"--stage-t {a.stage_t} ngoài [0, {model.num_timesteps - 1}]")
    dindex = DensityIndex(a.density_index, a.samples)
    size, _ = dot_sizes(model.vision.projection if policy else model.memory.ss_proj)
    os.makedirs(a.out, exist_ok=True)
    cases = pick_cases(a, dindex, T, np.random.default_rng(a.seed))
    name = {"mock": "mock", "ddpm": "DDPM"}.get(a.sampler, "DDIM")
    summary = []
    for cs in cases:
        split, br, turn, cls, holes = cs["split"], cs["branch"], cs["turn"], cs["class"], cs["holes"]
        hole = holes[-1]                                                          # lỗ mới nhất = đích
        hc = center(hole)
        tokens, tsha = (encode_class_tokens([cls], text_model_name(refiner_config(cfg["model"]["refiner"])), device=str(dev))
                        if refiner else (None, None))
        table = TextTable(encode_class_names([cls], cfg["model"]["clip_text"], device=str(dev), state_dict=clip_sd), tokens, tsha)
        if refiner:
            model.check_text_fingerprint(tsha)                                    # CLIP text tải lại phải đúng bản lúc train
        text = table([cls], dev)
        ncol = 8 if a.mode == "stages" else 6
        # compact: mỗi hàng hình theo vết MỘT loại box (x_t đỏ / U-Net xanh / refiner tím hồng); bố cục đầy đủ: hàng = (ảnh, density)
        tracks = (["x_t", "x0"] + (["refined"] if refiner else [])) if a.compact else [None]
        nfr = len(tracks) if a.compact else len(rows)
        fig, axes = plt.subplots(nfr, ncol, figsize=(ncol * 3.3, nfr * (3.7 if a.compact else 3.9 if refiner else 3.5)),
                                 squeeze=False)
        c_ss, c_traj = (1, 2) if a.compact else (ncol - 1, 1)                     # cột SpatialSoftmax / ô quỹ đạo đầu tiên
        case = {k: cs[k] for k in ("split", "branch", "turn", "class", "sample", "size")}
        case.update({"source": cs.get("source"), "density_gap_in_hole": cs["gap"], "rows": []})
        for r, (kind, dens) in enumerate(rows):
            img = cs["imgs"][kind]
            dsrc = cs["own"][kind] if dens == "own" else Image.new("RGB", img.size, (0, 0, 127))
            x, _, nw, nh = image_inputs(img, dsrc, T)
            xb, vhw = x[None].to(dev), torch.tensor([[nh, nw]], device=dev)
            g = torch.Generator(device=dev.type).manual_seed(a.seed)
            canvas = torch.full((4,), float(T))                                  # box chia canvas (CE-Loc / GAMMA1)
            if policy:                                                           # CE-Loc: mock 100 / DDPM 1000 bước, ghi mọi bước
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    u, traj = model.sample(xb, text, vhw, 1, generator=g, sampler=a.sampler, record="all",
                                           text_tokens=table.tokens([cls], dev) if refiner else None)
                steps = [{"t": s["t"], "x": s["x_t"],
                          "x0": sorted_box(unit_to_boxes(s["x0_hat"][0, 0], canvas).numpy())} for s in traj]
                if refiner:                                                      # box refiner đề xuất ở từng bước + TB |Δ| (đơn vị box)
                    for st, s in zip(steps, traj):
                        st["refined"] = sorted_box(unit_to_boxes(s["refined"][0, 0], canvas).numpy())
                        st["delta"] = float((s["refined"][0, 0] - s["x_t"][0, 0].float()).abs().mean())
            elif a.mode == "stages":                                               # MỘT lượt ở t = --stage-t: 6 tầng
                _, steps = model.sample(xb, text, vhw, 1, generator=g, steps=1, return_stages=True, t_start=a.stage_t)
                st0 = steps[0]
                boxes = [st0["noisy"][0, 0].numpy()] + [b for b in st0["stages"][:, 0, 0].numpy()]
                tlabel = [f"input t={st0['t']}"] + [f"stage {k}" for k in range(1, len(boxes))]
                snaps = list(range(1, len(boxes)))
            else:
                n_run = a.mock_steps if a.sampler == "mock" else a.steps
                u, steps = model.sample(xb, text, vhw, 1, generator=g, steps=n_run, eta=a.eta, return_stages=True,
                                        sampler=a.sampler, t_start=a.t_start)
            if a.mode == "steps":                    # trạng thái: x_t trước mỗi bước (t của bước đó) + đầu ra (t = 0); box THẬT, không kẹp
                boxes = [sorted_box(unit_to_boxes(st["x"][0, 0], canvas).numpy()) for st in steps]
                boxes.append(sorted_box(unit_to_boxes(u[0, 0].cpu(), canvas).numpy()))
                tlabel = [f"t={st['t']}" for st in steps] + ["output"]
                n = len(boxes) - 1                   # CE-Loc: sau bước N/4, N/2, 3N/4, N ; GAMMA1: cách đều từ bước 1
                snaps = sorted(set(np.linspace(n / 4 if policy else 1, n, 4).round().astype(int).tolist()))
            cents = [center(b) for b in boxes]
            trail = max(1, len(boxes) // 20)                                     # box vết: ~20 box trên cả quỹ đạo
            bg = x[:3].numpy().transpose(1, 2, 0)
            den = x[3].numpy()
            inp = kind == "inpainted"
            row = {"image": kind, "density": dens, "states": []}
            iou_h = lambda b: float(iou_to(b, hole)[0])  # noqa: E731
            ttls = []
            for k, si in enumerate(snaps):                                       # số đo từng ô (JSON) + tiêu đề bố cục đầy đủ
                is_out = si == len(boxes) - 1
                if a.mode == "stages":
                    ttl = f"stage {si}/{len(boxes) - 1} output (t = {steps[0]['t']})"
                    rec = {"stage": si, "t": steps[0]["t"], "box": boxes[si].tolist(), "center": cents[si].tolist()}
                else:
                    ttl = (f"after {name} step {si}/{len(steps)}: "
                           + ("output (t = 0)" if is_out else
                              f"x_t, t = {steps[si]['t']} (sqrt(abar) {float(ab[steps[si]['t']].sqrt()):.2f})"))
                    rec = {"after_step": si, "t": 0 if is_out else steps[si]["t"], "box": boxes[si].tolist(),
                           "center": cents[si].tolist()}
                if inp:
                    rec["iou_hole"] = iou_h(boxes[si])
                    rec["center_dist"] = float(np.linalg.norm(cents[si] - hc))
                    ttl += f"\nIoU(box, hole) {rec['iou_hole']:.2f} | center dist {rec['center_dist']:.0f} px"
                if refiner and a.mode == "steps":                                # bước dẫn tới trạng thái này: refiner vs U-Net
                    sp = steps[si - 1]
                    rec.update(step_t=sp["t"], refined=sp["refined"].tolist(), unet_x0=sp["x0"].tolist(),
                               iou_refined_unet=box_iou_any(sp["refined"], sp["x0"]), delta=sp["delta"])
                    ttl += f"\nstep t={sp['t']}: IoU(refined, U-Net x0) {rec['iou_refined_unet']:.2f}"
                    if inp:
                        rec["iou_refined_hole"], rec["iou_unet_hole"] = iou_h(sp["refined"]), iou_h(sp["x0"])
                        ttl += f"\nIoU w/ hole: refined {rec['iou_refined_hole']:.2f} | U-Net x0 {rec['iou_unet_hole']:.2f}"
                ttls.append(ttl)
                row["states"].append(rec)
            row["path"] = [[lab, *b.tolist()] for lab, b in zip(tlabel, boxes)]
            if refiner:                                                          # JSON: refiner vs U-Net ở MỌI bước
                row["per_step"] = [{"t": sp["t"], "iou_refined_unet": box_iou_any(sp["refined"], sp["x0"]), "delta": sp["delta"],
                                    **({"iou_xt_hole": iou_h(boxes[i]), "iou_refined_hole": iou_h(sp["refined"]),
                                        "iou_unet_hole": iou_h(sp["x0"])} if inp else {})} for i, sp in enumerate(steps)]
            xy, att = spatial_softmax(model, xb, vhw)
            eff = effective_cells(att).numpy()
            sharp = float((eff < SHARP).mean())
            row["sharp_frac"], row["median_cells"] = sharp, float(np.median(eff))
            case["rows"].append(row)

            def draw_traj(ax, k, si, tr):
                """Ô quỹ đạo sau bước si. tr None (bố cục đầy đủ): vết x_t + box U-Net / refiner của bước đó. tr = "x_t" | "x0" |
                "refined" (compact): vết (box các bước trước viền mảnh mờ dần + đường tâm) CHỈ của loại đó, hai loại kia chỉ box ở bước
                này. Box U-Net / refiner của trạng thái si = của bước si − 1 (bước dẫn tới nó)."""
                ax.imshow(bg)
                if inp:
                    draw_holes(ax, holes)
                key = tr or "x_t"
                col = {"x_t": "red", "x0": C_UNET, "refined": C_REFINER}[key]
                seq = boxes[: si + 1] if key == "x_t" else [steps[j][key] for j in range(si)]
                prev = list(range(0, len(seq) - 1, trail))                       # box các bước trước: viền mảnh, mờ dần
                for n_, j in enumerate(prev):
                    rect(ax, seq[j], col, lw=0.8, alpha=0.08 + 0.4 * (n_ + 1) / max(len(prev), 1))
                pk = center(np.stack(seq))                                       # đường tâm qua MỌI bước
                ax.plot(pk[:, 0], pk[:, 1], "-", color="black" if key == "x_t" else col, lw=0.8, alpha=0.85, zorder=5)
                if key == "x_t":                                                 # mốc t chỉ trên đường tâm box đỏ
                    for j in [0] + snaps[:k]:
                        ax.plot(*cents[j], "o", color="0.7", mec="black", mew=0.6, ms=5, zorder=6)
                        ax.annotate(tlabel[j], cents[j], xytext=(4, 4), textcoords="offset points", fontsize=7,
                                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7), zorder=7)
                if (a.show_x0 or refiner or a.compact) and a.mode == "steps":   # box ở bước này: U-Net, refiner, x_t
                    rect(ax, steps[si - 1]["x0"] if policy else steps[si - 1]["stages"][-1, 0, 0].numpy(), C_UNET,
                         lw=2.0 if refiner or a.compact else 1.2, ls="-" if refiner or a.compact else ":")
                if refiner:
                    rect(ax, steps[si - 1]["refined"], C_REFINER, lw=2.0)
                rect(ax, boxes[si], "red", lw=2.4)
                ax.plot(*pk[-1], "o", color=col, mec="black", mew=0.6, ms=7, zorder=8)

            for q, tr in enumerate(tracks):
                fr = q if a.compact else r
                ax = axes[fr, 0]
                if a.compact:                                                    # cột 1: ảnh test + lỗ
                    ax.imshow(bg)
                    draw_holes(ax, holes)
                    ax.set_title(f"Test image ('{cls}')", fontsize=10)
                    ax.set_ylabel({"x_t": "trace: predicted box x_t", "x0": "trace: U-Net box",
                                   "refined": "trace: refiner box"}[tr], fontsize=10,
                                  color={"x_t": "red", "x0": C_UNET, "refined": C_REFINER}[tr])
                else:
                    im = ax.imshow(den, cmap="viridis", vmin=0, vmax=1)
                    rect(ax, hole, "lime", lw=1.2)
                    ax.set_title(f"Density Map ({dens})\nmin {den.min():.3f} max {den.max():.3f} | in hole "
                                 f"{den[int(hole[1]):int(hole[3]), int(hole[0]):int(hole[2])].mean():.3f}", fontsize=8.5)
                    ax.set_ylabel(row_label(kind, dens, turn), fontsize=9.5)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                for k, si in enumerate(snaps):
                    ax = axes[fr, c_traj + k]
                    draw_traj(ax, k, si, tr)
                    ax.set_title(f"{name} step {si}/{len(steps)}" if a.compact else ttls[k], fontsize=10 if a.compact else 8.5)
                ax = axes[fr, c_ss]
                ss_scatter(fig, ax, bg, xy, att, size)
                ax.set_title("SpatialSoftmax" if a.compact else f"SpatialSoftmax Output ({att.shape[0]} ch)\nsharp (<{SHARP} cells) "
                             f"{sharp * 100:.1f}% | median cells {np.median(eff):.0f}", fontsize=10 if a.compact else 8.5)
                if inp:
                    rect(ax, hole, "lime", lw=1.2)
                tidy(axes[fr], T, T)
        if a.compact:
            handles = [Line2D([], [], color="red", lw=2.4, label="predicted box x_t"),
                       Line2D([], [], color=C_UNET, lw=2.0, label="U-Net box (x0 from eps)")]
            if refiner:
                handles.append(Line2D([], [], color=C_REFINER, lw=2.0, label="refiner box (x_t + delta)"))
            handles += [Line2D([], [], color="0.4", lw=0.8, alpha=0.6, label="traced type: earlier boxes (thin, fading)"),
                        Line2D([], [], color="0.2", lw=0.8, marker="o", mfc="0.7", label="traced type: center path (t labels on x_t)")]
            handles.append(Line2D([], [], color="lime", lw=1.6, marker="+", label="removed object (target)"))
            fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False)
            fig.suptitle(f"{a.name or os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))} | {split} {br}, turn {turn} | "
                         f"'{cls}'", fontsize=11)
            fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        else:
            handles = [Line2D([], [], color="red", lw=2.2, marker="o",
                              label="box + center of this stage" if a.mode == "stages" else
                              "box + center at this state (last = output)"),
                       Line2D([], [], color="red", alpha=0.3, label="earlier stages / input box (fading)" if a.mode == "stages"
                              else "boxes at earlier states (fading)"),
                       Line2D([], [], color="black", marker="o", mfc="0.7", lw=1.4,
                              label="center path: input -> stage 1..k" if a.mode == "stages" else "center path, every step")]
            if a.show_x0 or refiner:
                handles.append(Line2D([], [], color=C_UNET, lw=2.0 if refiner else 1.2, ls="-" if refiner else ":",
                                      label="U-Net x0 (from eps) of the step leading here" if refiner
                                      else "predicted x0 (from eps) of that step" if policy else "predicted x0 (stage 6) of that step"))
            if refiner:
                handles.append(Line2D([], [], color=C_REFINER, lw=2.0, label="refiner box (x_t + delta) of the step leading here"))
            handles += [Line2D([], [], color="lime", lw=1.6, marker="+", label="latest removed object (target)"),
                        Line2D([], [], color="lime", lw=1.0, ls="--", label="earlier removed objects")]
            fig.legend(handles=handles, loc="lower center", ncol=4 if refiner else 6, fontsize=8, frameon=False)
            seen = "train image, seen class" if split == "train" else "unseen class"
            fig.suptitle(f"{a.name or os.path.basename(os.path.dirname(os.path.abspath(a.ckpt))) + '/' + os.path.basename(a.ckpt)}"
                         f" ({it if isinstance(it, str) else f'iter {it}'}) | "
                         f"{split} {br}, turn {turn} ({seen} '{cls}') | text = '{cls}' | one box per generation, seed {a.seed} | "
                         + (f"ONE denoising pass at t={a.stage_t}: noisy box (clipped to canvas, as stage 1 sees it) -> 6 stages"
                            if a.mode == "stages" else
                            f"mock sampler (paper): pure noise labelled t={a.mock_steps - 1}..0, x -= eps/{a.mock_steps}"
                            if a.sampler == "mock" else
                            "DDPM 1000 steps (correct ancestral sampler)" if a.sampler == "ddpm" else
                            f"DDIM {a.steps} steps, eta {a.eta:g}, from t={model.num_timesteps - 1 if a.t_start is None else a.t_start}"),
                         fontsize=10)
            fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        fig.savefig(os.path.join(a.out, f"{split}_{br}_t{turn}.png"), dpi=100)
        plt.close(fig)
        summary.append(case)
        print(f"{split:5s} {br:10s} t{turn} ({cls}, hole size {cs['size']:.3f}, gap {cs['gap']:.3f}) | " + " | ".join(
            f"{r['image'][:4]}/{r['density']}: " + (f"IoU out {r['states'][-1]['iou_hole']:.2f}" if "iou_hole" in r["states"][-1]
                                                     else "no hole") for r in case["rows"]), flush=True)
    with open(os.path.join(a.out, "refiner_steps.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

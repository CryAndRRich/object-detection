"""Dữ liệu giả + model / config thu nhỏ dùng chung. Không tải gì: CLIP ViT của refiner thay bằng CLIP tí hon dựng từ config
(`_fake_load_clip`), text thay bằng embedding / token giả (`_fake_text_table`), mọi lượt train / eval ép `--device cpu`."""

import hashlib
import json
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image

from ce_localization.data.density import JET, build_index
from ce_localization.models.text import TextTable

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ce_localization", "config", "delta")
RUNS = ("delta4", "delta5", "delta5_1", "delta5_2", "delta5_3", "delta5_4", "delta5_5", "delta6", "delta6_1", "delta6_2", "delta7")
COCOUNT_NAMES = ("INTRA_FOO_TOM1_TOM2_00008_00006_0_200", "INTER_OFF_PEN0_PNC0_00007_00005_0_40")
FAKE_TEXT_SHA = hashlib.sha256(b"fake CLIP text").digest()
# ViT thật -> (ảnh, patch, hidden giả): cùng lưới g = ảnh / patch như bản thật
FAKE_VIT = {"openai/clip-vit-base-patch16": (224, 16, 32), "openai/clip-vit-base-patch32": (224, 32, 32)}
SMALL_REFINER = {"enabled": True, "clip_model_name": "openai/clip-vit-base-patch32", "text_model_name": "openai/clip-vit-base-patch16",
                 "d_model": 32, "num_heads": 2, "num_layers": 2, "aux_loss_weight": 0.1}


# ----------------------------------------------------------------------------- CLIP / text giả

def _fake_load_clip(name, text_name=None, *, seed=0):
    """Thay `models.clip_refiner.load_clip`: CLIPVisionModel tí hon (2 tầng), lưới như ViT `name`; weight tất định theo `seed`, KHÔNG
    đụng RNG toàn cục. -> (model frozen eval, hidden CLIP text 512)."""
    from transformers import CLIPVisionConfig, CLIPVisionModel
    size, patch, hid = FAKE_VIT[name]
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        m = CLIPVisionModel(CLIPVisionConfig(hidden_size=hid, intermediate_size=2 * hid, num_hidden_layers=2,
                                             num_attention_heads=2, image_size=size, patch_size=patch))
    return m.eval().requires_grad_(False), 512


def _patch_clip(monkeypatch, seed=0):
    import ce_localization.models.clip_refiner as cr
    monkeypatch.setattr(cr, "load_clip", lambda name, text_name: _fake_load_clip(name, text_name, seed=seed))


def _fake_text_table(names, cfg=None, dev=None):
    """Embedding pooler + token chữ giả, tất định theo tên (token dài 2 + len(tên) % 4)."""
    table, tok = {}, {}
    for n in names:
        table[n] = torch.randn(512, generator=torch.Generator().manual_seed(sum(map(ord, n))))
        tok[n] = torch.randn(2 + len(n) % 4, 512, generator=torch.Generator().manual_seed(7 + sum(map(ord, n))))
    return TextTable(table, tok, FAKE_TEXT_SHA)


def small_policy(seed=0, refiner=None, num_timesteps=100, **kw):
    """BoxPolicy thu nhỏ (U-Net (16, 32, 64), refiner d 32) cho test đơn vị."""
    from ce_localization.models.box_policy import BoxPolicy
    torch.manual_seed(seed)
    return BoxPolicy(pretrained_backbone=False, vis_dim=16, text_dim=16, step_embed_dim=32, down_dims=(16, 32, 64), n_groups=4,
                     num_timesteps=num_timesteps, refiner={**SMALL_REFINER, **(refiner or {})}, **kw)


def small_batch(B=2, seed=0, size=64):
    """-> (ảnh [B,4,T,T], text [B,512], valid_hw, x0 [B,4], token chữ, box vật list B)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(B, 4, size, size, generator=g)
    vhw = torch.tensor([[48, 64], [64, 56]])[:B]
    tok = (torch.randn(B, 5, 512, generator=g), torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)[:B])
    objs = [torch.tensor([[2.0, 3.0, 12.0, 20.0], [30.0, 5.0, 40.0, 18.0]]), torch.zeros(0, 4)][:B]
    return x, torch.randn(B, 512, generator=g), vhw, torch.rand(B, 4, generator=g) * 2 - 1, tok, objs


# ----------------------------------------------------------------------------- CE-130 giả

def _fake_ce130_turns(base, seed=0):
    """CE-130 giả theo (nhánh, lượt) + `samples/` của bài: mỗi ảnh gốc 1–2 nhánh, nhánh xoá cộng dồn T <= 4 vật (lỗ tô xám,
    `inpainted_bboxes` lệch vật `all_bboxes` 1 px như dữ liệu thật); file samples `{iid}_{j}` với j XÁO TRỘN; train + val -> samples/train,
    test -> samples/test như bài, trừ ảnh gốc val 2001 -> samples/test. -> (ce130_root, samples_root). Vật 0..1 không bao giờ bị xoá."""
    rng = np.random.default_rng(seed)
    root, samples = os.path.join(base, "all_phase2_V2"), os.path.join(base, "samples")
    plan = (("train", ["1000", "1001", "1002"], "apple"), ("val", ["2000", "2001"], "bird"), ("test", ["3000", "3001"], "cup"))
    for split, iids, cat in plan:
        for iid in iids:
            sdir = os.path.join(samples, "test" if split == "test" or iid == "2001" else "train")
            for d in ("images", "density", "annotation"):
                os.makedirs(os.path.join(sdir, d), exist_ok=True)
            W, H = 200, 150
            gt = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
            objs = []
            for k in range(7):
                x1, y1 = 10 + 26 * k, rng.uniform(10, 100)
                objs.append([float(x1), float(y1), float(x1 + 20), float(y1 + rng.uniform(15, 35))])
            outs = []
            for b, order in ((1, [2, 3, 4, 5]), (2, [6, 5]))[: 1 + int(iid[-1]) % 2]:
                br = os.path.join(root, split, f"{iid}_b{b}")
                os.makedirs(br, exist_ok=True)
                Image.fromarray(gt).save(os.path.join(br, "ground_truth.jpg"))
                holes = [[o + 1.0 for o in objs[i]] for i in order]
                img = np.asarray(Image.open(os.path.join(br, "ground_truth.jpg")).convert("RGB")).copy()
                for t, (x1, y1, x2, y2) in enumerate(holes, 1):
                    img[int(y1):int(y2), int(x1):int(x2)] = 128
                    Image.fromarray(img).save(os.path.join(br, f"inpainted_turn_{t}.png"))
                    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                    lv = np.zeros((H, W), dtype=np.uint8)
                    for i in range(len(objs)):
                        if i not in order[:t]:
                            ox, oy = int((objs[i][0] + objs[i][2]) / 2), int((objs[i][1] + objs[i][3]) / 2)
                            lv[oy - 2:oy + 3, ox - 2:ox + 3] = 255
                    outs.append((img.copy(), JET[lv], {"class": cat, "target_bbox": [cx, cy, w, h]}))
                with open(os.path.join(br, "annotation.json"), "w") as f:
                    json.dump({"all_bboxes": objs, "inpainted_bboxes": holes, "class_based_caption": cat}, f)
            for j, o in enumerate(rng.permutation(len(outs))):
                im, den, ann = outs[o]
                Image.fromarray(im).save(os.path.join(sdir, "images", f"{iid}_{j + 1}.png"))
                Image.fromarray(den).save(os.path.join(sdir, "density", f"{iid}_{j + 1}.png"))
                with open(os.path.join(sdir, "annotation", f"{iid}_{j + 1}.json"), "w") as f:
                    json.dump(ann, f)
    return root, samples


def _fake_turn_index(base):
    """CE-130 + samples giả, chỉ mục (nhánh, lượt) + chỉ mục density. -> (root, samples, đường dẫn chỉ mục, TurnIndex, report)."""
    from ce_localization.data.turns import TurnIndex, build_turn_index
    root, samples = _fake_ce130_turns(base)
    idx = build_turn_index(root, samples, workers=0, log=lambda *a: None)
    p = os.path.join(base, "turn_index.json")
    with open(p, "w") as f:
        json.dump(idx, f)
    with open(os.path.join(base, "density_index.json"), "w") as f:
        json.dump(build_index(samples, workers=0, log=lambda *a: None), f)
    return root, samples, p, TurnIndex(p), idx["report"]


def _fake_positives(base, path):
    """Bản gán box positive giả (khuôn `box_review/ce130_train.jsonl`): mẫu train của nhánh 1001_b1 = mọi lỗ tới lượt t + 1 box trùng
    lỗ mới nhất (lệch 2 px, phải bị lọc) + 1 chỗ trống dưới đáy ảnh (vật y2 <= 135, riêng từng lượt); mẫu khác không có trong file.
    -> {id: box}."""
    from ce_localization.data.turns import TurnIndex
    idx = TurnIndex(os.path.join(base, "turn_index.json"))
    out = {}
    for k, v in idx.turns.items():
        if v["branch"] == "train/1001_b1":
            sid = os.path.splitext(os.path.basename(v["sample"]))[0]
            holes = idx.branches[v["branch"]]["holes"][:v["t"]]
            out[sid] = holes + [[c + 2.0 for c in holes[-1]], [10.0 * v["t"], 137.0, 10.0 * v["t"] + 40, 149.0]]
    with open(path, "w") as f:
        for sid, pos in sorted(out.items()):
            f.write(json.dumps({"id": sid, "class": "apple", "positive": pos, "negative": [], "neutral": [], "meta": {}}) + "\n")
    return out


# ----------------------------------------------------------------------------- CE-CoCount giả

def _fake_cocount(root, seed=0):
    """Mỗi frame hai file `_positive` / `_negative`, vật hai lớp xen kẽ, 10 loc_bbox ở chỗ trống."""
    rng = np.random.default_rng(seed)
    for d in ("Image", "Anno", "Anno_with_exam_bbox"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    for fi, stem in enumerate(COCOUNT_NAMES):
        W, H = (200, 120) if fi == 0 else (90, 160)
        for pol in ("positive", "negative"):
            Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(os.path.join(root, "Image", f"{stem}_{pol}.jpg"))
        na, nb = int(stem.split("_")[4]), int(stem.split("_")[5])
        for pol, n, cls, off in (("positive", na, "normal tomato(Big round tomato).", 0), ("negative", nb, "baby tomato.", 9)):
            objs = [[float(5 + 11 * k), float(5 + off), float(13 + 11 * k), float(13 + off)] for k in range(n)]
            loc = [[float(4 + 8 * k), float(40 + (k % 2) * 10), float(12 + 8 * k), float(48 + (k % 2) * 10)] for k in range(10)]
            anno = {"class_name": cls, "loc_bbox": loc, "counting_anno": {"count": n, "exemplars": [list(map(int, o)) for o in objs[:3]]}}
            with open(os.path.join(root, "Anno", f"{stem}_{pol}.json"), "w") as f:
                json.dump(anno, f)
            with open(os.path.join(root, "Anno_with_exam_bbox", f"{stem}_{pol}.json"), "w") as f:
                json.dump({"class_name": cls, "exam_bbox": [{"bbox": o, "score": 0.9} for o in objs]}, f)
    return root


def _fake_review(root, path, drop_auto=None):
    """Bản rà giả (khuôn `box_review/app.py`): mỗi file loại tay box 0; `drop_auto` {tên: [chỉ số]} = lọc trùng tự động."""
    with open(path, "w") as f:
        for n in sorted(x[:-5] for x in os.listdir(os.path.join(root, "Anno_with_exam_bbox"))):
            m = len(json.load(open(os.path.join(root, "Anno_with_exam_bbox", n + ".json")))["exam_bbox"])
            f.write(json.dumps({"id": n, "bad": {n: [0]}, "auto": {n: (drop_auto or {}).get(n, [])}, "n": {n: m}}) + "\n")
    return path


# ----------------------------------------------------------------------------- config / chạy train, eval

def run_cfg(tmp_path, base, run, name="", **training):
    """Config thật `run` (đã gộp `base`) thu nhỏ cho dữ liệu giả: canvas 128, T = 100 (vòng mock 100 bước), 2 bộ nhiễu / ảnh, batch 2,
    4 iter, log mỗi iter; refiner / U-Net giữ kích thước thật, CLIP giả; `training` ghi đè thêm. -> (đường dẫn yaml, cfg)."""
    from ce_localization.utils.config import load_config
    cfg = load_config(os.path.join(CFG_DIR, f"{run}.yaml"))
    cfg["data"].update(root=os.path.join(base, "all_phase2_V2"), samples_root=os.path.join(base, "samples"),
                       turn_index=os.path.join(base, "turn_index.json"), density_index=os.path.join(base, "density_index.json"),
                       density_root=os.path.join(base, "samples"), image_size=128, num_workers=0)
    if cfg["data"].get("train_target") == "positive":
        cfg["data"]["positive_file"] = os.path.join(base, "ce130_train.jsonl")
        _fake_positives(base, cfg["data"]["positive_file"])
    cfg["model"]["pretrained_backbone"] = False
    cfg["diffusion"].update(num_timesteps=100, noise_per_image=2)
    cfg["training"].update(batch_size=2, max_iter=4, log_every=1, ckpt_every=2, cosine_t_max=100,
                           **({} if cfg["eval"]["split"] is None else {"eval_every": 2}), **training)
    cfg["eval"].update(n_samples=3, batch_size=2)
    p = str(tmp_path / f"cfg_{run}{name}.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f)
    return p, cfg


def _run_train(monkeypatch, argv):
    import ce_localization.train as ta
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    _patch_clip(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["train.py"] + argv + ["--device", "cpu"])
    ta.main()


def _run_eval(monkeypatch, argv, clip_seed=0):
    """eval.py trên CPU, 3 box / mẫu, batch 2 -> (JSON chỉ số, JSON dump)."""
    import ce_localization.eval as ea
    import ce_localization.train as ta
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    _patch_clip(monkeypatch, clip_seed)
    out = argv[argv.index("--out") + 1]
    monkeypatch.setattr(sys, "argv", ["eval.py", "--n-samples", "3", "--batch-size", "2", "--num-workers", "0", "--device", "cpu", *argv])
    ea.main()
    with open(out) as f, open(os.path.splitext(out)[0] + "_boxes.json") as g:
        return json.load(f), json.load(g)

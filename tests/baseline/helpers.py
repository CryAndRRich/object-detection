"""Dữ liệu giả + config tí hon cho test ZETA (không tải gì, CPU)."""

import json
import os

import numpy as np
from PIL import Image

COCOUNT_NAMES = ("INTRA_FOO_TOM1_TOM2_00008_00006_0_200", "INTER_OFF_PEN0_PNC0_00007_00005_0_40")
TINY = {
    "zeta1": {"model": {"backbone": "resnet18", "dilation": False, "hidden_dim": 32, "dropout": 0.0, "nheads": 2, "dim_feedforward": 64,
                            "enc_layers": 1, "dec_layers": 2, "num_queries": 12},
                  "finetune": {"batch_size": 2, "micro_batch": 1, "epochs": 2, "train_scales": [64, 80], "train_max_size": 96,
                               "num_workers": 0, "log_every": 1, "amp": False},
                  "predict": {"k": 5, "test_size": 64, "test_max_size": 96}},
    "zeta2": {"model": {"img_size": 32, "d_noise": 16, "mid": 16},
               "finetune": {"batch_size": 2, "micro_batch": 1, "epochs": 2, "num_workers": 0, "log_every": 1, "amp": False},
               "predict": {"k": 4, "batch_images": 3}},
}


# ----------------------------------------------------------------------------- CE-130 giả

def _save(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(img).save(path)


def fake_ce130(base, seed=0):
    """CE-130 giả: train (2 ảnh gốc, nhánh b1, xoá 2 vật) + test (2 ảnh gốc), 7 vật / ảnh, ảnh 150×100 / 120×160. Chỉ mục viết tay
    như `build_turn_index` (khoá turns `<iid>_b1_t<t>`, sample `<split>/images/<iid>_<t>.png`). -> (ce130_root, samples_root,
    turn_index path)."""
    rng = np.random.default_rng(seed)
    root, samples = os.path.join(base, "all_phase2_V2"), os.path.join(base, "samples")
    branches, turns = {}, {}
    for split, iids, cls in (("train", ["10", "11"], "apple"), ("test", ["30", "31"], "cup")):
        for n, iid in enumerate(iids):
            W, H = (150, 100) if n == 0 else (120, 160)
            gt = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
            objs = [[float(5 + 16 * k), float(10 + 3 * k), float(17 + 16 * k), float(30 + 4 * k)] for k in range(7)]
            br = f"{split}/{iid}_b1"
            _save(gt, os.path.join(root, br, "ground_truth.jpg"))
            img = np.asarray(Image.open(os.path.join(root, br, "ground_truth.jpg")).convert("RGB")).copy()
            holes = [objs[3], objs[5]]
            for t, (x1, y1, x2, y2) in enumerate(holes, 1):
                img[int(y1):int(y2), int(x1):int(x2)] = 128
                _save(img, os.path.join(root, br, f"inpainted_turn_{t}.png"))
                _save(img, os.path.join(samples, split, "images", f"{iid}_{t}.png"))
                _save(np.zeros((H, W, 3), dtype=np.uint8), os.path.join(samples, split, "density", f"{iid}_{t}.png"))
                turns[f"{iid}_b1_t{t}"] = {"branch": br, "t": t, "sample": f"{split}/images/{iid}_{t}.png",
                                           "density": f"{split}/density/{iid}_{t}.png"}
            branches[br] = {"split": split, "wh": [W, H], "class": cls, "holes": holes, "objects": objs,
                            "removed": [0, 0, 0, 1, 0, 2, 0]}
    p = os.path.join(base, "turn_index.json")
    with open(p, "w") as f:
        json.dump({"branches": branches, "turns": turns, "report": {}}, f)
    return root, samples, p


# ----------------------------------------------------------------------------- CE-CoCount giả

def fake_cocount(root, seed=0):
    """Hai frame × (`_positive`, `_negative`), vật hai lớp hai hàng, 10 loc_bbox; bản rà: loại tay box 0 mỗi file. -> review path."""
    rng = np.random.default_rng(seed)
    for d in ("Image", "Anno", "Anno_with_exam_bbox"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    review = []
    for fi, stem in enumerate(COCOUNT_NAMES):
        W, H = (200, 120) if fi == 0 else (90, 160)
        na, nb = int(stem.split("_")[4]), int(stem.split("_")[5])
        for pol, n, cls, off in (("positive", na, "normal tomato(Big round tomato).", 0), ("negative", nb, "baby tomato.", 30)):
            name = f"{stem}_{pol}"
            Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(os.path.join(root, "Image", name + ".jpg"))
            objs = [[float(5 + 10 * (k % 8)), float(5 + off + 12 * (k // 8)), float(13 + 10 * (k % 8)), float(14 + off + 12 * (k // 8))]
                    for k in range(n)]
            loc = [[float(4 + 8 * k), float(70 + (k % 2) * 10), float(12 + 8 * k), float(78 + (k % 2) * 10)] for k in range(10)]
            with open(os.path.join(root, "Anno", name + ".json"), "w") as f:
                json.dump({"class_name": cls, "loc_bbox": loc,
                           "counting_anno": {"count": n, "exemplars": [list(map(int, o)) for o in objs[:3]]}}, f)
            with open(os.path.join(root, "Anno_with_exam_bbox", name + ".json"), "w") as f:
                json.dump({"class_name": cls, "exam_bbox": [{"bbox": o, "score": 0.9} for o in objs]}, f)
            review.append({"id": name, "bad": {name: [0]}, "auto": {name: []}, "n": {name: n}})
    p = os.path.join(root, "review.jsonl")
    with open(p, "w") as f:
        for r in review:
            f.write(json.dumps(r) + "\n")
    return p


# ----------------------------------------------------------------------------- config / checkpoint tí hon

CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baseline", "config")


def tiny_cfg(tmp_path, run):
    """Config `<run>.yaml` thật (zeta1 = BootPlace, zeta2 = CSANet) + model / batch tí hon -> (đường dẫn yaml, cfg đã gộp)."""
    import yaml
    from ce_localization.utils.config import load_config
    p = os.path.join(str(tmp_path), f"{run}_tiny.yaml")
    with open(p, "w") as f:
        yaml.safe_dump({"base": os.path.abspath(os.path.join(CFG_DIR, f"{run}.yaml")), **TINY[run]}, f)
    return p, load_config(p)


def fake_ckpt(cfg, path, seed=0):
    """Checkpoint 'công bố' giả (weight ngẫu nhiên, khuôn `{'model': …}` như bản gốc, kèm khoá của module bản gốc đã bỏ)."""
    import torch
    from baseline.methods import METHODS
    torch.manual_seed(seed)
    m = METHODS[cfg["method"]].build(cfg)
    if cfg["method"] == "bootplace":
        sd = {**m.state_dict(), "transformer.patch_decoder.deconv1.weight": torch.zeros(1)}
    else:
        sd = {"generator": {**m[0].state_dict(), "att.convMod2.v.weight": torch.zeros(1)}, "discriminator": m[1].state_dict()}
    torch.save({"model": sd, "epoch": 18}, path)
    return path

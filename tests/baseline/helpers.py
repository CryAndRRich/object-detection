"""Dữ liệu giả + config tí hon cho test ZETA (không tải gì, CPU)."""

import json
import os

import numpy as np
from PIL import Image


TINY = {
    "zeta1": {"model": {"backbone": "resnet18", "dilation": False, "hidden_dim": 32, "dropout": 0.0, "nheads": 2, "dim_feedforward": 64,
                            "enc_layers": 1, "dec_layers": 2, "num_queries": 12},
                  "finetune": {"batch_size": 2, "micro_batch": 1, "epochs": 2, "train_scales": [64, 80], "train_max_size": 96,
                               "num_workers": 0, "log_every": 1, "amp": False},
                  "predict": {"k": 5, "test_size": 64, "test_max_size": 96}},
    "zeta2": {"model": {"img_size": 32, "d_noise": 16, "mid": 16},
               "finetune": {"batch_size": 2, "micro_batch": 1, "epochs": 2, "num_workers": 0, "log_every": 1, "amp": False},
               "predict": {"k": 4, "batch_images": 3}},
    "zeta7": {"predict": {"k": 5, "stride": 4, "workers": 1}},     # 1 tiến trình: spawn macOS tốn ~6 s, kết quả không phụ thuộc
    "zeta3": {"model": {"hidden_dim": 32, "nheads": 2, "enc_layers": 1, "dec_layers": 1, "dim_feedforward": 64, "dropout": 0.0,
                        "num_queries": 8, "image_size": 64, "backbone_weights": None},
              "finetune": {"batch_size": 2, "micro_batch": 1, "epochs": 2, "num_workers": 0, "log_every": 1, "amp": False},
              "predict": {"k": 5, "batch_images": 3, "num_workers": 0}},
}
TINY["zeta1_1"] = TINY["zeta1"]                     # ZETA1.1 = kiến trúc ZETA1, khởi tạo DETR (fake_ckpt khuôn DETR)
HO_CLASSES = ["apple", "cup", "dog"]


def fake_clip(names, model_name=None, device="cpu", **_):
    """Thay `encode_class_names` (không tải HF): vector 512-d tất định theo tên."""
    import hashlib
    import torch
    return {n: torch.randn(512, generator=torch.Generator().manual_seed(int(hashlib.md5(n.encode()).hexdigest()[:8], 16)))
            for n in names}


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
                m = np.zeros((H, W, 3), dtype=np.uint8)                   # mask vật đã xoá: lõi giữa lỗ (vật không đầy box)
                m[int(y1) + 3:int(y2) - 3, int(x1) + 2:int(x2) - 2] = 255
                _save(m, os.path.join(root, br, f"mask_{t}.png"))
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


# ----------------------------------------------------------------------------- config / checkpoint tí hon

CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "baseline", "config")


def tiny_cfg(tmp_path, run):
    """Config `<run>.yaml` thật (zeta1 = BootPlace, zeta2 = CSANet, zeta3 = HiddenObjects) + model / batch tí hon -> (đường dẫn yaml, cfg đã gộp)."""
    import yaml
    from ce_localization.utils.config import load_config
    p = os.path.join(str(tmp_path), f"{run}_tiny.yaml")
    with open(p, "w") as f:
        yaml.safe_dump({"base": os.path.abspath(os.path.join(CFG_DIR, f"{run}.yaml")), **TINY[run]}, f)
    return p, load_config(p)


def detr_state(sd, d):
    """State dict BootPlace -> khuôn DETR COCO: cross-attn tên `multihead_attn`, 92 lớp, 100 query, không có phần riêng BootPlace."""
    import torch
    out = {k.replace(".cross_attn.", ".multihead_attn."): v for k, v in sd.items()
           if not k.startswith(("bbox_embed.loc_proj.", "transformer.patch_encoder.", "logit_scale", "class_embed.", "query_embed."))}
    out.update({"class_embed.weight": torch.randn(92, d), "class_embed.bias": torch.zeros(92), "query_embed.weight": torch.randn(100, d)})
    return out


def fake_ckpt(cfg, path, seed=0):
    """Checkpoint 'công bố' giả (weight ngẫu nhiên, khuôn `{'model': …}` như bản gốc, kèm khoá của module bản gốc đã bỏ)."""
    import torch
    from baseline.methods import METHODS
    torch.manual_seed(seed)
    m = METHODS[cfg["method"]].build(cfg)
    if cfg["method"] == "hiddenobjects":
        sd = {k: v for k, v in m.state_dict().items() if not k.startswith(("backbone.", "text_proj."))}
        sd.update({"class_embed.weight": torch.randn(len(HO_CLASSES), cfg["model"]["hidden_dim"]), "input_proj.weight": torch.zeros(1)})
        torch.save({"model": sd, "classes": HO_CLASSES, "num_queries": cfg["model"]["num_queries"]}, path)
        return path
    if cfg["method"] == "bootplace" and "detr" in os.path.basename(cfg["finetune"].get("init", "")):
        torch.save({"model": detr_state(m.state_dict(), cfg["model"]["hidden_dim"])}, path)            # ZETA1.1: khởi tạo từ DETR COCO
        return path
    if cfg["method"] == "bootplace":
        sd = {**m.state_dict(), "transformer.patch_decoder.deconv1.weight": torch.zeros(1)}
    else:
        sd = {"generator": {**m[0].state_dict(), "att.convMod2.v.weight": torch.zeros(1)}, "discriminator": m[1].state_dict()}
    torch.save({"model": sd, "epoch": 18}, path)
    return path

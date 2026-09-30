"""Dữ liệu giả + config test dùng chung cho mọi file test của ce_localization. Không tải gì:
backbone `pretrained_backbone: false`, text thay bằng embedding giả, mọi lượt train ép `--device cpu`.
"""

import json
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image

from ce_localization.data.density import DensityIndex, JET, build_index
from ce_localization.data.points import PointTable
from ce_localization.engine import criterion as C
from ce_localization.models.text import TextTable


CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "ce_localization", "config")
CFG0 = os.path.join(CFG_DIR, "alpha", "alpha0.yaml")
CFG3 = {"full": os.path.join(CFG_DIR, "alpha", "alpha3_1.yaml"), "mix": os.path.join(CFG_DIR, "alpha", "alpha3_2.yaml")}
CFG_B0 = os.path.join(CFG_DIR, "beta", "beta0.yaml")
PSEUDO = {"knn": 3, "beta": 1.0, "min_frac": 0.02, "max_frac": 0.30}


CFG_L = {"alpha": 0.25, "gamma": 2.0, "class_weight": 2.0, "l1_weight": 5.0, "giou_weight": 2.0,
         "center_weight": 5.0, "size_weight": 1.0}
CFG_M = {"ota_k": 5, "center_radius": 2.5}


def _make_branch(root, split, iid, boxes_xyxy, category, w=200, h=150, seed=0):
    br = os.path.join(root, split, f"{iid}_b1")
    os.makedirs(br, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(
        os.path.join(br, "ground_truth.jpg"))
    with open(os.path.join(br, "annotation.json"), "w") as f:
        json.dump({"all_bboxes": boxes_xyxy, "inpainted_bboxes": boxes_xyxy[:1],
                   "class_based_caption": category}, f)
    return br


def _fake_ce130(root, n_train=4, n_val=2, n_test=2):
    rng = np.random.default_rng(1)
    k = 0
    for split, n, cats in (("train", n_train, ["apple", "egg"]), ("val", n_val, ["bird"]),
                           ("test", n_test, ["cup"])):
        for i in range(n):
            boxes = []
            for _ in range(int(rng.integers(3, 8))):
                x1, y1 = rng.uniform(0, 150), rng.uniform(0, 110)
                boxes.append([x1, y1, x1 + rng.uniform(8, 45), y1 + rng.uniform(8, 35)])
            _make_branch(root, split, str(1000 + k), boxes, cats[i % len(cats)], seed=k)
            k += 1


def _fake_text_table(names, cfg, dev):
    table = {}
    for n in names:
        g = torch.Generator().manual_seed(sum(map(ord, n)))
        table[n] = torch.randn(512, generator=g)
    return TextTable(table)


def _test_cfg(tmp_path, memory="none", data_root=None, density=None):
    with open(CFG0 if density is None else CFG3[density]) as f:
        cfg = yaml.safe_load(f)
    cfg["data"].update(root=data_root, image_size=128, num_workers=0)
    if density is not None:
        base = os.path.dirname(data_root)
        cfg["data"].update(density_root=os.path.join(base, "samples"),
                           density_index=os.path.join(base, "density_index.json"))
    cfg["model"].update(memory=memory, pretrained_backbone=False)
    cfg["diffusion"]["num_proposals"] = 20
    cfg["training"].update(max_iter=4, steps=[3], warmup_iters=2, log_every=1, ckpt_every=2,
                           eval_every=2)
    cfg["eval"]["batch_size"] = 2
    p = str(tmp_path / f"cfg_{memory}_{density}.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f)
    return p, cfg


def _run_train(monkeypatch, argv):
    import ce_localization.train as ta
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    monkeypatch.setattr(sys, "argv", ["train.py"] + argv + ["--device", "cpu"])
    ta.main()


def _fake_density(base, n_variants=None):
    """samples/{train,test}/density/{iid}_{k}.png tô jet cho mọi ảnh của CE-130 giả: bản k chỉ vẽ
    blob trên (số box − k) box đầu (mất dần vật, như lượt inpaint); ảnh đầu mỗi split chỉ 1 bản.
    Rồi dựng chỉ mục như `tools/build_density_index.py`."""
    root = os.path.join(base, "all_phase2_V2")
    for split in ("train", "val", "test"):
        for j, br in enumerate(sorted(os.listdir(os.path.join(root, split)))):
            iid = br.split("_b")[0]
            with open(os.path.join(root, split, br, "annotation.json")) as f:
                boxes = json.load(f)["all_bboxes"]
            w, h = Image.open(os.path.join(root, split, br, "ground_truth.jpg")).size
            n = 1 if j == 0 else (n_variants or 3)
            out = os.path.join(base, "samples", "test" if split == "test" else "train", "density")
            os.makedirs(out, exist_ok=True)
            for k in range(n):
                lv = np.zeros((h, w), dtype=np.uint8)
                for x1, y1, x2, y2 in boxes[: max(len(boxes) - k, 0)]:
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    lv[max(cy - 3, 0):cy + 4, max(cx - 3, 0):cx + 4] = 255
                    lv[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = 128
                Image.fromarray(JET[lv]).save(os.path.join(out, f"{iid}_{k + 1}.png"))
    idx = build_index(os.path.join(base, "samples"), workers=0, log=lambda *a: None)
    with open(os.path.join(base, "density_index.json"), "w") as f:
        json.dump(idx, f)
    return DensityIndex(os.path.join(base, "density_index.json"), os.path.join(base, "samples"))


def _targets(gt, wh=(100.0, 100.0)):
    return C.build_targets([gt], torch.tensor([[wh[0], wh[1], wh[0], wh[1]]]))


def _gauss(h, w, centers, sigma=2.5, peak=1.0):
    yy, xx = np.mgrid[0:h, 0:w] + 0.5
    d = sum(np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)) for cx, cy in centers)
    return np.round(d / d.max() * 255 * peak).astype(np.uint8)


def _write_points(path, table):
    with open(path, "w") as f:
        json.dump({"params": {"tau": 8, "radius": 2}, "points": table}, f)
    return PointTable(path)


def _beta_cfg(tmp_path, root, points):
    with open(CFG_B0) as f:
        cfg = yaml.safe_load(f)
    cfg["data"].update(root=root, image_size=128, num_workers=0, points=points)
    cfg["model"]["pretrained_backbone"] = False
    cfg["diffusion"]["num_proposals"] = 20
    cfg["training"].update(max_iter=4, steps=[3], warmup_iters=2, log_every=1, ckpt_every=2, eval_every=2)
    cfg["eval"]["batch_size"] = 2
    p = str(tmp_path / "cfg_beta0.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f)
    return p


def _run_g0(monkeypatch, base, out, report, extra=()):
    import ce_localization.tools.build_density_points as g0
    monkeypatch.setattr(sys, "argv", ["build_density_points.py", "--ce130", os.path.join(base, "all_phase2_V2"),
                                      "--samples", os.path.join(base, "samples"),
                                      "--density-index", os.path.join(base, "density_index.json"),
                                      "--out", out, "--report", report, "--workers", "0", *extra])
    g0.main()

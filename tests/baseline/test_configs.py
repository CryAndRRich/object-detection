"""Config BASELINE0–2: đúng ngân sách D.1 (12k iter × batch 2), đúng LR co theo batch, đúng giới hạn box.
Đọc YAML thô (không cần detectron2); `test_train_predict.py` kiểm lại bằng chính detectron2."""

import glob
import math
import os

import yaml

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "baseline", "configs")


def _load(name):
    with open(os.path.join(CFG_DIR, name)) as f:
        return yaml.safe_load(f)


def test_base_budget_matches_d1():
    b = _load("Base-CE130.yaml")
    s = b["SOLVER"]
    assert s["IMS_PER_BATCH"] == 2 and s["MAX_ITER"] == 12000 and s["STEPS"] == "(9000, 11000)"
    assert s["WARMUP_ITERS"] == 1000 and s["WARMUP_FACTOR"] == 0.01 and s["AMP"]["ENABLED"] is False
    assert b["DATASETS"]["TRAIN"] == '("ce130_agnostic_train",)'
    assert b["DATASETS"]["TEST"] == '("ce130_agnostic_val",)'          # chọn checkpoint trên VAL
    assert b["MODEL"]["WEIGHTS"].endswith("torchvision/R-50.pkl") and b["MODEL"]["RESNETS"]["STRIDE_IN_1X1"] is False
    assert b["INPUT"]["CROP"]["ENABLED"] is True and b["INPUT"]["FORMAT"] == "RGB"
    assert b["TEST"]["EVAL_PERIOD"] == 2000


def test_three_baselines():
    c0, c1, c2 = (_load(f) for f in ("baseline0_diffusiondet.yaml", "baseline1_sparsercnn.yaml",
                                     "baseline2_fasterrcnn.yaml"))
    lr_adamw = 2.5e-5 * math.sqrt(2 / 16)                               # = 1,25e-5 × √(2/4) của D.1
    assert abs(lr_adamw - 1.25e-5 * math.sqrt(2 / 4)) < 1e-15
    for c, arch, node, k in ((c0, "DiffusionDet", "DiffusionDet", "0"), (c1, "SparseRCNN", "SparseRCNN", "1")):
        assert c["_BASE_"] == "Base-CE130.yaml" and c["MODEL"]["META_ARCHITECTURE"] == arch
        assert c["MODEL"][node]["NUM_CLASSES"] == 1 and c["MODEL"][node]["NUM_PROPOSALS"] == 300
        assert c["SOLVER"]["OPTIMIZER"] == "ADAMW" and abs(c["SOLVER"]["BASE_LR"] - lr_adamw) < 1e-9
        assert c["SOLVER"]["CLIP_GRADIENTS"]["CLIP_TYPE"] == "full_model"
        assert c["BASELINE"]["NAME"] == f"BASELINE{k}" and c["OUTPUT_DIR"] == f"checkpoints/baseline{k}"
    assert c2["_BASE_"] == "Base-CE130.yaml" and c2["MODEL"]["META_ARCHITECTURE"] == "GeneralizedRCNN"
    assert c2["MODEL"]["ROI_HEADS"]["NUM_CLASSES"] == 1 and c2["MODEL"]["MASK_ON"] is False
    assert abs(c2["SOLVER"]["BASE_LR"] - 0.02 * 2 / 16) < 1e-12 and c2["SOLVER"]["MOMENTUM"] == 0.9
    assert c2["TEST"]["DETECTIONS_PER_IMAGE"] == 300 and c2["MODEL"]["ROI_HEADS"]["SCORE_THRESH_TEST"] == 0.0
    assert c2["MODEL"]["ROI_HEADS"]["NMS_THRESH_TEST"] == 0.5
    assert c2["BASELINE"]["NAME"] == "BASELINE2" and c2["OUTPUT_DIR"] == "checkpoints/baseline2"


def test_no_stray_configs():
    top = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CFG_DIR, "*.yaml")))
    assert top == ["Base-CE130.yaml", "baseline0_diffusiondet.yaml", "baseline1_sparsercnn.yaml",
                   "baseline2_fasterrcnn.yaml", "baseline3a_gdino_zeroshot.yaml", "baseline3b_gdino_finetune.yaml"]
    bench = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CFG_DIR, "benchmarks", "*.yaml")))
    assert "Base-Kaggle-T4x2.yaml" in bench and len(bench) == 5

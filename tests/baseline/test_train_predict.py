"""TRỌN LUỒNG BASELINE0–2 trên CE-130 giả (cạm bẫy 11): convert -> train 2 iter -> resume tới 4 ->
predict (dump) -> chấm. CPU, weight ngẫu nhiên (MODEL.WEIGHTS ""), ảnh nhỏ. Cần detectron2 -> tự skip ở
máy không có (local); chạy trên server / Kaggle: `python -m pytest tests/baseline -q`.
"""

import json
import os
import sys

import pytest

pytest.importorskip("detectron2")

from baseline.objdet.datasets import register_all  # noqa: E402
from baseline.tools.convert_ce130 import build_coco, scan_dedup  # noqa: E402
from tests.ce_localization.helpers import _fake_ce130  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "baseline", "configs")
SMALL = ["MODEL.WEIGHTS", "", "MODEL.DEVICE", "cpu", "SOLVER.IMS_PER_BATCH", "2", "SOLVER.WARMUP_ITERS", "1",
         "SOLVER.CHECKPOINT_PERIOD", "1", "TEST.EVAL_PERIOD", "2", "DATALOADER.NUM_WORKERS", "0",
         "INPUT.MIN_SIZE_TRAIN", "(160,)", "INPUT.MAX_SIZE_TRAIN", "256", "INPUT.MIN_SIZE_TEST", "160",
         "INPUT.MAX_SIZE_TEST", "256"]
PROPS = {"baseline0_diffusiondet": ["MODEL.DiffusionDet.NUM_PROPOSALS", "20"],
         "baseline1_sparsercnn": ["MODEL.SparseRCNN.NUM_PROPOSALS", "20"],
         "baseline2_fasterrcnn": ["TEST.DETECTIONS_PER_IMAGE", "20"]}


@pytest.fixture(scope="module")
def ce130(tmp_path_factory):
    root = tmp_path_factory.mktemp("objdet_data")
    img_root = str(root / "all_phase2_V2")
    _fake_ce130(img_root)
    os.makedirs(root / "ce130_coco")
    for split in ("train", "val", "test"):
        coco, _ = build_coco(scan_dedup(os.path.join(img_root, split), verbose=False), "class-agnostic",
                             image_root_for_relpath=img_root)
        with open(root / "ce130_coco" / f"ce130_agnostic_{split}.json", "w") as f:
            json.dump(coco, f)
    register_all(str(root))          # đăng ký một lần; main() gọi lại thì bỏ qua (đã có)
    return str(root), img_root


def _train(cfg, out, max_iter, resume=False):
    from baseline.train_net import get_parser, main
    argv = ["--config-file", cfg] + (["--resume"] if resume else []) + SMALL + [
        "OUTPUT_DIR", out, "SOLVER.MAX_ITER", str(max_iter), "SOLVER.STEPS", "(3,)"] + PROPS[os.path.basename(cfg)[:-5]]
    main(get_parser().parse_args(argv))


def _last_iter(out):
    import torch
    with open(os.path.join(out, "last_checkpoint")) as f:
        name = f.read().strip()
    return torch.load(os.path.join(out, name), map_location="cpu", weights_only=False)["iteration"]


@pytest.mark.parametrize("name", ["baseline0_diffusiondet", "baseline1_sparsercnn", "baseline2_fasterrcnn"])
def test_train_resume_predict_score(ce130, tmp_path, monkeypatch, name):
    root, img_root = ce130
    cfg = os.path.join(CFG_DIR, f"{name}.yaml")
    out = str(tmp_path / "ckpt")
    _train(cfg, out, 2)
    assert os.path.exists(os.path.join(out, "model_final.pth"))
    assert os.path.exists(os.path.join(out, "model_best.pth"))      # BestCheckpointer đọc được ce130/oracle_recall
    _train(cfg, out, 4, resume=True)
    assert _last_iter(out) == 3                                      # nối tiếp từ iter 2, không train lại từ 0

    import baseline.predict as P
    pred_dir = str(tmp_path / "pred")
    extra = (["--num-proposals", "20", "--steps", "1", "2"] if name.startswith("baseline0") else [])
    monkeypatch.setattr(sys, "argv", ["predict.py", "--config-file", cfg, "--weights",
                                      os.path.join(out, "model_final.pth"), "--split", "test", "--out-dir", pred_dir,
                                      "--data-root", img_root, "--device", "cpu", *extra, "--opts",
                                      "INPUT.MIN_SIZE_TEST", "160", "INPUT.MAX_SIZE_TEST", "256"] + PROPS[name])
    P.main()
    dumps = sorted(f for f in os.listdir(pred_dir) if not f.endswith("_metrics.json"))
    run = f"BASELINE{name[8]}"
    want = ([f"{run}_test_N20_s1.json", f"{run}_test_N20_s2.json"] if name.startswith("baseline0")
            else [f"{run}_test.json"])
    assert dumps == want
    for d in dumps:
        dump = json.load(open(os.path.join(pred_dir, d)))
        assert len(dump["pred"]) == 2 and dump["meta"]["iter"] == 3 and dump["meta"]["run"] == run
        assert all(len(p["scores"]) <= (40 if d.endswith("s2.json") else 20) for p in dump["pred"].values())
        m = json.load(open(os.path.join(pred_dir, d[:-5] + "_metrics.json")))
        r = next(iter(m["results"].values()))
        assert set(r) == {"topk_first", "nms_first"} and 0.0 <= r["nms_first"]["oracle_recall"] <= 1.0

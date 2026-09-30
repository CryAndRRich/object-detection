"""TRỌN LUỒNG BASELINE0–2 trên CE-130 giả (cạm bẫy 11): convert -> train 2 iter -> resume tới 4 ->
predict (dump) -> chấm. CPU, weight ngẫu nhiên (MODEL.WEIGHTS ""), ảnh nhỏ. Cần detectron2 -> tự skip ở
máy không có (local); chạy trên server / Kaggle: `python -m pytest tests/baseline -q`.
"""

import json
import os
import shutil
import sys

import pytest

pytest.importorskip("detectron2")

from baseline.objdet.datasets import register_all  # noqa: E402
from baseline.tools.convert_ce130 import build_coco, scan_dedup  # noqa: E402
from tests.ce_localization.helpers import _fake_ce130  # noqa: E402

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "baseline", "configs")
# CHECKPOINT_PERIOD lớn: last.pth chỉ ghi ở iter cuối (checkpoint R-50 kèm optimizer hàng trăm MB — /mnt/disk1
# của server có lúc chỉ còn vài GB); --resume vẫn được thử vì last.pth của lượt 1 mang iteration 1.
SMALL = ["MODEL.WEIGHTS", "", "MODEL.DEVICE", "cpu", "SOLVER.IMS_PER_BATCH", "2", "SOLVER.WARMUP_ITERS", "1",
         "SOLVER.CHECKPOINT_PERIOD", "1000", "TEST.EVAL_PERIOD", "2", "DATALOADER.NUM_WORKERS", "0",
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
        with open(root / "ce130_coco" / f"ce130_agnostic_{split}.json", "w", encoding="utf-8") as f:
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
    with open(os.path.join(out, "last_checkpoint"), encoding="utf-8") as f:
        name = f.read().strip()
    return torch.load(os.path.join(out, name), map_location="cpu", weights_only=False)["iteration"]


@pytest.mark.parametrize("name", ["baseline0_diffusiondet", "baseline1_sparsercnn", "baseline2_fasterrcnn"])
def test_train_resume_predict_score(ce130, tmp_path, monkeypatch, name):
    root, img_root = ce130
    cfg = os.path.join(CFG_DIR, f"{name}.yaml")
    out = str(tmp_path / "ckpt")
    try:
        _check_train_resume_predict(cfg, out, img_root, tmp_path, monkeypatch, name)
    finally:
        shutil.rmtree(out, ignore_errors=True)            # checkpoint lớn: không để lại trên đĩa dùng chung


def _check_train_resume_predict(cfg, out, img_root, tmp_path, monkeypatch, name):
    _train(cfg, out, 2)
    files = set(os.listdir(out))
    assert {"last.pth", "best.pth", "history.json", "last_checkpoint"} <= files
    assert not [f for f in files if f.startswith("model_") or f.endswith(".tmp")]     # không checkpoint định kỳ
    assert not ({"metrics.json", "inference"} & files) and not [f for f in files if f.startswith("events.")]
    h1 = json.load(open(os.path.join(out, "history.json"), encoding="utf-8"))
    assert [e["iter"] for e in h1["eval"]] == [2] and "ce130/oracle_recall" in h1["eval"][0]
    assert h1["best"]["iter"] == 2 and h1["train"][-1]["iter"] == 2 and "total_loss" in h1["train"][-1]
    import torch
    best = torch.load(os.path.join(out, "best.pth"), map_location="cpu", weights_only=False)
    assert set(best) == {"model", "iteration", "ce130/oracle_recall"}         # best.pth: CHỈ model (+ iteration, metric)

    _train(cfg, out, 4, resume=True)
    assert _last_iter(out) == 3                                      # nối tiếp từ iter 2, không train lại từ 0
    h2 = json.load(open(os.path.join(out, "history.json"), encoding="utf-8"))
    assert [e["iter"] for e in h2["eval"]] == [2, 4]                 # history cũ giữ, eval mới nối thêm
    assert [r["iter"] for r in h2["train"]] == [2, 4]
    assert h2["best"]["iter"] in (2, 4)
    assert torch.load(os.path.join(out, "best.pth"), map_location="cpu", weights_only=False)["iteration"] + 1 \
        == h2["best"]["iter"]

    import baseline.predict as P
    pred_dir = str(tmp_path / "pred")
    extra = (["--num-proposals", "20", "--steps", "1", "2"] if name.startswith("baseline0") else [])
    monkeypatch.setattr(sys, "argv", ["predict.py", "--config-file", cfg, "--weights",
                                      os.path.join(out, "last.pth"), "--split", "test", "--out-dir", pred_dir,
                                      "--data-root", img_root, "--device", "cpu", *extra, "--opts",
                                      "INPUT.MIN_SIZE_TEST", "160", "INPUT.MAX_SIZE_TEST", "256"] + PROPS[name])
    P.main()
    dumps = sorted(f for f in os.listdir(pred_dir) if not f.endswith("_metrics.json"))
    run = f"BASELINE{name[8]}"
    want = ([f"{run}_test_N20_s1.json", f"{run}_test_N20_s2.json"] if name.startswith("baseline0")
            else [f"{run}_test.json"])
    assert dumps == want
    for d in dumps:
        dump = json.load(open(os.path.join(pred_dir, d), encoding="utf-8"))
        assert len(dump["pred"]) == 2 and dump["meta"]["iter"] == 4 and dump["meta"]["run"] == run
        assert all(len(p["scores"]) <= (40 if d.endswith("s2.json") else 20) for p in dump["pred"].values())
        m = json.load(open(os.path.join(pred_dir, d[:-5] + "_metrics.json"), encoding="utf-8"))
        r = next(iter(m["results"].values()))
        assert set(r) == {"topk_first", "nms_first"} and 0.0 <= r["nms_first"]["oracle_recall"] <= 1.0


def test_sparsercnn_loss_when_more_gt_than_proposals():
    """CE-130 train có ảnh > 300 vật: Hungarian chỉ ghép min(Q, G) cặp. Bản gốc của Sparse R-CNN vỡ
    (RuntimeError ở loss_boxes, server 2026-09-30, iter ~112); bản đã sửa phải chạy và khi G <= Q phải cho
    đúng số của bản gốc."""
    import torch
    from baseline.sparsercnn.loss import HungarianMatcher, SetCriterion
    from baseline.train_net import build_cfg

    cfg = build_cfg(os.path.join(CFG_DIR, "baseline1_sparsercnn.yaml"))
    matcher = HungarianMatcher(cfg, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, use_focal=True)
    crit = SetCriterion(cfg, num_classes=1, matcher=matcher, weight_dict={}, eos_coef=0.1,
                        losses=["labels", "boxes"], use_focal=True)
    g = torch.Generator().manual_seed(0)

    def target(n, w=200.0, h=150.0):
        xy = torch.rand(n, 2, generator=g) * torch.tensor([w * 0.8, h * 0.8])
        xyxy = torch.cat([xy, xy + 5 + torch.rand(n, 2, generator=g) * 20], 1)
        whwh = torch.tensor([w, h, w, h])
        cxcywh = torch.cat([(xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]], 1) / whwh
        return {"labels": torch.zeros(n, dtype=torch.long), "boxes": cxcywh, "boxes_xyxy": xyxy,
                "image_size_xyxy": whwh, "image_size_xyxy_tgt": whwh.repeat(n, 1), "area": (xyxy[:, 2:] - xyxy[:, :2]).prod(1)}

    Q = 5
    xy = torch.rand(2, Q, 2, generator=g) * 150
    out = {"pred_logits": torch.randn(2, Q, 1, generator=g), "pred_boxes": torch.cat([xy, xy + 10], -1)}
    losses = crit(out, [target(8), target(3)])                          # ảnh 1: 8 GT > 5 proposal
    assert all(torch.isfinite(v) for v in losses.values())

    # G <= Q: trùng công thức bản gốc (image_size lấy toàn bộ dòng, không theo chỉ số)
    tg = [target(4), target(3)]
    idx = matcher(out, tg)
    ours = crit.loss_boxes(out, tg, idx, num_boxes=7.0)["loss_bbox"]
    src = out["pred_boxes"][crit._get_src_permutation_idx(idx)]
    tgt = torch.cat([t["boxes_xyxy"][j] for t, (_, j) in zip(tg, idx)])
    size = torch.cat([t["image_size_xyxy_tgt"] for t in tg])
    orig = torch.nn.functional.l1_loss(src / size, tgt / size, reduction="none").sum() / 7.0
    assert torch.allclose(ours, orig)

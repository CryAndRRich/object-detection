"""TRỌN LUỒNG `train.py` -> dừng -> `--resume` -> `eval.py` trên CE-130 giả cho mọi loại config (memory,
density, đích điểm), DDP 2 tiến trình, `--bench`, `--nan-debug`, và kiểm các config chỉ khác nhau đúng
chỗ đã chốt. Mọi lượt train / eval ép `--device cpu`: backward conv (cuDNN) và `roi_align` trên GPU
không tất định (resume lệch ~1e-5) và GPU server dùng chung có thể hết bộ nhớ (2026-09-29).
"""

import json
import numpy as np
import os
import sys

import pytest
import torch
import yaml

from ce_localization.data.dataset import CE130Dataset
from tests.ce_localization.helpers import (CFG0, CFG3, CFG_B0, _fake_ce130, _fake_text_table, _test_cfg,
                                           _run_train, _fake_density, _beta_cfg, _run_g0)


@pytest.mark.parametrize("kind", ["none", "spatial_softmax", "grid"])
def test_full_flow_train_resume_eval(tmp_path, monkeypatch, kind):
    """train 2 iter -> dừng -> --resume tới 4 iter == train liền 4 iter ; rồi eval ghi JSON."""
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root)
    cfg_path, _ = _test_cfg(tmp_path, kind, root)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")

    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", a, "--max-iter", "2"])
    ck = torch.load(os.path.join(a, "last.pth"), weights_only=False)
    assert ck["iter"] == 2 and ck["best"]["iter"] == 2
    with pytest.raises(SystemExit):                                  # last.pth có rồi mà thiếu --resume
        _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", a])
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", a, "--resume"])
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", b])
    ka = torch.load(os.path.join(a, "last.pth"), weights_only=False)
    kb = torch.load(os.path.join(b, "last.pth"), weights_only=False)
    assert ka["iter"] == kb["iter"] == 4
    for k in kb["model"]:
        assert torch.allclose(ka["model"][k].float(), kb["model"][k].float(), atol=1e-5), k
    assert sorted(os.listdir(a)) == ["best.pth", "history.json", "last.pth"]
    with open(os.path.join(a, "last.pth"), "rb") as f:               # pickle, KHÔNG zip (Kaggle)
        assert f.read(2) != b"PK"

    import ce_localization.eval as ea
    import ce_localization.train as ta
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    out = str(tmp_path / "res.json")
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", os.path.join(a, "best.pth"),
                                      "--split", "test", "--nms", "--oracle-score", "--steps", "1", "4",
                                      "--attn-diag", "1", "--batch-size", "2", "--num-workers", "0",
                                      "--out", out, "--device", "cpu"])
    ea.main()
    with open(out) as f:
        res = json.load(f)
    assert set(res["results"]) == {"steps1", "steps4", "steps1_nmsfirst", "steps4_nmsfirst"}
    for s in res["results"]:
        r = res["results"][s]
        assert set(r["density_recall"]) == {"<=30 vật", "31-100 vật", ">100 vật"}
        assert 0 <= r["oracle_recall"] <= 1 and len(r["oracle_recall_per_stage"]) == 6
        assert "AP50" in r["oracle_score"] and set(r["size_recall"]) == {"<1 ô P5", "1-4 ô P5", ">4 ô P5"}
    att = res["attention"]["999"]
    assert len(att["time"]) == 6
    assert ("ss" in att) == (kind == "spatial_softmax") and ("grid_lift_in_gt" in att) == (kind == "grid")


def test_resume_refuses_changed_model_config(tmp_path, monkeypatch):
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root, n_train=2, n_val=1, n_test=1)
    p0, cfg = _test_cfg(tmp_path, "none", root)
    save = str(tmp_path / "s")
    _run_train(monkeypatch, ["--config", p0, "--save-dir", save, "--max-iter", "2"])
    p1, _ = _test_cfg(tmp_path, "grid", root)
    with pytest.raises(SystemExit):
        _run_train(monkeypatch, ["--config", p1, "--save-dir", save, "--resume"])


def test_lr_override_used_and_recorded(tmp_path, monkeypatch):
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root, n_train=2, n_val=1, n_test=1)
    p0, _ = _test_cfg(tmp_path, "none", root)
    save = str(tmp_path / "lr")
    _run_train(monkeypatch, ["--config", p0, "--save-dir", save, "--max-iter", "2", "--lr", "1e-4"])
    ck = torch.load(os.path.join(save, "last.pth"), weights_only=False)
    assert ck["config"]["training"]["lr"] == 1e-4
    assert ck["optimizer"]["param_groups"][0]["initial_lr"] == 1e-4


def test_bench_writes_nothing(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root, n_train=2, n_val=1, n_test=1)
    p0, _ = _test_cfg(tmp_path, "none", root)
    save = str(tmp_path / "bench")
    _run_train(monkeypatch, ["--config", p0, "--save-dir", save, "--bench", "2"])
    assert "[bench]" in capsys.readouterr().out
    assert not os.path.exists(os.path.join(save, "last.pth"))


def _ddp_worker(rank, world, port, argv):
    import ce_localization.train as ta
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    ta.build_text_table = _fake_text_table
    sys.argv = ["train.py"] + argv + ["--device", "cpu"]
    ta.main()


def test_ddp_two_processes(tmp_path):
    """torchrun 2 tiến trình (gloo, CPU): batch toàn cục 2 = 1/GPU, chỉ rank 0 ghi, eval + dừng đúng."""
    import socket
    import torch.multiprocessing as tmp
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root)
    p0, _ = _test_cfg(tmp_path, "none", root)     # ALPHA0: P5 không có RoI -> tham số không dùng
    save = str(tmp_path / "ddp")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    tmp.spawn(_ddp_worker, args=(2, port, ["--config", p0, "--save-dir", save, "--max-iter", "2"]),
              nprocs=2, join=True)
    ck = torch.load(os.path.join(save, "last.pth"), weights_only=False)
    assert ck["iter"] == 2 and ck["best"] is not None
    assert sorted(os.listdir(save)) == ["best.pth", "history.json", "last.pth"]


def _ddp_worker_strict(rank, world, port, argv, unused):
    """`_ddp_worker` + cảnh báo lệch stride grad của DDP thành LỖI; `unused` ép find_unused_parameters (đường cũ)."""
    import warnings
    import ce_localization.train as ta
    warnings.filterwarnings("error", message="Grad strides do not match bucket view strides")
    warnings.filterwarnings("error", message="Detected call of `lr_scheduler.step()`")
    if unused is not None:
        ta.ddp_find_unused = lambda cfg: unused
    _ddp_worker(rank, world, port, argv)


def test_ddp_gamma2_celoc_bucket_view_no_stride_warning(tmp_path):
    """Pha 1 GAMMA2 (config thật, BN thường) trên 2 tiến trình gloo: grad = view bucket DDP => không cảnh báo lệch stride
    (weight Conv1d kernel 1 của U-Net trả grad stride khác ở chiều kích thước 1); đường cũ (find_unused, grad mới mỗi iter)
    thì có — để chắc phép thử bắt được cảnh báo."""
    import socket
    import torch.multiprocessing as tmp
    from tests.ce_localization.helpers import _fake_turn_index, _gamma2_cfg
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    cfg_path, cfg = _gamma2_cfg(tmp_path, base, "celoc2")
    assert cfg["training"]["sync_bn"] is False

    def run(save, unused, cfg=cfg_path, extra=()):
        with socket.socket() as so:
            so.bind(("127.0.0.1", 0))
            port = so.getsockname()[1]
        tmp.spawn(_ddp_worker_strict, args=(2, port, ["--config", cfg, "--save-dir", save, "--max-iter", "3",
                                                      "--eval-every", "3", *extra], unused), nprocs=2, join=True)
    import shutil
    rm = lambda name: shutil.rmtree(str(tmp_path / name), ignore_errors=True)   # noqa: E731 — mỗi run vài trăm MB–GB: xoá ngay
    run(str(tmp_path / "new"), None)
    ck = torch.load(os.path.join(str(tmp_path / "new"), "last.pth"), weights_only=False)
    assert ck["iter"] == 3 and all(np.isfinite(h["loss"]) and h["skipped"] == 0 for h in ck["history"] if "loss" in h)
    with pytest.raises(Exception, match="Grad strides"):
        run(str(tmp_path / "old"), True)
    rm("old")
    # pha 2 (đóng băng / train chung / GAMMA3 geo / GAMMA3.1 relation) trên DDP KHÔNG find_unused: tham số nào thiếu grad thì DDP báo lỗi ngay
    p_o, c_o = _gamma2_cfg(tmp_path, base, "obj")                           # GAMMA4: CE-Loc đứng một mình + box vật
    assert not __import__("ce_localization.train", fromlist=["x"]).ddp_find_unused(c_o)
    run(str(tmp_path / "obj"), None, p_o)
    k_o = torch.load(os.path.join(str(tmp_path / "obj"), "last.pth"), weights_only=False)
    assert k_o["iter"] == 3 and all(np.isfinite(h["loss"]) for h in k_o["history"] if "loss" in h)
    p_po, c_po = _gamma2_cfg(tmp_path, base, "pr_obj")                     # GAMMA4.1: refine trên CE-Loc GAMMA4
    assert not __import__("ce_localization.train", fromlist=["x"]).ddp_find_unused(c_po)
    run(str(tmp_path / "pr_obj"), None, p_po, ("--proposer-ckpt", os.path.join(str(tmp_path / "obj"), "last.pth")))
    k_po = torch.load(os.path.join(str(tmp_path / "pr_obj"), "last.pth"), weights_only=False)
    assert k_po["iter"] == 3 and all(np.isfinite(h["loss"]) for h in k_po["history"] if "loss" in h)
    rm("obj"), rm("pr_obj")
    for kind in ("pr", "pr_joint", "geo", "rel"):
        p2, c2 = _gamma2_cfg(tmp_path, base, kind)
        assert not __import__("ce_localization.train", fromlist=["x"]).ddp_find_unused(c2)
        run(str(tmp_path / kind), None, p2, ("--proposer-ckpt", os.path.join(str(tmp_path / "new"), "best.pth")))
        k2 = torch.load(os.path.join(str(tmp_path / kind), "last.pth"), weights_only=False)
        assert k2["iter"] == 3 and all(np.isfinite(h["loss"]) for h in k2["history"] if "loss" in h)
        rm(kind)
    rm("new")


def test_config_diff_ignores_data_root_and_workers():
    """Kaggle gắn dataset ở đường dẫn khác nhau giữa các phiên: resume vẫn phải chạy."""
    from ce_localization.train import config_diff
    with open(CFG0) as f:
        cfg = yaml.safe_load(f)
    other = json.loads(json.dumps(cfg))
    other["data"].update(root="/kaggle/input/x/all_phase2_V2", num_workers=2,
                         density_root="/kaggle/input/y/samples", density_index="/kaggle/temp/i.json")
    other["training"]["max_iter"] = 99
    assert config_diff(cfg, other) == ([], ["training"])
    other["model"]["memory"] = "grid"
    assert config_diff(cfg, other)[0] == ["model"]


def test_alpha3_configs_only_differ_from_alpha0_by_density():
    with open(CFG0) as f:
        c0 = yaml.safe_load(f)
    for mode, p in CFG3.items():
        with open(p) as f:
            c = yaml.safe_load(f)
        assert c["data"].pop("density") == mode and c["model"].pop("in_channels") == 4
        assert c["eval"].pop("density") == "full"
        c["data"].pop("density_root"), c["data"].pop("density_index")
        for k in ("experiment", "description"):
            c.pop(k), c0.get(k)
        assert {k: v for k, v in c.items()} == {k: v for k, v in c0.items() if k not in ("experiment", "description")}


def test_alpha3_2_36k_only_differs_from_alpha3_2_by_schedule():
    with open(CFG3["mix"]) as f:
        c0 = yaml.safe_load(f)
    with open(os.path.join(os.path.dirname(CFG3["mix"]), "alpha3_2_36k.yaml")) as f:
        c = yaml.safe_load(f)
    t, t0 = c.pop("training"), c0.pop("training")
    assert t.pop("max_iter") == 3 * t0.pop("max_iter")
    assert t.pop("steps") == [3 * s for s in t0.pop("steps")]
    assert t == t0
    for k in ("experiment", "description"):
        c.pop(k), c0.pop(k)
    assert c == c0


def test_beta0_config_only_differs_from_alpha0_by_beta_keys():
    with open(CFG0) as f:
        c0 = yaml.safe_load(f)
    with open(CFG_B0) as f:
        c = yaml.safe_load(f)
    assert c["data"].pop("targets") == "point" and c["data"].pop("points").endswith("density_points.json")
    assert set(c["data"].pop("pseudo_size")) == {"knn", "beta", "min_frac", "max_frac"}
    assert c["loss"].pop("center_weight") == c0["loss"]["l1_weight"] and "size_weight" in c["loss"]
    c["loss"].pop("size_weight")
    assert c["eval"].pop("select_metric") == "oracle_recall_pt"
    c0["eval"].pop("select_metric")
    for k in ("experiment", "description"):
        c.pop(k), c0.pop(k)
    assert c == c0


@pytest.mark.parametrize("mode", ["full", "mix"])
def test_full_flow_density_train_resume_eval(tmp_path, monkeypatch, mode):
    """ALPHA3.1 / 3.2: train 2 -> --resume tới 4 == train liền 4 (mix phải tái lập qua resume);
    eval full / partial / empty ghi đúng điều kiện; model 3 kênh + --density bị từ chối."""
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root)
    _fake_density(str(tmp_path))
    cfg_path, _ = _test_cfg(tmp_path, "none", root, density=mode)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", a, "--max-iter", "2"])
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", a, "--resume"])
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", b])
    ka = torch.load(os.path.join(a, "last.pth"), weights_only=False)
    kb = torch.load(os.path.join(b, "last.pth"), weights_only=False)
    assert ka["iter"] == kb["iter"] == 4
    for k in kb["model"]:
        assert torch.allclose(ka["model"][k].float(), kb["model"][k].float(), atol=1e-5), k
    w = kb["model"]["backbone.stem.0.weight"]
    assert w.shape[1] == 4 and w[:, 3].abs().max() > 0          # kênh density có gradient, đã học
    ev = [h for h in kb["history"] if "eval" in h]
    assert ev and ev[-1]["density_weight_ratio"] > 0

    import ce_localization.eval as ea
    import ce_localization.train as ta
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    for cond in (None, "partial", "empty"):
        out = str(tmp_path / f"res_{cond}.json")
        argv = ["eval.py", "--ckpt", os.path.join(a, "best.pth"), "--split", "test", "--nms",
                "--steps", "1", "--batch-size", "2", "--num-workers", "0", "--out", out, "--device", "cpu"]
        monkeypatch.setattr(sys, "argv", argv + ([] if cond is None else ["--density", cond]))
        ea.main()
        with open(out) as f:
            res = json.load(f)
        assert res["density"] == (cond or "full") and sum(res["density_kinds"].values()) == 2
        assert res["density_weight_ratio"] > 0 and set(res["results"]) == {"steps1", "steps1_nmsfirst"}
    with open(str(tmp_path / "res_partial.json")) as f:
        assert json.load(f)["density_kinds"] == {"full": 1, "partial": 1}   # ảnh đầu test chỉ 1 bản

    p0, _ = _test_cfg(tmp_path, "none", root)
    c3 = str(tmp_path / "c3")
    _run_train(monkeypatch, ["--config", p0, "--save-dir", c3, "--max-iter", "2"])
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", os.path.join(c3, "best.pth"),
                                      "--density", "full", "--device", "cpu", "--num-workers", "0"])
    with pytest.raises(SystemExit):
        ea.main()


def test_nan_debug_pinpoints_nan_input_channel(tmp_path, monkeypatch, capsys):
    """--nan-debug: NaN ở kênh density -> báo đúng kênh 3 + module đầu tiên (conv1) rồi DỪNG."""
    import ce_localization.train as ta
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root)
    _fake_density(str(tmp_path))
    cfg_path, _ = _test_cfg(tmp_path, "none", root, density="full")
    orig = CE130Dataset.__getitem__

    def poisoned(self, i):
        s = orig(self, i)
        s["image"][3, 0, 0] = float("nan")
        return s
    monkeypatch.setattr(CE130Dataset, "__getitem__", poisoned)
    with pytest.raises(SystemExit, match="NaN đầu tiên"):
        _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", str(tmp_path / "d"), "--nan-debug"])
    out = capsys.readouterr().out
    assert "ảnh kênh 3: không hữu hạn 2/" in out and "ảnh kênh 0: không hữu hạn 0/" in out
    assert "module ĐẦU TIÊN ra không hữu hạn: backbone.stem.0 (Conv2d) | đầu vào hữu hạn [False]" in out


def test_full_flow_beta0_train_resume_eval_and_gt_never_in_loss(tmp_path, monkeypatch):
    """G0 -> train 2 iter -> --resume tới 4 == train liền 4 ; eval ghi chỉ số điểm. Đổi box GT của
    ảnh train (giữ điểm) -> weight sau train Y HỆT: box GT không vào loss."""
    base = str(tmp_path)
    root = os.path.join(base, "all_phase2_V2")
    _fake_ce130(root)
    _fake_density(base)
    pts = str(tmp_path / "density_points.json")
    _run_g0(monkeypatch, base, pts, str(tmp_path / "rep.json"))
    cfg_path = _beta_cfg(tmp_path, root, pts)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", a, "--max-iter", "2"])
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", a, "--resume"])
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", b])
    ka = torch.load(os.path.join(a, "last.pth"), weights_only=False)
    kb = torch.load(os.path.join(b, "last.pth"), weights_only=False)
    assert ka["iter"] == kb["iter"] == 4
    for k in kb["model"]:
        assert torch.allclose(ka["model"][k].float(), kb["model"][k].float(), atol=1e-5), k
    assert "oracle_recall_pt" in kb["best"] and kb["config"]["data"]["targets"] == "point"

    for br in os.listdir(os.path.join(root, "train")):                        # box GT train khác hẳn
        p = os.path.join(root, "train", br, "annotation.json")
        with open(p) as f:
            ann = json.load(f)
        ann["all_bboxes"] = [[1.0, 1.0, 9.0, 9.0]]
        with open(p, "w") as f:
            json.dump(ann, f)
    c = str(tmp_path / "c")
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", c])
    kc = torch.load(os.path.join(c, "last.pth"), weights_only=False)
    for k in kb["model"]:
        assert torch.equal(kb["model"][k], kc["model"][k]), k

    import ce_localization.eval as ea
    import ce_localization.train as ta
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    out = str(tmp_path / "res.json")
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", os.path.join(a, "best.pth"), "--split", "test",
                                      "--nms", "--steps", "1", "--batch-size", "2", "--num-workers", "0",
                                      "--out", out, "--device", "cpu"])
    ea.main()
    with open(out) as f:
        res = json.load(f)
    for key in ("steps1", "steps1_nmsfirst"):
        p = res["results"][key]["point"]
        assert 0 <= p["oracle_recall_pt"] <= 1 and 0 <= p["AP_pt"] <= 1
        assert set(p["by_density"]) == {"<=30 vật", "31-100 vật", ">100 vật"}


# ----------------------------------------------------------------------------- GAMMA (bài add)

@pytest.mark.parametrize("kind", ["density", "rgb", "refiner", "refiner_coords"])
def test_full_flow_gamma_train_resume_eval(tmp_path, monkeypatch, kind):
    """GAMMA0 / 0.1 / 1: chỉ mục (nhánh, lượt) -> train 2 iter -> --resume tới 4 == train liền 4 -> eval.py trên ảnh
    inpaint + ảnh gốc, kèm mốc prior (GAMMA1: DDIM 1 và 2 bước, kèm attention lên [t ; text ; vis])."""
    from tests.ce_localization.helpers import _fake_turn_index, _gamma_cfg
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    cfg_path, cfg = _gamma_cfg(tmp_path, base, kind)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", a, "--max-iter", "2"])
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", a, "--resume"])
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", b])
    ka = torch.load(os.path.join(a, "last.pth"), weights_only=False)
    kb = torch.load(os.path.join(b, "last.pth"), weights_only=False)
    assert ka["iter"] == kb["iter"] == 4 and sorted(os.listdir(a)) == ["best.pth", "history.json", "last.pth"]
    for k in kb["model"]:
        assert torch.allclose(ka["model"][k].float(), kb["model"][k].float(), atol=1e-5), k
    assert "mean_iou_any" in kb["best"] and kb["best"]["iter"] in (2, 4)
    ev = [h["eval"] for h in kb["history"] if "eval" in h]
    assert len(ev) == 2 and ev[0]["n"] == 10 and 0 <= ev[0]["mean_iou_any"] <= 1    # val giả: 4 + 4 + 2 lượt

    import ce_localization.eval as ea
    import ce_localization.train as ta
    from tests.ce_localization.helpers import _fake_text_table
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    out = str(tmp_path / "res.json")
    refiner = kind.startswith("refiner")
    if refiner:
        assert all(len(h["loss_per_stage"]) == 6 for h in kb["history"] if "loss" in h)
        assert all(len(e["attn"]) == 6 for e in ev)
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", os.path.join(a, "best.pth"), "--split", "test",
                                      "--n-samples", "5", "--num-workers", "0", "--out", out, "--device", "cpu",
                                      "--steps", "1", "2", "--add-samplers", "ddpm"])               # T giả = 20 < 100 bước mock
    ea.main()
    with open(out) as f:
        res = json.load(f)
    keys = [f"{i}_steps{s}" for i in ("inpainted", "original") for s in (1, 2)] if refiner else ["inpainted", "original"]
    assert set(res["results"]) == set(res["prior"]) == set(keys)
    ri, ro = res["results"][keys[0]], res["results"][keys[-1]]
    assert ri["n"] == ro["n"] > 0 and 0 <= ri["best_iou@5_any"] <= 1 and "best_iou@5_any" not in ro
    assert res["density"] == ({"inpainted": None, "original": None} if kind == "rgb" else
                              {"inpainted": "sample", "original": "full"})
    assert ri["best_iou@5_any"] >= ri["best_iou@5_latest"] and res["prior"][keys[0]]["n_cnll"] == ri["n_cnll"]
    assert ("attn" in ri) == refiner


def test_gamma_configs_only_differ_by_density():
    from tests.ce_localization.helpers import CFG_G
    with open(CFG_G["density"]) as f:
        c0 = yaml.safe_load(f)
    with open(CFG_G["rgb"]) as f:
        c1 = yaml.safe_load(f)
    assert c0["task"] == "add" and c0["model"]["arch"] == "box_policy"
    assert (c0["data"].pop("density"), c1["data"].pop("density")) == ("sample", None)
    assert (c0["model"].pop("in_channels"), c1["model"].pop("in_channels")) == (4, 3)
    for k in ("experiment", "description"):
        c0.pop(k), c1.pop(k)
    assert c0 == c1


def test_gamma1_configs_only_differ_by_box_token():
    from tests.ce_localization.helpers import CFG_G
    with open(CFG_G["refiner"]) as f:
        c1 = yaml.safe_load(f)
    with open(CFG_G["refiner_coords"]) as f:
        c11 = yaml.safe_load(f)
    assert (c1["model"].pop("box_token"), c11["model"].pop("box_token")) == ("roi", "coords")
    for k in ("experiment", "description"):
        c1.pop(k), c11.pop(k)
    assert c1 == c11


def test_gamma_resume_refuses_changed_task(tmp_path, monkeypatch):
    import ce_localization.train as ta
    saved = {"task": "add", "model": {}, "diffusion": {}, "matcher": {}, "data": {}, "loss": {}}
    assert "task" in ta.config_diff(saved, {**saved, "task": "detect"})[0]
    assert ta.config_diff(saved, dict(saved))[0] == []


def test_plot_denoise_trajectory_full_flow(tmp_path, monkeypatch):
    """Tool vẽ quỹ đạo: checkpoint giả khuôn của bài trên samples/ giả -> PNG + JSON; ca chọn tay và chọn tự động."""
    import ce_localization.tools.plot_denoise_trajectory as tool
    from tests.ce_localization.helpers import _fake_ce130_turns, _fake_paper_ckpt
    _, samples = _fake_ce130_turns(str(tmp_path / "d"))
    ck = str(tmp_path / "best_model.pth")
    _fake_paper_ckpt(ck, T=100)                                           # vòng mock của bài: 100 bước
    monkeypatch.setattr(tool, "encode_class_names",
                        lambda names, *a, **k: {n: torch.randn(512, generator=torch.Generator().manual_seed(len(n)))
                                                for n in names})
    out = str(tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["plot_denoise_trajectory.py", "--ckpt", ck, "--samples", samples,
                                      "--files", "train/images/1000_1.png", "test/images/3000_1.png:apple",
                                      "--snapshots-ddpm", "99", "50", "0", "--snapshots-mock", "99", "0",
                                      "--out", out, "--device", "cpu"])
    tool.main()
    files = set(os.listdir(out))
    assert files == {"ddpm_case0_apple.png", "ddpm_case1_apple.png", "mock_case0_apple.png", "mock_case1_apple.png",
                     "trajectories.json"}
    with open(os.path.join(out, "trajectories.json")) as f:
        res = json.load(f)
    assert len(res["cases"]) == 4 and res["style"] == "paper" and res["track"] == "xt"
    r = res["cases"][2]                                                   # ddpm, case1: ảnh lớp cup, text apple
    assert r["text"] == "apple" and r["gt_class"] == "cup" and r["steps"] == [99, 50, 0]
    assert len(r["boxes_canvas"]) == 3 and all(0 <= v <= 1 for v in r["iou"]) and len(r["center_dist_px"]) == 3
    out2 = str(tmp_path / "out2")                                         # text rỗng / bỏ hẳn text
    monkeypatch.setattr(sys, "argv", ["plot_denoise_trajectory.py", "--ckpt", ck, "--samples", samples, "--samplers", "mock",
                                      "--files", "test/images/3000_1.png:<empty>", "test/images/3000_1.png:<zero>:empty",
                                      "test/images/3000_1.png:<zero>:blank",
                                      "--snapshots-mock", "99", "0", "--out", out2, "--device", "cpu"])
    tool.main()
    assert set(os.listdir(out2)) == {"mock_case0_empty.png", "mock_case1_zero_dempty.png", "mock_case2_zero_dblank.png",
                                     "trajectories.json"}
    with open(os.path.join(out2, "trajectories.json")) as f:
        assert [r["density"] for r in json.load(f)["cases"]] == ["sample", "empty", "blank"]
    from ce_localization.data.density import build_index                 # ảnh gốc chưa xoá + density đầy đủ nhất
    didx = str(tmp_path / "density_index.json")
    with open(didx, "w") as f:
        json.dump(build_index(samples, workers=0, log=lambda *x: None), f)
    out3 = str(tmp_path / "out3")
    monkeypatch.setattr(sys, "argv", ["plot_denoise_trajectory.py", "--ckpt", ck, "--samples", samples, "--samplers", "mock",
                                      "--image", "original", "--ce130", os.path.join(str(tmp_path / "d"), "all_phase2_V2"),
                                      "--density-index", didx, "--files", "test/images/3000_1.png:cup",
                                      "test/images/3000_1.png:<empty>:blank", "--snapshots-mock", "99", "0",
                                      "--out", out3, "--device", "cpu"])
    tool.main()
    with open(os.path.join(out3, "trajectories.json")) as f:
        assert [r["density"] for r in json.load(f)["cases"]] == ["full", "blank"]
    ann = {"train": {"train/images/a_1.png": "egg", "train/images/b_1.png": "cup"},
           "test": {"test/images/c_1.png": "egg", "test/images/d_1.png": "egg", "test/images/e_1.png": "kiwi"}}
    cases = tool.pick_cases(ann, None, np.random.default_rng(0))
    assert [c["name"] for c in cases] == ["train", "test_same", "test_unseen", "train_text"]
    assert cases[0]["text"] == "egg" and cases[2]["text"] == "kiwi" and cases[3]["file"] == cases[0]["file"]


def test_plot_spatial_softmax_full_flow(tmp_path, monkeypatch):
    """Tool SpatialSoftmax: toạ độ đổi đúng trục (toạ độ đầu của bài là DỌC, ±1 = tâm ô đầu / cuối), chạy trọn trên
    checkpoint giả khuôn của bài + ảnh gốc giả -> PNG + JSON (hàng density full / blank)."""
    import ce_localization.tools.plot_spatial_softmax as tool
    from ce_localization.data.density import build_index
    from tests.ce_localization.helpers import _fake_ce130_turns, _fake_paper_ckpt
    px, py = tool.keypoint_pixels(np.array([[-1.0, 1.0], [1.0, -1.0]]), 16)        # (dọc, ngang)
    assert np.allclose(px, [496, 16]) and np.allclose(py, [16, 496])
    root, samples = _fake_ce130_turns(str(tmp_path / "d"))
    ck = str(tmp_path / "best_model.pth")
    _fake_paper_ckpt(ck)
    didx = str(tmp_path / "density_index.json")
    with open(didx, "w") as f:
        json.dump(build_index(samples, workers=0, log=lambda *x: None), f)
    out = str(tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["plot_spatial_softmax.py", "--ckpt", ck, "--samples", samples, "--ce130", root,
                                      "--density-index", didx, "--files", "test/images/3000_1.png", "--out", out])
    tool.main()
    assert set(os.listdir(out)) == {"original_3000_1.png", "spatial_softmax.json"}
    with open(os.path.join(out, "spatial_softmax.json")) as f:
        rows = json.load(f)
    assert [r["density"] for r in rows] == ["full", "blank"] and all(0 <= r["sharp_frac"] <= 1 for r in rows)


def test_plot_refiner_steps_full_flow(tmp_path, monkeypatch):
    """Tool soi GAMMA1: checkpoint box_refiner giả (config gamma1 thật) trên CE-130 giả -> mỗi ca một PNG 4×6 + JSON (hàng
    inpaint own / blank rồi gốc own / blank; 4 trạng thái sau mỗi bước DDIM, ô cuối = đầu ra; IoU chỉ ở ảnh inpaint).
    `--files` dò ngược file samples/ ra (split, nhánh, lượt) bằng hash; `--quota` tự chọn phần còn lại; mock 10 bước."""
    import ce_localization.tools.plot_refiner_steps as tool
    from ce_localization.data.density import build_index
    from ce_localization.models.detector import build_model
    from tests.ce_localization.helpers import CFG_G, _fake_ce130_turns
    root, samples = _fake_ce130_turns(str(tmp_path / "d"))
    with open(CFG_G["refiner"]) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["image_size"] = 128
    cfg["diffusion"]["num_timesteps"] = 20
    torch.manual_seed(0)
    ck = str(tmp_path / "gamma1" / "best.pth")
    os.makedirs(os.path.dirname(ck))
    torch.save({"config": cfg, "model": build_model(cfg, pretrained_backbone=False).state_dict(), "iter": 7}, ck)
    didx = str(tmp_path / "density_index.json")
    with open(didx, "w") as f:
        json.dump(build_index(samples, workers=0, log=lambda *x: None), f)
    monkeypatch.setattr(tool, "encode_class_names", lambda names, *a, **k: {n: torch.randn(512) for n in names})
    base = ["plot_refiner_steps.py", "--ckpt", ck, "--ce130", root, "--samples", samples, "--density-index", didx]

    # file samples/ của nhánh test 3000_b1 lượt 2 -> dò ngược đúng (split, nhánh, lượt)
    a = type("A", (), {"samples": samples, "ce130": root})()
    hit = None
    for p in sorted(os.listdir(os.path.join(samples, "test", "images"))):
        f = tool.find_turn(a, f"test/images/{p}")
        if f == ("test", "3000_b1", 2):
            hit = f"test/images/{p}"
    assert hit is not None
    out = str(tmp_path / "out")
    monkeypatch.setattr(sys, "argv", base + ["--files", hit, "--cases", "test/3001_b1", "--quota", "1", "1", "2",
                                             "--pool", "3", "--out", out])
    tool.main()
    with open(os.path.join(out, "refiner_steps.json")) as f:
        cases = json.load(f)
    assert [c["split"] for c in cases] == ["train", "val", "test", "test"]
    assert {(c["branch"], c["turn"]) for c in cases if c["split"] == "test"} == {("3000_b1", 2), ("3001_b1", 1)}
    assert set(os.listdir(out)) == {f"{c['split']}_{c['branch']}_t{c['turn']}.png" for c in cases} | {"refiner_steps.json"}
    rows = cases[0]["rows"]
    assert [(r["image"], r["density"]) for r in rows] == list(tool.ROWS)
    assert [s["t"] for s in rows[0]["states"]] == [14, 9, 4, 0] and len(rows[0]["path"]) == 5   # sau bước 1..4; ô cuối = ra
    assert all(("iou_hole" in r["states"][-1]) == (r["image"] == "inpainted") for r in rows)
    assert rows[0]["path"][0][1:] == rows[2]["path"][0][1:]                      # cùng seed ⇒ cùng nhiễu ban đầu

    out3 = str(tmp_path / "out3")
    monkeypatch.setattr(sys, "argv", base + ["--cases", "test/3000_b1", "--quota", "0", "0", "0", "--sampler", "mock",
                                             "--mock-steps", "10", "--show-x0", "--out", out3])
    tool.main()
    with open(os.path.join(out3, "refiner_steps.json")) as f:
        r0 = json.load(f)[0]["rows"][0]
    assert [s["after_step"] for s in r0["states"]] == [1, 4, 7, 10] and len(r0["path"]) == 11

    out4 = str(tmp_path / "out4")                                               # 6 tầng trong MỘT lượt ở t cao nhất
    monkeypatch.setattr(sys, "argv", base + ["--cases", "test/3000_b1", "--quota", "0", "0", "0", "--mode", "stages",
                                             "--out", out4])
    tool.main()
    with open(os.path.join(out4, "refiner_steps.json")) as f:
        r0 = json.load(f)[0]["rows"][0]
    assert [s["stage"] for s in r0["states"]] == [1, 2, 3, 4, 5, 6] and {s["t"] for s in r0["states"]} == {19}
    assert r0["path"][0][0] == "input t=19" and len(r0["path"]) == 7


def test_plot_refiner_steps_box_policy(tmp_path, monkeypatch):
    """Tool soi nhận CE-Loc pha 1 (box_policy r18_paper, config gamma2_celoc thật): vòng mock 100 bước, 4 trạng thái sau bước
    25 / 50 / 75 / 100 (ô cuối = đầu ra), SpatialSoftmax 512 kênh C5 có mask; DDPM cũng chạy; stages / ddim bị từ chối."""
    import ce_localization.tools.plot_refiner_steps as tool
    from ce_localization.data.density import build_index
    from ce_localization.models.detector import build_model
    from tests.ce_localization.helpers import CFG_G, _fake_ce130_turns
    root, samples = _fake_ce130_turns(str(tmp_path / "d"))
    with open(CFG_G["celoc2"]) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["image_size"] = 128
    cfg["model"]["pretrained_backbone"] = False
    torch.manual_seed(0)
    ck = str(tmp_path / "gamma2_celoc" / "best.pth")
    os.makedirs(os.path.dirname(ck))
    torch.save({"config": cfg, "model": build_model(cfg, pretrained_backbone=False).state_dict(), "iter": 9}, ck)
    didx = str(tmp_path / "density_index.json")
    with open(didx, "w") as f:
        json.dump(build_index(samples, workers=0, log=lambda *x: None), f)
    monkeypatch.setattr(tool, "encode_class_names", lambda names, *a, **k: {n: torch.randn(512) for n in names})
    base = ["plot_refiner_steps.py", "--ckpt", ck, "--ce130", root, "--samples", samples, "--density-index", didx,
            "--cases", "test/3000_b1:2", "--quota", "0", "0", "0"]
    out = str(tmp_path / "out")
    monkeypatch.setattr(sys, "argv", base + ["--show-x0", "--out", out])
    tool.main()
    with open(os.path.join(out, "refiner_steps.json")) as f:
        cases = json.load(f)
    assert set(os.listdir(out)) == {"test_3000_b1_t2.png", "refiner_steps.json"}
    rows = cases[0]["rows"]
    assert [(r["image"], r["density"]) for r in rows] == list(tool.ROWS)
    assert [s["after_step"] for s in rows[0]["states"]] == [25, 50, 75, 100] and len(rows[0]["path"]) == 101
    assert [s["t"] for s in rows[0]["states"]] == [74, 49, 24, 0]
    assert all(("iou_hole" in r["states"][-1]) == (r["image"] == "inpainted") for r in rows)
    assert rows[0]["path"][0][1:] == rows[2]["path"][0][1:]                      # cùng seed ⇒ cùng nhiễu ban đầu
    assert all(0 <= r["sharp_frac"] <= 1 for r in rows)
    out2 = str(tmp_path / "out2")
    monkeypatch.setattr(sys, "argv", base + ["--sampler", "ddpm", "--out", out2])
    tool.main()
    with open(os.path.join(out2, "refiner_steps.json")) as f:
        r0 = json.load(f)[0]["rows"][0]
    assert [s["after_step"] for s in r0["states"]] == [250, 500, 750, 1000] and len(r0["path"]) == 1001
    for bad in (["--mode", "stages"], ["--sampler", "ddim"]):
        monkeypatch.setattr(sys, "argv", base + bad + ["--out", str(tmp_path / "bad")])
        with pytest.raises(SystemExit):
            tool.main()

    # checkpoint CE-Loc gốc của bài (`model_state_dict`, CLIP trong checkpoint, canvas 512, SpatialSoftmax không mask)
    from tests.ce_localization.helpers import _fake_paper_ckpt
    pk = str(tmp_path / "paper" / "best_model.pth")
    os.makedirs(os.path.dirname(pk))
    _fake_paper_ckpt(pk, T=100)
    seen = []
    monkeypatch.setattr(tool, "encode_class_names",
                        lambda names, *a, **k: seen.append(k.get("state_dict")) or {n: torch.randn(512) for n in names})
    out5 = str(tmp_path / "out5")
    monkeypatch.setattr(sys, "argv", [base[0], "--ckpt", pk] + base[3:] + ["--name", "CE-Loc (paper)", "--out", out5])
    tool.main()
    with open(os.path.join(out5, "refiner_steps.json")) as f:
        r0 = json.load(f)[0]["rows"][0]
    assert [s_["after_step"] for s_ in r0["states"]] == [25, 50, 75, 100]
    assert seen and all(sd is not None and "text_model.final_layer_norm.weight" in sd for sd in seen)


def test_plot_refine_stages_full_flow(tmp_path, monkeypatch):
    """Tool soi refine (propose_refine): mỗi ca một PNG 3 hàng (t0 / ce / noise) × 9 cột + JSON 6 stage có attention; model có
    box vật (GAMMA3) nhận hàng `_nogeo`; `--scan` ghi scan.json; hàng `_nogeo` với model không box vật bị từ chối."""
    import ce_localization.tools.plot_refine_stages as tool
    import ce_localization.tools.plot_refiner_steps as steps_tool
    from ce_localization.data.density import build_index
    from ce_localization.models.detector import build_model
    from tests.ce_localization.helpers import _fake_turn_index, _gamma2_cfg
    base_d = str(tmp_path / "d")
    os.makedirs(base_d)
    root, samples, tidx, _, _ = _fake_turn_index(base_d)
    didx = str(tmp_path / "density_index.json")
    with open(didx, "w") as f:
        json.dump(build_index(samples, workers=0, log=lambda *x: None), f)
    fake_text = lambda names, *a, **k: {n: torch.randn(512) for n in names}  # noqa: E731
    monkeypatch.setattr(tool, "encode_class_names", fake_text)
    monkeypatch.setattr(steps_tool, "encode_class_names", fake_text)

    def ckpt(kind):
        _, cfg = _gamma2_cfg(tmp_path, base_d, kind)
        cfg["diffusion"]["proposer"]["num_timesteps"] = 100                     # vòng mock của CE-Loc = 100 bước
        torch.manual_seed(0)
        p = str(tmp_path / kind / "best.pth")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        torch.save({"config": cfg, "model": build_model(cfg, pretrained_backbone=False).state_dict(), "iter": 5}, p)
        return p
    common = ["--ce130", root, "--samples", samples, "--density-index", didx, "--turn-index", tidx, "--refine-t", "10"]
    pr = ckpt("pr")
    out = str(tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["x", "--ckpt", pr, *common, "--cases", "test/3000_b1:2", "--quota", "0", "0", "0",
                                      "--name", "CE-Loc (frozen) + Refiner", "--out", out])
    tool.main()
    with open(os.path.join(out, "refine_stages.json")) as f:
        case = json.load(f)[0]
    assert set(os.listdir(out)) == {"test_3000_b1_t2.png", "refine_stages.json"}
    assert [r["mode"] for r in case["rows"]] == ["t0", "ce", "noise"] and [r["t"] for r in case["rows"]] == [0, 10, 19]
    assert all(len(r["stages"]) == 6 and len(r["stages"][0]["attn"]) == 3 for r in case["rows"])
    assert case["rows"][0]["ce"] == case["rows"][1]["ce"] and "ce" not in case["rows"][2]     # cùng seed ⇒ cùng box CE-Loc
    monkeypatch.setattr(sys, "argv", ["x", "--ckpt", pr, *common, "--rows", "ce", "ce_nogeo", "--cases", "test/3000_b1:2",
                                      "--quota", "0", "0", "0", "--out", str(tmp_path / "bad")])
    with pytest.raises(SystemExit):
        tool.main()
    geo = ckpt("geo")
    out2 = str(tmp_path / "out2")
    monkeypatch.setattr(sys, "argv", ["x", "--ckpt", geo, *common, "--rows", "ce", "ce_nogeo", "--scan", "2", "--pick", "1",
                                      "--out", out2])
    tool.main()
    with open(os.path.join(out2, "scan.json")) as f:
        scan = json.load(f)
    assert 1 <= len(scan) <= 2 and all(set(c["iou_latest_any"]) == {"ce", "ce_nogeo"} for c in scan)


def test_eval_paper_checkpoint_on_ce130_add(tmp_path, monkeypatch):
    """eval.py nhận checkpoint CE-Loc gốc của bài (khuôn `model_state_dict`) + `--config` GAMMA0 -> eval bài add trên CE-130
    (ảnh inpaint + ảnh gốc, ddpm + mock), text bằng CLIP của checkpoint; đếm mẫu samples/train (bài đã train)."""
    import ce_localization.eval as ea
    import ce_localization.train as ta
    from tests.ce_localization.helpers import _fake_paper_ckpt, _fake_text_table, _fake_turn_index, _gamma_cfg
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    cfg_path, _ = _gamma_cfg(tmp_path, base, "density")
    ck = str(tmp_path / "best_model.pth")
    _fake_paper_ckpt(ck, T=100)
    seen_sd = []
    monkeypatch.setattr(ta, "build_text_table",
                        lambda names, cfg, dev, state_dict=None: seen_sd.append(state_dict) or _fake_text_table(names, cfg, dev))
    with pytest.raises(SystemExit):                                       # thiếu --config
        monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", ck, "--device", "cpu"])
        ea.main()
    for split, n_seen in (("test", "none"), ("val", "some")):
        out = str(tmp_path / f"paper_{split}.json")
        monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", ck, "--config", cfg_path, "--split", split, "--n-samples",
                                          "3", "--add-samplers", "ddpm", "mock", "--num-workers", "0", "--out", out,
                                          "--device", "cpu"])
        ea.main()
        with open(out) as f:
            res = json.load(f)
        assert set(res["results"]) == {f"{i}_{s}" for i in ("inpainted", "original") for s in ("ddpm", "mock")}
        r = res["results"]["inpainted_ddpm"]
        assert res["density"] == {"inpainted": "sample", "original": "full"} and str(res["iter"]).startswith("epoch")
        assert 0 <= r["best_iou@3_latest"] <= 1
        if n_seen == "none":                                               # test giả: toàn samples/test
            assert r["n_paper_train"] == 0 and "excl_paper_train" not in r
        else:                                                              # val giả: 2000 ở samples/train, 2001 ở samples/test
            assert 0 < r["n_paper_train"] < r["n"] and r["excl_paper_train"]["n"] == r["n"] - r["n_paper_train"]
    assert seen_sd and all(sd is not None and "text_model.final_layer_norm.weight" in sd for sd in seen_sd)



# ----------------------------------------------------------------------------- GAMMA2: CE-Loc pha 1 -> refine pha 2

def test_full_flow_gamma2_celoc_then_refine(tmp_path, monkeypatch):
    """Pha 1 (CE-Loc ResNet18 + SpatialSoftmax mask, split samples) train -> pha 2 GAMMA2 nạp CE-Loc pha 1 (đóng băng):
    train 2 iter + --resume tới 4 == train liền 4, CE-Loc không đổi; eval quét refine_t (none / t* / noise) trên cùng box CE-Loc;
    GAMMA2.1 (train chung) đổi CE-Loc; proposer_ckpt lệch ss_mask bị từ chối."""
    import ce_localization.eval as ea
    import ce_localization.train as ta
    from ce_localization.models.detector import build_model, load_proposer
    from tests.ce_localization.helpers import _fake_text_table, _fake_turn_index, _gamma2_cfg
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c1_path, _ = _gamma2_cfg(tmp_path, base, "celoc2")
    p1 = str(tmp_path / "celoc")
    _run_train(monkeypatch, ["--config", c1_path, "--save-dir", p1])
    ck1 = torch.load(os.path.join(p1, "best.pth"), weights_only=False)
    assert ck1["config"]["model"]["ss_mask"] and ck1["iter"] in (2, 4)
    assert ck1["config"]["training"]["amp"] and "scaler" in ck1          # amp: true trên CPU -> tự tắt, vẫn lưu GradScaler
    ev = [h["eval"] for h in ck1["history"] if "eval" in h]
    assert ev and ev[0]["n"] == len(ta.TurnIndex(os.path.join(base, "turn_index.json")).keys("val", "samples"))

    c2_path, _ = _gamma2_cfg(tmp_path, base, "pr")
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _run_train(monkeypatch, ["--config", c2_path, "--save-dir", a, "--max-iter", "2",
                             "--proposer-ckpt", os.path.join(p1, "best.pth")])
    _run_train(monkeypatch, ["--config", c2_path, "--save-dir", a, "--resume", "--proposer-ckpt", os.path.join(p1, "best.pth")])
    _run_train(monkeypatch, ["--config", c2_path, "--save-dir", b, "--proposer-ckpt", os.path.join(p1, "best.pth")])
    ka = torch.load(os.path.join(a, "last.pth"), weights_only=False)
    kb = torch.load(os.path.join(b, "last.pth"), weights_only=False)
    assert ka["iter"] == kb["iter"] == 4
    for k in kb["model"]:
        assert torch.allclose(ka["model"][k].float(), kb["model"][k].float(), atol=1e-5), k
    for k, v in ck1["model"].items():                                     # CE-Loc đóng băng: y pha 1
        assert torch.equal(kb["model"]["proposer." + k], v), k
    assert all(len(h["loss_per_stage"]) == 6 for h in kb["history"] if "loss" in h)

    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    out = str(tmp_path / "res.json")
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", os.path.join(a, "best.pth"), "--split", "test", "--n-samples", "3",
                                      "--refine-t", "none", "5", "noise", "--proposer-sampler", "ddpm",
                                      "--num-workers", "0", "--out", out, "--device", "cpu"])
    ea.main()
    with open(out) as f:
        res = json.load(f)
    assert set(res["results"]) == {f"{i}_{v}" for i in ("inpainted", "original") for v in ("ce", "t5", "noise")}
    r = res["results"]["inpainted_t5"]
    assert r["n"] == res["results"]["inpainted_ce"]["n"] > 0 and 0 <= r["best_iou@3_latest"] <= 1 and len(r["attn"]) == 6

    cj_path, _ = _gamma2_cfg(tmp_path, base, "pr_joint")
    j = str(tmp_path / "joint")
    _run_train(monkeypatch, ["--config", cj_path, "--save-dir", j, "--proposer-ckpt", os.path.join(p1, "best.pth")])
    kj = torch.load(os.path.join(j, "last.pth"), weights_only=False)
    assert any(not torch.equal(kj["model"]["proposer." + k], v) for k, v in ck1["model"].items() if v.is_floating_point())

    _, cfg2 = _gamma2_cfg(tmp_path, base, "pr")
    bad = dict(ck1, config={**ck1["config"], "model": {**ck1["config"]["model"], "ss_mask": False}})
    torch.save(bad, str(tmp_path / "bad.pth"))
    with pytest.raises(ValueError):
        load_proposer(build_model(cfg2, pretrained_backbone=False), str(tmp_path / "bad.pth"))


def test_gamma2_configs():
    """gamma2 / gamma2_1 chỉ khác freeze_proposer (+ tên); proposer của pha 2 khớp model của pha 1."""
    from tests.ce_localization.helpers import CFG_G
    cfg = {}
    for k in ("celoc2", "pr", "pr_joint"):
        with open(CFG_G[k]) as f:
            cfg[k] = yaml.safe_load(f)
    a, b = cfg["pr"], cfg["pr_joint"]
    assert (a["model"].pop("freeze_proposer"), b["model"].pop("freeze_proposer")) == (True, False)
    for k in ("experiment", "description"):
        a.pop(k), b.pop(k)
    assert a == b
    m1 = cfg["celoc2"]["model"]
    for k, v in a["model"]["proposer"].items():
        assert m1[k] == v, k
    assert {k: cfg["celoc2"]["diffusion"][k] for k in ("num_timesteps", "beta_start", "beta_end")} == a["diffusion"]["proposer"]
    assert cfg["celoc2"]["data"] == {k: v for k, v in a["data"].items()}


# ----------------------------------------------------------------------------- GAMMA3: GAMMA2 + FiLM theo box vật đang có

def test_gamma3_config_only_adds_geo():
    """gamma3 = gamma2 + `model.geo` / `geo_hidden` (+ tên) ⇒ GAMMA3 − GAMMA2 = đóng góp của box vật."""
    from tests.ce_localization.helpers import CFG_G
    with open(CFG_G["pr"]) as f:
        a = yaml.safe_load(f)
    with open(CFG_G["geo"]) as f:
        b = yaml.safe_load(f)
    assert (b["model"].pop("geo"), b["model"].pop("geo_hidden")) == (True, 256)
    for k in ("experiment", "description"):
        a.pop(k), b.pop(k)
    assert a == b


def test_gamma3_objects_never_contain_target_hole(tmp_path):
    """Box vật đưa vào geo (`objects` của mẫu ảnh inpaint) không chứa lỗ đích hay lỗ cũ — không rò đáp án."""
    from ce_localization.data.turns import CE130AddDataset
    from ce_localization.utils.box_ops_np import box_iou
    from tests.ce_localization.helpers import _fake_turn_index
    base = str(tmp_path / "d")
    os.makedirs(base)
    root, samples, _, index, _ = _fake_turn_index(base)
    for split in ("train", "val", "test"):
        ds = CE130AddDataset(index, root, samples, split, 128, style="paper", split_source="samples")
        for i in range(len(ds)):
            s = ds[i]
            assert len(s["objects"]) and box_iou(s["holes"].numpy(), s["objects"].numpy())[0].max() < 0.5


def test_full_flow_gamma3_geo_train_eval(tmp_path, monkeypatch):
    """GAMMA3: pha 1 -> train gamma3 (thu nhỏ) nạp CE-Loc pha 1, nhánh geo học (lớp cuối rời 0), CE-Loc không đổi; eval có
    thêm `_nogeo` cho mọi biến thể refine, cùng số mẫu."""
    import ce_localization.eval as ea
    import ce_localization.train as ta
    from tests.ce_localization.helpers import _fake_text_table, _fake_turn_index, _gamma2_cfg
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c1_path, _ = _gamma2_cfg(tmp_path, base, "celoc2")
    p1 = str(tmp_path / "celoc")
    _run_train(monkeypatch, ["--config", c1_path, "--save-dir", p1, "--max-iter", "2"])
    ck1 = torch.load(os.path.join(p1, "last.pth"), weights_only=False)
    c3_path, _ = _gamma2_cfg(tmp_path, base, "geo")
    g = str(tmp_path / "geo")
    _run_train(monkeypatch, ["--config", c3_path, "--save-dir", g, "--proposer-ckpt", os.path.join(p1, "last.pth")])
    k3 = torch.load(os.path.join(g, "last.pth"), weights_only=False)
    assert k3["iter"] == 4 and k3["config"]["model"]["geo"]
    assert all(k3["model"][f"head.geo_films.{i}.2.weight"].abs().sum() > 0 for i in range(6))
    for k, v in ck1["model"].items():
        assert torch.equal(k3["model"]["proposer." + k], v), k

    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    out = str(tmp_path / "res.json")
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", os.path.join(g, "best.pth"), "--split", "test", "--n-samples", "3",
                                      "--refine-t", "none", "5", "noise", "--proposer-sampler", "ddpm",
                                      "--num-workers", "0", "--out", out, "--device", "cpu"])
    ea.main()
    with open(out) as f:
        res = json.load(f)
    assert set(res["results"]) == {f"{i}_{v}" for i in ("inpainted", "original")
                                   for v in ("ce", "t5", "noise", "t5_nogeo", "noise_nogeo")}
    r = res["results"]
    assert r["inpainted_t5_nogeo"]["n"] == r["inpainted_t5"]["n"] > 0 and len(r["inpainted_t5_nogeo"]["attn"]) == 6


# ----------------------------------------------------------------------------- GAMMA3.1: attention tới vật kiểu Relation-DETR

def test_gamma3_1_config_only_adds_relation():
    from tests.ce_localization.helpers import CFG_G
    with open(CFG_G["pr"]) as f:
        a = yaml.safe_load(f)
    with open(CFG_G["rel"]) as f:
        b = yaml.safe_load(f)
    assert (b["model"].pop("relation"), b["model"].pop("relation_k"), b["model"].pop("relation_embed")) == (True, 32, 16)
    for k in ("experiment", "description"):
        a.pop(k), b.pop(k)
    assert a == b


def test_full_flow_gamma3_1_relation_train_eval(tmp_path, monkeypatch):
    """GAMMA3.1: pha 1 -> train gamma3_1 (thu nhỏ), CE-Loc không đổi, có weight attention tới vật; eval có `_nogeo` + `_norel`."""
    import ce_localization.eval as ea
    import ce_localization.train as ta
    from tests.ce_localization.helpers import _fake_text_table, _fake_turn_index, _gamma2_cfg
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c1_path, _ = _gamma2_cfg(tmp_path, base, "celoc2")
    p1 = str(tmp_path / "celoc")
    _run_train(monkeypatch, ["--config", c1_path, "--save-dir", p1, "--max-iter", "2"])
    ck1 = torch.load(os.path.join(p1, "last.pth"), weights_only=False)
    c_path, _ = _gamma2_cfg(tmp_path, base, "rel")
    g = str(tmp_path / "rel")
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", g, "--proposer-ckpt", os.path.join(p1, "last.pth")])
    k = torch.load(os.path.join(g, "last.pth"), weights_only=False)
    assert k["iter"] == 4 and k["config"]["model"]["relation"]
    assert "head.stages.0.rel_attn.in_proj_weight" in k["model"] and "head.relation.rel_embed.pos_proj.weight" in k["model"]
    for n, v in ck1["model"].items():
        assert torch.equal(k["model"]["proposer." + n], v), n

    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    out = str(tmp_path / "res.json")
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", os.path.join(g, "best.pth"), "--split", "test", "--n-samples", "3",
                                      "--refine-t", "none", "5", "noise", "--proposer-sampler", "ddpm",
                                      "--num-workers", "0", "--out", out, "--device", "cpu"])
    ea.main()
    with open(out) as f:
        res = json.load(f)
    assert set(res["results"]) == {f"{i}_{v}" for i in ("inpainted", "original")
                                   for v in ("ce", "t5", "noise", "t5_nogeo", "noise_nogeo", "t5_norel", "noise_norel")}
    r = res["results"]
    assert r["inpainted_t5_norel"]["n"] == r["inpainted_t5"]["n"] > 0


# ----------------------------------------------------------------------------- GAMMA4: CE-Loc + cross-attn tới box vật

def test_gamma4_config_only_adds_obj_attn():
    """gamma4 = gamma2_celoc + `model.obj_attn` / `obj_heads` / `obj_max` (+ tên) ⇒ gamma2_celoc là đối chứng."""
    from tests.ce_localization.helpers import CFG_G
    with open(CFG_G["celoc2"]) as f:
        a = yaml.safe_load(f)
    with open(CFG_G["obj"]) as f:
        b = yaml.safe_load(f)
    assert (b["model"].pop("obj_attn"), b["model"].pop("obj_heads"), b["model"].pop("obj_max")) == (True, 4, 300)
    for k in ("experiment", "description"):
        a.pop(k), b.pop(k)
    assert a == b


def test_full_flow_gamma4_train_resume_eval(tmp_path, monkeypatch):
    """GAMMA4 thu nhỏ: train 2 + --resume tới 4 == train liền 4; có weight cross-attn; eval có `_noobj` cho cả 2 loại ảnh."""
    import ce_localization.eval as ea
    import ce_localization.train as ta
    from tests.ce_localization.helpers import _fake_text_table, _fake_turn_index, _gamma2_cfg
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c_path, _ = _gamma2_cfg(tmp_path, base, "obj")
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", a, "--max-iter", "2"])
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", a, "--resume"])
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", b])
    ka = torch.load(os.path.join(a, "last.pth"), weights_only=False)
    kb = torch.load(os.path.join(b, "last.pth"), weights_only=False)
    assert ka["iter"] == kb["iter"] == 4 and kb["config"]["model"]["obj_attn"]
    assert any("noise_net.obj_xattn.5.attn.out_proj.weight" == k for k in kb["model"])
    for k in kb["model"]:
        assert torch.allclose(ka["model"][k].float(), kb["model"][k].float(), atol=1e-5), k
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    out = str(tmp_path / "res.json")
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", os.path.join(b, "best.pth"), "--split", "test", "--n-samples", "3",
                                      "--add-samplers", "ddpm", "--num-workers", "0", "--out", out, "--device", "cpu"])
    ea.main()
    with open(out) as f:
        res = json.load(f)
    assert set(res["results"]) == {"inpainted", "inpainted_noobj", "original", "original_noobj"}
    assert res["results"]["inpainted_noobj"]["n"] == res["results"]["inpainted"]["n"] > 0


def test_gamma4_1_config_only_swaps_proposer():
    """gamma4_1 = gamma2 với CE-Loc đề xuất = gamma4 (khối proposer thêm 3 khoá obj_*, proposer_ckpt -> gamma4) (+ tên)."""
    from tests.ce_localization.helpers import CFG_G
    with open(CFG_G["pr"]) as f:
        a = yaml.safe_load(f)
    with open(CFG_G["pr_obj"]) as f:
        b = yaml.safe_load(f)
    with open(CFG_G["obj"]) as f:
        g4 = yaml.safe_load(f)
    pb = b["model"]["proposer"]
    assert {k: pb[k] for k in ("obj_attn", "obj_heads", "obj_max")} == {k: g4["model"][k] for k in ("obj_attn", "obj_heads", "obj_max")}
    for k in ("obj_attn", "obj_heads", "obj_max"):
        pb.pop(k)
    assert b["init"].pop("proposer_ckpt").endswith("gamma4/best.pth")
    a["init"].pop("proposer_ckpt")
    for k in ("experiment", "description"):
        a.pop(k), b.pop(k)
    assert a == b


def test_full_flow_gamma4_then_refine(tmp_path, monkeypatch):
    """GAMMA4 (CE-Loc + box vật) thu nhỏ -> GAMMA4.1 nạp nó làm CE-Loc đề xuất (đóng băng, CE-Loc không đổi) -> eval quét refine_t."""
    import ce_localization.eval as ea
    import ce_localization.train as ta
    from tests.ce_localization.helpers import _fake_text_table, _fake_turn_index, _gamma2_cfg
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c4, _ = _gamma2_cfg(tmp_path, base, "obj")
    p4 = str(tmp_path / "g4")
    _run_train(monkeypatch, ["--config", c4, "--save-dir", p4, "--max-iter", "2"])
    ck4 = torch.load(os.path.join(p4, "last.pth"), weights_only=False)
    c41, _ = _gamma2_cfg(tmp_path, base, "pr_obj")
    g = str(tmp_path / "g41")
    _run_train(monkeypatch, ["--config", c41, "--save-dir", g, "--proposer-ckpt", os.path.join(p4, "last.pth")])
    k = torch.load(os.path.join(g, "last.pth"), weights_only=False)
    assert k["iter"] == 4
    for n, v in ck4["model"].items():
        assert torch.equal(k["model"]["proposer." + n], v), n
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    out = str(tmp_path / "res.json")
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", os.path.join(g, "best.pth"), "--split", "test", "--n-samples", "3",
                                      "--refine-t", "none", "5", "noise", "--proposer-sampler", "ddpm",
                                      "--num-workers", "0", "--out", out, "--device", "cpu"])
    ea.main()
    with open(out) as f:
        res = json.load(f)
    assert set(res["results"]) == {f"{i}_{v}" for i in ("inpainted", "original") for v in ("ce", "t5", "noise")}

"""TRỌN LUỒNG (test nặng, `--runslow`): train -> resume -> eval trên dữ liệu giả với config thật thu nhỏ (refiner / U-Net kích thước
thật, CLIP giả) — hai cách chọn best.pth (val `mean_iou_any` / loss train TB mỗi epoch) và DDP 2 tiến trình."""

import os
import socket
import sys

import numpy as np
import pytest
import torch

from tests.ce_localization.helpers import (_fake_cocount, _fake_load_clip, _fake_review, _fake_text_table, _fake_turn_index, _run_eval,
                                           _run_train, run_cfg)


def _same_state(a, b):
    ka, kb = (torch.load(os.path.join(x, "last.pth"), weights_only=False) for x in (a, b))
    assert ka["iter"] == kb["iter"] and set(ka["model"]) == set(kb["model"]) and "clip_sha" in kb["model"]
    for k in kb["model"]:
        assert torch.allclose(ka["model"][k].float(), kb["model"][k].float(), atol=1e-5), k
    return ka, kb


def test_flow_val_select_resume_eval(tmp_path, monkeypatch):
    """DELTA5.2 (box log + GIoU, val ddim): train 2 + resume tới 4 == train liền 4; history có thống kê refiner; best theo val. Eval
    CE-130 (inpaint + gốc, mock + ddim, `--ablation`, `--obj-size`) + CE-CoCount review5: đủ khoá, prior, dump C-NLL từng box khớp chỉ
    số `cnll`; CLIP khác bản lúc train -> từ chối."""
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c_path, cfg = run_cfg(tmp_path, base, "delta5_2")
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", a, "--max-iter", "2"])
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", a, "--resume"])
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", b])
    _, kb = _same_state(a, b)
    logs = [h for h in kb["history"] if "loss" in h]
    assert all({"loss_eps", "loss_aux", "loss_giou", "delta_abs", "refiner_gates"} <= set(h) for h in logs) and len(logs) == 4
    ev = [h for h in kb["history"] if "eval" in h]
    assert [h["iter"] for h in ev] == [2, 4] and kb["best"]["mean_iou_any"] == max(h["eval"]["mean_iou_any"] for h in ev)
    ck = os.path.join(b, "best.pth")
    out = str(tmp_path / "ce.json")
    res, dump = _run_eval(monkeypatch, ["--ckpt", ck, "--image", "inpainted", "original", "--add-samplers", "mock", "ddim",
                                        "--ddim-steps", "7", "--ablation", "--obj-size", "mean", "--out", out])
    keys = {f"{img}_{s}{v}" for img in ("inpainted", "original") for s in ("mock", "ddim") for v in ("", "_noobj", "_norefine")}
    assert set(res["results"]) == keys | {k + "_objsize" for k in keys} and set(dump["results"]) == set(res["results"])
    assert set(res["prior"]) == {"inpainted", "original", "inpainted_objsize", "original_objsize"}
    r = res["results"]["inpainted_mock"]
    assert r["n"] == len(dump["results"]["inpainted_mock"]) > 0 and "best_iou@3_latest" in r and "mean_iou_any" not in res["results"]["original_mock"]
    rows = [x for x in dump["results"]["inpainted_mock"] if "cnll" in x]
    assert len(rows) == r["n_cnll"] > 0 and len(rows[0]["cnll"]) == 3 and len(rows[0]["cnll_gt"]) == rows[0]["t"]
    assert np.isclose(np.mean([min(x["cnll"]) for x in rows]), r["cnll"], rtol=1e-3, atol=1e-3)
    cc = _fake_cocount(str(tmp_path / "cc"))
    rv = _fake_review(cc, str(tmp_path / "rv.jsonl"))
    res_c, _ = _run_eval(monkeypatch, ["--ckpt", ck, "--dataset", "cocount", "--cocount-root", cc, "--cocount-review", rv,
                                       "--cocount-min-obj", "5", "--obj-size", "mean", "exemplar", "--out", str(tmp_path / "cc.json")])
    assert set(res_c["results"]) == {"cocount", "cocount_objsize", "cocount_objsize_exemplar"} and res_c["cocount_files"] == 3
    assert not any("latest" in k for k in res_c["results"]["cocount"]) and res_c["results"]["cocount"]["n_cnll"] == 3
    with pytest.raises(RuntimeError, match="vân tay"):
        _run_eval(monkeypatch, ["--ckpt", ck, "--out", str(tmp_path / "x.json")], clip_seed=1)


def test_flow_train_loss_select_resume(tmp_path, monkeypatch):
    """DELTA7 (= DELTA6, giao thức CE-Loc gốc, + đích train = box positive ngẫu nhiên): không dựng / chạy val; mỗi epoch TRỌN một dòng loss train TB = TB đúng các bước (log mỗi bước), phần
    lẻ cuối không chấm; best.pth = epoch thấp nhất; resume GIỮA epoch == train liền với cache token CLIP + snapshot mỗi epoch (b: cache /
    snapshot không đổi train; rút đích positive theo (seed, iter, rank) nên resume vẫn khớp); snapshot nạp + eval được; eval --split
    test_all; file positive có id ngoài samples/train -> dừng."""
    import ce_localization.train as ta
    from ce_localization.tools import build_clip_cache as bc
    from tests.ce_localization.helpers import _patch_clip
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c_path, cfg = run_cfg(tmp_path, base, "delta7")
    assert cfg["data"]["train_target"] == "positive" and cfg["eval"]["split"] is None and cfg["eval"]["select_metric"] == "train_loss" and cfg["training"]["snapshot_every_epochs"] == 10
    c_snap, _ = run_cfg(tmp_path, base, "delta7", name="snap", snapshot_every_epochs=1)
    monkeypatch.setattr(ta, "run_eval", lambda *x, **k: (_ for _ in ()).throw(AssertionError("không được chạy val")))
    clip_dir = str(tmp_path / "clip")
    _patch_clip(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["build_clip_cache.py", "--config", c_path, "--turn-index", os.path.join(base, "turn_index.json"),
                                      "--samples", os.path.join(base, "samples"), "--data-root", os.path.join(base, "all_phase2_V2"),
                                      "--batch-size", "4", "--workers", "0", "--device", "cpu", "--out", clip_dir])
    bc.main()
    ipe = len(ta.TurnIndex(os.path.join(base, "turn_index.json")).keys("train")) // 6
    n_it = 2 * ipe + 1
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    run = lambda c, d, n, *x: _run_train(monkeypatch, ["--config", c, "--save-dir", d, "--batch-size", "6", "--max-iter", str(n), *x])  # noqa: E731
    run(c_path, a, ipe + 1)                                           # dừng giữa epoch 2
    run(c_path, a, n_it, "--resume")
    run(c_snap, b, n_it, "--clip-cache", clip_dir)
    ka, kb = _same_state(a, b)
    assert sorted(os.listdir(os.path.join(b, "snapshots"))) == ["ep001.pth", "ep002.pth"] and not os.path.exists(os.path.join(a, "snapshots"))
    sn = torch.load(os.path.join(b, "snapshots", "ep002.pth"), weights_only=False)
    assert (sn["iter"], sn["epoch"], set(sn)) == (2 * ipe, 2, {"model", "iter", "epoch", "config", "env"})
    ep = [h for h in kb["history"] if "epoch_train_loss" in h]
    steps = [h["loss"] for h in kb["history"] if "loss" in h]
    assert [h["iter"] for h in ep] == [ipe, 2 * ipe] and [h["epoch_steps"] for h in ep] == [ipe, ipe] and len(steps) == n_it
    for h, lo in zip(ep, (0, ipe)):
        assert np.isclose(h["epoch_train_loss"], np.mean(steps[lo:lo + ipe]), rtol=1e-6)
    i = int(np.argmin([h["epoch_train_loss"] for h in ep]))
    assert kb["best"] == {"iter": ep[i]["iter"], "epoch": ep[i]["epoch"], "train_loss": ep[i]["epoch_train_loss"]} and ka["best"] == kb["best"]
    assert torch.load(os.path.join(b, "best.pth"), weights_only=False)["iter"] == ep[i]["iter"]
    res, _ = _run_eval(monkeypatch, ["--ckpt", os.path.join(b, "snapshots", "ep002.pth"), "--split", "test_all", "--out",
                                     str(tmp_path / "r.json")])
    assert res["iter"] == 2 * ipe and res["split"] == "test_all" and res["results"]["inpainted"]["n"] == len(ta.TurnIndex(os.path.join(base, "turn_index.json")).keys("test_all"))
    bad = str(tmp_path / "bad.jsonl")
    with open(bad, "w") as f:
        f.write('{"id": "9999_1", "positive": [[0, 0, 5, 5]]}\n')
    with pytest.raises(SystemExit, match="9999_1"):
        run(c_path, str(tmp_path / "c"), 1, "--positive-file", bad)


@pytest.mark.parametrize("run", ["delta7_0_1", "delta7_0_2"])
def test_flow_positive_all_pair_resume(tmp_path, monkeypatch, run):
    """DELTA7.0.1 (mọi box / ảnh mỗi iter) / 7.0.2 (mỗi cặp (mẫu, box) một mẫu): train 2 + resume tới 4 == train liền 4; 7.0.2: dataset
    = mọi cặp (epoch theo cặp); checkpoint nạp + eval được."""
    import ce_localization.train as ta
    from tests.ce_localization.helpers import _fake_positives
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c_path, cfg = run_cfg(tmp_path, base, run)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", a, "--max-iter", "2"])
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", a, "--resume"])
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", b])
    _, kb = _same_state(a, b)
    logs = [h for h in kb["history"] if "loss" in h]
    assert len(logs) == 4 and all(np.isfinite(h["loss"]) for h in logs)
    n = len(ta.TurnIndex(os.path.join(base, "turn_index.json")).keys("train"))
    if run == "delta7_0_2":                        # 1001_b1: mỗi lượt t lỗ + 1 chỗ trống; mẫu khác: t lỗ
        idx = ta.TurnIndex(os.path.join(base, "turn_index.json"))
        pos = _fake_positives(base, cfg["data"]["positive_file"])
        n = sum(v["t"] + (os.path.splitext(os.path.basename(v["sample"]))[0] in pos) for k, v in idx.turns.items()
                if k in idx.keys("train"))
    ds_tr, ds_ev = ta.add_datasets(cfg)
    assert len(ds_tr) == n and ds_ev is None and ds_tr.positive_sampling == cfg["data"]["positive_sampling"]
    res, _ = _run_eval(monkeypatch, ["--ckpt", os.path.join(b, "last.pth"), "--split", "test_all", "--out", str(tmp_path / "r.json")])
    assert res["results"]["inpainted"]["n"] > 0


def _ddp_worker(rank, world, port, argv):
    import warnings
    import ce_localization.models.clip_refiner as cr
    import ce_localization.train as ta
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    warnings.filterwarnings("error", message="Grad strides do not match bucket view strides")
    ta.build_text_table = _fake_text_table
    cr.load_clip = _fake_load_clip
    sys.argv = ["train.py"] + argv + ["--device", "cpu"]
    ta.main()


def test_ddp_two_processes(tmp_path):
    """2 tiến trình gloo, DDP không find_unused + gradient_as_bucket_view (tham số thiếu grad / lệch stride -> lỗi); loss train TB của epoch
    cộng qua cả hai rank; chỉ rank 0 ghi. Config DELTA7.0.1 (mọi box đích / ảnh, `x0_index`)."""
    import torch.multiprocessing as tmp
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c_path, _ = run_cfg(tmp_path, base, "delta7_0_1")           # mọi box / ảnh: số hàng đích khác nhau giữa 2 rank
    save = str(tmp_path / "ddp")
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    tmp.spawn(_ddp_worker, args=(2, port, ["--config", c_path, "--save-dir", save, "--max-iter", "3", "--batch-size", "6"]), nprocs=2,
              join=True)
    ck = torch.load(os.path.join(save, "last.pth"), weights_only=False)
    ep = [h for h in ck["history"] if "epoch_train_loss" in h]
    assert ck["iter"] == 3 and ep[0]["epoch_steps"] == 6 and ck["best"]["iter"] == 3                # 3 bước × 2 rank
    assert all(np.isfinite(h["loss"]) and h["skipped"] == 0 for h in ck["history"] if "loss" in h)
    assert sorted(os.listdir(save)) == ["best.pth", "history.json", "last.pth"]

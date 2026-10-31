"""Finetune + predict trọn luồng (model tí hon, CPU): nạp checkpoint công bố giả (bỏ `drop_keys`, khoá thừa chỉ cảnh báo), dừng giữa
epoch -> `--resume` đúng chỗ, snapshot / history, predict CE-130 + CoCount; cộng dồn grad = batch lớn; CSANet: grad của g_loss không vào D;
`sync_grads` 2 tiến trình."""

import json
import os
import sys

import pytest
import torch
import torch.multiprocessing as mp

from baseline.data import train_items
from baseline.methods import load_model, make_trainer, split_batch
from baseline.models.csanet import _bce, gan_losses
from ce_localization.data.turns import TurnIndex
from tests.baseline.helpers import fake_ce130, fake_ckpt, fake_cocount, tiny_cfg


def _run(monkeypatch, mod, argv):
    monkeypatch.setattr(sys, "argv", [mod.__name__] + [str(a) for a in argv])
    mod.main()


@pytest.mark.parametrize("run", ["zeta1", "zeta2"])
def test_finetune_resume_predict(tmp_path, monkeypatch, run):
    """BootPlace: dừng giữa epoch -> resume -> hết 2 epoch -> predict CE-130 + CoCount. CSANet (vòng lặp chung đã test ở trên): 1 bước ->
    predict CE-130 từ `last.pth`."""
    import baseline.finetune as finetune
    import baseline.predict as predict
    root, samples, ti = fake_ce130(str(tmp_path / "data"))
    review = fake_cocount(str(tmp_path / "data" / "cocount"))
    cfg_path, cfg = tiny_cfg(tmp_path, run)
    init = fake_ckpt(cfg, tmp_path / "init.pth")
    w = tmp_path / "w"
    common = ["--config", cfg_path, "--save-dir", w, "--init", init, "--turn-index", ti, "--samples-root", samples, "--gt-root", root,
              "--device", "cpu"]
    _run(monkeypatch, finetune, common + ["--max-steps", 1])        # 2 bước / epoch: dừng giữa epoch 1
    st = torch.load(w / "last.pth", weights_only=False)
    assert (st["epoch"], st["step"], st["iter"]) == (0, 1, 1) and st["ep_acc"][1] == 1
    ckpt, sets = w / "last.pth", [("ce130", "inpainted", 4)]
    if cfg["method"] == "bootplace":
        m = load_model(cfg, init, drop_keys=cfg["finetune"]["drop_keys"])     # bỏ query_embed: Q vẫn lấy từ loc_proj
        assert m.num_queries == cfg["model"]["num_queries"]
        _run(monkeypatch, finetune, common + ["--resume"])
        st = torch.load(w / "last.pth", weights_only=False)
        hist = json.load(open(w / "history.json"))
        assert (st["epoch"], st["step"], st["iter"]) == (2, 0, 4) and [h["epoch"] for h in hist] == [1, 2] and hist[0]["steps_ok"] == 2 \
            and hist[0]["select_by"] == "loss"
        assert sorted(os.listdir(w / "snapshots")) == ["ep000.pth", "ep001.pth", "ep002.pth"] and (w / "best.pth").exists()
        ckpt, sets = w / "snapshots" / "ep002.pth", sets + [("cocount", "cocount_objsize_exemplar", 3)]
    data = ["--turn-index", ti, "--samples-root", samples, "--cocount-root", tmp_path / "data" / "cocount", "--cocount-review", review,
            "--device", "cpu"]
    K = cfg["predict"]["k"]
    for ds, key, n in sets:
        out = tmp_path / "o" / ds / "test.json"
        _run(monkeypatch, predict, ["--config", cfg_path, "--ckpt", ckpt, "--dataset", ds, "--out", out] + data)
        d = json.load(open(out))
        raw = [json.loads(x) for x in open(str(out).replace(".json", "_raw.jsonl"))]
        assert d["n_samples"] == K and d["results"][key]["n"] == n and all(len(r["boxes"]) == K for r in raw)


def _batch(cfg, tmp_path, n=2):
    root, samples, ti = fake_ce130(str(tmp_path / "data"))
    model = load_model(cfg, fake_ckpt(cfg, tmp_path / "init.pth"), train=True)
    tr = make_trainer(cfg, torch.device("cpu"), model)
    ds, collate = tr.dataset(train_items(TurnIndex(ti), samples, root))
    items = [ds[i] for i in range(n)]
    return tr, (collate(items) if collate else torch.utils.data.default_collate(items))


def test_bootplace_accumulation_equals_full_batch(tmp_path):
    _, cfg = tiny_cfg(tmp_path, "zeta1")
    cfg["finetune"]["train_scales"] = [64]                         # cùng cỡ (2 mẫu đầu cùng ảnh) ⇒ đệm như nhau ở mọi cách chia micro
    torch.manual_seed(0)
    tr, batch = _batch(cfg, tmp_path)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    grads = []
    for micro in (2, 1):
        for p in tr.params:
            p.grad = None
        ctx = tr.context(batch, 1)
        for chunk in split_batch(batch, micro):
            tr.backward(chunk, ctx, len(chunk) / len(batch), False, scaler)
        grads.append([p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in tr.params])
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_csanet_generator_loss_not_in_discriminator(tmp_path):
    _, cfg = tiny_cfg(tmp_path, "zeta2")
    tr, batch = _batch(cfg, tmp_path)
    tr.train()
    torch.manual_seed(1)
    tr.backward(batch, None, 1.0, False, torch.cuda.amp.GradScaler(enabled=False))
    got = [p.grad.clone() for p in tr.d_params]
    assert all(p.grad is not None for p in tr.g_params)
    torch.manual_seed(1)
    f = lambda k: batch[k].float() / 255  # noqa: E731
    _, d_loss, _ = gan_losses(tr.G, tr.D, f("bg"), f("fg"), f("msk"), batch["fg_bbox"], f("comp"), f("comp_msk"), batch["trans"])
    for a, b in zip(got, torch.autograd.grad(d_loss, tr.d_params)):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-7)


def test_csanet_bce_matches_torch_and_passes_nan():
    p, y = torch.tensor([1e-50, 0.2, 0.7, 1.0], dtype=torch.float64), torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(_bce(p, y), torch.nn.functional.binary_cross_entropy(p, y))
    assert torch.isnan(_bce(torch.tensor([float("nan")]), torch.ones(1)))         # F.binary_cross_entropy: device assert trên CUDA


def test_sync_grads_two_processes(tmp_path):
    from tests.baseline._dist_worker import sync_worker
    mp.spawn(sync_worker, args=(str(tmp_path / "store"),), nprocs=2, join=True)

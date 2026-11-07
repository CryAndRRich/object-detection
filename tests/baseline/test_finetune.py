"""Finetune + predict trọn luồng (model tí hon, CPU): nạp checkpoint công bố giả (bỏ `drop_keys`, khoá thừa chỉ cảnh báo), dừng giữa
epoch -> `--resume` đúng chỗ, snapshot / history, predict CE-130 + CoCount; cộng dồn grad = batch lớn; CSANet: grad của g_loss không vào D,
ảnh thật = ảnh ghép ở trans GT, gamma không train, eval mode, best = epoch cuối; BootPlace: box cảnh pixel khung 960×540 đổi được box;
`sync_grads` 2 tiến trình."""

import json
import os
import random
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


@pytest.mark.parametrize("run", ["zeta1", "zeta2", "zeta3"])
def test_finetune_resume_predict(tmp_path, monkeypatch, run):
    """BootPlace / HiddenObjects: dừng giữa epoch -> resume -> hết 2 epoch -> predict CE-130 + CoCount. CSANet (vòng lặp chung đã test ở trên): 1 bước ->
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
    sets = [("ce130", "inpainted", 4, w / "last.pth")]
    if cfg["method"] != "csanet":
        m = load_model(cfg, init, drop_keys=cfg["finetune"].get("drop_keys", ()))   # BootPlace: bỏ query_embed, Q vẫn lấy từ loc_proj
        assert m.num_queries == cfg["model"]["num_queries"]
        _run(monkeypatch, finetune, common + ["--resume"])
        st = torch.load(w / "last.pth", weights_only=False)
        hist = json.load(open(w / "history.json"))
        assert (st["epoch"], st["step"], st["iter"]) == (2, 0, 4) and [h["epoch"] for h in hist] == [1, 2] and hist[0]["steps_ok"] == 2 \
            and hist[0]["select_by"] == "loss"
        assert sorted(os.listdir(w / "snapshots")) == ["ep000.pth", "ep001.pth", "ep002.pth"] and (w / "best.pth").exists()
        if cfg["method"] == "hiddenobjects":                                     # checkpoint không mang backbone đóng băng
            assert not any(k.startswith("backbone.") for k in torch.load(w / "snapshots" / "ep002.pth", weights_only=False)["model"])
        # best.pth lồng {"model": {"model": sd}} (lỗi eval server 2026-11-02), snapshot một lớp — predict phải nạp được cả hai
        sets = [("ce130", "inpainted", 4, w / "best.pth"), ("cocount", "cocount_objsize_exemplar", 3, w / "snapshots" / "ep002.pth")]
    data = ["--turn-index", ti, "--samples-root", samples, "--cocount-root", tmp_path / "data" / "cocount", "--cocount-review", review,
            "--device", "cpu"]
    K = cfg["predict"]["k"]
    for ds, key, n, ckpt in sets:
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
    random.seed(0)                                                 # lật ngang của dataset
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
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-5)    # fp32 cộng theo thứ tự khác (đích [0, 1] ⇒ grad nhỏ, lệch ~2e-4 tương đối)


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
    comp, comp_msk = tr.real_composite(f("bg"), f("fg"), f("bmsk"), batch["fg_bbox"], batch["trans"])
    _, d_loss, _ = gan_losses(tr.G, tr.D, f("bg"), f("fg"), f("msk"), batch["fg_bbox"], comp, comp_msk, batch["trans"], blend_msk=f("bmsk"))
    for a, b in zip(got, torch.autograd.grad(d_loss, tr.d_params)):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-7)


def test_csanet_real_composite_gamma_eval_mode_best_last(tmp_path):
    """Ảnh thật = ảnh vật dán ở trans GT (mask phủ đúng lỗ trên ô S×S); gamma khối chú ý đứng yên + ngoài optimizer; `eval_mode` ⇒ G / D
    eval cả lúc train; best = epoch cuối."""
    _, cfg = tiny_cfg(tmp_path, "zeta2")
    tr, batch = _batch(cfg, tmp_path)
    tr.train()
    assert cfg["finetune"]["eval_mode"] and not tr.G.training and not tr.D.training and tr.best_last
    gam = [p for n, p in tr.G.named_parameters() if n.endswith(("gamma1", "gamma2"))]
    assert gam and all(not p.requires_grad for p in gam)
    opt_ids = {id(p) for o in tr.optimizers for g in o.param_groups for p in g["params"]}
    assert not any(id(p) in opt_ids for p in gam) and all(id(p) in opt_ids for p in tr.g_params + tr.d_params)
    f = lambda k: batch[k].float() / 255  # noqa: E731
    comp, m = tr.real_composite(f("bg"), f("fg"), f("msk"), batch["fg_bbox"], batch["trans"])
    S = cfg["model"]["img_size"]
    from baseline.methods.csanet import trans_to_box
    for i in range(len(m)):                                       # mask ghép ≈ box của trans GT (cùng công thức với suy luận)
        fx, fy, fw, fh = batch["fg_bbox"][i].tolist()
        x1, y1, x2, y2 = trans_to_box(batch["trans"][i], (S, S), (fw, fh))
        ys, xs = torch.nonzero(m[i, 0] > 0.5, as_tuple=True)
        assert abs(xs.min().item() - x1) <= 2 and abs(ys.min().item() - y1) <= 2 and abs(xs.max().item() + 1 - x2) <= 2 \
            and abs(ys.max().item() + 1 - y2) <= 2
    out = (m == 0).expand_as(comp)
    torch.testing.assert_close(comp[out], f("bg")[out])                  # ngoài vùng dán = nền
    # zeta2: dán bằng mask vật thật (nằm trong mask chữ nhật của G, nhỏ hơn hẳn); ảnh thật ngoài mask vật = nền
    assert cfg["finetune"]["blend_mask"] == "object"
    bm, rm = f("bmsk"), f("msk")
    assert (bm <= rm + 1e-6).all() and 0 < bm.sum() < 0.8 * rm.sum()
    comp, m = tr.real_composite(f("bg"), f("fg"), bm, batch["fg_bbox"], batch["trans"])
    out = (m == 0).expand_as(comp)
    torch.testing.assert_close(comp[out], f("bg")[out])
    assert (m > 0.5).sum() < 0.8 * (tr.real_composite(f("bg"), f("fg"), rm, batch["fg_bbox"], batch["trans"])[1] > 0.5).sum()
    tr.cfg["finetune"]["eval_mode"] = False
    tr.train()
    assert tr.G.training and tr.D.training


def test_bootplace_scene_boxes_change_boxes(tmp_path):
    """Box cảnh pixel khung 960×540 đi qua `cMLP` làm box ĐỔI theo query và theo cảnh (bản chuẩn hoá [0, 1] cho vô hướng ≈ 0 ⇒ 30 box
    trùng hệt — lỗi port cũ)."""
    from baseline.methods.bootplace import FRAME, patch_canvas, prep_images, resize_size, scene_boxes
    _, cfg = tiny_cfg(tmp_path, "zeta1")
    tr, batch = _batch(cfg, tmp_path)
    f = cfg["finetune"]
    assert all(s["size"] in [resize_size(FRAME, sc, f["train_max_size"]) for sc in f["train_scales"]] for s in batch)   # khung 960×540
    model = tr.model.eval()
    s = batch[0]
    x, mask = prep_images([s["image"]], [s["size"]], "cpu")
    p = patch_canvas([s["patches"]], "cpu")
    with torch.no_grad():
        a = model(x, mask, p, [s["scene"]])["pred_boxes"][0]
        b = model(x, mask, p, [s["scene"] * 0.5])["pred_boxes"][0]
    assert a.std(0).max() > 1e-3 and (a - b).abs().max() > 1e-4
    assert s["scene"].max() > 1.5 and s["scene"][:, 2].max() <= FRAME[0] + 1e-3          # pixel khung, không chuẩn hoá
    torch.testing.assert_close(scene_boxes([[0, 0, 50, 25]], (100, 50), 4), torch.tensor([[0.0, 0.0, 480.0, 270.0]]))
    # đích train KHÔNG đi qua `scene_boxes`: cxcywh chuẩn hoá như `pred_boxes` (bản 2026-11-07 sáng lẫn hai thứ ⇒ đích pixel, loss ~45k
    # đứng yên cả 10 epoch)
    from baseline.methods.bootplace import target_boxes
    torch.testing.assert_close(target_boxes([[0, 0, 50, 25]], (100, 50)), torch.tensor([[0.25, 0.25, 0.5, 0.5]]))
    for s in batch:
        assert len(s["boxes"]) and 0 <= s["boxes"].min() and s["boxes"].max() <= 1
    model.train()
    out = model(x, mask, p, [s["scene"]])
    box, mean, _ = tr.crit(out, [{"boxes": s["boxes"], "labels": torch.zeros(len(s["boxes"]), dtype=torch.int64)}])
    assert float(box + mean) < 200                                                       # 6 tầng × (5·L1 + 2·GIoU) + clip, đích [0, 1]


def test_csanet_bce_matches_torch_and_passes_nan():
    p, y = torch.tensor([1e-50, 0.2, 0.7, 1.0], dtype=torch.float64), torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(_bce(p, y), torch.nn.functional.binary_cross_entropy(p, y))
    assert torch.isnan(_bce(torch.tensor([float("nan")]), torch.ones(1)))         # F.binary_cross_entropy: device assert trên CUDA


def test_sync_grads_two_processes(tmp_path):
    from tests.baseline._dist_worker import sync_worker
    mp.spawn(sync_worker, args=(str(tmp_path / "store"),), nprocs=2, join=True)


def test_csanet_predict_batch_invariant(tmp_path):
    """`predict.batch_images` chỉ là nút bộ nhớ: nhiễu + ảnh vật mẫu seed theo mẫu, BN ở eval ⇒ cùng box ở mọi cỡ batch."""
    from baseline.data import load_samples
    from baseline.methods import predict
    _, cfg = tiny_cfg(tmp_path, "zeta2")
    root, samples, ti = fake_ce130(str(tmp_path / "data"))
    test, _ = load_samples("ce130", turn_index=ti, samples_root=samples)
    model = load_model(cfg, fake_ckpt(cfg, tmp_path / "init.pth"))
    rows = []
    for b in (1, 3):
        cfg["predict"]["batch_images"] = b
        rows.append(predict(cfg, model, test, seed=0))
    for a, b in zip(*rows):
        assert a["image_id"] == b["image_id"] and a["fg"] == b["fg"]
        torch.testing.assert_close(torch.tensor(a["boxes"]), torch.tensor(b["boxes"]), rtol=1e-4, atol=1e-3)

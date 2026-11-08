"""HiddenObjects (ZETA3): nạp checkpoint công bố (lstsq ra đúng nhúng lớp cũ), backbone đóng băng ở train(), loss batch hoá = vòng lặp
từng mẫu, hình học canvas khứ hồi, cache memmap = đọc PNG, cộng dồn grad, suy luận xếp theo điểm, khoá `class` của mẫu."""

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment

from baseline.data import load_samples, train_items
from ce_localization.data.cocount import load_review
from baseline.methods import load_model, make_trainer, split_batch
from baseline.methods import hiddenobjects as ho
from baseline.models.hiddenobjects import hungarian_loss, pairwise_giou, xywh_to_xyxy
from ce_localization.data.turns import TurnIndex, letterbox
from tests.baseline.helpers import HO_CLASSES, fake_ce130, fake_ckpt, fake_clip, tiny_cfg
from tests.fakes import fake_cocount_review


@pytest.fixture
def setup(tmp_path):
    _, cfg = tiny_cfg(tmp_path, "zeta3")
    root, samples, ti = fake_ce130(str(tmp_path / "data"))
    return cfg, fake_ckpt(cfg, tmp_path / "init.pth"), train_items(TurnIndex(ti), samples, root), (samples, ti)


def test_published_ckpt_text_proj_reproduces_class_embed(setup):
    cfg, init, _, _ = setup
    model = load_model(cfg, init)
    emb = torch.load(init, weights_only=False)["model"]["class_embed.weight"]
    for i, c in enumerate(HO_CLASSES):
        t = torch.nn.functional.normalize(fake_clip([c])[c], dim=0)
        torch.testing.assert_close(model.text_proj(t), emb[i], rtol=1e-4, atol=1e-4)
    zeta = {"model": {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}}   # checkpoint ZETA: nạp thẳng
    torch.save(zeta, init.parent / "zeta.pth")
    again = load_model(cfg, init.parent / "zeta.pth")
    torch.testing.assert_close(again.text_proj.weight, model.text_proj.weight)


def test_backbone_frozen_and_eval_in_train_mode(setup):
    cfg, init, items, _ = setup
    tr = make_trainer(cfg, torch.device("cpu"), load_model(cfg, init, train=True))
    ds, collate = tr.dataset(items)
    bn = tr.model.backbone.stage1[1]
    before = bn.running_mean.clone()
    tr.train()
    assert tr.model.training and not tr.model.backbone.training
    tr.backward(collate([ds[0], ds[1]]), None, 1.0, False, torch.cuda.amp.GradScaler(enabled=False))
    torch.testing.assert_close(bn.running_mean, before)
    assert all(p.grad is None for p in tr.model.backbone.parameters()) and all(p.grad is not None for p in tr.params)
    assert not any(k.startswith("backbone.") for k in tr.weights()["model"])


def _loop_loss(boxes, logits, targets):
    """Tham chiếu: từng mẫu, như `hungarian_loss` bản gốc (reward = 1, đích plausibility detach)."""
    out = []
    for b, lg, t in zip(boxes, logits, targets):
        l1 = (b[:, None] - t[None]).abs().sum(-1)
        g = pairwise_giou(xywh_to_xyxy(b), xywh_to_xyxy(t))
        p, q = linear_sum_assignment((l1 + 1 - g).detach().numpy())
        box = sum(5 * l1[i, j] + 2 * (1 - g[i, j]) for i, j in zip(p, q)) / len(p)
        out.append(box + 0.5 * ((torch.sigmoid(lg) - g.detach().max(1).values.clamp(0, 1)) ** 2).mean())
    return torch.stack(out)


def test_hungarian_loss_batched_matches_loop():
    g = torch.Generator().manual_seed(0)
    boxes = (torch.rand(3, 6, 4, generator=g) * 0.5).requires_grad_()
    logits = torch.randn(3, 6, generator=g, requires_grad=True)
    targets = [torch.rand(m, 4, generator=g) * 0.5 for m in (1, 2, 4)]
    got, _ = hungarian_loss(boxes, logits, targets)
    want = _loop_loss(boxes, logits, targets)
    torch.testing.assert_close(got, want)
    ga = torch.autograd.grad(got.sum(), (boxes, logits))
    gb = torch.autograd.grad(want.sum(), (boxes, logits))
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b)
    with pytest.raises(ValueError):
        hungarian_loss(boxes[:1], logits[:1], [torch.zeros(0, 4)])


def test_canvas_geometry_roundtrip_and_dataset_target(setup):
    cfg, init, items, _ = setup
    holes = np.array([[10.0, 20.0, 50.0, 41.0], [0.0, 0.0, 120.0, 160.0]])
    for wh in ((150, 100), (120, 160)):
        np.testing.assert_allclose(ho.to_orig(ho.to_canvas(holes, wh, 64), wh, 64), holes, atol=1e-4)
    tr = make_trainer(cfg, torch.device("cpu"), load_model(cfg, init, train=True))
    ds, _ = tr.dataset(items)
    it, x = items[1], ds[1]
    from PIL import Image
    np.testing.assert_array_equal(x["image"].permute(1, 2, 0).numpy(), letterbox(Image.open(it["image"]).convert("RGB"), 64)[0])
    torch.testing.assert_close(x["boxes"], ho.to_canvas(it["holes"], it["wh"], 64))
    assert len(x["boxes"]) == it["t"]


@pytest.mark.parametrize("procs", [1, 2])
def test_cache_equals_png_and_rebuilds_on_mismatch(setup, tmp_path, monkeypatch, procs):
    cfg, init, items, _ = setup
    d = tmp_path / "cache"
    path = ho.ensure_cache(items, str(d), 64, procs=procs, chunk=1)
    cached, plain = ho.HOTrain(items, {"apple": 0}, 64, path), ho.HOTrain(items, {"apple": 0}, 64)
    for i in range(len(items)):
        torch.testing.assert_close(cached[i]["image"], plain[i]["image"])
    assert ho.ensure_cache(items, str(d), 64, rank=1) == path                      # rank khác: meta khớp -> dùng luôn
    rev = items[::-1]
    p2 = ho.ensure_cache(rev, str(d), 64, procs=1)                                  # đổi thứ tự -> dựng lại
    torch.testing.assert_close(ho.HOTrain(rev, {"apple": 0}, 64, p2)[0]["image"], plain[len(items) - 1]["image"])
    monkeypatch.setattr(ho.shutil, "disk_usage", lambda p: type("U", (), {"free": 0})())
    assert ho.ensure_cache(items, str(tmp_path / "small"), 64) is None              # thiếu đĩa -> không cache
    assert ho.ensure_cache(items, str(tmp_path / "small"), 64, rank=1, wait_s=0.01) is None


def test_accumulation_equals_full_batch(setup):
    cfg, init, items, _ = setup
    torch.manual_seed(0)
    tr = make_trainer(cfg, torch.device("cpu"), load_model(cfg, init, train=True))
    ds, collate = tr.dataset(items)
    batch = collate([ds[i] for i in range(len(ds))])
    tr.train()
    grads = []
    for micro in (len(ds), 1):
        for p in tr.params:
            p.grad = None
        for chunk in split_batch(batch, micro):
            tr.backward(chunk, None, len(chunk["cls"]) / len(ds), False, torch.cuda.amp.GradScaler(enabled=False))
        grads.append([p.grad.clone() for p in tr.params])
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_predict_sorted_by_score_and_samples_have_class(setup, tmp_path):
    cfg, init, items, (samples_root, ti) = setup
    assert {it["class"] for it in items} == {"apple"}
    test, _ = load_samples("ce130", turn_index=ti, samples_root=samples_root)
    review = fake_cocount_review(str(tmp_path / "cc"))
    cc, _ = load_samples("cocount", cocount_root=str(tmp_path / "cc"), review=load_review(review), min_obj=0)
    assert {s["class"] for s in test} == {"cup"} and all(s["class"] for s in cc)
    rows = ho.predict(load_model(cfg, init), test, 5, cfg["predict"])
    assert [r["image_id"] for r in rows] == [s["image_id"] for s in test]
    for r in rows:
        assert len(r["boxes"]) == 5 and r["score"] == sorted(r["score"], reverse=True)

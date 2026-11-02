"""Baseline ZETA theo `cfg["method"]` (bootplace | csanet | hiddenobjects): nạp checkpoint, suy luận K box, trainer finetune, chia
micro-batch."""

import torch

from baseline.methods import bootplace, csanet, hiddenobjects
from baseline.models.hiddenobjects import lstsq_text_proj

__all__ = ["METHODS", "load_model", "predict", "make_trainer", "split_batch", "batch_len"]

METHODS = {"bootplace": bootplace, "csanet": csanet, "hiddenobjects": hiddenobjects}


def _load(module, sd, what, drop=()):
    """Nạp state dict: THIẾU khoá (trừ `drop`) = lỗi (sai kiến trúc); khoá THỪA (module bản gốc đã bỏ vì không dùng) = cảnh báo."""
    missing, unexpected = module.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k not in drop]
    if missing:
        raise RuntimeError(f"{what}: thiếu {len(missing)} khoá (vd {missing[:5]}) — sai kiến trúc / config?")
    if unexpected:
        print(f"[nạp] {what}: bỏ {len(unexpected)} khoá không dùng (vd {unexpected[:3]})", flush=True)


def model_state(ck):
    """Checkpoint -> state dict model. `best.pth` / `last.pth` của finetune lưu `{"model": Trainer.state_dict()}`, mà BootPlace /
    HiddenObjects tự bọc `{"model": sd}` ⇒ lồng hai lớp; snapshot và checkpoint công bố chỉ một lớp. CSANet: `{"generator", ...}`."""
    sd = ck["model"]
    return sd["model"] if isinstance(sd.get("model"), dict) else sd


def load_model(cfg, path, dev="cpu", drop_keys=(), train=False):
    """-> BootPlace: `BOOTPLACE`; CSANet: (G, D); HiddenObjects: `PlacementDETR` (`_load_hiddenobjects`). `path`: checkpoint công bố hay của ZETA (khoá `model`). `drop_keys`: bỏ trước khi nạp
    (khởi tạo lại). BootPlace: num_queries lấy từ checkpoint (`bbox_embed.loc_proj` [Q, 4Q]); `train` ⇒ train layer2–4 của ResNet."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = model_state(ck)
    if cfg["method"] == "hiddenobjects":
        return _load_hiddenobjects(cfg, ck, path, dev)
    if cfg["method"] == "bootplace":
        sd = {k: v for k, v in sd.items() if k not in drop_keys}
        q = int(sd["bbox_embed.loc_proj.0.weight"].shape[0])
        model = bootplace.build({**cfg, "model": {**cfg["model"], "num_queries": q}},
                                train_backbone=train and cfg["finetune"]["lr_backbone"] > 0)
        _load(model, sd, f"BootPlace {path}", drop_keys)
        return model.to(dev)
    G, D = csanet.build(cfg)
    _load(G, sd["generator"], f"CSANet G {path}")
    if "discriminator" in sd:
        _load(D, sd["discriminator"], f"CSANet D {path}")
    elif train:
        raise RuntimeError(f"{path}: finetune CSANet cần cả discriminator")
    return G.to(dev), D.to(dev)


def _load_hiddenobjects(cfg, ck, path, dev):
    """Checkpoint công bố (`class_embed` + `classes`): `text_proj` = nghiệm lstsq trên CLIP text của `classes` (ra đúng nhúng cũ ⇒ model
    như bản tác giả trên các lớp đó = hàng zero-shot). Checkpoint ZETA (`text_proj.*`): nạp thẳng. Q lấy từ `query_offsets`; backbone
    không có trong checkpoint (dựng từ torchvision)."""
    sd = dict(model_state(ck))
    model = hiddenobjects.build({**cfg, "model": {**cfg["model"], "num_queries": int(sd["query_offsets"].shape[0])}})
    if "text_proj.weight" not in sd:
        classes, emb = list(ck["classes"]), sd.pop("class_embed.weight")
        if len(classes) != len(emb):
            raise RuntimeError(f"{path}: {len(classes)} lớp nhưng class_embed {tuple(emb.shape)}")
        _, text = hiddenobjects.class_texts(classes, model.clip_text, "cpu")
        order = sorted(range(len(classes)), key=lambda i: classes[i])                  # class_texts sắp theo tên
        sd["text_proj.weight"], sd["text_proj.bias"] = lstsq_text_proj(text, emb[order])
        err = (text.double() @ sd["text_proj.weight"].T.double() + sd["text_proj.bias"].double() - emb[order].double()).abs().max().item()
        print(f"[nạp] HiddenObjects {path}: class_embed {len(classes)} lớp -> text_proj (lstsq, sai số max {err:.2e})", flush=True)
    _load(model, sd, f"HiddenObjects {path}", drop=tuple(k for k in model.state_dict() if k.startswith("backbone.")))
    return model.to(dev)


def predict(cfg, model, samples, dev="cpu", seed=0, log=print):
    p = cfg["predict"]
    m = model[0] if cfg["method"] == "csanet" else model
    return METHODS[cfg["method"]].predict(m, samples, p["k"], p, seed=seed, dev=dev, log=log)


def make_trainer(cfg, dev, model):
    return METHODS[cfg["method"]].Trainer(cfg, dev, model)


def batch_len(batch):
    return len(batch) if isinstance(batch, list) else len(next(iter(batch.values())))


def split_batch(batch, micro):
    """Batch / rank (list mẫu | dict tensor) -> các micro-batch <= `micro` mẫu."""
    n = batch_len(batch)
    return [batch[i:i + micro] if isinstance(batch, list) else {k: v[i:i + micro] for k, v in batch.items()} for i in range(0, n, micro)]

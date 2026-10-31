"""Baseline ZETA theo `cfg["method"]` (bootplace | csanet): nạp checkpoint, suy luận K box, trainer finetune, chia micro-batch."""

import torch

from baseline.methods import bootplace, csanet

__all__ = ["METHODS", "load_model", "predict", "make_trainer", "split_batch", "batch_len"]

METHODS = {"bootplace": bootplace, "csanet": csanet}


def _load(module, sd, what, drop=()):
    """Nạp state dict: THIẾU khoá (trừ `drop`) = lỗi (sai kiến trúc); khoá THỪA (module bản gốc đã bỏ vì không dùng) = cảnh báo."""
    missing, unexpected = module.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k not in drop]
    if missing:
        raise RuntimeError(f"{what}: thiếu {len(missing)} khoá (vd {missing[:5]}) — sai kiến trúc / config?")
    if unexpected:
        print(f"[nạp] {what}: bỏ {len(unexpected)} khoá không dùng (vd {unexpected[:3]})", flush=True)


def load_model(cfg, path, dev="cpu", drop_keys=(), train=False):
    """-> BootPlace: `BOOTPLACE`; CSANet: (G, D). `path`: checkpoint công bố hay của ZETA (khoá `model`). `drop_keys`: bỏ trước khi nạp
    (khởi tạo lại). BootPlace: num_queries lấy từ checkpoint (`bbox_embed.loc_proj` [Q, 4Q]); `train` ⇒ train layer2–4 của ResNet."""
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
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


def predict(cfg, model, samples, dev="cpu", seed=0, log=print):
    p = cfg["predict"]
    m = model if cfg["method"] == "bootplace" else model[0]
    return METHODS[cfg["method"]].predict(m, samples, p["k"], p, seed=seed, dev=dev, log=log)


def make_trainer(cfg, dev, model):
    return METHODS[cfg["method"]].Trainer(cfg, dev, model)


def batch_len(batch):
    return len(batch) if isinstance(batch, list) else len(next(iter(batch.values())))


def split_batch(batch, micro):
    """Batch / rank (list mẫu | dict tensor) -> các micro-batch <= `micro` mẫu."""
    n = batch_len(batch)
    return [batch[i:i + micro] if isinstance(batch, list) else {k: v[i:i + micro] for k, v in batch.items()} for i in range(0, n, micro)]

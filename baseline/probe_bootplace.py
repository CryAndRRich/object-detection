#!/usr/bin/env python3
"""Chẩn đoán BootPlace (ZETA1, `docs/EXPERIMENT_ZETA.md` mục 11.3–11.4): 30 box trùng + xác suất lớp hằng trên MỌI ảnh ⇒ đầu ra decoder
`hs` gần như không phụ thuộc ảnh / query. Chạy CPU (vài giây / ảnh), không ghi gì, in:

1. `args` của checkpoint (pre_norm, is_mask, số tầng …) + khoá checkpoint THỪA / THIẾU / LỆCH CỠ so với port — vd
   `transformer.encoder.norm.*` chỉ có khi train pre-norm (port là post-norm ⇒ khoá đó bị bỏ, model chạy sai kiến trúc).
2. Qua N ảnh test (đầu vào y hệt `predict`, `bootplace.sample_inputs`), từng tầng: độ lớn RMS, độ lệch chuẩn THEO TOKEN (vị trí ảnh /
   query) và THEO ẢNH (cùng vị trí / query, khác ảnh) — chia RMS; cos TB giữa các query; entropy chú ý cross-attn decoder (1 = trải đều
   mọi vị trí ảnh ⇒ không nhìn ảnh); vô hướng box cảnh (`loc_proj`); xác suất lớp 0 + box theo query / ảnh.

  cd /mnt/disk1/aiotlab/haitn/object-detection/baseline
  python probe_bootplace.py --config config/zeta1.yaml --dataset ce130 --limit 8 \\
      --ckpt ../weights/add/zeta1/pretrained/bootplace_cityscapes.pth ../weights/add/zeta1/snapshots/ep010.pth
"""

import argparse
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from baseline.methods import load_model, model_state  # noqa: E402
from baseline.methods.bootplace import sample_inputs  # noqa: E402
from baseline.score import add_data_args, load_data  # noqa: E402
from ce_localization.utils.config import load_config  # noqa: E402

ARGS = ("epoch", "pre_norm", "is_mask", "enc_layers", "dec_layers", "hidden_dim", "nheads", "dim_feedforward", "dropout", "num_queries",
        "backbone", "dilation", "position_embedding", "aux_loss", "eos_coef", "ce_loss_coef", "bbox_loss_coef", "giou_loss_coef", "lr",
        "lr_backbone", "batch_size", "epochs", "resume", "masks")


def log(*a):
    print(*a, flush=True)


def ckpt_report(path, model):
    """In `args` + khoá thừa / thiếu / lệch cỡ của checkpoint so với `model` (port)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    log(f"  khoá gốc checkpoint: {sorted(ck)}" + (f" | epoch {ck['epoch']}" if "epoch" in ck else ""))
    a = ck.get("args")
    if a is None:
        log("  không có `args`")
    else:
        d = vars(a) if hasattr(a, "__dict__") else dict(a)
        log("  args: " + " | ".join(f"{k}={d[k]!r}" for k in ARGS if k in d))
        log(f"  args khác: {sorted(k for k in d if k not in ARGS)}")
    sd, own = model_state(ck), model.state_dict()
    extra = [k for k in sd if k not in own]
    missing = [k for k in own if k not in sd]
    shape = [k for k in sd if k in own and tuple(sd[k].shape) != tuple(own[k].shape)]
    group = lambda ks: dict(Counter(".".join(k.split(".")[:3]) for k in ks))  # noqa: E731
    log(f"  khoá: checkpoint {len(sd)} | port {len(own)} | THỪA {len(extra)} {group(extra)} | THIẾU {len(missing)} {group(missing)} | "
        f"LỆCH CỠ {[(k, tuple(sd[k].shape), tuple(own[k].shape)) for k in shape]}")
    bn = [k for k in sd if k.startswith("backbone.") and k.endswith("running_var")]
    if bn:
        v = torch.cat([sd[k].flatten().float() for k in bn])
        log(f"  BN backbone: {len(bn)} lớp running_var, TB {v.mean():.4f} (toàn 1 = chưa có thống kê thật)")


def spread(xs):
    """xs: list [N, D] (mỗi ảnh một tensor, cùng N). -> (rms, std theo token / rms, std theo ảnh / rms)."""
    x = torch.stack(xs).float()                                           # [I, N, D]
    rms = x.pow(2).mean().sqrt().item()
    tok = x.std(1, unbiased=False).mean().item() if x.shape[1] > 1 else float("nan")
    img = x.std(0, unbiased=False).mean().item() if x.shape[0] > 1 else float("nan")
    return rms, tok / max(rms, 1e-12), img / max(rms, 1e-12)


def query_cos(xs):
    """cos TB giữa các cặp query khác nhau (TB trên ảnh); 1 = mọi query như nhau."""
    out = []
    for x in xs:
        z = torch.nn.functional.normalize(x.float(), dim=-1)
        c = z @ z.T
        n = len(c)
        out.append(((c.sum() - c.diagonal().sum()) / (n * n - n)).item())
    return sum(out) / len(out)


@torch.no_grad()
def probe(model, samples, p, seed):
    model.eval()
    rec = {"proj": [], "enc": [], "dec": [], "attn": [], "loc": []}
    tr = model.transformer
    hooks = [model.input_proj.register_forward_hook(lambda m, i, o: rec["proj"].append(o[0].flatten(1).T)),
             tr.decoder.norm.register_forward_hook(lambda m, i, o: rec["dec"].append(o[:, 0])),
             model.bbox_embed.loc_proj.register_forward_hook(lambda m, i, o: rec["loc"].append(o.flatten()[0].item()))]
    hooks += [l.register_forward_hook(lambda m, i, o: rec["enc"].append(o[:, 0])) for l in tr.encoder.layers]
    hooks += [l.cross_attn.register_forward_hook(lambda m, i, o: rec["attn"].append(o[1][0])) for l in tr.decoder.layers]
    prob, box = [], []
    try:
        for s in samples:
            args, _, _ = sample_inputs(model, s, p, seed)
            o = model(*args)
            prob.append(o["pred_logits"][0].softmax(-1)[:, 0])
            box.append(o["pred_boxes"][0])
    finally:
        for h in hooks:
            h.remove()
    I, Le, Ld = len(samples), len(tr.encoder.layers), len(tr.decoder.layers)
    log(f"  {'tầng':14s} {'RMS':>9s} {'std/token':>10s} {'std/ảnh':>9s}   (/RMS; token = vị trí ảnh hoặc query)")
    log("  {:14s} {:9.4f} {:10.4f} {:9.4f}".format("backbone→proj", *spread(rec["proj"])))
    for j in range(Le):
        log("  {:14s} {:9.4f} {:10.4f} {:9.4f}".format(f"encoder {j + 1}", *spread(rec["enc"][j::Le])))
    for j in range(Ld):
        xs = rec["dec"][j::Ld]
        a = torch.stack(rec["attn"][j::Ld]).float().clamp_min(1e-12)       # [I, Q, HW]
        ent = (-(a * a.log()).sum(-1) / math.log(a.shape[-1])).mean().item()
        log("  {:14s} {:9.4f} {:10.4f} {:9.4f}".format(f"decoder {j + 1}", *spread(xs))
            + f" | cos query {query_cos(xs):.4f} | cross-attn entropy {ent:.3f}, max {a.max(-1).values.mean():.4f}")
    P, B = torch.stack(prob), torch.stack(box)                             # [I, Q], [I, Q, 4]
    log(f"  vô hướng box cảnh (loc_proj): {[round(v, 4) for v in rec['loc']]}")
    log(f"  xác suất lớp 0: min {P.min():.5f} max {P.max():.5f} | std theo query {P.std(1, unbiased=False).mean():.2e} | "
        f"std theo ảnh {P.std(0, unbiased=False).mean():.2e}")
    log(f"  box cxcywh: TB {B.mean((0, 1)).tolist()} | std theo query {B.std(1, unbiased=False).mean():.2e} | "
        f"std theo ảnh {B.std(0, unbiased=False).mean():.2e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    add_data_args(ap)
    a = ap.parse_args(argv)
    if a.limit is None:
        a.limit = 8
    torch.set_grad_enabled(False)
    cfg = load_config(a.config)
    samples, _, _ = load_data(a)
    log(f"[probe] {a.dataset}: {len(samples)} ảnh | CPU | {cfg['predict']}")
    for path in a.ckpt:
        log(f"\n=== {path}")
        model = load_model(cfg, path, "cpu")
        ckpt_report(path, model)
        probe(model, samples, cfg["predict"], a.seed)


if __name__ == "__main__":
    main()

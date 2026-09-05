#!/usr/bin/env python3
"""Pre-flight check — run this BEFORE committing to a long training run.

Two training attempts died in the val loop after ~7 minutes each, both on things
a 30-second check would have caught. This script walks every code path training
will take, on the REAL data and the REAL cache, but touching each of them only
briefly.

The failures it is built to catch are the ones that only appear at scale:

  * a device mismatch that only exists on GPU (invisible on a CPU dev box)
  * an image_id or class present in the dataset but MISSING from the cache
    (a KeyError that could first fire on image 1500, an hour in)
  * cache metadata that disagrees with the model (wrong token count, wrong dim)
  * an OOM that only shows up once the largest image in the split is reached
  * the val loop specifically, since that is what died both times

  python3 tools/run_on_free_gpu.py -- tools/preflight.py --cache ../../data/cache_clip
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection, PatchCache, normalize_for_clip  # noqa: E402
from models.criterion import SetCriterion  # noqa: E402
from models.detector import CELocDetector  # noqa: E402

OK, BAD = "[ok]", "[FAIL]"
failures = []


def check(name, fn):
    """Run one check, report, and keep going so every problem surfaces at once."""
    t0 = time.time()
    try:
        detail = fn()
        print(f"  {OK}   {name:52s} {detail}  ({time.time()-t0:.1f}s)", flush=True)
        return True
    except Exception as e:                                    # noqa: BLE001
        failures.append((name, f"{type(e).__name__}: {e}"))
        print(f"  {BAD} {name:52s} {type(e).__name__}: {e}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    B = a.batch_size or cfg["training"]["batch_size"]
    N = cfg["diffusion"]["num_proposals_train"]

    print("=" * 78)
    print(f"  PRE-FLIGHT — device={dev} batch={B} N={N} cache={a.cache}")
    print("=" * 78, flush=True)

    # ---------------------------------------------------------------- datasets
    print("\n[1/6] datasets", flush=True)
    ds = {}
    for split in ["train", "val"]:
        check(f"load {split}",
              lambda s=split: (ds.__setitem__(s, CE130Detection(
                  cfg["data"]["root"], s, cfg["data"]["image_size"],
                  cfg["data"]["flip_prob"] if s == "train" else 0.0,
                  seed=cfg["training"]["seed"])), ds[s].stats())[1])
    if len(ds) < 2:
        print("\nCannot continue without both splits."); return _finish()

    # ------------------------------------------------------------------ cache
    print("\n[2/6] cache completeness  (a KeyError here would fire mid-training)",
          flush=True)
    caches = {}
    if a.cache:
        for split in ["train", "val"]:
            def _cache(s=split):
                c = PatchCache(a.cache, s)
                caches[s] = c
                missing_ids = [it["image_id"] for it in ds[s].items
                               if it["image_id"] not in c.idx_image]
                missing_cls = sorted({it["text"] for it in ds[s].items
                                      if it["text"] not in c.idx_class})
                if missing_ids:
                    raise KeyError(f"{len(missing_ids)} image_id absent from cache, "
                                   f"e.g. {missing_ids[:3]}")
                if missing_cls:
                    raise KeyError(f"{len(missing_cls)} class(es) absent from cache: "
                                   f"{missing_cls[:5]}")
                return (f"{len(ds[s].items)} images / {len(c.idx_class)} classes "
                        f"all present, {c.n_ver} versions")
            check(f"{split}: every image_id + class is cached", _cache)

        def _shape():
            c = caches["train"]
            _, _, n_tok, d = c.meta["shape"]
            grid = cfg["data"]["image_size"] // 16          # ViT-B/16
            if n_tok != grid * grid:
                raise ValueError(f"cache has {n_tok} tokens, model expects {grid*grid} "
                                 f"at image_size={cfg['data']['image_size']}")
            if d != 768:
                raise ValueError(f"cache dim {d} != 768 (ViT-B hidden size)")
            if c.meta.get("clip") != cfg["model"]["clip_name"]:
                raise ValueError(f"cache built with {c.meta.get('clip')} but config "
                                 f"says {cfg['model']['clip_name']}")
            return f"{n_tok} tokens x {d} dim, {c.meta.get('clip')}"
        check("cache shape/model agree with the config", _shape)
    else:
        print("  (skipped: no --cache given)")

    # ------------------------------------------------------------------ model
    print("\n[3/6] model", flush=True)
    model = {}

    def _build():
        m = CELocDetector(
            cfg["model"]["clip_name"], cfg["model"]["d_model"], cfg["model"]["n_layer"],
            cfg["model"]["n_head"], cfg["data"]["image_size"],
            cfg["diffusion"]["num_timesteps"], cfg["diffusion"]["snr_scale"],
            cfg["diffusion"]["sampling_steps"], cfg["model"]["dropout"],
            cfg["model"]["freeze_clip"]).to(dev)
        model["m"] = m
        tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return f"{tr/1e6:.2f}M trainable / {sum(p.numel() for p in m.parameters())/1e6:.1f}M"
    if not check("build + move to device", _build):
        return _finish()
    m = model["m"]
    crit = SetCriterion(cfg["matcher"]["method"])

    def _frozen():
        bad = [n for n, p in m.named_parameters()
               if (n.startswith("encoder.vision.") or n.startswith("encoder.text."))
               and p.requires_grad]
        if bad:
            raise ValueError(f"{len(bad)} CLIP params still trainable, e.g. {bad[:2]}")
        return "CLIP fully frozen"
    check("CLIP is frozen (else ~19 GB backward -> OOM)", _frozen)

    # ------------------------------------------- one real train step, per split
    print("\n[4/6] one real step on the LARGEST images of each split", flush=True)

    def _batch(split, idxs):
        """Build a batch exactly the way TorchWrap+collate does."""
        samples = [ds[split][i] for i in idxs]
        tg = [torch.from_numpy(s["boxes"]).float().to(dev) for s in samples]
        vh = [s["valid_h"] for s in samples]
        if split in caches:
            c = caches[split]
            pr, tr_ = zip(*[c.get(s["image_id"], s["text"], s["flipped"]) for s in samples])
            kw = {"patch_raw": torch.from_numpy(np.stack(pr)).to(dev),
                  "text_raw": torch.from_numpy(np.stack(tr_)).to(dev)}
        else:
            kw = {"pixel_values": torch.stack(
                      [torch.from_numpy(normalize_for_clip(s["image"])) for s in samples]).to(dev),
                  "texts": [s["text"] for s in samples]}
        return tg, vh, kw

    for split in ["train", "val"]:
        # the worst case for memory is the image with the most boxes
        order = sorted(range(len(ds[split])),
                       key=lambda i: -len(ds[split].items[i]["boxes_xyxy_px"]))
        idxs = order[:B]
        n_max = len(ds[split].items[order[0]]["boxes_xyxy_px"])

        def _step(sp=split, ix=idxs, nm=n_max):
            tg, vh, kw = _batch(sp, ix)
            x_t, tt, _ = m.build_inputs(tg, N, vh)
            pb, lg = m(x_t, tt, **kw)
            loss, st, _ = crit(pb, lg, tg)
            if not torch.isfinite(loss):
                raise ValueError(f"loss is not finite: {loss}")
            loss.backward()
            m.zero_grad(set_to_none=True)
            return (f"max {nm} GT/img, loss {float(loss):.3f}, "
                    f"matched {st['n_matched']}, IoU {st['iou_matched']:.3f}")
        check(f"{split}: forward+backward on the {B} biggest images", _step)

    # ------------------------------------------------ the val loop that crashed
    print("\n[5/6] the val loop (this is what died twice)", flush=True)

    def _val():
        tg, vh, kw = _batch("val", list(range(B)))
        m.eval()                        # as run_val does; without it dropout stays
                                        # active and nothing is reproducible
        g = torch.Generator(device=dev).manual_seed(1234)   # exactly as run_val does
        with torch.no_grad():
            x_t, tt, _ = m.build_inputs(tg, N, vh, generator=g)
            pb, lg = m(x_t, tt, **kw)
            _, st, _ = crit(pb, lg, tg)
        return f"val step OK, loss {st['loss']:.3f}, seeded generator on {dev.type}"
    check("seeded generator through build_inputs (device match)", _val)

    def _repeat():
        tg, vh, kw = _batch("val", list(range(B)))
        m.eval()                        # dropout off, else this can never pass
        out = []
        for _ in range(2):
            g = torch.Generator(device=dev).manual_seed(1234)
            with torch.no_grad():
                x_t, tt, _ = m.build_inputs(tg, N, vh, generator=g)
                pb, lg = m(x_t, tt, **kw)
                out.append(crit(pb, lg, tg)[1]["loss"])
        if abs(out[0] - out[1]) > 1e-4:
            raise ValueError(f"same seed gave {out[0]:.6f} vs {out[1]:.6f} — val loss "
                             f"will not be comparable across epochs")
        return f"reproducible: {out[0]:.4f} twice"
    check("same seed -> same val loss", _repeat)

    def _ddim():
        # keep `texts` if present: ddim_sample needs either cached text_raw OR the
        # raw strings, and dropping both makes the CLIP tokenizer raise
        _, _, kw = _batch("val", list(range(min(2, B))))
        m.eval()
        with torch.no_grad():
            b, lg = m.ddim_sample(cfg["diffusion"]["num_proposals_eval"], **kw)
        if not (torch.isfinite(b).all() and (b >= 0).all() and (b <= 1).all()):
            raise ValueError(f"boxes out of range: [{b.min():.3f}, {b.max():.3f}]")
        return f"eval N={cfg['diffusion']['num_proposals_eval']} -> {tuple(b.shape)} in [0,1]"
    check("ddim_sample at eval N (used by eval.py)", _ddim)

    # ------------------------------------------------------- memory + writability
    print("\n[6/6] resources", flush=True)
    if dev.type == "cuda":
        def _mem():
            gb = torch.cuda.max_memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gb > 0.6 * total:
                raise MemoryError(f"peak {gb:.1f} GB of {total:.0f} GB — too close to OOM")
            return f"peak {gb:.2f} GB / {total:.0f} GB (worst-case images, batch {B})"
        check("GPU memory headroom", _mem)

    def _writable():
        d = cfg["training"]["save_dir"]
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, ".preflight")
        with open(p, "w") as f:
            json.dump({"ok": True}, f)
        os.remove(p)
        return d
    check("save_dir is writable", _writable)

    return _finish()


def _finish():
    print("\n" + "=" * 78)
    if failures:
        print(f"NOT READY — {len(failures)} check(s) failed:")
        for n, e in failures:
            print(f"  - {n}\n      {e}")
        print("\nFix these before starting a long run.")
        print("=" * 78)
        return 1
    print("ALL CHECKS PASSED — safe to start training.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

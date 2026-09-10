#!/usr/bin/env python3
"""ONE command to run before committing GPU hours. Nothing here needs a GPU.

    python3 tools/check_before_train.py

It runs three things in order, cheapest first, and stops at the first failure:

  1. the test suite                     (~2 min, CPU)
  2. one real training step per config   (~1 min each, CPU, real data)
  3. a cross-config comparison           (instant)

WHY A SEPARATE SCRIPT FROM tools/preflight.py
  preflight runs ON THE SERVER, needs the GPU, and answers "will this run survive
  the next 4 hours". This runs on the LAPTOP before pushing, and answers "is this
  the experiment I think it is". They overlap deliberately: the cheap one should
  catch what it can before the expensive one is even reachable.

STEP 2 EXISTS BECAUSE UNIT TESTS MISSED A REAL BUG. Every dataset test passed
while training died immediately on `KeyError: 'labels'` in collate -- the dataset
returned the key, the wrapper did not forward it, and nothing exercised the two
together. Only an actual step through DataLoader -> collate -> model -> loss ->
backward covers that seam, so that is what this does.
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIGS = ["config/experiment_a.yaml", "config/experiment_b.yaml",
           "config/experiment_a1.yaml", "config/experiment_a2.yaml",
           "config/experiment_c1.yaml", "config/experiment_c1b.yaml",
           "config/experiment_c1c.yaml", "config/experiment_e1.yaml"]

failures = []


def head(title):
    print("\n" + "=" * 78, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 78, flush=True)


def report(name, ok, detail=""):
    print(f"  {'[ok]  ' if ok else '[FAIL]'} {name:46s} {detail}", flush=True)
    if not ok:
        failures.append(name)
    return ok


def run_tests(python):
    head("1/3  test suite")
    t0 = time.time()
    r = subprocess.run([python, "-m", "pytest", "tests/", "-q", "--no-header"],
                       capture_output=True, text=True)
    tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-1:]
    ok = r.returncode == 0
    report("pytest tests/", ok, f"{tail[0] if tail else ''}  ({time.time()-t0:.0f}s)")
    if not ok:
        for line in r.stdout.splitlines():
            if line.startswith("FAILED") or line.startswith("ERROR"):
                print(f"        {line}", flush=True)
    return ok


def one_step(cfg_path):
    """A real batch through DataLoader -> collate -> model -> loss -> backward.

    Deliberately NOT a hand-built tensor: the bug this catches lived in the seam
    between the dataset and the training loop, which a synthetic batch skips.
    """
    import numpy as np
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from data.factory import build_dataset
    from models.criterion import SetCriterion, loss_from_output
    from models.detector import build_model
    from train import TorchWrap, collate, run_val

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ds = build_dataset(cfg, "train", cfg["data"]["flip_prob"], seed=0)
    n_total = len(ds)
    ds.items = ds.items[:4]
    loader = DataLoader(TorchWrap(ds), batch_size=2, collate_fn=collate, num_workers=0)
    batch = next(iter(loader))
    for k in ("boxes", "labels", "text", "valid_h", "image_id"):
        if k not in batch:
            raise KeyError(f"collate dropped {k!r}")

    model = build_model(cfg)
    model.train()
    tg = [b for b in batch["boxes"]]
    tl = [l for l in batch["labels"]]
    for b_, l_ in zip(tg, tl):
        if b_.shape[0] != l_.shape[0]:
            raise ValueError(f"labels {l_.shape} not aligned with boxes {b_.shape}")

    N = cfg["diffusion"]["num_proposals_train"]
    x_t, tt, _ = model.build_inputs(tg, N, batch["valid_h"])
    out = model(x_t, tt, pixel_values=batch["pixel_values"], texts=batch["text"])

    # C1 returns a LIST of (boxes, logits), one per refinement round.
    rounds = cfg["model"].get("refine_rounds", 0)
    if rounds:
        if not isinstance(out, list) or len(out) != rounds:
            raise ValueError(f"refine_rounds={rounds} but forward returned "
                             f"{type(out).__name__} of len "
                             f"{len(out) if isinstance(out, list) else 'n/a'}")
        pairs = out
    else:
        if isinstance(out, list):
            raise ValueError("model returned rounds but refine_rounds=0 in config")
        pairs = [out]

    n_class = cfg["model"].get("n_class", 1)
    want = (2, N) if n_class == 1 else (2, N, n_class)
    for r, (pb, lg) in enumerate(pairs):
        if tuple(lg.shape) != want:
            raise ValueError(f"round {r}: logits {tuple(lg.shape)}, expected {want}")
        if not (torch.isfinite(pb).all() and (pb >= 0).all() and (pb <= 1).all()):
            raise ValueError(f"round {r}: boxes outside [0,1]: "
                             f"[{pb.min():.3f}, {pb.max():.3f}]")
    pb, lg = pairs[-1]

    crit = SetCriterion(cfg["matcher"]["method"])
    loss, st, _, lg = loss_from_output(crit, out, tg, labels=tl)
    if not torch.isfinite(loss):
        raise ValueError(f"loss is {loss}")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads:
        raise ValueError("backward produced no gradients")
    if not all(torch.isfinite(g).all() for g in grads):
        raise ValueError("non-finite gradient")

    # Sanity on the class pathway, cheap and it catches a mis-wired config.
    use_text = cfg["model"].get("use_text", True)
    if not use_text and model.encoder.text is not None:
        raise ValueError("use_text=false but the text tower was built")

    # A REAL VALIDATION PASS, for the same reason step 2 runs a real training step.
    # run_val is a separate seam: it calls the model under `no_grad`, reads the
    # per-round stats, and converts logits with `.numpy()`. None of that is touched
    # by the training step above. It went untested once and E1 died three minutes
    # into its first GPU run -- `run_val` had lost its @torch.no_grad() to a
    # function inserted above it, so `.numpy()` hit a tensor carrying a graph.
    # Called with grad ENABLED on purpose: that is the state train.py is in, and a
    # missing decorator only shows up from there.
    with torch.enable_grad():
        v = run_val(model, loader, crit, N, torch.device("cpu"))
    for k in ("loss", "iou_matched", "oracle_recall_final", "score"):
        if k not in v:
            raise KeyError(f"run_val dropped {k!r}")
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # `n_matched` is an int for A/B but a MEAN over rounds for C1 -- format it as a
    # float so a valid C1 run does not fail the check with "Unknown format code 'd'".
    return (f"n={n_total:,} loss={float(loss):8.2f} matched={float(st['n_matched']):5.1f} "
            f"iou={st['iou_matched']:.3f} params={trainable/1e6:.2f}M "
            f"grads={len(grads)} val_orec={v['oracle_recall_final']:.3f}")


def compile_and_import_tools(python):
    """Import every tool with its module-level code executed.

    `py_compile` (in the test suite) catches SyntaxError but NOT NameError: a name
    used inside a function is only resolved when that function RUNS. preflight.py
    shipped with `tl` referenced in four nested callbacks that never saw it, passed
    every local check, and failed on the server -- which is the one place these
    tools are ever run.

    Importing is still not execution, so the guarantee is limited; the real fix is
    the smoke run below, which executes preflight end to end on tiny data.
    """
    head("2b/3  tools import cleanly")
    import importlib
    import glob
    ok_all = True
    for f in sorted(glob.glob("tools/*.py")):
        mod = "tools." + os.path.basename(f)[:-3]
        try:
            importlib.import_module(mod)
            report(os.path.basename(f), True)
        except Exception as e:                                    # noqa: BLE001
            report(os.path.basename(f), False, f"{type(e).__name__}: {e}")
            ok_all = False
    return ok_all


def smoke_preflight(python, cfg_path):
    """Run tools/preflight.py itself, on a handful of samples, on the CPU.

    preflight is what guards the GPU budget, so it is the LAST thing that should
    be discovered broken on the server. Running it here on --limit-sized data
    executes every one of its callbacks, which is what actually catches a
    NameError inside them.
    """
    r = subprocess.run(
        [python, "tools/preflight.py", "--config", cfg_path,
         "--device", "cpu", "--batch-size", "2", "--limit", "8"],
        capture_output=True, text=True)
    fails = [l.strip() for l in r.stdout.splitlines() if "[FAIL]" in l]
    return r.returncode == 0, (fails[0][:70] if fails else
                               ("all checks passed" if r.returncode == 0
                                else (r.stderr.strip().splitlines() or ["?"])[-1][:70]))


def compare_configs():
    """The comparisons only mean something if the runs differ where intended and
    nowhere else. Checked here rather than trusted to review."""
    import yaml

    head("3/3  cross-config comparison")
    cfgs = {}
    for p in CONFIGS:
        with open(p) as f:
            cfgs[os.path.basename(p)] = yaml.safe_load(f)
    a, b = cfgs["experiment_a.yaml"], cfgs["experiment_b.yaml"]
    a1, a2 = cfgs["experiment_a1.yaml"], cfgs["experiment_a2.yaml"]
    c1 = cfgs.get("experiment_c1.yaml")
    c1b, c1c = cfgs.get("experiment_c1b.yaml"), cfgs.get("experiment_c1c.yaml")
    D = {"n_class": 1, "use_text": True, "roi_k": 0, "refine_rounds": 0}

    def model_diff(x, y):
        return {k for k in set(x["model"]) | set(y["model"])
                if x["model"].get(k, D.get(k)) != y["model"].get(k, D.get(k))}

    report("B differs from A only in roi_k", model_diff(a, b) == {"roi_k"},
           str(model_diff(a, b)))
    report("A.1 model identical to A", model_diff(a, a1) == set(),
           str(model_diff(a, a1)) or "identical")
    report("A.2 differs from A.1 only in the class pathway",
           model_diff(a1, a2) == {"n_class", "use_text"}, str(model_diff(a1, a2)))
    if c1:
        report("C1 differs from A only in refine_rounds",
               model_diff(a, c1) == {"refine_rounds"}, str(model_diff(a, c1)))
        # One read per EXISTING layer: C1 adds no attention layer.
        report("C1 refine_rounds == n_layer",
               c1["model"]["refine_rounds"] == c1["model"]["n_layer"],
               f"{c1['model']['refine_rounds']} vs {c1['model']['n_layer']}")
    if c1 and c1b:
        # C1b must differ ONLY in how many rounds are read. If it also changed
        # n_layer, a worse result would be explained by lost depth instead.
        report("C1b differs from C1 only in refine_rounds",
               model_diff(c1, c1b) == {"refine_rounds"}, str(model_diff(c1, c1b)))
        report("C1b keeps C1's decoder depth",
               c1b["model"]["n_layer"] == c1["model"]["n_layer"],
               f"{c1b['model']['n_layer']} vs {c1['model']['n_layer']}")
        report("C1b reads exactly 1 round", c1b["model"]["refine_rounds"] == 1,
               str(c1b["model"]["refine_rounds"]))
    if c1 and c1c:
        # C1c changes NO model field at all -- only which epoch gets saved.
        report("C1c model identical to C1", model_diff(c1, c1c) == set(),
               str(model_diff(c1, c1c)) or "identical")
        report("C1c selects on oracle_recall",
               c1c["training"].get("select_metric") == "oracle_recall",
               str(c1c["training"].get("select_metric")))
        report("C1/C1b still select on loss_final (unchanged)",
               all(c["training"].get("select_metric", "loss_final") == "loss_final"
                   for c in (c1, c1b)))
    e1 = cfgs.get("experiment_e1.yaml")
    if e1:
        # E1 must differ from A in exactly the three RoI/score fields. In
        # particular `roi_to_tgt: false` is what separates it from B -- with it
        # true, E1 would silently be B+E1 and measure two variables.
        report("E1 differs from A only in the score-path fields",
               model_diff(a, e1) == {"roi_k", "roi_to_tgt", "score_roi"},
               str(model_diff(a, e1)))
        report("E1 does NOT add RoI to tgt (that would be B)",
               e1["model"]["roi_to_tgt"] is False,
               str(e1["model"].get("roi_to_tgt")))
        report("E1 has score_roi on and roi_k>0",
               e1["model"]["score_roi"] is True and e1["model"]["roi_k"] > 0)
        report("E1 keeps refine_rounds off (score_roi+refine is refused)",
               e1["model"].get("refine_rounds", 0) == 0)
        report("B still adds RoI to tgt (unchanged)",
               b["model"].get("roi_to_tgt", True) is True)
    for s in ("diffusion", "loss", "matcher", "eval"):
        report(f"{s} identical across all configs",
               len({str(c[s]) for c in cfgs.values()}) == 1)

    # Budget: A.1 has 73,531 samples, A.2 only 25,000. Equal EPOCHS would give A.1
    # ~3x the compute, and the comparison would measure budget, not conditioning.
    v1 = a1["training"]["epochs"] * 73531
    v2 = a2["training"]["epochs"] * 25000
    report("A.1 and A.2 matched in image-views", abs(v1 - v2) / v1 < 0.02,
           f"{v1:,} vs {v2:,} ({100*abs(v1-v2)/v1:.1f}% apart)")
    report("AMP off everywhere (project rule)",
           not any(c["training"]["amp"] for c in cfgs.values()))
    report("save_dirs are distinct",
           len({c["training"]["save_dir"] for c in cfgs.values()}) == len(cfgs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip-tests", action="store_true",
                    help="skip step 1 (it is the slow one) — use while iterating")
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    a = ap.parse_args()

    t0 = time.time()
    if not a.skip_tests and not run_tests(a.python):
        print("\nTests failed — fix them before looking at anything else.")
        return 1

    head("2/3  one real training step per config  (data -> collate -> loss -> backward)")
    for p in a.configs:
        if not os.path.exists(p):
            report(os.path.basename(p), False, "config not found")
            continue
        try:
            report(os.path.basename(p), True, one_step(p))
        except Exception as e:                                    # noqa: BLE001
            report(os.path.basename(p), False, f"{type(e).__name__}: {e}")

    compile_and_import_tools(a.python)

    head("2c/3  preflight itself, on tiny data (CPU)")
    for p in a.configs:
        if os.path.exists(p):
            ok, detail = smoke_preflight(a.python, p)
            report(f"preflight {os.path.basename(p)}", ok, detail)

    compare_configs()

    print("\n" + "=" * 78)
    if failures:
        print(f"NOT READY — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        print("=" * 78)
        return 1
    print(f"ALL CHECKS PASSED in {time.time()-t0:.0f}s — push, then run "
          f"tools/preflight.py on the server.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Verification step 7: does ``groundingdino_swint_ogc.pth`` actually load?

    python tools/check_checkpoint.py -c config/cfg_odvg_diffusion.py \
        --checkpoint ../weights/diffu_grounding_dino/groundingdino_swint_ogc.pth

Builds the model exactly as ``main.py`` would, loads the checkpoint with
``strict=False``, and classifies every key:

  * **expected missing** -- the diffusion modules (``time_``, ``diffusion``), which
    a pretrained non-diffusion checkpoint cannot possibly contain;
  * **UNEXPECTED missing** -- a module we renamed during the rewrite. These stay
    randomly initialised, so the run silently trains part of the network from
    scratch. This is the failure this script exists to catch;
  * **unexpected in checkpoint** -- keys the checkpoint has and we do not.

Exits non-zero if anything is in the second or third bucket, so it can gate a
training launch.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import build_model  # noqa: E402
from util.config import Config, DictAction  # noqa: E402
from util.misc import clean_state_dict  # noqa: E402

DIFFUSION_KEYWORDS = ("time_", "diffusion")

# Keys the released checkpoint carries that this rewrite deliberately has no use
# for -- not a sign of a renamed module, so they should not fail the gate.
#   * bert.embeddings.position_ids: a deterministic arange() buffer, persistent
#     in the transformers version the checkpoint was saved with; nothing learned
#     is lost by not loading it.
#   * label_enc.weight: feeds CDN (contrastive denoising queries), which this
#     project does not implement (dn_number=0, same as upstream Open-GroundingDino).
KNOWN_HARMLESS_UNEXPECTED = ("bert.embeddings.position_ids", "label_enc.weight")


def group_prefix(key: str, depth: int = 2) -> str:
    return ".".join(key.split(".")[:depth])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config_file", "-c", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--options", nargs="+", action=DictAction)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--finetune-ignore", nargs="+", default=list(DIFFUSION_KEYWORDS), help="keys to skip when loading"
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="datasets json (same format as main.py --datasets); only needed to build the real "
        "eval category list. Without it, use_coco_eval is forced off and a dummy 1-class prompt "
        "is used instead -- fine here, since this script only checks checkpoint keys/shapes, "
        "which PostProcess's category list has no effect on.",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)
    if args.options:
        cfg.merge_from_dict(args.options)
    for key, value in cfg.to_dict().items():
        setattr(args, key, value)

    if getattr(args, "use_coco_eval", False):
        if args.datasets:
            import json

            with open(args.datasets, encoding="utf-8") as f:
                dataset_meta = json.load(f)
            args.coco_val_path = dataset_meta["val"][0]["anno"]
        else:
            print("no --datasets given: forcing use_coco_eval=False, using a dummy 1-class prompt")
            args.use_coco_eval = False
            args.label_list = list(getattr(args, "label_list", None) or ["object"])

    print(f"building model from {args.config_file} (use_diffusion={getattr(args, 'use_diffusion', False)})")
    model, _, _ = build_model(args)
    model_keys = set(model.state_dict().keys())
    print(f"model has {len(model_keys)} tensors, {sum(p.numel() for p in model.parameters()):,} parameters")

    print(f"loading {args.checkpoint}")
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = clean_state_dict(raw.get("model", raw))
    print(f"checkpoint has {len(state)} tensors")

    skipped = [k for k in state if any(kw in k for kw in args.finetune_ignore)]
    filtered = {k: v for k, v in state.items() if k not in skipped}
    result = model.load_state_dict(filtered, strict=False)

    expected_missing = [k for k in result.missing_keys if any(kw in k for kw in DIFFUSION_KEYWORDS)]
    unexpected_missing = [k for k in result.missing_keys if k not in expected_missing]
    harmless_unexpected = [k for k in result.unexpected_keys if k in KNOWN_HARMLESS_UNEXPECTED]
    real_unexpected = [k for k in result.unexpected_keys if k not in KNOWN_HARMLESS_UNEXPECTED]

    print("\n--- summary ---")
    print(f"loaded            : {len(filtered) - len(result.unexpected_keys)}")
    print(f"skipped by ignore : {len(skipped)}")
    print(f"expected missing  : {len(expected_missing)}  (diffusion modules, will train from scratch)")
    print(f"UNEXPECTED missing: {len(unexpected_missing)}")
    print(f"unexpected in ckpt: {len(result.unexpected_keys)} ({len(harmless_unexpected)} known harmless)")

    if expected_missing:
        print("\nexpected missing, grouped:")
        for prefix, count in sorted(Counter(group_prefix(k, 3) for k in expected_missing).items()):
            print(f"  {prefix:50s} {count}")

    if unexpected_missing:
        print("\nUNEXPECTED missing -- these modules were renamed and will NOT be initialised from the checkpoint:")
        for prefix, count in sorted(Counter(group_prefix(k) for k in unexpected_missing).items()):
            print(f"  {prefix:50s} {count}")
        for key in unexpected_missing[:30]:
            print(f"    {key}")

    if harmless_unexpected:
        print("\nin the checkpoint but not in the model (known harmless, see KNOWN_HARMLESS_UNEXPECTED):")
        for key in harmless_unexpected:
            print(f"  {key}")

    if real_unexpected:
        print("\nin the checkpoint but not in the model:")
        for prefix, count in sorted(Counter(group_prefix(k) for k in real_unexpected).items()):
            print(f"  {prefix:50s} {count}")
        for key in real_unexpected[:30]:
            print(f"    {key}")

    # Shape check on everything that did load, since load_state_dict(strict=False)
    # would otherwise report a mismatch only as a missing key.
    mismatches = []
    model_state = model.state_dict()
    for key, value in filtered.items():
        if key in model_state and tuple(model_state[key].shape) != tuple(value.shape):
            mismatches.append((key, tuple(model_state[key].shape), tuple(value.shape)))
    if mismatches:
        print(f"\nSHAPE MISMATCHES: {len(mismatches)}")
        for key, ours, theirs in mismatches[:20]:
            print(f"  {key}: model {ours} vs checkpoint {theirs}")

    failed = bool(unexpected_missing or real_unexpected or mismatches)
    print("\nRESULT:", "FAIL" if failed else "OK -- every pretrained tensor found its place")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

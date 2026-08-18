"""Run every test suite and report against the plan's verification checklist.

    python tests/run_all.py

Each suite is run in a subprocess so one crash cannot take the rest down, and so
the per-suite output stays separable. Steps that need artefacts we may not have
(the pretrained checkpoint, a real dataset) are reported as SKIPPED with the exact
command to run once those exist.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT.parents[0] / "weights" / "diffu_grounding_dino"

SUITES = [
    ("test_diffusion.py", "1-2: noise schedule, q_sample, reference-point construction, DDIM"),
    ("test_util.py", "config, param groups, warm-up freeze, box ops"),
    ("test_backbone_text.py", "Swin key layout, deformable attention, sub-sentence text mask"),
    ("test_model.py", "3-4, 6: forward pass, DDIM sampling, baseline parity"),
    ("test_data_pipeline.py", "5: transforms, ODVG dataset, training loop"),
    ("test_ddp.py", "8: 2-process DDP training step (gloo/CPU stand-in for real 2-GPU nccl)"),
    ("test_lora.py", "LoRA injection: no-op at init, correct freeze, resume-ordering"),
]


def run_suite(name: str) -> bool:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests" / name)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    # Suites print one line per test plus a tally; the loop logs from
    # train_one_epoch are noise here, so keep only the test lines.
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(("ok ", "FAIL", "(")) or "passed" in stripped:
            print(line)
    if result.returncode != 0:
        print(result.stderr[-3000:])
    return result.returncode == 0


def main():
    results = []
    for name, description in SUITES:
        ok = run_suite(name)
        results.append((name, description, ok))

    print(f"\n{'=' * 78}\nVERIFICATION SUMMARY\n{'=' * 78}")
    for name, description, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:26s} {description}")

    checkpoint = WEIGHTS / "groundingdino_swint_ogc.pth"
    print("\nSteps needing downloaded artefacts:")
    if checkpoint.exists():
        print(f"  step 7: checkpoint present -- run")
        print(f"    python tools/check_checkpoint.py -c config/cfg_odvg_diffusion.py --checkpoint {checkpoint}")
    else:
        print(f"  step 7: SKIPPED, {checkpoint} not downloaded")
        print("    python tools/download_weights.py --dest ../weights/diffu_grounding_dino")
    print("  step 5 (real overfit on 20-50 COCO images): see README, section 'Verification'")

    if not all(ok for _, _, ok in results):
        raise SystemExit(1)
    print("\nall suites passed")


if __name__ == "__main__":
    main()

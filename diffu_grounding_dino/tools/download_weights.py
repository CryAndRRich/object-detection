"""Fetch the pretrained weights this project needs.

    python tools/download_weights.py --dest ../weights/diffu_grounding_dino            # both
    python tools/download_weights.py --dest ../weights/diffu_grounding_dino --only bert

Downloads (nothing is fetched if the target already exists):

  ``groundingdino_swint_ogc.pth``  694MB  Swin-T GroundingDINO, trained on
      O365 + GoldG + Cap4M. This is what makes the open-vocabulary behaviour work;
      training from scratch is not a realistic substitute.
  ``bert-base-uncased/``           440MB  text encoder, in safetensors form
      (``pytorch_model.bin`` is deliberately avoided: torch >= 2.6 refuses to load
      pickled checkpoints by default).

After downloading, point the config at the local copy so no run depends on network
access::

    text_encoder_type = "../weights/diffu_grounding_dino/bert-base-uncased"
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

GDINO_URLS = [
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
]

BERT_FILES = [
    "config.json",
    "vocab.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
]
BERT_BASE_URL = "https://huggingface.co/bert-base-uncased/resolve/main"


def _download(urls, target: Path) -> bool:
    if target.exists() and target.stat().st_size > 0:
        print(f"  exists, skipping: {target.name} ({target.stat().st_size / 1e6:.0f} MB)")
        return True

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")

    for url in urls:
        try:
            print(f"  fetching {target.name} from {url}")

            def progress(block_num, block_size, total_size):
                if total_size > 0:
                    done = min(block_num * block_size, total_size)
                    sys.stdout.write(f"\r    {done / 1e6:7.1f} / {total_size / 1e6:.1f} MB")
                    sys.stdout.flush()

            urllib.request.urlretrieve(url, partial, reporthook=progress)
            print()
            # Rename only after a complete download, so an interrupted run never
            # leaves a truncated file that later looks valid.
            partial.replace(target)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"\n    failed: {exc}")
            partial.unlink(missing_ok=True)
    return False


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default="../weights/diffu_grounding_dino", help="destination directory")
    parser.add_argument("--only", choices=["gdino", "bert"], help="fetch just one of the two")
    parser.add_argument("--checksums", action="store_true", help="print sha256 of what is on disk")
    args = parser.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    print(f"destination: {dest}")
    ok = True

    if args.only in (None, "gdino"):
        print("GroundingDINO Swin-T checkpoint:")
        ok &= _download(GDINO_URLS, dest / "groundingdino_swint_ogc.pth")

    if args.only in (None, "bert"):
        print("bert-base-uncased:")
        bert_dir = dest / "bert-base-uncased"
        for name in BERT_FILES:
            ok &= _download([f"{BERT_BASE_URL}/{name}"], bert_dir / name)

    if args.checksums:
        print("\nsha256:")
        for path in sorted(dest.rglob("*")):
            if path.is_file() and path.suffix in (".pth", ".safetensors"):
                print(f"  {sha256(path)}  {path.relative_to(dest)}")

    if not ok:
        raise SystemExit("some downloads failed; re-run or fetch manually (URLs are in this file's docstring)")
    print("\ndone")


if __name__ == "__main__":
    main()

"""Dataset, transforms and training-loop tests.

Builds a two-image ODVG dataset on disk and runs real training steps through
``engine.train_one_epoch`` with the tiny model. Verification step 5's overfit run
needs real COCO images; this is the wiring check that must pass first.

    python tests/test_data_pipeline.py
"""

import json
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gdino_datasets.transforms as T  # noqa: E402
from gdino_datasets.odvg import ODVGDataset, make_coco_transforms  # noqa: E402
from tests.tiny import build_tiny_model  # noqa: E402
from util.misc import collate_fn  # noqa: E402
from util.param_dicts import get_param_dict  # noqa: E402


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def _image_and_target(width=100, height=50):
    image = Image.new("RGB", (width, height), color=(128, 128, 128))
    target = {
        "boxes": torch.tensor([[10.0, 10.0, 30.0, 40.0], [60.0, 5.0, 80.0, 25.0]]),
        "labels": torch.tensor([0, 1]),
        "size": torch.tensor([height, width]),
    }
    return image, target


def test_hflip_mirrors_boxes_exactly():
    image, target = _image_and_target(width=100)
    _, flipped = T.hflip(image, target)

    # x1' = W - x2, x2' = W - x1; y untouched.
    assert torch.allclose(flipped["boxes"][0], torch.tensor([70.0, 10.0, 90.0, 40.0]))
    assert torch.allclose(flipped["boxes"][1], torch.tensor([20.0, 5.0, 40.0, 25.0]))
    assert (flipped["boxes"][:, 2] > flipped["boxes"][:, 0]).all(), "flip must keep boxes well-ordered"


def test_resize_scales_boxes():
    image, target = _image_and_target(width=100, height=50)
    resized, out = T.resize(image, target, size=100)  # short side 50 -> 100, so 2x

    assert resized.size == (200, 100), resized.size
    assert torch.allclose(out["boxes"], target["boxes"] * 2)
    assert out["size"].tolist() == [100, 200]


def test_resize_respects_max_size():
    image, target = _image_and_target(width=1000, height=100)
    resized, _ = T.resize(image, target, size=800, max_size=1333)
    assert max(resized.size) <= 1333, resized.size


def test_crop_drops_boxes_and_labels_together():
    image, target = _image_and_target(width=100, height=50)
    # Keep only the left half: box 1 (x 60..80) falls entirely outside.
    _, out = T.crop(image, target, region=(0, 0, 50, 50))

    assert out["boxes"].shape[0] == 1, "the out-of-view box must be dropped"
    assert out["labels"].tolist() == [0], "labels must be filtered with the boxes"
    assert out["boxes"].shape[0] == out["labels"].shape[0]
    assert out["size"].tolist() == [50, 50]


def test_normalize_converts_to_normalized_cxcywh():
    image, target = _image_and_target(width=100, height=50)
    tensor_image, _ = T.ToTensor()(image, target)
    _, out = T.Normalize([0.5] * 3, [0.5] * 3)(tensor_image, target)

    boxes = out["boxes"]
    assert (boxes >= 0).all() and (boxes <= 1).all(), "normalized boxes must be in [0, 1]"
    # First box: cx=20/100, cy=25/50, w=20/100, h=30/50
    assert torch.allclose(boxes[0], torch.tensor([0.20, 0.50, 0.20, 0.60]), atol=1e-6)


def test_train_transform_produces_normalized_boxes():
    class Args:
        data_aug_scales = [32, 48]
        data_aug_max_size = 64
        data_aug_scales2_resize = [32]
        data_aug_scales2_crop = [16, 24]
        data_aug_scale_overlap = None

    transform = make_coco_transforms("train", args=Args())
    for _ in range(20):  # the pipeline is random; check every branch it can take
        image, target = _image_and_target()
        out_image, out_target = transform(image, target)
        assert out_image.ndim == 3
        if out_target["boxes"].numel():
            assert (out_target["boxes"] >= 0).all() and (out_target["boxes"] <= 1).all()
        assert out_target["boxes"].shape[0] == out_target["labels"].shape[0]


# --------------------------------------------------------------------------- #
# ODVG dataset
# --------------------------------------------------------------------------- #
def _write_dataset(tmpdir: Path, num_images: int = 2):
    """A minimal on-disk ODVG dataset with a car/carrot vocabulary."""
    image_dir = tmpdir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    label_map = {"0": "person", "1": "car", "2": "dog", "3": "carrot"}
    (tmpdir / "label_map.json").write_text(json.dumps(label_map), encoding="utf-8")

    lines = []
    for i in range(num_images):
        name = f"img{i}.jpg"
        Image.new("RGB", (64, 48), color=(100 + i, 120, 140)).save(image_dir / name)
        instances = [
            {"bbox": [5.0, 5.0, 25.0, 30.0], "label": i % 4},
            {"bbox": [30.0, 10.0, 55.0, 40.0], "label": (i + 1) % 4},
        ]
        lines.append(
            json.dumps({"filename": name, "height": 48, "width": 64, "detection": {"instances": instances}})
        )
    (tmpdir / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return image_dir, tmpdir / "train.jsonl", tmpdir / "label_map.json", label_map


def test_odvg_dataset_builds_prompts():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image_dir, anno, label_map_path, label_map = _write_dataset(tmp)

        dataset = ODVGDataset(str(image_dir), str(anno), str(label_map_path), max_labels=4)
        assert len(dataset) == 2

        _, target = dataset[0]
        cap_list = target["cap_list"]

        assert target["caption"].endswith(" ."), "the trailing separator is required by the text mask"
        assert target["caption"] == " . ".join(cap_list) + " ."
        assert len(cap_list) == 4, "max_labels=4 means all four categories appear in the prompt"
        assert set(cap_list) == set(label_map.values())

        # Labels index into this image's prompt, not the global label map.
        assert target["labels"].max() < len(cap_list)
        present = {cap_list[i] for i in target["labels"].tolist()}
        assert present == {"person", "car"}, present


def test_odvg_negatives_are_sampled_and_capped():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image_dir, anno, label_map_path, _ = _write_dataset(tmp)

        dataset = ODVGDataset(str(image_dir), str(anno), str(label_map_path), max_labels=2)
        _, target = dataset[0]
        assert len(target["cap_list"]) == 2, "max_labels must cap the prompt length"
        # Both positives must survive the cap: negatives are only ever added.
        present = {target["cap_list"][i] for i in target["labels"].tolist()}
        assert present == {"person", "car"}


def test_odvg_prompt_order_varies():
    """Shuffling matters: a fixed order teaches the model a positional shortcut."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image_dir, anno, label_map_path, _ = _write_dataset(tmp)
        dataset = ODVGDataset(str(image_dir), str(anno), str(label_map_path), max_labels=4)

        orders = {tuple(dataset[0][1]["cap_list"]) for _ in range(30)}
        assert len(orders) > 1, "prompt order never changed across 30 samples"


def test_odvg_with_transforms_is_collatable():
    class Args:
        data_aug_scales = [32]
        data_aug_max_size = 48
        data_aug_scales2_resize = [32]
        data_aug_scales2_crop = [16, 24]
        data_aug_scale_overlap = None
        max_labels = 4
        fix_size = False
        strong_aug = False

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        image_dir, anno, label_map_path, _ = _write_dataset(tmp, num_images=3)
        dataset = ODVGDataset(
            str(image_dir),
            str(anno),
            str(label_map_path),
            max_labels=4,
            transforms=make_coco_transforms("train", args=Args()),
        )

        samples, targets = collate_fn([dataset[i] for i in range(3)])
        assert samples.tensors.shape[0] == 3
        assert samples.mask.shape[0] == 3
        assert len(targets) == 3
        for target in targets:
            assert isinstance(target["caption"], str)
            assert target["boxes"].shape[1] == 4


# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #
class _EngineArgs:
    amp = False
    debug = False
    onecyclelr = False
    clip_max_norm = 0.1
    use_diffusion = True
    diff_warmup_iters = 3
    diff_warmup_freeze_keywords = ["backbone.0", "bert"]
    freeze_keywords = None
    param_dict_type = "ddetr_in_mmdet"
    lr = 1e-4
    lr_backbone = 1e-5
    lr_linear_proj_mult = 1e-5
    lr_backbone_names = ["backbone.0", "bert"]
    lr_linear_proj_names = ["ref_point_head", "sampling_offsets"]
    weight_decay = 1e-4


def _tiny_loader(cfg, num_images=2):
    from tests.tiny import fake_batch

    samples, targets = fake_batch(cfg, num_boxes=(2, 2), image_size=(3, 64, 64))
    batch = (samples, targets)
    return [batch, batch]  # two identical steps: enough to exercise the loop


def test_train_one_epoch_runs_with_diffusion():
    from engine import train_one_epoch

    model, criterion, _, cfg = build_tiny_model(use_diffusion=True)
    args = _EngineArgs()
    optimizer = torch.optim.AdamW(get_param_dict(args, model, include_frozen=True), lr=1e-3)

    stats = train_one_epoch(
        model, criterion, _tiny_loader(cfg), optimizer, torch.device("cpu"), epoch=0, max_norm=0.1, args=args
    )

    assert "loss" in stats and stats["loss"] > 0
    assert all(v == v for v in stats.values()), f"NaN in stats: {stats}"
    assert "diff_t" in stats, "the diffusion timestep should be logged"


def test_warmup_freezes_then_releases_during_training():
    from engine import train_one_epoch

    model, criterion, _, cfg = build_tiny_model(use_diffusion=True)
    args = _EngineArgs()
    args.diff_warmup_iters = 2
    optimizer = torch.optim.AdamW(get_param_dict(args, model, include_frozen=True), lr=1e-3)

    bert_weight = dict(model.named_parameters())["bert.embeddings.word_embeddings.weight"]
    loader = _tiny_loader(cfg)  # 2 steps -> global_step 0 and 1, both inside warm-up

    train_one_epoch(model, criterion, loader, optimizer, torch.device("cpu"), epoch=0, args=args)
    assert not bert_weight.requires_grad, "bert must still be frozen during warm-up"

    # Epoch 1 starts at global_step 2, which is past the warm-up.
    train_one_epoch(model, criterion, loader, optimizer, torch.device("cpu"), epoch=1, args=args)
    assert bert_weight.requires_grad, "bert must be released once the warm-up is over"


def test_loss_decreases_when_overfitting_one_batch():
    """Sanity that the whole chain learns; verification #5 at miniature scale."""
    from engine import train_one_epoch

    torch.manual_seed(0)
    model, criterion, _, cfg = build_tiny_model(use_diffusion=True)
    args = _EngineArgs()
    args.diff_warmup_iters = 0
    optimizer = torch.optim.AdamW(get_param_dict(args, model, include_frozen=True), lr=1e-3)

    loader = _tiny_loader(cfg)
    first = train_one_epoch(model, criterion, loader, optimizer, torch.device("cpu"), epoch=0, args=args)["loss"]
    for epoch in range(1, 6):
        last = train_one_epoch(
            model, criterion, loader, optimizer, torch.device("cpu"), epoch=epoch, args=args
        )["loss"]

    assert last < first, f"loss did not decrease over 6 epochs on one batch: {first:.3f} -> {last:.3f}"


def test_baseline_train_step_ignores_diffusion_plumbing():
    from engine import train_one_epoch

    model, criterion, _, cfg = build_tiny_model(use_diffusion=False)
    args = _EngineArgs()
    args.use_diffusion = False
    optimizer = torch.optim.AdamW(get_param_dict(args, model, include_frozen=True), lr=1e-3)

    stats = train_one_epoch(model, criterion, _tiny_loader(cfg), optimizer, torch.device("cpu"), epoch=0, args=args)
    assert "diff_t" not in stats, "the baseline must not report a diffusion timestep"
    assert stats["loss"] > 0


# --------------------------------------------------------------------------- #
def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            torch.manual_seed(0)
            fn()
            print(f"  ok    {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

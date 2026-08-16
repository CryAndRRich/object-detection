"""Tests for the config loader, param groups and the warm-up freeze.

    python tests/test_util.py
"""

import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou  # noqa: E402
from util.config import Config, DictAction  # noqa: E402
from util.misc import inverse_sigmoid, nested_tensor_from_tensor_list  # noqa: E402
from util.param_dicts import apply_diffusion_warmup, get_param_dict  # noqa: E402


# --------------------------------------------------------------------------- #
def test_config_base_inheritance():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "base.py").write_text("lr = 1e-4\nnum_queries = 900\nfreeze_keywords = ['bert']\n")
        (tmp / "child.py").write_text("_base_ = 'base.py'\nlr = 5e-5\nuse_diffusion = True\n")

        cfg = Config.fromfile(str(tmp / "child.py"))
        assert cfg.lr == 5e-5, "child must override the base"
        assert cfg.num_queries == 900, "base fields must be inherited"
        assert cfg.use_diffusion is True
        assert cfg.freeze_keywords == ["bert"]
        assert "num_queries" in cfg and "nope" not in cfg


def test_config_merge_and_dump():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "c.py").write_text("lr = 1e-4\ndiff = {'steps': 3}\n")
        cfg = Config.fromfile(str(tmp / "c.py"))

        cfg.merge_from_dict({"lr": 2e-4, "diff.steps": 5, "brand_new": True})
        assert cfg.lr == 2e-4 and cfg.diff["steps"] == 5 and cfg.brand_new is True

        text = cfg.dump(tmp / "dumped.py")
        assert "lr = 0.0002" in text
        reloaded = Config.fromfile(str(tmp / "dumped.py"))
        assert reloaded.to_dict() == cfg.to_dict(), "a dumped config must round-trip"


def test_dict_action_parses_literals():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--options", nargs="+", action=DictAction)
    args = parser.parse_args(["--options", "a=5", "b=0.1", "c=True", "d=hello", "e=[1,2]"])
    assert args.options == {"a": 5, "b": 0.1, "c": True, "d": "hello", "e": [1, 2]}


# --------------------------------------------------------------------------- #
class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleList([nn.Linear(4, 4)])
        self.bert = nn.Linear(4, 4)
        self.transformer = nn.Module()
        self.transformer.ref_point_head = nn.Linear(4, 4)
        self.transformer.body = nn.Linear(4, 4)
        self.time_embed = nn.Linear(4, 4)


class _Args:
    param_dict_type = "ddetr_in_mmdet"
    lr = 1e-4
    lr_backbone = 1e-5
    lr_linear_proj_mult = 1e-5
    lr_backbone_names = ["backbone.0", "bert"]
    lr_linear_proj_names = ["ref_point_head", "sampling_offsets"]
    weight_decay = 1e-4
    use_diffusion = True
    diff_warmup_iters = 2000
    diff_warmup_freeze_keywords = ["backbone.0", "bert"]
    freeze_keywords = None


def test_param_groups_cover_every_parameter_once():
    model = _Toy()
    groups = get_param_dict(_Args(), model)
    assert len(groups) == 3

    seen = [p for g in groups for p in g["params"]]
    all_params = list(model.parameters())
    assert len(seen) == len(all_params), "every parameter must land in exactly one group"
    assert {id(p) for p in seen} == {id(p) for p in all_params}
    assert [g["lr"] for g in groups] == [1e-4, 1e-5, 1e-5]


def test_frozen_params_still_enter_the_optimizer():
    """The whole point of include_frozen: a warm-up must be reversible."""
    model = _Toy()
    model.bert.weight.requires_grad_(False)

    kept = sum(len(g["params"]) for g in get_param_dict(_Args(), model, include_frozen=True))
    dropped = sum(len(g["params"]) for g in get_param_dict(_Args(), model, include_frozen=False))
    assert kept == len(list(model.parameters()))
    assert dropped == kept - 1, "include_frozen=False should drop exactly the frozen tensor"


def test_warmup_freeze_is_resume_safe():
    model = _Toy()
    args = _Args()

    def bert_trainable():
        return model.bert.weight.requires_grad

    assert apply_diffusion_warmup(args, model, global_step=0) is True
    assert not bert_trainable(), "bert must be frozen during warm-up"
    assert model.time_embed.weight.requires_grad, "the timestep module must stay trainable"

    assert apply_diffusion_warmup(args, model, global_step=1999) is True
    assert not bert_trainable()

    # A resume that lands past the warm-up must unfreeze with no persisted flag.
    fresh = _Toy()
    assert apply_diffusion_warmup(args, fresh, global_step=5000) is False
    assert fresh.bert.weight.requires_grad, "warm-up must be over at step 5000 after a resume"

    assert apply_diffusion_warmup(args, model, global_step=2000) is False
    assert bert_trainable()


def test_permanent_freeze_wins_over_warmup():
    model = _Toy()
    args = _Args()
    args.freeze_keywords = ["bert"]
    apply_diffusion_warmup(args, model, global_step=999_999)
    assert not model.bert.weight.requires_grad, "freeze_keywords must not be undone by the warm-up"
    args.freeze_keywords = None


def test_warmup_is_a_noop_without_diffusion():
    model = _Toy()
    args = _Args()
    args.use_diffusion = False
    assert apply_diffusion_warmup(args, model, global_step=0) is False
    assert model.bert.weight.requires_grad
    args.use_diffusion = True


# --------------------------------------------------------------------------- #
def test_box_conversions():
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4], [0.1, 0.9, 0.05, 0.05]])
    assert torch.allclose(box_xyxy_to_cxcywh(box_cxcywh_to_xyxy(boxes)), boxes, atol=1e-6)

    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    assert abs(generalized_box_iou(a, a).item() - 1.0) < 1e-6
    far = torch.tensor([[10.0, 10.0, 11.0, 11.0]])
    assert generalized_box_iou(a, far).item() < 0.0, "disjoint boxes must give a negative GIoU"


def test_inverse_sigmoid_is_finite_at_the_boundary():
    x = torch.tensor([0.0, 1e-9, 0.5, 1.0 - 1e-9, 1.0])
    out = inverse_sigmoid(x)
    assert torch.isfinite(out).all(), "inverse_sigmoid must never return +/-inf"
    assert abs(out[2].item()) < 1e-6
    assert torch.allclose(inverse_sigmoid(torch.tensor([0.3, 0.7])).sigmoid(), torch.tensor([0.3, 0.7]), atol=1e-5)


def test_nested_tensor_padding():
    imgs = [torch.rand(3, 20, 30), torch.rand(3, 25, 10)]
    nt = nested_tensor_from_tensor_list(imgs)
    assert nt.tensors.shape == (2, 3, 25, 30)
    assert nt.mask.shape == (2, 25, 30)
    assert not nt.mask[0, :20, :30].any(), "real pixels must be unmasked"
    assert nt.mask[0, 20:, :].all(), "padded rows must be masked"
    assert nt.mask[1, :, 10:].all(), "padded columns must be masked"


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

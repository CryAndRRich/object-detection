"""Build a small but complete model offline, for tests.

The real model needs ``bert-base-uncased`` (440MB) and a real tokenizer. Tests must
run without either, so this builds a genuine ``BertTokenizerFast`` from a
hand-written vocab file and a randomly initialised ``BertModel``. It is a real
tokenizer -- ``char_to_token`` works, offsets are correct -- so the category/token
mapping under test is the same code path as production.
"""

import sys
import tempfile
from pathlib import Path
from typing import List, Sequence

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models.diffu_groundingdino as model_module  # noqa: E402
import models.postprocess as postprocess_module  # noqa: E402
from models.text.bert import force_eager_attention  # noqa: E402
from util.config import Config  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

TINY_CATEGORIES = ["person", "car", "dog", "cat", "carrot"]
_EXTRA_VOCAB = ["a", "b", "the", "on", "of", "and", "bus", "bird", "boat", "chair"]


_CACHE: dict = {}


def make_tiny_tokenizer(words: Sequence[str] = None) -> BertTokenizerFast:
    """A WordPiece tokenizer over a handful of whole words.

    Every test word is its own vocab entry, so tokenization is trivial and the
    expected token positions are easy to reason about in assertions.
    """
    if "tokenizer" in _CACHE:
        return _CACHE["tokenizer"]

    words = list(words or (TINY_CATEGORIES + _EXTRA_VOCAB))
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", "?"] + sorted(set(words))

    tmpdir = Path(tempfile.mkdtemp(prefix="diffugdino-vocab-"))
    vocab_file = tmpdir / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    _CACHE["tokenizer"] = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)
    _CACHE["vocab_size"] = len(vocab)
    return _CACHE["tokenizer"]


def make_tiny_bert(hidden_size: int = 32) -> BertModel:
    key = f"bert{hidden_size}"
    if key in _CACHE:
        return _CACHE[key]

    make_tiny_tokenizer()  # populates _CACHE["vocab_size"]
    config = BertConfig(
        vocab_size=_CACHE["vocab_size"],
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=hidden_size * 2,
        max_position_embeddings=128,
    )
    _CACHE[key] = force_eager_attention(BertModel(config, add_pooling_layer=True))
    return _CACHE[key]


def patch_text_stack(hidden_size: int = 32):
    """Point the model's text loaders at the tiny stack.

    Patching the module globals (rather than adding constructor arguments) keeps the
    production API free of test-only hooks.
    """
    tokenizer = make_tiny_tokenizer()
    bert = make_tiny_bert(hidden_size)

    model_module.get_tokenizer = lambda _type: tokenizer
    model_module.get_pretrained_language_model = lambda _type: bert
    postprocess_module.get_tokenizer = lambda _type: tokenizer

    # PostProcess imports the loader lazily inside __init__, so patch the source too.
    import models.text as text_pkg
    import models.text.bert as bert_module

    text_pkg.get_tokenizer = lambda _type: tokenizer
    bert_module.get_tokenizer = lambda _type: tokenizer
    return tokenizer, bert


def tiny_config(use_diffusion: bool = False, **overrides) -> Config:
    """The real config file, shrunk to something a CPU test can run."""
    name = "cfg_odvg_diffusion.py" if use_diffusion else "cfg_odvg.py"
    cfg = Config.fromfile(str(CONFIG_DIR / name))

    cfg.merge_from_dict(
        {
            "device": "cpu",
            "hidden_dim": 32,
            "dim_feedforward": 64,
            "nheads": 8,
            "enc_layers": 1,
            "dec_layers": 2,
            "num_queries": 20,
            "num_select": 10,
            "dropout": 0.0,
            "use_checkpoint": False,
            "use_transformer_ckpt": False,
            "use_coco_eval": False,
            "label_list": list(TINY_CATEGORIES),
            "max_text_len": 64,
            "max_labels": 5,
            "batch_size": 2,
        }
    )
    cfg.merge_from_dict(overrides)
    return cfg


def build_tiny_model(use_diffusion: bool = False, **overrides):
    """``(model, criterion, postprocessors, cfg)`` on CPU, no downloads."""
    cfg = tiny_config(use_diffusion, **overrides)
    patch_text_stack(hidden_size=cfg.hidden_dim)
    model, criterion, postprocessors = model_module.build_diffu_groundingdino(cfg)
    return model, criterion, postprocessors, cfg


def fake_batch(cfg, num_boxes: Sequence[int] = (3, 0), image_size=(3, 96, 128)):
    """``(samples, targets)`` with the shapes the dataset produces.

    ``num_boxes`` includes a 0 on purpose: an image with no annotation is the case
    that breaks naive diffusion code.
    """
    from util.misc import nested_tensor_from_tensor_list
    from util.vl_utils import build_caption

    images = [torch.rand(*image_size) for _ in num_boxes]
    samples = nested_tensor_from_tensor_list(images)

    targets: List[dict] = []
    for i, count in enumerate(num_boxes):
        # Two categories per image, deliberately including the car/carrot pair that
        # a substring-based positive map gets wrong.
        cat_list = ["carrot", "car"] if i % 2 == 0 else ["person", "dog"]
        boxes = torch.rand(count, 4) * 0.4 + 0.3  # keep boxes inside the image
        targets.append(
            {
                "boxes": boxes,
                "labels": torch.randint(0, len(cat_list), (count,)),
                "size": torch.as_tensor(image_size[1:]),
                "caption": build_caption(cat_list),
                "cap_list": cat_list,
            }
        )
    return samples, targets


__all__ = [
    "TINY_CATEGORIES",
    "build_tiny_model",
    "fake_batch",
    "make_tiny_bert",
    "make_tiny_tokenizer",
    "patch_text_stack",
    "tiny_config",
]

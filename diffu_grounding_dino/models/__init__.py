from .criterion import SetCriterion
from .diffu_groundingdino import DiffuGroundingDINO, build_diffu_groundingdino
from .matcher import build_matcher
from .postprocess import PostProcess
from .transformer import EncoderOutput, Transformer, build_transformer


def build_model(args):
    """Entry point used by ``main.py``."""
    assert args.modelname == "diffu_groundingdino", f"unknown modelname {args.modelname!r}"
    return build_diffu_groundingdino(args)


__all__ = [
    "DiffuGroundingDINO",
    "EncoderOutput",
    "PostProcess",
    "SetCriterion",
    "Transformer",
    "build_diffu_groundingdino",
    "build_matcher",
    "build_model",
    "build_transformer",
]

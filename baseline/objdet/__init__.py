"""Phần bổ sung của repo này quanh DiffusionDet gốc: đăng ký dataset và evaluator.

Import lazy (PEP 562): ``objdet.mmr`` là numpy thuần nên phải import được mà không cần
detectron2 — nhờ vậy chạy được ``tests/test_mmr.py`` ở máy không cài detectron2.
"""

__all__ = ["DATA_ROOT", "register_all", "dataset_num_classes", "CrowdHumanEvaluator"]


def __getattr__(name):
    if name in ("DATA_ROOT", "register_all", "dataset_num_classes"):
        from . import datasets
        return getattr(datasets, name)
    if name == "CrowdHumanEvaluator":
        from .crowdhuman_eval import CrowdHumanEvaluator
        return CrowdHumanEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

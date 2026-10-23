"""Đọc config thí nghiệm: YAML + khoá `base` (một file hoặc list, đường dẫn tương đối so với file đang đọc) — nạp base trước, rồi
GỘP SÂU phần của file (dict gộp đệ quy, giá trị khác thay hẳn). Checkpoint lưu config đã gộp."""

import os

import yaml

__all__ = ["load_config", "merge"]


def merge(a, b):
    """Gộp sâu b vào bản sao của a."""
    out = dict(a)
    for k, v in b.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    bases = cfg.pop("base", None) or []
    out = {}
    for b in [bases] if isinstance(bases, str) else bases:
        out = merge(out, load_config(os.path.join(os.path.dirname(os.path.abspath(path)), b)))
    return merge(out, cfg)

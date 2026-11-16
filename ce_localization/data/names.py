"""Chuẩn hoá tên lớp CE-CoCount — module KHÔNG import torch (app chấm `box_review/cegen.py` chạy trong env không có torch)."""

import re


def clean_class(name):
    """'normal tomato(Big round tomato).' -> 'normal tomato' (bỏ phần trong ngoặc, dấu chấm cuối, khoảng trắng thừa)."""
    return re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", name)).strip().rstrip(".").strip()

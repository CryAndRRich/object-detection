"""Gốc `object-detection/` vào sys.path để import theo tên package đầy đủ
(`ce_localization.*`, `diffuse2seg.*`). Các project dùng chung tên package con (`data`,
`utils`, `config`), nên import trần kiểu `from utils import ...` sẽ nạp nhầm project."""

import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    # matplotlib < 3.10 gọi API camelCase của pyparsing (đã deprecated từ pyparsing 3.3) ngay lúc IMPORT mathtext — mà
    # backend Agg luôn import mathtext, nên mọi test lưu hình đều dính dù hình không có chữ toán. Lỗi của thư viện, không
    # của code: chỉ bỏ ĐÚNG loại này từ ĐÚNG module đó. (Nhãn trục log của tool đã bỏ mathtext — các lần parse không còn.)
    config.addinivalue_line("filterwarnings", r"ignore::DeprecationWarning:matplotlib\._mathtext")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    rep = (yield).get_result()
    if rep.when == "call":
        item._call_passed = rep.passed


@pytest.fixture(autouse=True)
def _rm_tmp_path_if_passed(request):
    """Test QUA thì xoá ngay `tmp_path` của nó (checkpoint / cache giả mỗi test vài GB — dồn tới cuối phiên từng ngốn > 40 GB
    ổ local). Test lỗi giữ lại để soi."""
    yield
    p = request.node.funcargs.get("tmp_path")
    if p is not None and getattr(request.node, "_call_passed", False):
        shutil.rmtree(p, ignore_errors=True)

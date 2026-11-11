"""Gốc `object-detection/` vào sys.path (import theo tên package đầy đủ `ce_localization.*`). Test NẶNG (trọn luồng train / eval trên CPU,
DDP 2 tiến trình, tool cục bộ Gradio / hình) gắn `@pytest.mark.slow`: mặc định bỏ qua, chạy khi `--runslow` (hoặc env CE_RUN_SLOW=1) —
khi sửa lõi train / eval / model, và ở ô test của notebook Kaggle."""

import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="chạy cả test nặng (trọn luồng train / eval, DDP)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow") or os.environ.get("CE_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="test nặng — chạy với --runslow")
    for it in items:
        if "slow" in it.keywords:
            it.add_marker(skip)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: test nặng (trọn luồng train / eval, DDP, tool cục bộ) — chỉ chạy với --runslow")
    # matplotlib < 3.10 gọi API deprecated của pyparsing khi import mathtext / font — lỗi thư viện, bỏ đúng loại đó
    config.addinivalue_line("filterwarnings", r"ignore::DeprecationWarning:matplotlib\._mathtext")
    config.addinivalue_line("filterwarnings", r"ignore::DeprecationWarning:matplotlib\._fontconfig_pattern")
    config.addinivalue_line("filterwarnings", r"ignore:'parseAll' argument is deprecated")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    rep = (yield).get_result()
    if rep.when == "call":
        item._call_passed = rep.passed


@pytest.fixture(autouse=True)
def _restore_grad_mode():
    """Test gọi `main()` trong cùng tiến trình có thể tắt autograd toàn cục — trả lại sau mỗi test (lỗi chỉ hiện khi chạy cả bộ)."""
    import torch
    mode = torch.is_grad_enabled()
    yield
    torch.set_grad_enabled(mode)


@pytest.fixture(autouse=True)
def _rm_tmp_path_if_passed(request):
    """Test QUA thì xoá ngay `tmp_path` (checkpoint giả ~350 MB mỗi cái); test lỗi giữ lại để soi."""
    yield
    p = request.node.funcargs.get("tmp_path")
    if p is not None and getattr(request.node, "_call_passed", False):
        shutil.rmtree(p, ignore_errors=True)

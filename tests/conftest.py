"""Gốc `object-detection/` vào sys.path (import theo tên package đầy đủ `ce_localization.*`). Test NẶNG (trọn luồng train / eval trên CPU,
DDP 2 tiến trình — danh sách `SLOW`) mặc định bỏ qua, chạy khi `--runslow` (hoặc env CE_RUN_SLOW=1): khi sửa lõi train / eval / model,
và ở ô test của notebook Kaggle."""

import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SLOW = {"test_flow_val_select_resume_eval", "test_flow_train_loss_select_resume", "test_ddp_two_processes"}


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="chạy cả test nặng (trọn luồng train / eval, DDP)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow") or os.environ.get("CE_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="test nặng — chạy với --runslow")
    for it in items:
        if getattr(it, "originalname", it.name) in SLOW:
            it.add_marker(skip)


def pytest_configure(config):
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
def _rm_tmp_path_if_passed(request):
    """Test QUA thì xoá ngay `tmp_path` (checkpoint giả ~350 MB mỗi cái); test lỗi giữ lại để soi."""
    yield
    p = request.node.funcargs.get("tmp_path")
    if p is not None and getattr(request.node, "_call_passed", False):
        shutil.rmtree(p, ignore_errors=True)

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


# Test NẶNG (train / eval thật trên CPU, DDP 2 tiến trình, tool vẽ chạy model): mặc định BỎ QUA, chỉ chạy khi `--runslow` (hoặc
# env CE_RUN_SLOW=1) — khi sửa lõi train.py / eval.py / model, và ở ô test của notebook Kaggle. Người dùng 2026-10-14: test nhỏ chạy
# mỗi lần, test trọn luồng không (local CPU mất ~1 giờ).
SLOW = {
    # trọn luồng train -> eval
    "test_full_flow_train_resume_eval", "test_full_flow_density_train_resume_eval",
    "test_full_flow_beta0_train_resume_eval_and_gt_never_in_loss", "test_full_flow_gamma_train_resume_eval",
    "test_full_flow_gamma2_celoc_then_refine", "test_full_flow_gamma3_geo_train_eval", "test_full_flow_gamma3_1_relation_train_eval",
    "test_full_flow_gamma4_train_resume_eval", "test_full_flow_gamma4_0_x_train_resume_eval", "test_full_flow_gamma4_then_refine",
    "test_full_flow_delta_train_resume_eval", "test_full_flow_delta1_1_train_eval", "test_full_flow_delta6_train_loss_select",
    "test_resume_refuses_changed_model_config", "test_lr_override_used_and_recorded", "test_bench_writes_nothing",
    "test_nan_debug_pinpoints_nan_input_channel", "test_train_max_iter_without_steps_and_batch_override",
    # eval checkpoint giả
    "test_eval_paper_checkpoint_on_ce130_add", "test_eval_obj_k_override_on_gamma4_checkpoint", "test_eval_cocount_full_flow",
    "test_eval_cocount_delta",
    # DDP 2 tiến trình
    "test_ddp_two_processes", "test_ddp_gamma2_celoc_bucket_view_no_stride_warning", "test_ddp_delta_no_find_unused",
    # tool vẽ / user study chạy model
    "test_plot_denoise_trajectory_full_flow", "test_plot_spatial_softmax_full_flow", "test_plot_refiner_steps_full_flow",
    "test_plot_refiner_steps_box_policy", "test_plot_refiner_steps_delta", "test_plot_refine_stages_full_flow",
    "test_full_flow_dump_build_rate_score", "test_cocount_flow_two_datasets",
}


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="chạy cả test nặng (trọn luồng train / eval, DDP, tool vẽ)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow") or os.environ.get("CE_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="test nặng — chạy với --runslow")
    for it in items:
        if getattr(it, "originalname", it.name) in SLOW:
            it.add_marker(skip)


def pytest_configure(config):
    # matplotlib < 3.10 gọi API camelCase của pyparsing (đã deprecated từ pyparsing 3.3) ngay lúc IMPORT mathtext — mà
    # backend Agg luôn import mathtext, nên mọi test lưu hình đều dính dù hình không có chữ toán. Lỗi của thư viện, không
    # của code: chỉ bỏ ĐÚNG loại này từ ĐÚNG module đó. (Nhãn trục log của tool đã bỏ mathtext — các lần parse không còn.)
    config.addinivalue_line("filterwarnings", r"ignore::DeprecationWarning:matplotlib\._mathtext")
    # cùng lỗi thư viện ở bộ phân tích tên font (`_fontconfig_pattern`, lần đầu dựng font trong phiên)
    config.addinivalue_line("filterwarnings", r"ignore::DeprecationWarning:matplotlib\._fontconfig_pattern")


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

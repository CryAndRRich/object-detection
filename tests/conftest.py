"""Gốc `object-detection/` vào sys.path để import theo tên package đầy đủ
(`ce_localization.*`, `diffuse2seg.*`). Các project dùng chung tên package con (`data`,
`utils`, `config`), nên import trần kiểu `from utils import ...` sẽ nạp nhầm project."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def pytest_configure(config):
    # matplotlib < 3.10 gọi API camelCase của pyparsing (đã deprecated từ pyparsing 3.3) ngay lúc IMPORT mathtext — mà
    # backend Agg luôn import mathtext, nên mọi test lưu hình đều dính dù hình không có chữ toán. Lỗi của thư viện, không
    # của code: chỉ bỏ ĐÚNG loại này từ ĐÚNG module đó. (Nhãn trục log của tool đã bỏ mathtext — các lần parse không còn.)
    config.addinivalue_line("filterwarnings", r"ignore::DeprecationWarning:matplotlib\._mathtext")

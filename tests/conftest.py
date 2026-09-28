"""Gốc `object-detection/` vào sys.path để import theo tên package đầy đủ
(`ce_localization.*`, `diffuse2seg.*`). Các project dùng chung tên package con (`data`,
`utils`, `config`), nên import trần kiểu `from utils import ...` sẽ nạp nhầm project."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

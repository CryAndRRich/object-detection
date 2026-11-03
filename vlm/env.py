"""Key API: `vlm/.env` (gitignore; mẫu `vlm/.env.example`) dạng `KEY=VALUE`. Biến môi trường đã có được ưu tiên. Không bao giờ in
giá trị — log chỉ ghi tên biến + 4 ký tự cuối."""

import os

__all__ = ["ENV_PATH", "load_env", "require_key", "mask"]

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def load_env(path=ENV_PATH):
    """Nạp `path` vào `os.environ` (không ghi đè biến có sẵn). -> tên các biến đọc được trong file (không phải giá trị)."""
    if not os.path.exists(path):
        return []
    names = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().removeprefix("export ").strip()
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            if k:
                names.append(k)
                if v:
                    os.environ.setdefault(k, v)
    return names


def mask(value):
    return f"…{value[-4:]}" if value and len(value) >= 8 else "…"


def require_key(name):
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"thiếu {name}: điền vào {ENV_PATH} (cấu trúc xem .env.example) hoặc export trước khi chạy")
    return v

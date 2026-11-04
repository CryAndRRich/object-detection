"""Key API: `vlm/.env` (gitignore; mẫu `vlm/.env.example`) dạng `KEY=VALUE`. Biến môi trường đã có được ưu tiên. Không bao giờ in
giá trị — log chỉ ghi tên biến + 4 ký tự cuối."""

import os

__all__ = ["ENV_PATH", "load_env", "get_key", "require_key", "mask"]

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
_from_file = {}                     # tên biến -> True nếu giá trị lấy từ .env (không phải biến export sẵn)


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
                if v and k not in os.environ:
                    os.environ[k] = v
                    _from_file[k] = True
    return names


def mask(value):
    return f"…{value[-4:]}" if value and len(value) >= 8 else "…"


PLACEHOLDER = {"none", "null", "nil", "false", "0", "-", "...", "…", "todo", "changeme", "your_key", "your-key"}


def get_key(name):
    """-> key, hoặc None nếu chưa có / để trống / giữ chỗ (`None`, `null`, `<...>`, …) ⇒ coi như không có key."""
    v = os.environ.get(name, "").strip()
    if not v or v.lower() in PLACEHOLDER or (v.startswith("<") and v.endswith(">")):
        return None
    return v


def require_key(name):
    """-> key; thiếu / giữ chỗ (`get_key`) / có ký tự trắng hay ngoài ASCII bên trong (dán lỗi; OpenRouter trả 401 "Missing
    Authentication header") -> dừng (SystemExit; backend có `fallback` thì chuyển key sau), chỉ báo vị trí + mã ký tự, không in key."""
    v = get_key(name)
    if v is None:
        raise SystemExit(f"thiếu {name} (trống / None): điền vào {ENV_PATH} (cấu trúc xem .env.example) hoặc export trước khi chạy")
    bad = [(i, f"U+{ord(c):04X}") for i, c in enumerate(v) if c.isspace() or not 33 <= ord(c) <= 126]
    if bad:
        src = "biến môi trường đã export (được ưu tiên hơn .env)" if name in os.environ and not _from_file.get(name) else ENV_PATH
        raise SystemExit(f"{name} (dài {len(v)}, nguồn: {src}) có ký tự trắng / lạ ở vị trí {bad[:5]} — key phải viết liền một dòng, "
                         f"không dấu cách, không chú thích cùng dòng")
    return v

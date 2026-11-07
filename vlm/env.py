"""Key API: `vlm/.env` (gitignore; mẫu `vlm/.env.example`) dạng `KEY=VALUE`. Biến môi trường đã export sẵn được ưu tiên; trong file,
dòng SAU ghi đè dòng trước. Không bao giờ in giá trị — log chỉ ghi tên biến, nguồn (`key_source`) + 4 ký tự cuối."""

import os

__all__ = ["ENV_PATH", "load_env", "get_key", "require_key", "key_source", "mask"]

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
_from_file = {}                     # tên biến -> giá trị đã nạp từ .env (os.environ khác giá trị này = export sẵn)
_shadowed = set()                   # biến export sẵn khác giá trị trong .env (giá trị .env bị bỏ qua)


def load_env(path=ENV_PATH):
    """Nạp `path` vào `os.environ` (không ghi đè biến export sẵn; trong file dòng sau thắng). -> tên các biến đọc được trong file."""
    if not os.path.exists(path):
        return []
    vals, names = {}, []
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
                    vals[k] = v
    for k, v in vals.items():
        if k not in os.environ or _from_file.get(k) == os.environ[k]:
            os.environ[k] = _from_file[k] = v
            _shadowed.discard(k)
        elif os.environ[k] != v:
            _shadowed.add(k)
    return names


def key_source(name):
    """Nguồn giá trị đang dùng của `name` (để log, không in giá trị)."""
    if name in os.environ and _from_file.get(name) == os.environ[name]:
        return ".env"
    if name in _shadowed and name in os.environ:
        return (f"biến môi trường export sẵn (KHÁC giá trị trong .env — .env bị bỏ qua; `unset {name}` hoặc xoá export trong "
                f"~/.bashrc)")
    return "biến môi trường export sẵn" if name in os.environ else "không có"


def mask(value):
    return f"…{value[-4:]}" if value and len(value) >= 8 else "…"


PLACEHOLDER = {"none", "null", "nil", "false", "0", "-", "...", "…", "todo", "changeme", "your_key", "your-key"}


def get_key(name):
    """-> key, hoặc None nếu chưa có / để trống / giữ chỗ (`None`, `null`, `<...>`, chữ mẫu `sk-your-…-key`, `xxxx`)."""
    v = os.environ.get(name, "").strip()
    low = v.lower()
    if not v or low in PLACEHOLDER or (v.startswith("<") and v.endswith(">")) or "your" in low or "xxxx" in low:   # sk-your-…-key
        return None
    return v


def require_key(name):
    """-> key; thiếu / giữ chỗ (`get_key`) / có ký tự trắng hay ngoài ASCII bên trong (dán lỗi; OpenRouter trả 401 "Missing
    Authentication header") -> dừng (SystemExit), chỉ báo vị trí + mã ký tự, không in key."""
    v = get_key(name)
    if v is None:
        raise SystemExit(f"thiếu {name} (trống / None): điền vào {ENV_PATH} (cấu trúc xem .env.example) hoặc export trước khi chạy")
    bad = [(i, f"U+{ord(c):04X}") for i, c in enumerate(v) if c.isspace() or not 33 <= ord(c) <= 126]
    if bad:
        src = ENV_PATH if key_source(name) == ".env" else "biến môi trường đã export (được ưu tiên hơn .env)"
        raise SystemExit(f"{name} (dài {len(v)}, nguồn: {src}) có ký tự trắng / lạ ở vị trí {bad[:5]} — key phải viết liền một dòng, "
                         f"không dấu cách, không chú thích cùng dòng")
    return v

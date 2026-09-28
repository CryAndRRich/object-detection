"""Lưu / nạp checkpoint để train bị ngắt vẫn giữ được weight và TRAIN TIẾP được.

HAI FILE, ghi sau MỖI epoch:
  - `last.pt` : trạng thái mới nhất — đủ để train tiếp (`--resume`).
  - `best.pt` : epoch có `oracle_recall` cao nhất — dùng cho eval.
Cả hai cùng định dạng: `eval.py` đọc `model` / `config` / `epoch` như trước, còn các khoá
còn lại (optimizer, history, RNG...) chỉ `train.py --resume` dùng.

GHI NGUYÊN TỬ: ghi ra `*.tmp` rồi `os.replace` sang tên thật. Nếu job bị giết đúng lúc
đang ghi (ssh rớt, `kill`, GPU dùng chung bị OOM), file cũ vẫn nguyên vẹn — `torch.save`
thẳng vào `last.pt` thì một lần ngắt giữa chừng làm mất CẢ bản cũ lẫn bản mới.
"""

import os
import shutil

import torch

__all__ = ["CheckpointManager", "MODEL_KEYS_MUST_MATCH", "rng_state", "set_rng_state"]

# Các nhánh config quyết định KIẾN TRÚC / BÀI TOÁN. Khác nhau thì weight cũ không còn
# nghĩa với model mới -> từ chối resume. Nhánh `training` (batch, lr, epochs) được phép
# đổi giữa chừng, chỉ in cảnh báo.
MODEL_KEYS_MUST_MATCH = ("model", "diffusion", "matcher", "data")
# Khoá trong các nhánh trên mà đổi thì KHÔNG ảnh hưởng weight (chỉ tốc độ đọc dữ liệu).
_IGNORE = {"data": {"num_workers"}}


def _strip(branch, section):
    if not isinstance(section, dict):
        return section
    return {k: v for k, v in section.items() if k not in _IGNORE.get(branch, ())}


class CheckpointManager:
    LAST, BEST = "last.pt", "best.pt"

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    @property
    def last_path(self):
        return os.path.join(self.save_dir, self.LAST)

    @property
    def best_path(self):
        return os.path.join(self.save_dir, self.BEST)

    def has_last(self):
        return os.path.exists(self.last_path)

    # ------------------------------------------------------------------ ghi

    @staticmethod
    def _atomic_save(obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        with open(tmp, "rb") as f:          # đẩy xuống đĩa trước khi đổi tên
            os.fsync(f.fileno())
        os.replace(tmp, path)               # nguyên tử trên cùng một filesystem

    def save(self, state, is_best):
        """Ghi `last.pt`; nếu `is_best` thì chép sang `best.pt`.

        Chép file thay vì `torch.save` lần hai: tránh serialize ~700 MB hai lần. Chép
        cũng qua `.tmp` + `os.replace` nên `best.pt` không bao giờ ở trạng thái dở.
        """
        self._atomic_save(state, self.last_path)
        if is_best:
            tmp = self.best_path + ".tmp"
            shutil.copyfile(self.last_path, tmp)
            os.replace(tmp, self.best_path)

    # ------------------------------------------------------------------ đọc

    def load_last(self, map_location="cpu"):
        return torch.load(self.last_path, map_location=map_location, weights_only=False)

    @staticmethod
    def config_mismatch(saved_cfg, cfg):
        """-> (lỗi, cảnh báo): danh sách nhánh config khác nhau giữa checkpoint và hiện tại."""
        errors = [k for k in MODEL_KEYS_MUST_MATCH
                  if _strip(k, saved_cfg.get(k)) != _strip(k, cfg.get(k))]
        warns = [k for k in ("training",) if saved_cfg.get(k) != cfg.get(k)]
        return errors, warns


def rng_state(generator):
    """Gói mọi nguồn ngẫu nhiên ảnh hưởng tới train: `t` / nhiễu của diffusion đến từ
    `generator`, còn thứ tự xáo DataLoader và dropout đến từ RNG toàn cục."""
    st = {"torch": torch.get_rng_state(), "generator": generator.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st, generator):
    torch.set_rng_state(st["torch"])
    generator.set_state(st["generator"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])

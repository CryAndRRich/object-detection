"""Lưu / nạp checkpoint GHI NGUYÊN TỬ: ghi ra `*.tmp`, fsync, rồi `os.replace` sang tên thật — job bị
giết đúng lúc đang ghi (ssh rớt, `kill`, OOM trên GPU dùng chung) thì file cũ vẫn nguyên.
`last` = trạng thái mới nhất (đủ để `--resume`), `best` = bản chép của `last` khi chỉ số chọn cải thiện.
`engine/train_utils.PthCheckpoints` đổi tên file sang `.pth` (pickle không zip, quy tắc Kaggle)."""

import os
import shutil

import torch

__all__ = ["CheckpointManager"]


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

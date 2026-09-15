"""Chọn nguồn SD2: thư mục local nếu có, ngược lại tên repo HF.

VÌ SAO CÓ FILE NÀY: server aiotlab KHÔNG ra được HuggingFace — một repo CÔNG
KHAI trả về 401 / "Repository Not Found" / "Invalid username or password" trong
khi HF_TOKEN rỗng và không có file token nào (đo 2026-09-15). Không có credential
nào để mà sai, nên chặn nằm ở tầng mạng và token cũng không cứu được.

Nên SD2 tải ở local rồi đưa lên theo quy ước weights/. Logic chọn nguồn phải
đúng, và phải đúng KHÔNG CẦN SD2 để test — đó là việc của file này.

Run:  python -m pytest tests/test_model_source.py -q
      python tests/test_model_source.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base import Diffu2SegConfig  # noqa: E402


def test_falls_back_to_hub_id_when_dir_missing():
    cfg = Diffu2SegConfig(local_model_dir="/khong/ton/tai/o/dau/ca")
    assert cfg.model_source == cfg.hf_model_id


def test_uses_local_dir_when_it_has_model_index():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "model_index.json"), "w") as f:
            json.dump({"_class_name": "StableDiffusionPipeline"}, f)
        cfg = Diffu2SegConfig(local_model_dir=d)
        assert cfg.model_source == d


def test_empty_dir_is_not_accepted():
    """Thư mục rỗng do giải nén hỏng phải bị từ chối NGAY.

    Chỉ kiểm os.path.isdir là chưa đủ: một thư mục rỗng sẽ lọt qua rồi mới chết
    ở from_pretrained với thông báo khó hiểu, sau khi đã tốn thời gian khởi động.
    model_index.json là file from_pretrained đọc đầu tiên nên nó là điều kiện
    đúng để kiểm.
    """
    with tempfile.TemporaryDirectory() as d:
        cfg = Diffu2SegConfig(local_model_dir=d)
        assert cfg.model_source == cfg.hf_model_id, \
            "thư mục rỗng phải rơi về hub id, không được dùng"


def test_relative_path_resolves_against_project_root():
    """Đường dẫn mặc định là tương đối so với diffu2seg/, không phải cwd.

    Nếu phân giải theo cwd thì chạy tool từ thư mục khác sẽ im lặng không thấy
    model rồi quay ra gọi mạng — đúng thứ đang hỏng trên server.
    """
    cfg = Diffu2SegConfig()
    assert cfg.local_model_dir.startswith("..")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as tmp:
        old = os.getcwd()
        try:
            os.chdir(tmp)                    # cwd khác hẳn project root
            src = cfg.model_source
        finally:
            os.chdir(old)

    expected = os.path.normpath(os.path.join(root, cfg.local_model_dir))
    assert src in (expected, cfg.hf_model_id)
    assert not src.startswith(tmp), "không được phân giải theo cwd"


def test_default_points_into_weights_dir():
    """Khớp quy ước weights/ của dự án (zip thủ công, không scp/rsync)."""
    cfg = Diffu2SegConfig()
    assert "weights" in cfg.local_model_dir
    assert "stable-diffusion" in cfg.local_model_dir


def test_to_dict_records_the_actual_source():
    """Log phải ghi nguồn THẬT đã dùng, không chỉ tên repo.

    Nếu chỉ log hf_model_id thì đọc log cũ sẽ không biết lần chạy đó lấy model
    từ đâu — mà đó đúng là câu hỏi đã tốn thời gian với detectron2 của D.1.
    """
    d = Diffu2SegConfig().to_dict()
    assert "model_source" in d
    assert "local_model_dir" in d and "hf_model_id" in d


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

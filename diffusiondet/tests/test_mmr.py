"""Self-test cho metric mMR/Recall/AP50. Chạy: python tests/test_mmr.py

Không cần detectron2 — chỉ numpy. Mục đích là bắt lỗi logic trong metric trước khi đốt
giờ GPU, vì một metric sai sẽ cho ra bảng kết quả trông rất hợp lý mà vô nghĩa.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from objdet.mmr import compute_mmr_and_recall

NO_IGNORE = np.zeros((0, 4))
GTS = np.array([[10, 10, 50, 50], [100, 100, 150, 150]], dtype=float)
DETS = np.array([[10, 10, 50, 50, 0.99], [100, 100, 150, 150, 0.98]], dtype=float)


def check(name, got, **want):
    line = "  ".join(f"{k}={got[k]:.2f}" for k in ("AP50", "mMR", "Recall"))
    print(f"  {name:34s} {line}")
    for k, v in want.items():
        assert abs(got[k] - v) < 0.01, f"{name}: {k} = {got[k]}, kỳ vọng {v}"


def main():
    print("test_mmr:")

    # detector hoàn hảo: trúng hết, không FP -> recall 100, mMR ~ 0
    check("hoàn hảo", compute_mmr_and_recall({1: (DETS, GTS, NO_IGNORE)}),
          Recall=100.0, mMR=0.0, AP50=100.0)

    # bỏ sót 1 trong 2 GT -> recall 50, MR = 50 ở mọi mốc FPPI nên mMR = 50
    check("bỏ sót 1/2", compute_mmr_and_recall({1: (DETS[:1], GTS, NO_IGNORE)}),
          Recall=50.0, mMR=50.0)

    # Detection thứ 3 nằm trong vùng ignore. Score của nó phải CAO hơn mọi true positive,
    # nếu không thì phép thử vô nghĩa: AP kiểu VOC bỏ qua false positive xếp sau khi
    # recall đã đạt 100%, nên một FP score thấp không làm AP giảm và hai nhánh có/không
    # khai ignore sẽ ra cùng kết quả.
    ign = np.array([[200, 200, 300, 300]], dtype=float)
    d3 = np.concatenate([[[210, 210, 290, 290, 0.995]], DETS])
    with_ign = compute_mmr_and_recall({1: (d3, GTS, ign)})
    check("có khai ignore", with_ign, Recall=100.0, mMR=0.0, AP50=100.0)

    # cùng detection đó nhưng không khai ignore -> thành FP xếp đầu, AP50 và mMR đều xấu đi
    without = compute_mmr_and_recall({1: (d3, GTS, NO_IGNORE)})
    check("không khai ignore", without, Recall=100.0)
    assert without["AP50"] < with_ign["AP50"], \
        f"vùng ignore phải loại được false positive: {without['AP50']} vs {with_ign['AP50']}"
    assert without["mMR"] > with_ign["mMR"], \
        f"FP score cao phải làm mMR xấu đi: {without['mMR']} vs {with_ign['mMR']}"

    # ảnh không có detection nào vẫn phải tính vào recall
    two_img = compute_mmr_and_recall({
        1: (DETS, GTS, NO_IGNORE),
        2: (np.zeros((0, 5)), GTS, NO_IGNORE),
    })
    check("1/2 ảnh không detect", two_img, Recall=50.0, mMR=50.0)

    # detector lệch hẳn (IoU < 0.5) -> không trúng gì
    off = np.array([[300, 300, 340, 340, 0.9]], dtype=float)
    check("detect sai chỗ", compute_mmr_and_recall({1: (off, GTS, NO_IGNORE)}),
          Recall=0.0, mMR=100.0, AP50=0.0)

    # không có GT -> nan chứ không crash
    nan_res = compute_mmr_and_recall({1: (DETS, NO_IGNORE, NO_IGNORE)})
    assert np.isnan(nan_res["mMR"]), nan_res
    print("  không có GT                        -> nan (không crash)")

    print("TẤT CẢ ĐẠT")


if __name__ == "__main__":
    main()

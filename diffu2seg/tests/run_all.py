#!/usr/bin/env python3
"""Chạy mọi test suite, báo cáo theo checklist.

    python tests/run_all.py

Mỗi suite chạy trong một subprocess riêng, để một crash không kéo đổ phần còn
lại và để output tách bạch được. Suite nào cần thứ có thể chưa có (dữ liệu
CE-130) thì tự skip sạch bên trong.

MỌI SUITE Ở ĐÂY CHẠY ĐƯỢC TRÊN CPU, KHÔNG CẦN GPU, KHÔNG CẦN SD2. Chúng kiểm
phần toán học và phần ghép nối. Việc SD2 thật có tách được vật trên CE-130 hay
không thì KHÔNG suite nào ở đây trả lời được — đó là tools/check_attention_
separates.py, chạy trên server. Một bộ test xanh ở đây KHÔNG có nghĩa là
"phương pháp chạy được".
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUITES = [
    ("test_plaplacian.py",
     "Algorithm 1: khai triển matmul == vòng lặp ngây thơ (fp64), clamp g, hội tụ"),
    ("test_affinity.py",
     "(h,w,h,w) -> (N,N) row-stochastic, nhiệt độ, không IPF"),
    ("test_prompts.py",
     "lưới prompt, f0 one-hot chính xác, loại vùng pad, độ phủ"),
    ("test_masks_boxes.py",
     "connected components, rel_floor, box outer-edge, round-trip toạ độ"),
    ("test_pipeline.py",
     "ghép toàn mạch trên affinity giả lập; p<2 chặn rò nền"),
    ("test_metrics.py",
     "cộng dồn thô, ảnh rỗng trả n_gt, không có score_AUC"),
    ("test_ce130_loader.py",
     "COCO json thật: 908 ảnh / 38289 box, pad CLIP-mean ở dưới"),
]


def run_suite(name: str) -> bool:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests" / name)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(("ok ", "SKIP", "FAIL")) or "passed" in stripped:
            print(line)
    if result.returncode != 0:
        print(result.stderr[-3000:])
    return result.returncode == 0


def main() -> int:
    print("Diffu2Seg — bộ test (CPU, không cần GPU/SD2)")
    results = []
    for name, desc in SUITES:
        path = ROOT / "tests" / name
        if not path.exists():
            print(f"\n{'=' * 78}\n{name}\n{'=' * 78}\nMISSING: {path}")
            results.append((name, desc, False))
            continue
        results.append((name, desc, run_suite(name)))

    print(f"\n{'=' * 78}\nTỔNG KẾT\n{'=' * 78}")
    for name, desc, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:26s} {desc}")

    n_ok = sum(1 for _, _, ok in results if ok)
    print(f"\n{n_ok}/{len(results)} suite pass")

    if n_ok == len(results):
        print("\nBước tiếp theo (TRÊN SERVER, người dùng tự chạy):")
        print("  export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache")
        print("  python tools/check_attention_separates.py --split val --limit 30")
        print("     ^ CỬA CHẶN 0, ~5 phút, có thể kết luận sớm cả hướng")
        print("\n⚠️ Bộ test xanh KHÔNG có nghĩa là phương pháp chạy được — nó chỉ")
        print("   nói phần toán và phần ghép nối đúng. Cửa chặn mới trả lời.")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

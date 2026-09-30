#!/usr/bin/env python3
"""Chạy `main.py` của Open-GroundingDino NGAY TRONG tiến trình này, sau khi cài shim op
(`runtime.install_msda_fallback`) — không sửa file nào của repo ngoài. `gdino/train.py` gọi script
này (1 tiến trình hoặc qua `torch.distributed.run`); tham số dòng lệnh chuyển nguyên cho main.py.
Repo lấy từ biến môi trường BASELINE_OG_REPO.
"""

import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.gdino.runtime import setup_og  # noqa: E402


def main():
    repo = os.environ["BASELINE_OG_REPO"]
    shim = setup_og(repo)
    print(f"[gdino launch] repo {repo} | op MultiScaleDeformableAttention: "
          f"{'shim PyTorch thuần' if shim else 'CUDA'}", flush=True)
    os.chdir(repo)
    sys.argv = [os.path.join(repo, "main.py"), *sys.argv[1:]]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Đóng gói web chấm (docs/EXPERIMENT_GAMMA.md mục 17) thành MỘT file zip để người khác chấm trên máy họ, không cần repo / GPU.

Zip chứa: `app.py` (chỉ cần Pillow + gradio), mỗi bộ dữ liệu một thư mục `data/<i>_<tên>/` với items*.json (giữ tên file ⇒ nhãn
`ratings*.jsonl` ghi cạnh đó), ảnh `images/<j>/...` — CHỈ những ảnh items dùng tới, giữ đường dẫn tương đối (bộ dùng chung thư mục ảnh
thì dùng chung một bản), `run.sh` / `run.bat` (lệnh mở web đủ mọi bộ, `--rater` nếu có), `requirements.txt`, `README.txt`.
Không kèm nhãn của người đóng gói (người chấm mới bắt đầu từ đầu). Ảnh lưu nguyên (ZIP_STORED — PNG / JPG nén thêm không được gì).

  cd /mnt/disk1/aiotlab/haitn/object-detection/ce_localization
  python user_study/pack.py --out /mnt/disk1/aiotlab/haitn/output/gamma/user_study_pkg/user_study_<tên>.zip --rater <tên> \\
      --data "CE-130" ../../output/gamma/user_study/items.json ../data/samples \\
      --data "CE-CoCount" ../../output/gamma/user_study_cocount/items.json ../data/cocount \\
      --data "CE-CoCount (box resize)" ../../output/gamma/user_study_cocount/items_objsize.json ../data/cocount
"""

import argparse
import json
import os
import re
import sys
import time
import zipfile

__all__ = ["slug", "pack"]

REQUIREMENTS = "gradio==6.17.3\nhuggingface_hub>=0.36,<1.0\npillow\n"
README = """User study — chấm box đề xuất (CE-Loc)

1. Cài Python >= 3.10, rồi mở terminal TRONG thư mục này:
     python -m venv .venv
     macOS / Linux:  .venv/bin/pip install -r requirements.txt
     Windows:        .venv\\Scripts\\pip install -r requirements.txt
2. Mở web:
     macOS / Linux:  bash run.sh        (dùng .venv nếu có)
     Windows:        run.bat
   Trình duyệt tự mở http://127.0.0.1:7860. Ô "Dataset" đổi bộ dữ liệu, ô "Models" chọn model.
3. Chấm: bấm vào box (hoặc phím 1..4) để đánh dấu box KHÔNG ổn + chọn lý do; Enter = lưu và sang màn sau;
   ← / → = màn trước / sau (không lưu). Tắt web lúc nào cũng được, mở lại sẽ tiếp từ màn chưa lưu.
4. Nhãn được lưu ngay mỗi lần Save vào các file data/*/ratings*.jsonl.
   Chấm xong (hoặc giữa chừng) gửi lại TẤT CẢ các file đó (nén cả thư mục data/ cũng được, không cần gửi ảnh).
"""


def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower() or "data"


def pack(data, out, app_path, rater=None, log=print):
    """data: list (tên, items.json, thư mục ảnh) -> ghi zip `out` (ghi ra .tmp rồi đổi tên). -> dict thống kê."""
    roots, plan, cmd = {}, [], []
    for i, (name, items, root) in enumerate(data):
        with open(items) as f:
            d = json.load(f)
        j = roots.setdefault(os.path.abspath(root), len(roots))
        imgs = sorted({it["image"] for it in d["items"].values()})
        miss = [p for p in imgs if not os.path.isfile(os.path.join(root, p))]
        if miss:
            raise FileNotFoundError(f"{name}: thiếu {len(miss)} ảnh trong {root} (vd {miss[:3]})")
        ddir = f"data/{i}_{slug(name)}"
        plan.append((items, f"{ddir}/{os.path.basename(items)}", root, j, imgs))
        cmd.append((name, f"{ddir}/{os.path.basename(items)}", f"images/{j}"))
    files = {}                                                  # đường dẫn trong zip -> nguồn (ảnh chung thư mục: một bản)
    for _, _, root, j, imgs in plan:
        for p in imgs:
            files[f"images/{j}/{p}"] = os.path.join(root, p)
    rater_arg = f" --rater {rater}" if rater else ""
    sh = ('#!/usr/bin/env bash\ncd "$(dirname "$0")"\nPY=python3\n[ -x .venv/bin/python ] && PY=.venv/bin/python\n'
          '"$PY" app.py' + "".join(f' \\\n    --data "{n}" {it} {im}' for n, it, im in cmd) + rater_arg + "\n")
    bat = ('@echo off\r\ncd /d "%~dp0"\r\nset PY=python\r\nif exist .venv\\Scripts\\python.exe set PY=.venv\\Scripts\\python.exe\r\n'
           "%PY% app.py" + "".join(f' --data "{n}" {it} {im}' for n, it, im in cmd) + rater_arg + "\r\n")
    total = sum(os.path.getsize(s) for s in files.values())
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    t0, done = time.time(), 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(app_path, "app.py")
        for items, arc, _, _, _ in plan:
            z.write(items, arc)
        z.writestr("requirements.txt", REQUIREMENTS)
        z.writestr("README.txt", README)
        zi = zipfile.ZipInfo("run.sh", date_time=time.localtime()[:6])
        zi.external_attr = 0o755 << 16
        z.writestr(zi, sh, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("run.bat", bat)
        step = max(len(files) // 20, 1)
        for k, (arc, src) in enumerate(sorted(files.items()), 1):
            z.write(src, arc, compress_type=zipfile.ZIP_STORED)
            done += os.path.getsize(src)
            if k % step == 0 or k == len(files):
                el = time.time() - t0
                log(f"  [{k}/{len(files)} ảnh] {done / 1e6:.0f} / {total / 1e6:.0f} MB | {el:.0f}s | còn ~"
                    f"{el / max(done, 1) * (total - done):.0f}s", flush=True)
    os.replace(tmp, out)
    return {"n_images": len(files), "mb": total / 1e6, "datasets": [c[0] for c in cmd], "zip_mb": os.path.getsize(out) / 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs=3, action="append", required=True, metavar=("NAME", "ITEMS", "IMAGE_ROOT"),
                    help="như app.py: tên hiện trên web, items*.json của build.py, thư mục ảnh")
    ap.add_argument("--out", required=True, help="file .zip (vd /mnt/disk1/aiotlab/haitn/output/gamma/user_study_pkg/...)")
    ap.add_argument("--rater", default=None, help="tên người chấm: nhãn ghi ra ratings*_<tên>.jsonl (gộp về không trùng file)")
    a = ap.parse_args()
    if len({n for n, _, _ in a.data}) != len(a.data):
        sys.exit("tên bộ dữ liệu trùng nhau")
    res = pack(a.data, a.out, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"), a.rater)
    print(f"-> {a.out}: {res['zip_mb']:.0f} MB, {res['n_images']} ảnh, bộ {res['datasets']}", flush=True)


if __name__ == "__main__":
    main()

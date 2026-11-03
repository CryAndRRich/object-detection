"""DÙNG MỘT LẦN (2026-11-03), chạy xong thì xoá file: dời log/ + output/ sang bố cục mới (như local).
  cd /mnt/disk1/aiotlab/haitn/object-detection && python3 tools/reorg_output_1103.py          # chỉ in kế hoạch
  python3 tools/reorg_output_1103.py --run                                                     # dời thật
Không ghi đè: đích có sẵn và giống hệt -> xoá nguồn; khác -> dừng. File không khớp quy tắc -> liệt kê, giữ nguyên chỗ cũ."""
import filecmp
import os
import re
import sys

R = os.environ.get("REORG_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # gốc: log/ output/ object-detection/
O, L, D = "output/add", "log/add", "object-detection/data"
RUN = r"(delta\d+(?:_\d+)*|zeta\d+)"
ONE_RUN_LOG = [r"eval_(delta7_0_1)_gpu\d_.*", r"eval_(delta7_0_2)_gpu\d_.*", r"eval_(delta7_1)_gpu\d_.*", r"eval_(delta8)_gpu\d_.*",
               r"eval_(delta8)_train_gpu\d_.*", r"eval_(zeta\d)_snapshots_.*", r"eval_(delta5_5)_\d{4}_\d{4}\.log", r"eval_(delta7)_[ABC]_.*"]
j = os.path.join


def ex(p):
    return os.path.exists(j(R, p))


def ls(p):
    return sorted(os.listdir(j(R, p))) if os.path.isdir(j(R, p)) else []


moves, dels, skip = [], [], []
for run in ls(j(O, "snapshots")):                                       # snapshots/<run>/<tập>/* -> <run>/<tập>/*
    for ds in ls(j(O, "snapshots", run)):
        for f in ls(j(O, "snapshots", run, ds)):
            moves.append((j(O, "snapshots", run, ds, f), j(O, run, ds, f)))
for ds in ls(j(O, "eval_v2")):                                          # eval_v2/<tập>/test_<run>[_boxes].json -> <run>/<tập>/best*
    for f in ls(j(O, "eval_v2", ds)):
        m = re.fullmatch(rf"test_{RUN}(_boxes)?\.json", f)
        if m:
            moves.append((j(O, "eval_v2", ds, f), j(O, m[1], ds, f"best{m[2] or ''}.json")))
        else:
            skip.append(j(O, "eval_v2", ds, f))
for sub in ("filt3", "review5"):                                        # cocount/<x>/... -> <run>/legacy/cocount_<x>/
    for f in ls(j(O, "cocount", sub)):
        if f == "gt_cnll.json":
            moves.append((j(O, "cocount", sub, f), j(O, "gt", "legacy", f"cocount_{sub}_gt_cnll.json")))
            continue
        m = re.fullmatch(rf"(boxes|test|rescore)_{RUN}(_[a-z][a-z0-9_]*)?\.json", f)
        if m:
            moves.append((j(O, "cocount", sub, f), j(O, m[2], "legacy", f"cocount_{sub}", f"{m[1]}{m[3] or ''}.json")))
        else:
            skip.append(j(O, "cocount", sub, f))
for sub in ("ce130_cnll_test", "ce130_cnll_train"):                     # ce130_cnll_*/<kind>_<run>.json -> <run>/legacy/<sub>/
    for f in ls(j(O, sub)):
        m = re.fullmatch(rf"(boxes|test)_{RUN}\.json", f)
        if m:
            moves.append((j(O, sub, f), j(O, m[2], "legacy", sub, f"{m[1]}.json")))
        else:
            skip.append(j(O, sub, f))
for run in ls(O):                                                       # <run>/test_*.json trần (giao thức cũ) -> <run>/legacy/ce130_test/
    if re.fullmatch(RUN, run):
        for f in ls(j(O, run)):
            if os.path.isfile(j(R, O, run, f)) and f.startswith("test_"):
                moves.append((j(O, run, f), j(O, run, "legacy", "ce130_test", f)))
br = j(O, "box_review")                                                 # box_review: input -> data/, kết quả rà -> review/
for f in ls(br):
    if f == "cocount.jsonl":
        moves.append((j(br, f), j(D, "cocount_review.jsonl")))
    elif f == "ce130_train.jsonl":
        moves.append((j(br, f), j(D, "ce130_train.jsonl")))
    elif f.startswith("cocount.jsonl.bak"):
        dels.append(j(br, f))
    elif re.fullmatch(r"pred_delta8_ep010_.*_user_\d\.jsonl", f):
        moves.append((j(br, f), j(O, "review", "delta8_ep010", f)))
    elif f in ("pred_ce130.jsonl", "pred_cocount.jsonl"):
        ds = "ce130_test_all" if "ce130" in f else "cocount_review5"
        moves.append((j(br, f), j(O, "review", "delta8_ep010", f"pred_delta8_ep010_{ds}_merged.jsonl")))
    else:
        skip.append(j(br, f))
for sub, dst in (("snapshots", "compare"), ("box_review", "single")):   # viz: png trần -> thư mục con
    for f in ls(j(O, "viz", sub)):
        if os.path.isfile(j(R, O, "viz", sub, f)):
            moves.append((j(O, "viz", sub, f), j(O, "viz", sub, dst, f)))
for f in ls(j(L, "eval_chain")):                                        # log eval của một run -> log/add/<run>/
    m = next((re.fullmatch(p, f) for p in ONE_RUN_LOG if re.fullmatch(p, f)), None)
    if m:
        moves.append((j(L, "eval_chain", f), j(L, m[1], f)))
for f in ls(j(L, "vlm")):
    if f.endswith(".log") and "_launch_" in f:
        moves.append((j(L, "vlm", f), j(L, "vlm", "launch", f)))
dels += [j(L, f) for f in ls(L) if f.startswith("zip_kaggle_") and f.endswith(".log")]

dst = [d for _, d in moves]
assert len(set(dst)) == len(dst), "hai nguồn cùng một đích"
same = [(s, d) for s, d in moves if ex(d) and os.path.isfile(j(R, d)) and filecmp.cmp(j(R, s), j(R, d), shallow=False)]
clash = [d for s, d in moves if ex(d) and (s, d) not in same]
print(f"{len(moves)} mv ({len(same)} đích đã có giống hệt -> xoá nguồn), {len(dels)} xoá | đích khác nội dung: {len(clash)} {clash[:5]}")
print(f"bỏ qua (không khớp quy tắc, giữ nguyên chỗ cũ): {len(skip)}", *skip[:30], sep="\n  ")
OLD = ("snapshots", "eval_v2", "cocount", "ce130_cnll_test", "ce130_cnll_train", "box_review")
print("thư mục khác ở output/add (không đụng):", [p for p in ls(O) if p not in OLD + ("viz", "vlm", "review", "gt") and not re.fullmatch(RUN, p)])
if "--run" not in sys.argv or clash:
    sys.exit("DỪNG: có đích khác nội dung" if clash else "chỉ in kế hoạch (thêm --run để dời)")
for s, d in moves:
    if (s, d) in same:
        os.remove(j(R, s))
        continue
    os.makedirs(os.path.dirname(j(R, d)), exist_ok=True)
    os.rename(j(R, s), j(R, d))
for d in dels:
    os.remove(j(R, d))
for top in OLD:
    p = j(R, O, top)
    for dp, _, _ in (sorted(os.walk(p), key=lambda x: -len(x[0])) if os.path.isdir(p) else []):
        if not os.listdir(dp):
            os.rmdir(dp)
print("xong; thư mục cũ còn lại (có file bỏ qua):", [t for t in OLD if ex(j(O, t))])

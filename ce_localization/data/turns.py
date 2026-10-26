"""Dữ liệu CE-130 của bài ADD: một mẫu = (nhánh, lượt t).

Nhánh `all_phase2_V2/<split>/<iid>_b<k>/` xoá CỘNG DỒN tối đa 4 vật: ảnh lượt t thiếu vật 1..t (`inpainted_bboxes[:t]`, xyxy pixel ảnh
gốc). Đích train = lỗ MỚI NHẤT (như CE-Loc gốc); các lỗ cũ cũng là chỗ thêm hợp lệ (chỉ để chấm). `samples/{train,test}/images/{iid}_{j}.png`
của bài = đúng các ảnh inpaint đó, kèm `density/` (density CountGD của chính ảnh) và `annotation/` (`target_bbox` cxcywh); chỉ mục
`data/turn_index.json` (`tools/build_turn_index.py`) ghép (nhánh, lượt) <-> file samples bằng hash pixel.

Split như CE-Loc gốc: train = mọi mẫu `samples/train`; val / test = mẫu `samples/test` có ảnh gốc thuộc val / test CE-130 (1.174 / 4.948);
`test_all` = toàn bộ `samples/test` (6.122, như tác giả). Đầu vào kiểu bài: letterbox góc trên-trái (BILINEAR, đệm đen), ảnh /255,
density `.convert("L")` (NEAREST). `image="original"` = `ground_truth.jpg` của nhánh (không lỗ) + density `full` của ảnh gốc.
Dataset trả ảnh uint8 HWC; `to_device_add` mới /255 trên thiết bị. `AddCache` (Kaggle): ảnh + density đã letterbox, uint8 [T,T,4] trong
một file (zlib hoặc memmap raw).
"""

import glob
import hashlib
import json
import os
import zlib
from collections import defaultdict
from multiprocessing import get_context

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ce_localization.utils.box_ops_np import box_iou, filter_degenerate

__all__ = ["SPLITS", "IMAGE_KINDS", "letterbox", "scale_boxes", "read_annotation", "pixel_hash", "assign_removed", "build_turn_index",
           "TurnIndex", "AddCache", "build_add_cache", "convert_add_cache", "CE130AddDataset", "collate_add", "to_device_add",
           "paper_to_float", "load_positives", "hand_positives", "POSITIVE_SAMPLING"]

SPLITS = ("train", "val", "test")
IMAGE_KINDS = ("inpainted", "original")
# đích train khi có box positive (`CE130AddDataset.set_positives`, `train.step_loss`): random = 1 box ngẫu nhiên / ảnh / bước (DELTA7);
# all = mọi box của ảnh trong cùng bước, loss TB trong ảnh (DELTA7.0.1); pair = mỗi cặp (ảnh, box) là một phần tử (DELTA7.0.2)
POSITIVE_SAMPLING = ("random", "all", "pair")
TARGET_TOL_PX = 1.0                             # target_bbox (cxcywh, làm tròn) vs inpainted_bboxes (xyxy)


def letterbox(img, target=512):
    """PIL RGB -> (canvas uint8 [T,T,3], scale, nw, nh). Y như `resize_and_pad` của CE-Loc gốc."""
    w, h = img.size
    scale = min(target / w, target / h)
    nw, nh = int(w * scale), int(h * scale)
    canvas = np.zeros((target, target, 3), dtype=np.uint8)            # đệm ĐEN như bản gốc
    canvas[:nh, :nw] = np.asarray(img.resize((nw, nh), resample=Image.BILINEAR), dtype=np.uint8)
    return canvas, scale, nw, nh


def scale_boxes(boxes_xyxy_px, scale, nw, nh):
    """Box xyxy pixel ảnh gốc -> xyxy pixel canvas, kẹp vào vùng ảnh thật, bỏ box rỗng."""
    b = np.asarray(boxes_xyxy_px, dtype=np.float64).reshape(-1, 4) * scale
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, nw)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, nh)
    keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    return b[keep]


def load_positives(path):
    """`box_review/ce130_train.jsonl` (một dòng một mẫu `samples/train`: `id` = tên file, `positive` = box xyxy pixel ảnh gốc người gán
    nhãn chấm đặt được vật — đề xuất của model, gồm cả chính các lỗ; `negative` / `neutral` không dùng) -> {id: mảng [P,4]}."""
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = np.asarray(r["positive"], dtype=np.float64).reshape(-1, 4)
    return out


def hand_positives(boxes, holes, hole_iou):
    """Box positive gán tay KHÔNG trùng lỗ nào (IoU với mọi lỗ <= `hole_iou`): box trùng lỗ là chính lỗ đó (model đề xuất lại) — giữ
    thì lỗ bị tính nhiều lần khi rút đích."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if not len(boxes) or not len(holes):
        return boxes
    return boxes[box_iou(boxes, holes)[0].max(1) <= hole_iou]


def read_annotation(branch_dir):
    """`fixed_annotation.json` (chỉ val / test) hoặc `annotation.json`; None nếu không có."""
    for name in ("fixed_annotation.json", "annotation.json"):
        path = os.path.join(branch_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def _pool(workers):
    """Pool `spawn`: tiến trình gọi đã nạp torch (đa luồng) — fork từ đó có thể treo con (Python 3.12 cảnh báo)."""
    return get_context("spawn").Pool(workers)


def pixel_hash(path):
    """md5 của mảng pixel RGB + kích thước — hai file khác byte nhưng cùng pixel cho cùng hash."""
    a = np.asarray(Image.open(path).convert("RGB"))
    return hashlib.md5(a.tobytes() + str(a.shape).encode()).hexdigest()


def assign_removed(objects_xyxy, holes_xyxy, min_iou=0.5):
    """Mỗi lỗ (theo thứ tự lượt) ghép vật `all_bboxes` chưa ghép có IoU lớn nhất (>= min_iou).
    `all_bboxes` và `inpainted_bboxes` lệch nhẹ (SPATIAL_SOFTMAX mục 3: 69 % nhánh IoU < 0,9, trung vị 0,83).
    -> (removed [M] int: lượt bị xoá, 0 = còn trong ảnh mọi lượt ; số lỗ không ghép được)."""
    removed = np.zeros(len(objects_xyxy), dtype=int)
    miss = 0
    for t, h in enumerate(holes_xyxy, 1):
        if not len(objects_xyxy):
            miss += 1
            continue
        iou = box_iou(np.asarray([h]), objects_xyxy)[0][0]
        iou[removed > 0] = -1.0
        j = int(iou.argmax())
        if iou[j] >= min_iou:
            removed[j] = t
        else:
            miss += 1
    return removed, miss


def _hash_iid(job):
    """Một iid: hash mọi ảnh inpaint của mọi nhánh + mọi file samples -> ghép một-một."""
    iid, turns, samples = job                     # turns: [(key, path)], samples: [(rel, path)]
    th = [(k, pixel_hash(p)) for k, p in turns]
    sh = defaultdict(list)
    for rel, p in samples:
        sh[pixel_hash(p)].append(rel)
    match, issues = {}, []
    used = set()
    for k, h in th:
        cand = sh.get(h, [])
        if len(cand) != 1:
            issues.append(f"{k}: {len(cand)} file samples cùng pixel")
            continue
        if cand[0] in used:
            issues.append(f"{k}: file {cand[0]} đã ghép với lượt khác")
            continue
        used.add(cand[0])
        match[k] = cand[0]
    extra = [rel for rel, _ in samples if rel not in used]
    return iid, match, issues, extra


def build_turn_index(ce130_root, samples_root, workers=8, log=print):
    """-> dict chỉ mục (đường dẫn TƯƠNG ĐỐI so với ce130_root / samples_root) + `report`. Không ném lỗi:
    người gọi (`tools/build_turn_index.py`) quyết dừng theo `report`."""
    branches, turns_by_iid, report = {}, defaultdict(list), {"n_holes_unassigned": 0, "turn_count_mismatch": []}
    for split in SPLITS:
        for br in sorted(glob.glob(os.path.join(ce130_root, split, "*"))):
            name = os.path.basename(br)
            ann = read_annotation(br)
            if ann is None:
                continue
            T = len(glob.glob(os.path.join(br, "inpainted_turn_*.png")))
            holes = [list(map(float, h)) for h in ann.get("inpainted_bboxes", [])]
            if len(holes) != T:
                report["turn_count_mismatch"].append(f"{split}/{name}: {T} ảnh, {len(holes)} inpainted_bboxes")
                continue
            objects, _ = filter_degenerate(np.asarray(ann.get("all_bboxes", []), dtype=np.float64).reshape(-1, 4))
            removed, miss = assign_removed(objects, holes)
            report["n_holes_unassigned"] += miss
            W, H = Image.open(os.path.join(br, "ground_truth.jpg")).size
            branches[f"{split}/{name}"] = {"split": split, "wh": [W, H], "class": ann.get("class_based_caption", ""),
                                           "holes": holes, "objects": objects.tolist(), "removed": removed.tolist()}
            iid = name.split("_b")[0]
            for t in range(1, T + 1):
                turns_by_iid[iid].append((f"{name}_t{t}", os.path.join(br, f"inpainted_turn_{t}.png")))
    samples_by_iid = defaultdict(list)
    for s in sorted(os.listdir(samples_root)):
        d = os.path.join(samples_root, s, "images")
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith((".png", ".jpg", ".jpeg")):
                samples_by_iid[os.path.splitext(f)[0].rsplit("_", 1)[0]].append((f"{s}/images/{f}", os.path.join(d, f)))
    n_turns = sum(len(v) for v in turns_by_iid.values())
    n_samples = sum(len(v) for v in samples_by_iid.values())
    log(f"  {len(branches)} nhánh, {n_turns} (nhánh, lượt) | {n_samples} file samples — hash pixel {len(turns_by_iid)} ảnh gốc")
    jobs = [(iid, turns_by_iid[iid], samples_by_iid.get(iid, [])) for iid in sorted(turns_by_iid)]
    match, issues, extra = {}, [], []
    pool = _pool(workers) if workers > 0 else None
    for done, (iid, m, iss, ex) in enumerate(pool.imap_unordered(_hash_iid, jobs, chunksize=4) if pool else
                                             map(_hash_iid, jobs), 1):
        match.update(m)
        issues += iss
        extra += ex
        if done % 200 == 0 or done == len(jobs):
            log(f"  hash {done}/{len(jobs)} ảnh gốc | ghép {len(match)} lượt")
    if pool:
        pool.close()
        pool.join()
    extra += [rel for iid, v in samples_by_iid.items() if iid not in turns_by_iid for rel, _ in v]
    turns, tgt_bad, cls_bad = {}, [], []
    for key, rel in sorted(match.items()):
        name, t = key.rsplit("_t", 1)
        t = int(t)
        bkey = next(k for k in (f"{s}/{name}" for s in SPLITS) if k in branches)
        b = branches[bkey]
        stem = os.path.splitext(os.path.basename(rel))[0]
        sdir = rel.split("/")[0]
        with open(os.path.join(samples_root, sdir, "annotation", stem + ".json")) as f:
            sa = json.load(f)
        cx, cy, w, h = map(float, sa["target_bbox"])
        x1, y1, x2, y2 = b["holes"][t - 1]
        if max(abs(cx - w / 2 - x1), abs(cy - h / 2 - y1), abs(cx + w / 2 - x2), abs(cy + h / 2 - y2)) > TARGET_TOL_PX:
            tgt_bad.append(f"{key}: target_bbox {sa['target_bbox']} vs lỗ lượt {t} {b['holes'][t - 1]}")
        if sa.get("class") != b["class"]:
            cls_bad.append(f"{key}: class {sa.get('class')!r} vs {b['class']!r}")
        turns[key] = {"branch": bkey, "t": t, "sample": rel, "density": f"{sdir}/density/{stem}.png"}
    report.update({"n_branches": len(branches), "n_turns": n_turns, "n_samples": n_samples, "n_matched": len(turns),
                   "match_issues": issues, "unmatched_samples": sorted(extra), "target_mismatch": tgt_bad,
                   "class_mismatch": cls_bad,
                   "per_split": {s: sum(1 for v in turns.values() if branches[v["branch"]]["split"] == s) for s in SPLITS}})
    report["ok"] = (len(turns) == n_turns == n_samples and not issues and not extra and not tgt_bad and not cls_bad
                    and not report["turn_count_mismatch"])
    return {"branches": branches, "turns": turns, "report": report}


class TurnIndex:
    """Đọc `data/turn_index.json`."""

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"thiếu chỉ mục (nhánh, lượt) {path} — dựng một lần (cửa G0): python "
                                    f"tools/build_turn_index.py --out {path}")
        with open(path) as f:
            d = json.load(f)
        self.branches, self.turns = d["branches"], d["turns"]

    def keys(self, split):
        """Khoá (nhánh, lượt) của `split` (train | val | test | test_all — docstring module)."""
        if split not in SPLITS + ("test_all",):
            raise ValueError(f"split {split!r} không thuộc {SPLITS + ('test_all',)}")
        sdir = "train" if split == "train" else "test"
        return sorted(k for k, v in self.turns.items() if v["sample"].split("/")[0] == sdir
                      and (split in ("train", "test_all") or self.branches[v["branch"]]["split"] == split))


class AddCache:
    """Đọc cache của `build_add_cache` (`meta.json`: image_size, format, khoá -> hàng / (offset, độ dài), (scale, nw, nh)).
    Mỗi mẫu = uint8 [T,T,4] (RGB letterbox + density `.convert("L")` letterbox):
      raw   `data.u8` memmap [N,T,T,4] (~1 MiB / mẫu ở canvas 512)
      zlib  `data.zlib` các khối zlib nối liền (~0,41 MiB / mẫu đo trên samples/train — vừa giới hạn output ~20 GB của Kaggle
            nên LƯU LẠI được giữa các phiên); giải nén vài ms / mẫu, vẫn nhanh hơn giải mã PNG nhiều.
    File mở lười (mỗi worker DataLoader mở riêng)."""

    def __init__(self, cache_dir):
        with open(os.path.join(cache_dir, "meta.json")) as f:
            m = json.load(f)
        self.dir, self.size, self.rows, self.geom = cache_dir, m["image_size"], m["rows"], m["geom"]
        self.format = m.get("format", "raw")
        self._fh = None

    def __contains__(self, key):
        return key in self.rows

    def get(self, key):
        """-> (uint8 [T,T,4], scale, nw, nh)."""
        scale, nw, nh = self.geom[key]
        if self.format == "zlib":
            if self._fh is None:
                self._fh = open(os.path.join(self.dir, "data.zlib"), "rb")
            off, n = self.rows[key]
            self._fh.seek(off)
            arr = np.frombuffer(bytearray(zlib.decompress(self._fh.read(n))), dtype=np.uint8)   # ghi được (torch.from_numpy)
            return arr.reshape(self.size, self.size, 4), scale, nw, nh
        if self._fh is None:
            self._fh = np.memmap(os.path.join(self.dir, "data.u8"), dtype=np.uint8, mode="r",
                                 shape=(len(self.rows), self.size, self.size, 4))
        return np.array(self._fh[self.rows[key]]), scale, nw, nh          # bản sao ghi được (memmap chỉ đọc)


def _letterbox_l_u8(gray_u8, nw, nh, target):
    """Density kiểu bài (`.convert("L")`, uint8 [H,W]) -> uint8 [T,T]: resize NEAREST, dán góc trên-trái lên nền 0.
    `/255` ra đúng density kiểu bài (float)."""
    den = np.zeros((target, target), dtype=np.uint8)
    small = Image.fromarray(np.asarray(gray_u8, dtype=np.uint8)).resize((nw, nh), resample=Image.NEAREST)
    den[:nh, :nw] = np.asarray(small, dtype=np.uint8)
    return den


def _cache_one(job):
    key, img_path, den_path, size, fmt = job
    img = Image.open(img_path).convert("RGB")
    canvas, scale, nw, nh = letterbox(img, size)
    den = _letterbox_l_u8(np.asarray(Image.open(den_path).convert("L"), dtype=np.uint8), nw, nh, size)
    arr = np.ascontiguousarray(np.concatenate([canvas, den[..., None]], axis=-1))
    return key, (zlib.compress(arr.tobytes(), 1) if fmt == "zlib" else arr), (float(scale), int(nw), int(nh))


def build_add_cache(index, samples_root, keys, out_dir, image_size=512, workers=4, log=print, fmt="zlib"):
    """Ghi cache uint8 của `keys` (ảnh inpaint + density của mẫu, letterbox kiểu bài) vào `out_dir`, định dạng `fmt`."""
    if fmt not in ("zlib", "raw"):
        raise ValueError(f"format {fmt!r} không thuộc ('zlib', 'raw')")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(k, os.path.join(samples_root, index.turns[k]["sample"]), os.path.join(samples_root, index.turns[k]["density"]),
             image_size, fmt) for k in keys]
    if fmt == "raw":
        sink = np.memmap(os.path.join(out_dir, "data.u8"), dtype=np.uint8, mode="w+",
                         shape=(len(jobs), image_size, image_size, 4))
    else:
        sink = open(os.path.join(out_dir, "data.zlib"), "wb")
    rows, geom, off = {}, {}, 0
    pool = _pool(workers) if workers > 0 else None
    it = pool.imap(_cache_one, jobs, chunksize=16) if pool else map(_cache_one, jobs)
    t0 = __import__("time").time()
    for i, (k, data, g) in enumerate(it):
        if fmt == "raw":
            sink[i] = data
            rows[k] = i
        else:
            sink.write(data)
            rows[k] = [off, len(data)]
            off += len(data)
        geom[k] = g
        if (i + 1) % 2000 == 0 or i + 1 == len(jobs):
            el = __import__("time").time() - t0
            log(f"  [cache] {i + 1}/{len(jobs)} | {el:.0f}s | còn ~{el / (i + 1) * (len(jobs) - i - 1):.0f}s"
                + ("" if fmt == "raw" else f" | {off / 2 ** 30:.2f} GB"))
    if pool:
        pool.close()
        pool.join()
    if fmt == "raw":
        sink.flush()
    else:
        sink.close()
    with open(os.path.join(out_dir, "meta.json"), "w") as f:      # ghi CUỐI: có meta.json = cache đầy đủ
        json.dump({"image_size": image_size, "format": fmt, "rows": rows, "geom": geom}, f)
    return len(jobs)


def _unzip_rows(job):
    src_dir, dst_path, n, size, items = job
    src = AddCache(src_dir)
    dst = np.memmap(dst_path, dtype=np.uint8, mode="r+", shape=(n, size, size, 4))
    for i, key in items:
        dst[i] = src.get(key)[0]
    dst.flush()
    return len(items)


def convert_add_cache(src_dir, out_dir, workers=4, log=print):
    """Cache `zlib` -> `raw` (memmap `data.u8`, ~1 MiB / mẫu, đọc = memcpy, KHÔNG giải nén mỗi lượt đọc). Cùng khoá / geom, trùng
    từng byte (có test). Kaggle: cache zlib nằm trong output (lưu giữa phiên), bản raw bung ra /kaggle/temp đầu mỗi phiên —
    giải nén zlib ~5 ms / mẫu là phần chính của thời gian worker."""
    src = AddCache(src_dir)
    if src.format != "zlib":
        raise ValueError(f"{src_dir}: cache đã là {src.format}")
    os.makedirs(out_dir, exist_ok=True)
    keys = sorted(src.rows)
    n, size = len(keys), src.size
    dst_path = os.path.join(out_dir, "data.u8")
    np.memmap(dst_path, dtype=np.uint8, mode="w+", shape=(n, size, size, 4)).flush()
    rows = {k: i for i, k in enumerate(keys)}
    items = list(enumerate(keys))
    chunk = max(1, -(-n // max(workers * 8, 1)))
    jobs = [(src_dir, dst_path, n, size, items[j:j + chunk]) for j in range(0, n, chunk)]
    t0, done = __import__("time").time(), 0
    pool = _pool(workers) if workers > 0 else None
    for m in (pool.imap_unordered(_unzip_rows, jobs) if pool else map(_unzip_rows, jobs)):
        done += m
        if done == n or done // 4000 != (done - m) // 4000:
            el = __import__("time").time() - t0
            log(f"  [cache raw] {done}/{n} | {el:.0f}s | còn ~{el / done * (n - done):.0f}s")
    if pool:
        pool.close()
        pool.join()
    with open(os.path.join(out_dir, "meta.json"), "w") as f:     # ghi CUỐI: có meta.json = cache đầy đủ
        json.dump({"image_size": size, "format": "raw", "rows": rows, "geom": src.geom}, f)
    return n


class CE130AddDataset(Dataset):
    """Một phần tử = một (nhánh, lượt): ảnh uint8 [4,T,T] (RGB + density kiểu bài), box xyxy PIXEL CANVAS. Density: ảnh inpaint =
    density của chính mẫu; ảnh gốc = bản `full` của ảnh gốc (`density_index`, `data/density.py`). Box positive (`set_positives`):
    tập đích = mọi lỗ tới lượt t + box gán tay không trùng lỗ (`hand_positives`); `random` / `all`: thêm khoá `positives` (cả tập);
    `pair`: mỗi phần tử = một cặp (mẫu, box), `target` = box đó."""

    def __init__(self, index, ce130_root, samples_root, split, image_size=512, image="inpainted", density_index=None, cache=None,
                 clip_cache=None, positives=None, positive_hole_iou=None, positive_sampling="random"):
        if image not in IMAGE_KINDS:
            raise ValueError(f"image {image!r} không thuộc {IMAGE_KINDS}")
        if image == "original" and density_index is None:
            raise ValueError("ảnh gốc cần density_index (density `full` của ảnh gốc)")
        self.index, self.ce130_root, self.samples_root = index, ce130_root, samples_root
        self.keys = index.keys(split)
        self.image_size, self.image, self.density_index = image_size, image, density_index
        if cache is not None and cache.size != image_size:
            raise ValueError(f"cache image_size {cache.size} != {image_size}")
        self.cache = cache if image == "inpainted" else None          # cache = ảnh inpaint + density của mẫu
        self.clip_cache = clip_cache                                   # token CLIP tính sẵn (`data/clip_cache.py`, chỉ train)
        self.positives = self.positive_hole_iou = self.positive_sampling = self.pairs = None
        if positives is not None:
            self.set_positives(positives, positive_hole_iou, positive_sampling)

    def set_positives(self, positives, hole_iou, sampling="random"):
        """Bật đích = box positive ({tên file samples: box}, `load_positives`). `pair`: dựng danh sách (chỉ số mẫu, chỉ số box) trên
        `self.keys` HIỆN TẠI (đặt sau khi cắt keys)."""
        if hole_iou is None:
            raise ValueError("positives cần positive_hole_iou")
        if sampling not in POSITIVE_SAMPLING:
            raise ValueError(f"positive_sampling {sampling!r} không thuộc {POSITIVE_SAMPLING}")
        self.positives, self.positive_hole_iou, self.positive_sampling = positives, float(hole_iou), sampling
        self.pairs = ([(i, j) for i, k in enumerate(self.keys) for j in range(len(self.positive_boxes(k)))] if sampling == "pair"
                      else None)

    def positive_boxes(self, key):
        """Tập đích của mẫu `key`, xyxy pixel ảnh gốc: mọi lỗ tới lượt t, rồi box gán tay không trùng lỗ."""
        e = self.index.turns[key]
        holes = np.asarray(self.index.branches[e["branch"]]["holes"][:e["t"]], dtype=np.float64).reshape(-1, 4)
        hand = hand_positives(self.positives.get(os.path.splitext(os.path.basename(e["sample"]))[0], ()), holes, self.positive_hole_iou)
        return np.concatenate([holes, hand])

    def __len__(self):
        return len(self.pairs) if self.pairs is not None else len(self.keys)

    def classes(self):
        return sorted({self.index.branches[self.index.turns[k]["branch"]]["class"] for k in self.keys})

    def __getitem__(self, i):
        i, j = self.pairs[i] if self.pairs is not None else (i, None)
        key = self.keys[i]
        e = self.index.turns[key]
        b = self.index.branches[e["branch"]]
        t = e["t"]
        if self.cache is not None and key in self.cache:
            x, scale, nw, nh = self.cache.get(key)
        else:
            path = (os.path.join(self.samples_root, e["sample"]) if self.image == "inpainted" else
                    os.path.join(self.ce130_root, e["branch"], "ground_truth.jpg"))
            x, scale, nw, nh = letterbox(Image.open(path).convert("RGB"), self.image_size)
            path_d = (os.path.join(self.samples_root, e["density"]) if self.image == "inpainted" else
                      self.density_index.full(e["branch"].split("/")[1].split("_b")[0]))
            den = _letterbox_l_u8(np.asarray(Image.open(path_d).convert("L"), dtype=np.uint8), nw, nh, self.image_size)
            x = np.concatenate([x, den[..., None]], axis=-1)
        holes = scale_boxes(b["holes"][:t], scale, nw, nh)
        if len(holes) != t:
            raise ValueError(f"{key}: lỗ rơi ra ngoài vùng ảnh thật sau letterbox")
        removed = np.asarray(b["removed"], dtype=int)
        objs = np.asarray(b["objects"], dtype=np.float64).reshape(-1, 4)
        if self.image == "inpainted":                    # vật còn trong ảnh lượt t: chưa bị xoá tới lượt t
            objs = objs[(removed == 0) | (removed > t)]
        target = holes[-1]                               # lỗ MỚI NHẤT = đích train (mặc định)
        if self.positives is not None:
            raw = self.positive_boxes(key)
            pos = scale_boxes(raw, scale, nw, nh)
            if len(pos) != len(raw):                     # `pair` đánh chỉ số box trên tập chưa letterbox: phải giữ đủ
                raise ValueError(f"{key}: box positive rơi ra ngoài vùng ảnh thật sau letterbox")
            if j is not None:
                target = pos[j]
        return {
            # [C,T,T] uint8 là VIEW permute của mảng HWC liền (= bố cục channels_last), `collate_add` gom HWC
            "image": torch.from_numpy(np.ascontiguousarray(x, dtype=np.uint8)).permute(2, 0, 1),
            "target": torch.from_numpy(target).float(),
            "holes": torch.from_numpy(holes).float(),                    # mọi lỗ tới lượt t (chấm)
            "objects": torch.from_numpy(scale_boxes(objs, scale, nw, nh)).float(),   # vật đang có
            "valid_hw": (nh, nw),
            "text": b["class"],
            "image_id": key,
            "t": t,
            **({"clip": self.clip_cache.get(key)} if self.clip_cache is not None else {}),
            **({"positives": torch.from_numpy(pos).float()} if self.positives is not None and j is None else {}),
        }


def collate_add(batch):
    """Ảnh uint8 -> `images` [B,T,T,C] liền (NHWC: gom = memcpy, không hoán trục trên CPU); `to_device_add` permute thành NCHW
    channels_last SAU khi lên thiết bị."""
    nh = torch.tensor([b["valid_hw"][0] for b in batch], dtype=torch.float32)
    nw = torch.tensor([b["valid_hw"][1] for b in batch], dtype=torch.float32)
    return {
        "images": torch.stack([b["image"].permute(1, 2, 0) for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "holes": [b["holes"] for b in batch],
        "objects": [b["objects"] for b in batch],
        "whwh": torch.stack([nw, nh, nw, nh], dim=1),
        "valid_hw": torch.stack([nh, nw], dim=1).long(),
        "text": [b["text"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
        "t": [b["t"] for b in batch],
        **({"objects_all": [b["objects_all"] for b in batch]} if "objects_all" in batch[0] else {}),   # CE-CoCount: cả hai lớp
        **({"exemplars": [b["exemplars"] for b in batch]} if "exemplars" in batch[0] else {}),         # CE-CoCount: 3 box mẫu gán tay
        **({"clip": torch.stack([b["clip"] for b in batch])} if "clip" in batch[0] else {}),          # token CLIP tính sẵn (train)
        **({"positives": [b["positives"] for b in batch]} if "positives" in batch[0] else {}),       # tập đích (DELTA7 / 7.0.1)
    }


_LUT = {}


def paper_to_float(images_u8):
    """Ảnh kiểu bài uint8 -> float32 `/255` qua bảng tra 256 giá trị tính bằng numpy float32: trùng TỪNG BIT với
    `to_tensor` / đọc PNG (phép chia `/` của PyTorch CUDA theo vô hướng là nhân nghịch đảo, có thể lệch 1 ulp). Bảng giữ sẵn trên
    từng thiết bị; chỉ số int32 (int64 tạm gấp đôi bộ nhớ: 268 MB / batch 32 ảnh 4 kênh 512²)."""
    lut = _LUT.get(images_u8.device)
    if lut is None:
        lut = _LUT[images_u8.device] = torch.from_numpy(np.arange(256, dtype=np.float32) / np.float32(255.0)).to(images_u8.device)
    return lut[images_u8.int()]


def to_device_add(batch, dev):
    """Đưa batch lên `dev`. Ảnh đi qua PCIe dạng uint8 NHWC (nhẹ 4×), permute thành NCHW (view, bố cục channels_last) rồi mới
    `/255` trên thiết bị."""
    nb = dev.type == "cuda"
    out = dict(batch)
    for k in ("images", "target", "whwh", "valid_hw", "clip"):
        if k in batch:
            out[k] = batch[k].to(dev, non_blocking=nb)
    out["images"] = paper_to_float(out["images"].permute(0, 3, 1, 2))
    return out

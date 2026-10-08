"""Dữ liệu bài ADD (docs/EXPERIMENT_GAMMA.md mục 2): một mẫu = (nhánh, lượt t) của CE-130.

Nhánh `all_phase2_V2/<split>/<iid>_b<k>/` xoá CỘNG DỒN tối đa 4 vật: `inpainted_turn_t.png` thiếu đúng vật
1..t (`inpainted_bboxes[:t]`, xyxy pixel ảnh gốc). Đích train = lỗ MỚI NHẤT `inpainted_bboxes[t-1]` (như CE-Loc
gốc); các lỗ cũ vẫn trong ảnh và cũng là chỗ thêm hợp lệ (chỉ để chấm).

`samples/{train,test}/images/{iid}_{j}.png` của bài = đúng các ảnh inpaint đó (trùng từng pixel), kèm
`density/{iid}_{j}.png` (density CountGD của CHÍNH ảnh đó) và `annotation/{iid}_{j}.json` (`target_bbox`
cxcywh). Thứ tự j không theo nhánh / lượt và split của `samples/` lẫn lớp giữa các split ⇒ chỉ mục
`data/turn_index.json` (`tools/build_turn_index.py`, cửa G0) khớp (nhánh, lượt) <-> file samples bằng hash pixel,
còn split lấy theo thư mục `all_phase2_V2` (lớp 3 split rời nhau).

Dataset đọc ảnh + density từ `samples/` (Kaggle chỉ cần zip samples + annotation CE-130), letterbox góc trên-trái
+ chuẩn hoá ImageNet như ALPHA (`data/dataset.py`), density giải mã jet như ALPHA3 (`data/density.py`).
`image="original"` thay ảnh vào bằng `ground_truth.jpg` của nhánh (không lỗ) — phép thử lối tắt "tìm vết inpaint".

`split_source` (GAMMA2): `ce130` = split theo thư mục `all_phase2_V2` (mặc định); `samples` = split NHƯ CE-Loc GỐC TRAIN:
train = mọi mẫu `samples/train` (gồm cả ảnh gốc val / test CE-130 bài đã dùng), val / test = mẫu `samples/test` có ảnh gốc
thuộc val / test CE-130 (`samples/` train và test không chung ảnh gốc ⇒ chưa model nào train trên val / test này).
`AddCache` (GAMMA2, Kaggle): ảnh inpaint + density của mẫu đã letterbox, uint8 [T,T,4] trong MỘT file (zlib hoặc memmap thô) —
`/255` ra đúng từng bit đầu vào kiểu bài (có test); bỏ nghẽn giải mã PNG. Style `paper` trả ảnh uint8 (cả khi không cache):
`to_device_add` mới `/255` trên thiết bị (`paper_to_float`) — worker nhẹ hơn, PCIe chở 1/4 số byte.
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

from ce_localization.data.dataset import _read_annotation, letterbox, normalize, scale_boxes
from ce_localization.data.density import decode_jet, letterbox_density, load_density_levels
from ce_localization.utils.box_ops_np import box_iou, filter_degenerate

__all__ = ["SPLITS", "IMAGE_KINDS", "ADD_DENSITY", "INPUT_STYLES", "SPLIT_SOURCES", "pixel_hash", "assign_removed",
           "build_turn_index", "TurnIndex", "AddCache", "build_add_cache", "convert_add_cache", "CE130AddDataset", "collate_add", "to_device_add",
           "paper_to_float", "image_inputs"]

SPLITS = ("train", "val", "test")
IMAGE_KINDS = ("inpainted", "original")
ADD_DENSITY = ("sample", "full", "empty")       # density của chính mẫu | bản `full` của ảnh gốc (ALPHA3) | trống
TARGET_TOL_PX = 1.0                             # target_bbox (cxcywh, làm tròn) vs inpainted_bboxes (xyxy)
INPUT_STYLES = ("ours", "paper")
SPLIT_SOURCES = ("ce130", "samples")


def image_inputs(img_rgb, density_path=None, target=512, style="ours"):
    """Một ảnh PIL RGB (+ PNG density: đường dẫn hoặc ảnh PIL) -> (tensor [3|4,T,T], scale, nw, nh), letterbox góc trên-trái.
      ours : chuẩn hoá ImageNet + density giải mã jet (như `CE130AddDataset`, ALPHA3)
      paper: CE-Loc gốc — chỉ `to_tensor` (/255) + density `.convert("L")` (độ sáng của PNG jet, nền 14/255), resize
             NEAREST + độn 0: đúng `ObjectPlacementDataset.resize_and_pad` (có test so với công thức gốc)."""
    if style not in INPUT_STYLES:
        raise ValueError(f"style {style!r} không thuộc {INPUT_STYLES}")
    canvas, scale, nw, nh = letterbox(img_rgb, target)
    x = normalize(canvas) if style == "ours" else canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
    if density_path is not None:
        den = Image.open(density_path) if isinstance(density_path, str) else density_path
        lv = (decode_jet(np.asarray(den.convert("RGB")))[0] if style == "ours" else
              np.asarray(den.convert("L"), dtype=np.uint8))
        x = np.concatenate([x, letterbox_density(lv, nw, nh, target)[None]], axis=0)
    return torch.from_numpy(np.ascontiguousarray(x)), scale, nw, nh


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
            ann = _read_annotation(br)
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

    def keys(self, split, source="ce130"):
        """Khoá (nhánh, lượt) của `split`. `source` (mục docstring module): ce130 | samples."""
        if source not in SPLIT_SOURCES:
            raise ValueError(f"split_source {source!r} không thuộc {SPLIT_SOURCES}")
        if source == "ce130":
            return sorted(k for k, v in self.turns.items() if self.branches[v["branch"]]["split"] == split)
        sdir = "train" if split == "train" else "test"
        return sorted(k for k, v in self.turns.items() if v["sample"].split("/")[0] == sdir
                      and (split == "train" or self.branches[v["branch"]]["split"] == split))


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
    `/255` ra đúng `letterbox_density` (float)."""
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
    """Một phần tử = một (nhánh, lượt). Box ra là xyxy PIXEL CANVAS như ALPHA.
    `style`: `ours` = chuẩn hoá ImageNet + density giải mã jet (ALPHA3); `paper` = CE-Loc gốc: `to_tensor` (/255) + density
    `.convert("L")` (density `empty` = PNG jet trống (0, 0, 127) -> độ sáng 14/255 trên vùng thật), như `image_inputs`."""

    def __init__(self, index, ce130_root, samples_root, split, image_size=512, density=None, density_index=None,
                 image="inpainted", style="ours", split_source="ce130", cache=None):
        if image not in IMAGE_KINDS:
            raise ValueError(f"image {image!r} không thuộc {IMAGE_KINDS}")
        if style not in INPUT_STYLES:
            raise ValueError(f"style {style!r} không thuộc {INPUT_STYLES}")
        if density is not None and density not in ADD_DENSITY:
            raise ValueError(f"density {density!r} không thuộc {ADD_DENSITY}")
        if density == "full" and density_index is None:
            raise ValueError("density 'full' cần density_index (data/density_index.json của ALPHA3)")
        self.index, self.ce130_root, self.samples_root = index, ce130_root, samples_root
        self.keys = index.keys(split, split_source)
        self.image_size, self.density, self.density_index, self.image = image_size, density, density_index, image
        self.style = style
        # cache chỉ dùng được khi đầu vào đúng thứ đã cache: ảnh inpaint kiểu bài + density của mẫu (hoặc không density)
        usable = cache is not None and image == "inpainted" and style == "paper" and density in (None, "sample")
        if cache is not None and cache.size != image_size:
            raise ValueError(f"cache image_size {cache.size} != {image_size}")
        self.cache = cache if usable else None

    def __len__(self):
        return len(self.keys)

    def classes(self):
        return sorted({self.index.branches[self.index.turns[k]["branch"]]["class"] for k in self.keys})

    def entry(self, i):
        e = self.index.turns[self.keys[i]]
        return e, self.index.branches[e["branch"]]

    def __getitem__(self, i):
        key = self.keys[i]
        e, b = self.entry(i)
        t = e["t"]
        paper = self.style == "paper"
        ch = -1 if paper else 0                          # kiểu bài: uint8 HWC (không chép sang CHW trong worker)
        if self.cache is not None and key in self.cache:
            arr, scale, nw, nh = self.cache.get(key)
            x = arr if self.density is not None else arr[..., :3]
        else:
            path = (os.path.join(self.samples_root, e["sample"]) if self.image == "inpainted" else
                    os.path.join(self.ce130_root, e["branch"], "ground_truth.jpg"))
            img = Image.open(path).convert("RGB")
            canvas, scale, nw, nh = letterbox(img, self.image_size)
            x = canvas if paper else normalize(canvas)
        if self.density is not None and x.shape[ch] == 3:
            if self.density == "empty" and paper:
                den = _letterbox_l_u8(np.asarray(Image.new("RGB", img.size, (0, 0, 127)).convert("L")), nw, nh,
                                      self.image_size)
            elif self.density == "empty":
                den = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            else:
                iid = e["branch"].split("/")[1].split("_b")[0]
                path_d = (os.path.join(self.samples_root, e["density"]) if self.density == "sample" else
                          self.density_index.path(self.density_index.pick(iid, "full")[0]))
                if paper:
                    den = _letterbox_l_u8(np.asarray(Image.open(path_d).convert("L"), dtype=np.uint8), nw, nh,
                                          self.image_size)
                else:
                    den = letterbox_density(load_density_levels(path_d), nw, nh, self.image_size)
            x = np.concatenate([x, np.expand_dims(den, ch)], axis=ch)
        holes = scale_boxes(b["holes"][:t], scale, nw, nh)
        if len(holes) != t:
            raise ValueError(f"{key}: lỗ rơi ra ngoài vùng ảnh thật sau letterbox")
        removed = np.asarray(b["removed"], dtype=int)
        objs = np.asarray(b["objects"], dtype=np.float64).reshape(-1, 4)
        if self.image == "inpainted":                    # vật còn trong ảnh lượt t: chưa bị xoá tới lượt t
            objs = objs[(removed == 0) | (removed > t)]
        return {
            # kiểu bài: [C,T,T] uint8 là VIEW permute của mảng HWC liền (= bố cục channels_last), `collate_add` gom HWC
            "image": (torch.from_numpy(np.ascontiguousarray(x, dtype=np.uint8)).permute(2, 0, 1) if paper else
                      torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))),
            "target": torch.from_numpy(holes[-1]).float(),               # lỗ MỚI NHẤT = đích train
            "holes": torch.from_numpy(holes).float(),                    # mọi lỗ tới lượt t (chấm)
            "objects": torch.from_numpy(scale_boxes(objs, scale, nw, nh)).float(),   # vật đang có (chấm)
            "valid_hw": (nh, nw),
            "text": b["class"],
            "image_id": key,
            "t": t,
        }


def collate_add(batch):
    """Ảnh float (kiểu ours) -> `images` [B,C,T,T]. Ảnh uint8 (kiểu bài) -> `images` [B,T,T,C] liền (NHWC: gom = memcpy, không
    hoán trục trên CPU) + `images_nhwc` True; `to_device_add` permute thành NCHW channels_last SAU khi lên thiết bị."""
    nh = torch.tensor([b["valid_hw"][0] for b in batch], dtype=torch.float32)
    nw = torch.tensor([b["valid_hw"][1] for b in batch], dtype=torch.float32)
    nhwc = batch[0]["image"].dtype == torch.uint8
    return {
        "images": torch.stack([b["image"].permute(1, 2, 0) if nhwc else b["image"] for b in batch]),
        "images_nhwc": nhwc,
        "target": torch.stack([b["target"] for b in batch]),
        "holes": [b["holes"] for b in batch],
        "objects": [b["objects"] for b in batch],
        "whwh": torch.stack([nw, nh, nw, nh], dim=1),
        "valid_hw": torch.stack([nh, nw], dim=1).long(),
        "text": [b["text"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
        "t": [b["t"] for b in batch],
        **({"objects_all": [b["objects_all"] for b in batch]} if "objects_all" in batch[0] else {}),   # CE-CoCount: cả hai lớp
        **({"exemplars": [b["exemplars"] for b in batch]} if "exemplars" in batch[0] else {}),         # CE-CoCount: box mẫu gán tay
    }


def paper_to_float(images_u8):
    """Ảnh kiểu bài uint8 -> float32 `/255` qua bảng tra 256 giá trị tính bằng numpy float32: trùng TỪNG BIT với
    `to_tensor` / đọc PNG (phép chia `/` của PyTorch CUDA theo vô hướng là nhân nghịch đảo, có thể lệch 1 ulp)."""
    lut = torch.from_numpy(np.arange(256, dtype=np.float32) / np.float32(255.0)).to(images_u8.device)
    return lut[images_u8.long()]


def to_device_add(batch, dev):
    """Đưa batch lên `dev`. Ảnh kiểu bài đi qua PCIe dạng uint8 (nhẹ 4×) rồi mới `/255` trên thiết bị."""
    nb = dev.type == "cuda"
    out = dict(batch)
    for k in ("images", "target", "whwh", "valid_hw"):
        out[k] = batch[k].to(dev, non_blocking=nb)
    if out.pop("images_nhwc", False):                   # NHWC -> NCHW: view, bố cục channels_last
        out["images"] = out["images"].permute(0, 3, 1, 2)
    if out["images"].dtype == torch.uint8:
        out["images"] = paper_to_float(out["images"])
    return out

"""Suy luận + chấm bài ADD. Mỗi mẫu: model sinh K box ĐỘC LẬP từ nhiễu thuần (ảnh mã hoá một lần, K mẫu khử nhiễu song song; không dùng
số trong log train). Chấm bằng numpy trên box xyxy PIXEL CANVAS (IoU không đổi qua letterbox). Box suy biến kẹp về w / h = 0.

`add_metrics` (K box / mẫu):
  best_iou@K_any / hit50@K_any         max IoU giữa K box và lỗ bất kỳ (best-of-K) / tỉ lệ mẫu >= 0,5
  best_iou@K_latest / hit50@K_latest   như trên với lỗ MỚI NHẤT (CE-130; CE-CoCount không có thứ tự lỗ ⇒ không có)
  mean_iou_any / box_hit50_any         IoU / tỉ lệ >= 0,5 của TỪNG box (không oracle) — chọn best.pth theo val (DELTA4 / 5.x)
  mean_iou_latest / box_hit50_latest
  hole_cover                           mẫu >= 2 lỗ: tỉ lệ lỗ được >= 1 box IoU >= 0,5 phủ
  on_object / in_image / degenerate    box có IoA >= 0,5 với một vật (CE-CoCount: vật cả hai lớp) / nằm trọn trong ảnh / suy biến
  cnll                                 C-NLL như bài Add One Take One: mỗi ảnh lấy box C-NLL THẤP nhất trong K (= q cao nhất),
                                       TRUNG BÌNH trên các ảnh có >= 5 vật cùng lớp (n_cnll). cnll_gt: cùng cách trên lỗ GT (mốc).
C-NLL(x) = −log q(x) − min_i(−log q(z_i)), q = Gaussian 4-D fit trên vật CÙNG lớp z_i, đặc trưng [w, h, IoU lớn nhất với vật, TB khoảng
cách tới tâm 3 vật gần nhất] (w, h chia (nw, nh), khoảng cách chia √(nw·nh)). Hai chiều quan hệ tính với MỌI vật trong ảnh (CE-CoCount:
cả lớp kia — "F1all", box đè / sát vật lớp kia bị phạt; CE-130 một lớp / ảnh). Trung bình bị vài ảnh vật xếp hàng đều chi phối (Gaussian
suy biến) — C-NLL TỪNG box mọi ảnh nằm trong dump (`record_cnll`) để tính biến thể khác (trung vị, bỏ 1 %…, `tools/cnll_outliers.py`).
"""

import hashlib
import time

import numpy as np
import torch

from ce_localization.data.turns import to_device_add
from ce_localization.models.box_policy import unit_to_boxes
from ce_localization.utils.box_ops_np import box_iou, ioa
from ce_localization.utils.log import fmt_time

__all__ = ["CNLL_MIN_OBJ", "SIZE_STATS", "predict_add", "prior_unit_boxes", "prior_records", "resize_to_objects", "clip_boxes", "cnll",
           "select_nms", "other_class", "record_cnll", "add_metrics"]

HIT = 0.5
CNLL_MIN_OBJ = 5                 # Gaussian 4-D cần >= 5 điểm để hiệp phương sai có hạng đủ
CNLL_RIDGE = 1e-6
KNN = 3
SIZE_STATS = ("mean", "exemplar")


@torch.no_grad()
def predict_add(model, loader, text_table, n_samples=30, seed=0, variants=(("", {"sampler": "mock"}),), amp=False, log_every=0,
                log=print):
    """-> {hậu tố: list record numpy (image_id, t, wh (nw, nh), boxes [K,4] thô, holes, objects (+ objects_all, exemplars))}.
    `variants`: (hậu tố, kw của `BoxPolicy.sample_from_cond` — sampler, ddim_steps, eta, use_objects, use_refiner). Phần không phụ thuộc
    t (`encode`) tính MỘT lần mỗi batch cho mọi biến thể; mỗi biến thể một generator seed `seed` ⇒ trùng từng bit với chạy riêng.
    `amp`: autocast fp16 (CUDA) như lúc train."""
    model.eval()
    dev = next(model.parameters()).device
    gens = {s: torch.Generator(device=dev.type).manual_seed(seed) for s, _ in variants}
    records, t0, n = {s: [] for s, _ in variants}, time.time(), len(loader.dataset)
    for bi, batch in enumerate(loader):
        batch = to_device_add(batch, dev)
        text = text_table(batch["text"], dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
            cond, obj, ctx = model.encode(batch["images"], text, batch["valid_hw"], batch["objects"],
                                          text_table.tokens(batch["text"], dev))
            us = {}
            for s, kw in variants:
                kw = dict(kw)
                o = obj if kw.pop("use_objects", True) else None
                c = ctx if kw.pop("use_refiner", True) else None
                us[s] = model.sample_from_cond(cond, n_samples, gens[s], obj=o, ctx=c, **kw)
        canvas = torch.full_like(batch["whwh"], float(batch["images"].shape[-1]))
        wh = batch["whwh"][:, :2].cpu().numpy()
        for s, u in us.items():
            boxes = unit_to_boxes(u.float(), canvas[:, None, :], model.box_coder).cpu().numpy().astype(np.float64)
            for i in range(len(batch["image_id"])):
                records[s].append({"image_id": batch["image_id"][i], "t": batch["t"][i], "wh": wh[i].astype(np.float64),
                                   "boxes": boxes[i], "holes": batch["holes"][i].numpy().astype(np.float64),
                                   "objects": batch["objects"][i].numpy().astype(np.float64),
                                   **{k: batch[k][i].numpy().astype(np.float64) for k in ("objects_all", "exemplars") if k in batch}})
        done = len(next(iter(records.values())))
        if log_every and (bi % log_every == 0 or done == n):
            el = time.time() - t0
            log(f"    [eval {done:5d}/{n}] {el / done * 1000:.0f} ms/mẫu | {fmt_time(el)} | còn ~{fmt_time(el / done * (n - done))}")
    return records


def prior_unit_boxes(index, split="train"):
    """Lỗ MỚI NHẤT của mọi mẫu `split`, cxcywh chia (W, H) ảnh gốc -> [N,4]. Mốc `prior`."""
    out = []
    for k in index.keys(split):
        e = index.turns[k]
        b = index.branches[e["branch"]]
        W, H = b["wh"]
        x1, y1, x2, y2 = b["holes"][e["t"] - 1]
        out.append([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H])
    return np.asarray(out, dtype=np.float64)


def prior_records(records, prior_unit, n_samples=30, seed=0):
    """Thay box của mỗi record bằng n_samples lỗ train ngẫu nhiên, quy về vùng thật (nw, nh) của mẫu đó."""
    rng = np.random.default_rng(seed)
    out = []
    for r in records:
        c = prior_unit[rng.integers(0, len(prior_unit), size=n_samples)] * np.tile(r["wh"], 2)
        out.append({**r, "boxes": np.stack([c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2,
                                            c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2], 1)})
    return out


def resize_to_objects(records, stat="mean"):
    """Giữ tâm mỗi box, (w, h) = TB box vật cùng lớp (`mean`) | TB 3 box mẫu gán tay (`exemplar`, CE-CoCount). Không có box nguồn:
    giữ nguyên."""
    if stat not in SIZE_STATS:
        raise ValueError(f"stat {stat!r} không thuộc {SIZE_STATS}")
    out = []
    for r in records:
        src = np.asarray(r.get("exemplars" if stat == "exemplar" else "objects", np.zeros((0, 4))), dtype=np.float64).reshape(-1, 4)
        if not len(src):
            out.append(r)
            continue
        w, h = float((src[:, 2] - src[:, 0]).mean()), float((src[:, 3] - src[:, 1]).mean())
        b = np.asarray(r["boxes"], dtype=np.float64).reshape(-1, 4)
        cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
        out.append({**r, "boxes": np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)})
    return out


def clip_boxes(boxes):
    """w hoặc h âm -> 0 (giữ mép trái / trên). -> (box đã kẹp, mask suy biến)."""
    b = np.array(boxes, dtype=np.float64).reshape(-1, 4)
    deg = (b[:, 2] <= b[:, 0]) | (b[:, 3] <= b[:, 1])
    b[:, 2] = np.maximum(b[:, 2], b[:, 0])
    b[:, 3] = np.maximum(b[:, 3], b[:, 1])
    return b, deg


def _knn_mean(centers, ref, k, exclude_self=False):
    d = np.linalg.norm(centers[:, None] - ref[None], axis=-1)
    if exclude_self:
        np.fill_diagonal(d, np.inf)
    kk = min(k, ref.shape[0] - (1 if exclude_self else 0))
    if kk <= 0:
        return np.zeros(len(centers))
    return np.sort(d, axis=1)[:, :kk].mean(1)


def _features(boxes, ref, wh, is_objects=False):
    """[w, h, IoU lớn nhất với `ref`, TB khoảng cách tới tâm 3 vật gần nhất]. `is_objects`: boxes = ref[:len(boxes)] ⇒ bỏ chính nó."""
    nw, nh = wh
    c = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    iou = box_iou(boxes, ref)[0]
    if is_objects:
        np.fill_diagonal(iou, 0.0)
    oc = np.stack([(ref[:, 0] + ref[:, 2]) / 2, (ref[:, 1] + ref[:, 3]) / 2], 1)
    knn = _knn_mean(c, oc, KNN, exclude_self=is_objects)
    return np.stack([w / nw, h / nh, iou.max(1) if iou.size else np.zeros(len(boxes)), knn / np.sqrt(nw * nh)], 1)


def cnll(boxes, objects, wh, others=None):
    """C-NLL của từng box [K] (đã kẹp). None nếu < CNLL_MIN_OBJ vật cùng lớp. `others`: box vật lớp KHÁC (đặc trưng quan hệ tính trên
    [objects ; others], Gaussian + hiệu chỉnh chỉ trên `objects`)."""
    if len(objects) < CNLL_MIN_OBJ:
        return None
    ref = objects if others is None or not len(others) else np.concatenate([objects, others])
    mu, inv, logdet, zmin = _cnll_fit(objects, ref, wh)
    d = _features(boxes, ref, wh) - mu
    return 0.5 * (np.einsum("ni,ij,nj->n", d, inv, d) + logdet + 4 * np.log(2 * np.pi)) - zmin


_FIT, _FIT_MAX = {}, 4096


def _cnll_fit(objects, ref, wh):
    """Gaussian của C-NLL trên box vật (chỉ phụ thuộc ảnh) — nhớ theo nội dung (mỗi ảnh chấm nhiều lần: model / prior / box GT / đổi
    cỡ). -> (mu, inv, logdet, min NLL của chính vật)."""
    objects, ref = np.ascontiguousarray(objects), np.ascontiguousarray(ref)
    h = hashlib.blake2b(digest_size=16)
    for a in (objects, ref):
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    key = (tuple(float(v) for v in wh), h.digest())
    hit = _FIT.get(key)
    if hit is not None:
        return hit
    z = _features(objects, ref, wh, is_objects=True)                # objects = ref[:M] ⇒ bỏ chính nó = đường chéo
    mu = z.mean(0)
    cov = np.cov(z.T, bias=True) + CNLL_RIDGE * np.eye(4)
    inv = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(cov)
    d = z - mu
    zmin = (0.5 * (np.einsum("ni,ij,nj->n", d, inv, d) + logdet + 4 * np.log(2 * np.pi))).min()
    if len(_FIT) >= _FIT_MAX:
        _FIT.clear()
    _FIT[key] = (mu, inv, logdet, zmin)
    return _FIT[key]


def select_nms(boxes, c, k=5, iou=0.5):
    """k box C-NLL thấp nhất sau NMS (minh hoạ): duyệt box theo C-NLL `c` tăng dần, bỏ box có IoU > `iou` với box đã giữ (K mẫu hay
    trùng nhau). Gaussian fit MỘT lần trên vật đang có — box đã chọn KHÔNG được coi là vật mới. `boxes` đã kẹp. -> list chỉ số."""
    keep = []
    for i in np.argsort(c, kind="stable"):
        if len(keep) == k:
            break
        if not keep or box_iou(boxes[i:i + 1], boxes[keep])[0].max() <= iou:
            keep.append(int(i))
    return keep


def other_class(objects, objects_all):
    """Box vật lớp kia = `objects_all` bỏ phần đầu `objects` (`read_cocount` ghép [cùng lớp ; lớp kia])."""
    objects_all = np.asarray(objects_all, dtype=np.float64).reshape(-1, 4)
    m = len(objects)
    if len(objects_all) < m or not np.array_equal(objects_all[:m], objects):
        raise ValueError("objects_all phải bắt đầu bằng đúng `objects` (cùng lớp) rồi mới tới lớp kia")
    return objects_all[m:]


def record_cnll(r):
    """C-NLL TỪNG box của một record (box kẹp) + của lỗ GT -> (c [K], c_gt [t] | None) | None khi < CNLL_MIN_OBJ vật cùng lớp."""
    b, _ = clip_boxes(r["boxes"])
    obj = np.asarray(r["objects"], dtype=np.float64).reshape(-1, 4)
    oth = other_class(obj, r["objects_all"]) if "objects_all" in r else None
    c = cnll(b, obj, r["wh"], others=oth)
    if c is None:
        return None
    holes = np.asarray(r.get("holes", np.zeros((0, 4))), dtype=np.float64).reshape(-1, 4)
    return c, (cnll(holes, obj, r["wh"], others=oth) if len(holes) else None)


def add_metrics(records, with_holes=True, latest=True):
    """records (box thô) -> dict chỉ số (docstring module). `with_holes=False` (ảnh gốc không lỗ): bỏ chỉ số IoU; `latest=False`
    (CE-CoCount): bỏ chỉ số `_latest`."""
    acc = {k: [] for k in ("best_any", "best_latest", "mean_any", "mean_latest", "boxhit_any", "boxhit_latest",
                           "on_object", "in_image", "degenerate", "cnll", "cnll_gt")}
    cover = []
    for r in records:
        b, deg = clip_boxes(r["boxes"])
        nw, nh = r["wh"]
        acc["degenerate"].append(deg.mean())
        acc["in_image"].append(((b[:, 0] >= 0) & (b[:, 1] >= 0) & (b[:, 2] <= nw) & (b[:, 3] <= nh) & ~deg).mean())
        obj = r["objects"].reshape(-1, 4)
        obj_on = r["objects_all"].reshape(-1, 4) if "objects_all" in r else obj
        acc["on_object"].append((ioa(b, obj_on).max(1) >= HIT).mean() if len(obj_on) else 0.0)
        if with_holes:
            iou = box_iou(b, r["holes"])[0]                                  # [K, t]
            any_, lat = iou.max(1), iou[:, -1]
            for k, v in (("best_any", any_.max()), ("best_latest", lat.max()), ("mean_any", any_.mean()),
                         ("mean_latest", lat.mean()), ("boxhit_any", (any_ >= HIT).mean()),
                         ("boxhit_latest", (lat >= HIT).mean())):
                acc[k].append(float(v))
            if iou.shape[1] >= 2:
                cover.append(float((iou.max(0) >= HIT).mean()))
        rc = record_cnll(r if with_holes else {k: v for k, v in r.items() if k != "holes"})
        if rc is not None:
            acc["cnll"].append(float(rc[0].min()))
            if rc[1] is not None:
                acc["cnll_gt"].append(float(rc[1].min()))
    m = lambda v: float(np.mean(v)) if len(v) else float("nan")  # noqa: E731
    res = {"n": len(records), "on_object": m(acc["on_object"]), "in_image": m(acc["in_image"]), "degenerate": m(acc["degenerate"]),
           "n_cnll": len(acc["cnll"]), "cnll": m(acc["cnll"])}
    if with_holes:
        K = len(records[0]["boxes"]) if records else 0
        res.update({"cnll_gt": m(acc["cnll_gt"]), f"best_iou@{K}_any": m(acc["best_any"]), f"hit50@{K}_any": m(np.asarray(acc["best_any"]) >= HIT),
                    "mean_iou_any": m(acc["mean_any"]), "box_hit50_any": m(acc["boxhit_any"]), "hole_cover": m(cover),
                    "n_multi_hole": len(cover)})
        if latest:
            res.update({f"best_iou@{K}_latest": m(acc["best_latest"]), f"hit50@{K}_latest": m(np.asarray(acc["best_latest"]) >= HIT),
                        "mean_iou_latest": m(acc["mean_latest"]), "box_hit50_latest": m(acc["boxhit_latest"])})
    return res

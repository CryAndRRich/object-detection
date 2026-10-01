"""Đánh giá bài ADD — docs/EXPERIMENT_GAMMA.md mục 5.

Mỗi mẫu: model sinh K box ĐỘC LẬP (ảnh mã hoá một lần, K nhiễu khử song song từ nhiễu thuần: DDPM ở GAMMA0, DDIM ở
GAMMA1 — không dùng số trong log train, cạm bẫy 9). Mọi thứ sau model là numpy, box xyxy PIXEL CANVAS; IoU không đổi qua phép
co giãn đẳng hướng của letterbox nên chấm trên canvas = chấm trên ảnh gốc. Box suy biến (w hoặc h <= 0) kẹp về
w / h = 0 trước khi chấm (IoU 0), tỉ lệ của chúng báo riêng (`degenerate`).

IoU với LỖ (ảnh inpaint — `with_holes=True`):
  best_iou@K_latest / hit50@K_latest  max IoU giữa K box và lỗ MỚI NHẤT (đích của bài; oracle best-of-K)
  best_iou@K_any    / hit50@K_any     như trên với lỗ BẤT KỲ (ảnh lượt t có t lỗ, lỗ nào cũng hợp lệ)
  mean_iou_any / box_hit50_any        IoU / tỉ lệ IoU >= 0,5 của MỘT box bất kỳ (không oracle) — CHỌN best.pth
  mean_iou_latest / box_hit50_latest  như trên với lỗ mới nhất
  hole_cover                          mẫu t >= 2: tỉ lệ lỗ được >= 1 box IoU >= 0,5 phủ
  by_turn                             theo lượt t
Không cần GT (cả ảnh gốc không lỗ):
  on_object      tỉ lệ box có IoA (giao / diện tích box) >= 0,5 với một vật đang có
  in_image       tỉ lệ box nằm trọn trong vùng ảnh thật
  C-NLL (paper)  `−log q(x) − min_i(−log q(z_i))`, q = Gaussian 4-D fit trên vật đang có của ảnh. Paper tự mâu
                 thuẫn về đặc trưng ⇒ báo cả hai: F1 = [w, h, IoU lớn nhất với vật có sẵn, TB khoảng cách tới tâm
                 3 vật gần nhất] (paper mục 3.2 / 4), F2 = [cx, cy, w, h] (phụ lục A.2); toạ độ / cỡ chia (nw, nh),
                 khoảng cách chia sqrt(nw·nh). Hai chế độ: `n1` = mọi box (một mẫu ngẫu nhiên — chất lượng mô hình),
                 `sel` = box có q lớn nhất trong K (bộ chọn của paper; paper chọn theo box CountGD + SAM, ở đây theo
                 cùng `all_bboxes` — ghi rõ khi báo cáo). Ảnh có < CNLL_MIN_OBJ vật đang có thì bỏ (n_cnll).
"""

import time

import numpy as np
import torch

from ce_localization.data.turns import to_device_add
from ce_localization.models.box_policy import unit_to_boxes
from ce_localization.utils.box_ops_np import box_iou
from ce_localization.utils.log import fmt_time

__all__ = ["CNLL_MIN_OBJ", "predict_add", "prior_unit_boxes", "prior_records", "clip_boxes", "cnll", "add_metrics"]

HIT = 0.5
CNLL_MIN_OBJ = 5                 # Gaussian 4-D cần >= 5 điểm để hiệp phương sai có hạng đủ
CNLL_RIDGE = 1e-6
KNN = 3


@torch.no_grad()
def predict_add(model, loader, text_table, n_samples=30, seed=0, log_every=0, log=print, steps=None):
    """-> list record numpy: image_id, t, wh (nw, nh), boxes [K,4] (thô), holes [t,4], objects [M,4].
    `steps`: số bước DDIM của `BoxRefiner` (GAMMA1; None = mặc định của model). BoxRefiner còn cộng dồn attention lên
    [t ; text ; vis] — lấy bằng `model.pop_attn()` sau khi gọi."""
    model.eval()
    kw = {"steps": steps} if steps else {}
    if hasattr(model, "track_attn"):
        model.track_attn = True
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev.type).manual_seed(seed)
    records, t0, n = [], time.time(), len(loader.dataset)
    for bi, batch in enumerate(loader):
        batch = to_device_add(batch, dev)
        u = model.sample(batch["images"], text_table(batch["text"], dev), batch["valid_hw"], n_samples, generator=gen,
                         **kw)
        boxes = unit_to_boxes(u.float(), batch["whwh"][:, None, :]).cpu().numpy().astype(np.float64)
        wh = batch["whwh"][:, :2].cpu().numpy()
        for i in range(len(batch["image_id"])):
            records.append({"image_id": batch["image_id"][i], "t": batch["t"][i], "wh": wh[i].astype(np.float64),
                            "boxes": boxes[i], "holes": batch["holes"][i].numpy().astype(np.float64),
                            "objects": batch["objects"][i].numpy().astype(np.float64)})
        if log_every and (bi % log_every == 0 or len(records) == n):
            el = time.time() - t0
            log(f"    [eval {len(records):5d}/{n}] {el / len(records) * 1000:.0f} ms/mẫu | {fmt_time(el)} | "
                f"còn ~{fmt_time(el / len(records) * (n - len(records)))}")
    if hasattr(model, "track_attn"):
        model.track_attn = False
    return records


def prior_unit_boxes(index, split="train"):
    """Lỗ MỚI NHẤT của mọi mẫu `split`, cxcywh chia (W, H) ảnh gốc -> [N,4]. Mốc `prior` không nhìn ảnh."""
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


def clip_boxes(boxes):
    """w hoặc h âm -> 0 (giữ mép trái / trên). -> (box đã kẹp, mask suy biến)."""
    b = np.array(boxes, dtype=np.float64).reshape(-1, 4)
    deg = (b[:, 2] <= b[:, 0]) | (b[:, 3] <= b[:, 1])
    b[:, 2] = np.maximum(b[:, 2], b[:, 0])
    b[:, 3] = np.maximum(b[:, 3], b[:, 1])
    return b, deg


def _ioa(boxes, objects):
    """[K,4], [M,4] -> [K,M] giao / diện tích box (box diện tích 0 -> 0)."""
    lt = np.maximum(boxes[:, None, :2], objects[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], objects[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(area > 0, inter / area, 0.0)


def _knn_mean(centers, ref, k, exclude_self=False):
    d = np.linalg.norm(centers[:, None] - ref[None], axis=-1)
    if exclude_self:
        np.fill_diagonal(d, np.inf)
    kk = min(k, ref.shape[0] - (1 if exclude_self else 0))
    if kk <= 0:
        return np.zeros(len(centers))
    return np.sort(d, axis=1)[:, :kk].mean(1)


def _features(boxes, objects, wh, kind, is_objects=False):
    nw, nh = wh
    c = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    if kind == "F2":
        return np.stack([c[:, 0] / nw, c[:, 1] / nh, w / nw, h / nh], 1)
    iou = box_iou(boxes, objects)[0]
    if is_objects:
        np.fill_diagonal(iou, 0.0)
    oc = np.stack([(objects[:, 0] + objects[:, 2]) / 2, (objects[:, 1] + objects[:, 3]) / 2], 1)
    knn = _knn_mean(c, oc, KNN, exclude_self=is_objects)
    return np.stack([w / nw, h / nh, iou.max(1) if iou.size else np.zeros(len(boxes)), knn / np.sqrt(nw * nh)], 1)


def cnll(boxes, objects, wh, kind="F1"):
    """C-NLL của từng box [K] (đã kẹp). None nếu < CNLL_MIN_OBJ vật đang có."""
    if len(objects) < CNLL_MIN_OBJ:
        return None
    z = _features(objects, objects, wh, kind, is_objects=True)
    x = _features(boxes, objects, wh, kind)
    mu = z.mean(0)
    cov = np.cov(z.T, bias=True) + CNLL_RIDGE * np.eye(4)
    inv = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(cov)

    def nll(v):
        d = v - mu
        return 0.5 * (np.einsum("ni,ij,nj->n", d, inv, d) + logdet + 4 * np.log(2 * np.pi))

    return nll(x) - nll(z).min()


def add_metrics(records, with_holes=True):
    """records (box thô) -> dict chỉ số (mục docstring). Hàm thuần."""
    acc = {k: [] for k in ("best_any", "best_latest", "mean_any", "mean_latest", "boxhit_any", "boxhit_latest",
                           "on_object", "in_image", "degenerate")}
    cover, by_turn = [], {}
    cn = {f"{f}_{m}": [] for f in ("F1", "F2") for m in ("n1", "sel")}
    n_cnll = 0
    for r in records:
        b, deg = clip_boxes(r["boxes"])
        nw, nh = r["wh"]
        acc["degenerate"].append(deg.mean())
        acc["in_image"].append(((b[:, 0] >= 0) & (b[:, 1] >= 0) & (b[:, 2] <= nw) & (b[:, 3] <= nh) & ~deg).mean())
        obj = r["objects"].reshape(-1, 4)
        acc["on_object"].append((_ioa(b, obj).max(1) >= HIT).mean() if len(obj) else 0.0)
        if with_holes:
            iou = box_iou(b, r["holes"])[0]                                  # [K, t]
            any_, lat = iou.max(1), iou[:, -1]
            for k, v in (("best_any", any_.max()), ("best_latest", lat.max()), ("mean_any", any_.mean()),
                         ("mean_latest", lat.mean()), ("boxhit_any", (any_ >= HIT).mean()),
                         ("boxhit_latest", (lat >= HIT).mean())):
                acc[k].append(float(v))
            if iou.shape[1] >= 2:
                cover.append(float((iou.max(0) >= HIT).mean()))
            bt = by_turn.setdefault(int(r["t"]), {"best_any": [], "mean_any": [], "best_latest": []})
            bt["best_any"].append(float(any_.max()))
            bt["mean_any"].append(float(any_.mean()))
            bt["best_latest"].append(float(lat.max()))
        c1 = cnll(b, obj, r["wh"], "F1")
        if c1 is not None:
            n_cnll += 1
            c2 = cnll(b, obj, r["wh"], "F2")
            for f, c in (("F1", c1), ("F2", c2)):
                cn[f"{f}_n1"] += c.tolist()
                cn[f"{f}_sel"].append(float(c.min()))
    m = lambda v: float(np.mean(v)) if len(v) else float("nan")  # noqa: E731
    res = {"n": len(records), "on_object": m(acc["on_object"]), "in_image": m(acc["in_image"]),
           "degenerate": m(acc["degenerate"]), "n_cnll": n_cnll}
    for k, v in cn.items():
        res[f"cnll_{k}_mean"] = m(v)
        res[f"cnll_{k}_median"] = float(np.median(v)) if v else float("nan")
    if with_holes:
        K = len(records[0]["boxes"]) if records else 0
        res.update({f"best_iou@{K}_any": m(acc["best_any"]), f"best_iou@{K}_latest": m(acc["best_latest"]),
                    f"hit50@{K}_any": m(np.asarray(acc["best_any"]) >= HIT),
                    f"hit50@{K}_latest": m(np.asarray(acc["best_latest"]) >= HIT),
                    "mean_iou_any": m(acc["mean_any"]), "mean_iou_latest": m(acc["mean_latest"]),
                    "box_hit50_any": m(acc["boxhit_any"]), "box_hit50_latest": m(acc["boxhit_latest"]),
                    "hole_cover": m(cover), "n_multi_hole": len(cover),
                    "by_turn": {t: {k: m(v) for k, v in d.items()} | {"n": len(d["best_any"])}
                                for t, d in sorted(by_turn.items())}})
    return res

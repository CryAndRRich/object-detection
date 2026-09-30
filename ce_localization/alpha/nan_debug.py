"""Chẩn đoán NaN cho `train_alpha.py --nan-debug` (ALPHA3.1 lần đầu: NaN từ bước 3, 2026-09-30).

Hai câu hỏi phải tách:
  (a) một bước optimizer với grad HỮU HẠN có làm weight thành không hữu hạn không -> `bad_params`
      gọi SAU mỗi `opt.step()`;
  (b) nếu weight hữu hạn thì NaN đến từ đâu: đầu vào (từng kênh ảnh, box, text), buffer, hay một
      module cụ thể -> `report` chạy lại forward với hook, chỉ ra module ĐẦU TIÊN ra không hữu hạn
      (hook chạy theo thứ tự xong-trước: module con xong trước module cha, nên module đầu tiên bị
      bắt là module trong cùng; kèm cờ "đầu vào hữu hạn" để biết NaN sinh ra BÊN TRONG nó hay chỉ
      truyền qua nó).
"""

import torch

__all__ = ["bad_params", "tensor_stats", "report"]


def _finite(x):
    return bool(torch.isfinite(x).all())


def tensor_stats(x):
    x = x.detach().float()
    bad = int((~torch.isfinite(x)).sum())
    fx = x[torch.isfinite(x)]
    rng = f"min {fx.min():.4g} max {fx.max():.4g}" if fx.numel() else "không có giá trị hữu hạn"
    return f"không hữu hạn {bad}/{x.numel()} | {rng}"


def bad_params(model):
    """-> tên các tham số có phần tử không hữu hạn."""
    return [n for n, p in model.named_parameters() if not _finite(p)]


def _tensors(o):
    if torch.is_tensor(o):
        return [o]
    if isinstance(o, dict):
        o = list(o.values())
    if isinstance(o, (tuple, list)):
        return [t for v in o for t in _tensors(v)]
    return []


def report(model, batch, boxes, t, text, log):
    """In mọi thứ cần để khoanh vùng NaN của MỘT batch (boxes/t tái tạo đúng như bước lỗi)."""
    imgs = batch["images"]
    for c in range(imgs.shape[1]):
        log(f"  [nan] ảnh kênh {c}: {tensor_stats(imgs[:, c])}")
    log(f"  [nan] box GT: {sum(int((~torch.isfinite(b)).sum()) for b in batch['boxes'])} phần tử không hữu hạn | "
        f"số box/ảnh {[len(b) for b in batch['boxes']]} | whwh {batch['whwh'].tolist()}")
    log(f"  [nan] box nhiễu: {tensor_stats(boxes)} | t {t.tolist()}")
    log(f"  [nan] text: {tensor_stats(text)} | density {batch.get('density_kind')} | ảnh {batch.get('image_id')}")
    bp = bad_params(model)
    bb = [n for n, b in model.named_buffers() if b.is_floating_point() and not _finite(b)]
    log(f"  [nan] tham số không hữu hạn: {len(bp)} {bp[:10]}")
    log(f"  [nan] buffer không hữu hạn: {len(bb)} {bb[:10]}")
    w = model.backbone.stem[0].weight.detach()
    for c in range(w.shape[1]):
        log(f"  [nan] conv1 weight kênh {c}: {tensor_stats(w[:, c])} | norm {w[:, c].norm():.4g}")

    first = []

    def hook(name):
        def f(m, inp, out):
            if first:
                return
            for o in _tensors(out):
                if o.is_floating_point() and not _finite(o):
                    ins = [x for x in _tensors(inp) if x.is_floating_point()]
                    first.append((name, type(m).__name__, [_finite(x) for x in ins], tensor_stats(o)))
                    return
        return f

    hs = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if n]
    try:
        with torch.no_grad():
            logits, pred = model(imgs, text, batch["valid_hw"], boxes, t)
    finally:
        for h in hs:
            h.remove()
    if first:
        n, typ, fin, st = first[0]
        log(f"  [nan] module ĐẦU TIÊN ra không hữu hạn: {n} ({typ}) | đầu vào hữu hạn {fin} | đầu ra {st}")
    else:
        log("  [nan] forward (no_grad, cùng box/t) HỮU HẠN ở mọi module -> NaN nằm ở loss / backward "
            "hoặc phụ thuộc dropout")
    log(f"  [nan] logits {tensor_stats(logits)} | box ra {tensor_stats(pred)}")

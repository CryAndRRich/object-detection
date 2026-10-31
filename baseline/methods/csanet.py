"""CSANet cho bài ADD: tiền xử lý (như `refs/CSANet/loader/datasets.py: OPADst1`), hình học box, dataset finetune CE-130 train,
trainer, suy luận K box.

Ô vuông `img_size` (256): nền = ảnh resize méo về S×S; ảnh vật = giữ tỉ lệ cạnh theo tỉ lệ của nền, đệm đen giữa ô (`fg_letterbox`);
mask vật = chữ nhật đầy ở `fg_bbox`. Dataset trả uint8, chia 255 trên GPU. trans (tỉ lệ, x, y) -> box: vật giữ tỉ lệ cạnh của ảnh vật,
cỡ theo cạnh "khớp" của nền, góc trên trái = (nền − vật) × (x, y) (`trans_to_box`, như `gen_composite_image`).
Train: nền = ảnh đã xoá vật 1..t, ảnh vật = cắt `ground_truth.jpg` ở lỗ mới nhất, ảnh ghép thật = ảnh còn vật t, mask ghép = chữ nhật lỗ,
nhãn dương; trans GT = `get_trans_label`. Suy luận: K lần rút, mỗi lần một vật cùng lớp có sẵn làm ảnh vật (`fg_select`), nhiễu seed
theo mẫu.
"""

import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from baseline.data import int_box, pick_objects, sample_rng
from baseline.models.csanet import CSANetConfig, Discriminator, SysNet, gan_losses

__all__ = ["trans_to_box", "get_trans_label", "fg_inputs", "CSANetTrain", "Trainer", "predict", "build"]


def trans_to_box(trans, bg_wh, fg_wh):
    """trans -> box xyxy pixel nền (phần vật nằm trong ảnh)."""
    bw, bh = bg_wh
    fw, fh = fg_wh
    s, rx, ry = (float(v) for v in trans)
    nw, nh = (bh * s * fw / fh, bh * s) if bw / bh > fw / fh else (bw * s, bw * s * fh / fw)
    x = min(max(round((bw - nw) * rx), 0), bw - 1)
    y = min(max(round((bh - nh) * ry), 0), bh - 1)
    return [float(x), float(y), float(min(x + max(round(nw), 1), bw)), float(min(y + max(round(nh), 1), bh))]


def get_trans_label(bg_wh, fg_wh, box_xywh):
    """trans GT ∈ [0, 1]³ của box (x, y, w, h nguyên, nằm trong nền và nhỏ hơn nền)."""
    bw, bh = bg_wh
    x, y, w, h = box_xywh
    return np.clip(np.array([h / bh if bw / bh > fg_wh[0] / fg_wh[1] else w / bw, x / max(bw - w, 1), y / max(bh - h, 1)],
                            dtype=np.float32), 0, 1)


def _resize_sq(img, S):
    return torch.from_numpy(np.array(img.resize((S, S), Image.BILINEAR), dtype=np.uint8)).permute(2, 0, 1).contiguous()


def fg_inputs(fg, bg_wh, S):
    """Ảnh vật PIL -> (uint8 [3,S,S] letterbox, uint8 [1,S,S] mask chữ nhật, fg_bbox [x, y, w, h] trong ô)."""
    bw, bh = bg_wh
    fw, fh = fg.size
    if bw / bh > fw / fh:
        w = max(1, (S * fw * bh) // (fh * bw))
        box = [(S - w) // 2, 0, w, S]
    else:
        h = max(1, (S * fh * bw) // (fw * bh))
        box = [0, (S - h) // 2, S, h]
    x, y, w, h = box
    out = np.zeros((S, S, 3), np.uint8)
    out[y:y + h, x:x + w] = np.asarray(fg.resize((w, h), Image.BILINEAR))
    msk = np.zeros((1, S, S), np.uint8)
    msk[0, y:y + h, x:x + w] = 255
    return torch.from_numpy(out).permute(2, 0, 1).contiguous(), torch.from_numpy(msk), torch.tensor(box, dtype=torch.float32)


class CSANetTrain(Dataset):
    def __init__(self, items, S):
        self.items, self.S = items, S

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it, S = self.items[i], self.S
        bg = Image.open(it["image"]).convert("RGB")
        W, H = bg.size
        x1, y1, x2, y2 = int_box(it["holes"][-1], (W, H))
        x2, y2 = min(x2, x1 + W - 1), min(y2, y1 + H - 1)               # box phải nhỏ hơn nền (trans x, y xác định)
        fg = Image.open(it["gt"]).convert("RGB").crop((x1, y1, x2, y2))
        f, m, fb = fg_inputs(fg, (W, H), S)
        cm = Image.new("L", (W, H), 0)
        cm.paste(255, (x1, y1, x2, y2))
        return {"bg": _resize_sq(bg, S), "fg": f, "msk": m, "fg_bbox": fb, "comp": _resize_sq(Image.open(it["before"]).convert("RGB"), S),
                "comp_msk": torch.from_numpy(np.array(cm.resize((S, S), Image.BILINEAR), dtype=np.uint8))[None],
                "trans": torch.from_numpy(get_trans_label((W, H), fg.size, (x1, y1, x2 - x1, y2 - y1)))}


def _f(x, dev, amp):
    x = x.to(dev, non_blocking=True).float() / 255
    return x.contiguous(memory_format=torch.channels_last) if amp else x


class Trainer:
    """Finetune CSANet: generator + discriminator, hai AdamW (VGG chung lr, phần còn lại của G lr × 10, wd 5e-4 cho VGG như bản gốc;
    D lr). Mỗi micro-batch: `g_loss.backward(inputs=G)`, `d_loss.backward(inputs=D)` ⇒ grad của g_loss không lọt vào D (bản gốc xoá bằng
    `zero_grad` giữa hai bước) ⇒ cộng dồn tương đương batch lớn. Chọn best theo `g_rec` (sai số hồi quy trans so với box GT khi mã lấy từ
    ảnh ghép thật — phần có giám sát của loss), KHÔNG theo `g_loss` (loss đối kháng, lên xuống theo độ mạnh của D)."""

    select = "g_rec"

    def __init__(self, cfg, dev, model):
        f = cfg["finetune"]
        self.cfg, self.dev, (self.G, self.D) = cfg, dev, model
        vgg = list(self.G.shared_net.parameters())
        ids = {id(p) for p in vgg}
        rest = [p for p in self.G.parameters() if id(p) not in ids]
        self.g_params, self.d_params = vgg + rest, list(self.D.parameters())
        self.params = self.g_params + self.d_params
        fused, betas = dev.type == "cuda", (f["b1"], f["b2"])
        self.optimizers = [torch.optim.AdamW([{"params": vgg, "lr": f["lr"], "weight_decay": 5e-4}, {"params": rest, "lr": 10 * f["lr"],
                                               "weight_decay": 0.0}], betas=betas, fused=fused),
                           torch.optim.AdamW(self.d_params, lr=f["lr"], betas=betas, weight_decay=0.0, fused=fused)]
        self.clip = 0.0

    def dataset(self, items):
        return CSANetTrain(items, self.cfg["model"]["img_size"]), None

    def train(self):
        self.G.train()
        self.D.train()

    def context(self, batch, world):
        return None

    def backward(self, chunk, ctx, frac, amp, scaler):
        d = self.dev
        g, dl, parts = gan_losses(self.G, self.D, _f(chunk["bg"], d, amp), _f(chunk["fg"], d, amp), _f(chunk["msk"], d, amp),
                                  chunk["fg_bbox"].to(d), _f(chunk["comp"], d, amp), _f(chunk["comp_msk"], d, amp), chunk["trans"].to(d), amp)
        scaler.scale(g * frac).backward(inputs=self.g_params)
        scaler.scale(dl * frac).backward(inputs=self.d_params)
        return {"loss": (g * frac).detach(), "d_loss": (dl * frac).detach(), **{k: v.detach() * frac for k, v in parts.items()}}

    def state_dict(self):
        return {"generator": self.G.state_dict(), "discriminator": self.D.state_dict()}

    def load_state_dict(self, sd):
        self.G.load_state_dict(sd["generator"])
        self.D.load_state_dict(sd["discriminator"])

    def weights(self):
        return {"model": {"generator": self.G.state_dict()}}


@torch.no_grad()
def predict(G, samples, k, p, seed=0, dev="cpu", log=print, log_every=500):
    """-> list {"image_id", "boxes" [K,4] pixel ảnh gốc, "fg" [K], "trans" [K,3]}. `p` = khối `predict` của config."""
    G.eval()
    S, n_img = G.img_size, p.get("batch_images", 4)
    out, t0 = [], time.time()
    for b0 in range(0, len(samples), n_img):
        rows, metas, noise = {"bg": [], "fg": [], "msk": [], "fb": []}, [], []
        for s in samples[b0:b0 + n_img]:
            img = Image.open(s["image"]).convert("RGB")
            objs = np.asarray(s["objects"], dtype=np.float64).reshape(-1, 4)
            rng = sample_rng(seed, s["image_id"])
            fgs = pick_objects(objs, k, p["fg_select"], rng)
            noise.append(torch.randn(k, G.opt.d_noise, generator=torch.Generator().manual_seed(int(rng.integers(0, 2 ** 62)))))
            bg, sizes = _resize_sq(img, S), []
            for j in fgs:
                fg = img.crop(int_box(objs[j], img.size))
                f, m, fb = fg_inputs(fg, img.size, S)
                for key, v in (("bg", bg), ("fg", f), ("msk", m), ("fb", fb)):
                    rows[key].append(v)
                sizes.append(fg.size)
            metas.append((s, img.size, fgs, sizes))
        trans = G(*(torch.stack(rows[key]).to(dev).float() / 255 for key in ("bg", "fg", "msk")), torch.cat(noise).to(dev)).float().cpu().numpy()
        for i, (s, wh, fgs, sizes) in enumerate(metas):
            tr = trans[i * k:(i + 1) * k]
            out.append({"image_id": s["image_id"], "boxes": [trans_to_box(tr[j], wh, sizes[j]) for j in range(k)], "fg": fgs,
                        "trans": tr.round(5).tolist()})
        if log_every and (len(out) // log_every != (len(out) - len(metas)) // log_every or len(out) == len(samples)):
            el = time.time() - t0
            log(f"    [csanet {len(out):5d}/{len(samples)}] {el / len(out) * 1000:.0f} ms/ảnh | còn ~{el / len(out) * (len(samples) - len(out)):.0f} s")
    return out


def build(cfg, train_backbone=False):
    return SysNet(CSANetConfig(**cfg["model"])), Discriminator()

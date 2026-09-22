"""Các khối dựng nên EXPERIMENT A — xem `docs/EXPERIMENT_A_PLAN.md`.

KHÔNG CÓ CỜ CHỌN THÍ NGHIỆM. Mỗi lớp làm đúng một việc.

Thân mô hình vòng 1 (`box_transformer.py`) phục vụ đồng thời A/B/A.1/A.2/C1/C1b/E1 bằng
các cờ (`roi_to_tgt`, `score_roi`, `refine_rounds`), và thứ tự khởi tạo module ở đó bị
ràng buộc để giữ "bit-exact ở step 0" giữa các thí nghiệm ấy. File đó đã bị xoá cùng
toàn bộ code vòng 1; nguyên văn còn trong `docs/old/ROUND_1_ARCHIVE.md`.

BA ĐIỀU KHÁC VÒNG 1
-------------------
1. Toạ độ `x` là LUỒNG CHÍNH: `x = update_box(x, delta)` nằm TRONG vòng lặp tầng, nên
   toạ độ tồn tại ở mọi tầng và `RoIFeatureSampler` gọi được ở mọi tầng. Vòng 1 chỉ có
   toạ độ ở đầu và cuối (`box_head` sau cả 6 tầng) nên chỉ lấy ảnh được một lần.
2. Hai loại token cho mỗi box: `h` (hình học) và `r` (ảnh), nối thành chuỗi 2N rồi cho
   self-attention với mask bất đối xứng. Vòng 1 cộng thẳng `tgt = tgt + roi(...)`, ép
   trộn 1:1 (OminiControl §3.2.2 đo được cách cộng này cho loss cao hơn).
3. Thời gian vào bằng adaLN-Zero (DiT) thay vì một token trong memory.
"""

import torch
import torch.nn as nn

from models.roi_sampler import RoIFeatureSampler

__all__ = [
    "MIN_WH",
    "update_box",
    "SinusoidalCoordEmbedding",
    "SinusoidalTimeEmbedding",
    "BoxCoordEmbedder",
    "TimestepConditioner",
    "build_cross_mask",
    "RegionGate",
    "DiTBlock",
]

# Box thật nhỏ nhất của CE-130 rộng 0,0059 nên ngưỡng này không cắt mất box nào.
MIN_WH = 0.005


def update_box(anchor, delta):
    """Cập nhật box theo kiểu OBJECT-NORMALIZED của V-DETR
    (`refs/repos/V-DETR/models/vdetr_transformer.py:271,278`):

        cx' = cx + d_cx * w          w' = w * exp(d_w)
        cy' = cy + d_cy * h          h' = h * exp(d_h)

    Tâm dịch theo ĐƠN VỊ KÍCH THƯỚC CHÍNH BOX ĐÓ, nên box nhỏ được nhích nhẹ còn box to
    đi được xa; kích thước nhân nên không bao giờ âm và không cần clamp để giữ dương.
    V-DETR đo +3,9 AP50 so với bản cộng thẳng (Table 7).

    `delta = 0` trả về ĐÚNG `anchor` (exp(0)=1, 0*w=0). Đây là thứ khiến `box_delta`
    zero-init làm mọi tầng thành ánh xạ đồng nhất ở bước 0.

    CLAMP `w`, `h` TRƯỚC KHI NHÂN (kế hoạch mục 7.1b): ở `t` lớn `x_t` gần nhiễu thuần,
    sau `decode_diffusion` thì `w` có đuôi chạm 0. Khi đó `cx + d_cx * w` gần như không
    dịch được BẤT KỂ `delta` lớn cỡ nào, làm chuỗi cộng dồn vô nghĩa ở đúng những bước
    mà nó cần hoạt động nhất. Vòng 1 không gặp lỗi này vì `box_head` dự đoán toạ độ
    tuyệt đối, không nhân với `w`.
    """
    cx, cy, w, h = anchor.unbind(-1)
    d_cx, d_cy, d_w, d_h = delta.unbind(-1)
    w_safe = w.clamp(min=MIN_WH)
    h_safe = h.clamp(min=MIN_WH)
    return torch.stack([
        cx + d_cx * w_safe,
        cy + d_cy * h_safe,
        (w_safe * d_w.exp()).clamp(MIN_WH, 1.0),
        (h_safe * d_h.exp()).clamp(MIN_WH, 1.0),
    ], dim=-1)


def clamp_to_valid(boxes_norm, valid_h):
    """Kéo tâm box về vùng ảnh THẬT, bỏ phần đệm ở đáy.

    Mọi ảnh CE-130 cao 384px và rộng >= 384, nên sau `resize_and_pad(512)` phần đệm
    LUÔN nằm ở đáy và `valid_h` là một ngưỡng vô hướng. Đo được: `valid_h` trung vị
    0,707 (train) / 0,686 (val), nhỏ nhất 0,199 — tức khoảng 30 % dưới của canvas là
    đệm, xấu nhất tới 80 %.

    Lấy mẫu RoI trong vùng đệm cho feature CLIP của một mảng phẳng (đệm bằng CLIP mean),
    hoàn toàn vô nghĩa. Ở `t` lớn có quá nửa số box rơi vào đó. Vòng 1 chịu được vì chỉ
    lấy RoI một lần; A lấy 6 lần mỗi khối nên phải chặn.

    `valid_h`: [B] hoặc vô hướng. Trả về bản sao, KHÔNG sửa tại chỗ.
    """
    if valid_h is None:
        return boxes_norm
    vh = valid_h if torch.is_tensor(valid_h) else torch.as_tensor(
        valid_h, device=boxes_norm.device, dtype=boxes_norm.dtype)
    vh = vh.to(boxes_norm.device, boxes_norm.dtype).reshape(-1, 1)   # [B,1]
    cx, cy, w, h = boxes_norm.unbind(-1)
    return torch.stack([cx, cy.clamp(max=vh), w, h], dim=-1)


class SinusoidalCoordEmbedding(nn.Module):
    """Mỗi toạ độ -> `dim` chiều sin/cos ở nhiều tần số; 4 toạ độ nối lại.

    VÌ SAO KHÔNG DÙNG THẲNG `Linear(4 -> D)`: Linear là tuyến tính, nên vị trí vào mạng
    dưới dạng ĐỘ LỚN — box ở x=0,4 cho vector gấp đôi box ở x=0,2. Sin/cos cho mỗi vị
    trí một CHỮ KÝ mà tích vô hướng giảm dần theo khoảng cách, đúng thứ attention cần.
    """

    def __init__(self, dim=64, temperature=10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim, self.temperature = dim, temperature

    def forward(self, boxes):
        """[..., 4] trong [0,1] -> [..., 4*dim]."""
        half = self.dim // 2
        freq = torch.arange(half, device=boxes.device, dtype=torch.float32)
        freq = self.temperature ** (2 * freq / self.dim)
        x = boxes.unsqueeze(-1) * 100.0 / freq       # scale 100: [0,1] -> dải hữu dụng
        emb = torch.cat([x.sin(), x.cos()], dim=-1)
        return emb.flatten(-2)


class SinusoidalTimeEmbedding(nn.Module):
    """Time embedding DDPM chuẩn."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        import math
        half = self.dim // 2
        f = math.log(10000) / (half - 1)
        f = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -f)
        a = t.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class BoxCoordEmbedder(nn.Module):
    """[B,N,4] -> [B,N,d_model]. Gói `coord_emb` + `box_proj` lại vì chúng luôn đi cùng
    nhau và được gọi ở HAI chỗ: dựng `h` đầu khối, và đánh dấu box cho token `r`."""

    def __init__(self, d_model=256, coord_dim=64):
        super().__init__()
        self.coord_emb = SinusoidalCoordEmbedding(coord_dim)
        self.proj = nn.Linear(4 * coord_dim, d_model)

    def forward(self, boxes_norm):
        return self.proj(self.coord_emb(boxes_norm))


class TimestepConditioner(nn.Module):
    """t -> vector điều kiện [B, d_model] cho adaLN và cho gate.

    Tính MỘT lần mỗi khối rồi dùng lại ở cả 6 tầng, vì `t` không đổi trong khối.
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, timesteps):
        return self.mlp(self.time_emb(timesteps))


def build_cross_mask(n, device):
    """Mask self-attention cho chuỗi `[h(N); r(N)]`, theo quy ước của PyTorch:
    True = CHẶN.

                  key h     key r
        query h     cho       cho       box biết nhau; box đọc vùng ảnh
        query r    CHẶN       cho       vùng so vùng; vùng KHÔNG biết box ở đâu

    Ô `r -> h` bị chặn để thay cho `.detach()` của vòng 1. Nếu `r` đọc được `h` thì
    loss của score chảy ngược về nhánh toạ độ và mạng học DỊCH BOX TỚI CHỖ DỄ CHẤM ĐIỂM
    thay vì chỗ có vật. Ba ô còn lại mở:
      h->h  N box biết nhau (intra-category coherence, đúng tên bài)
      h->r  box đọc vùng của chính nó VÀ của box khác — ĐƯỜNG CHÍNH để `delta` học
      r->r  vùng so sánh vùng; CE-130 có 20-30 vật cùng loại nên có lý

    Lưu ý đây KHÁC mục đích với mask của OminiControl2: họ chặn condition->noisy để
    cache KV (điều kiện của họ cố định). Ở đây `r` PHẢI đổi mỗi tầng.
    """
    mask = torch.zeros(2 * n, 2 * n, dtype=torch.bool, device=device)
    mask[n:, :n] = True                                   # r không đọc h
    return mask


class RegionGate(nn.Module):
    """Trộn quan sát RoI mới vào luồng ảnh `r`:

        g = sigmoid(MLP(t_emb))
        r = (1 - g) * r + g * roi_moi

    VÌ SAO TRỘN CHỨ KHÔNG CỘNG: cộng 6 lần liên tiếp thì `r` thành tổng chồng của 6 vùng
    khác nhau, không phân biệt được vùng nào là vùng nào, và chuẩn của `r` phình dần.
    Trộn giữ chuẩn ổn định và để mạng tự quyết tin quan sát mới bao nhiêu.

    `t_emb` làm điều kiện vì ở `t` lớn box còn là nhiễu nên RoI lấy tại chỗ vô nghĩa —
    mạng học đóng gate; ở `t` nhỏ thì mở. Bias khởi tạo 0 cho g ~ 0,5 ở bước đầu.
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model))
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

    def forward(self, r_prev, r_new, t_emb):
        g = torch.sigmoid(self.mlp(t_emb)).unsqueeze(1)    # [B,1,D]
        return (1.0 - g) * r_prev + g * r_new


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Một tầng của EXPERIMENT A. Thứ tự bên trong đúng theo kế hoạch mục 5.3:

        (1) lấy RoI tại `x` HIỆN TẠI, trộn vào `r` qua gate
        (2) nối [h; r] thành chuỗi 2N
        (3) self-attention có mask, điều biến bằng adaLN-Zero theo `t`
        (4) chỉ `r` cross-attention vào memory; mỗi luồng có FFN riêng
        (5) `delta = box_delta(h)` -> `x = update_box(x, delta)`

    `h` KHÔNG đọc memory. `h` mã hoá toạ độ còn memory là đặc trưng ảnh CLIP — tích vô
    hướng giữa hai không gian đó không có gì bảo đảm ăn khớp, và vòng 1 đã đo hệ quả:
    chỉ cross-attention cho AP50 1,52; thêm RoI lấy mẫu thẳng bằng toạ độ cho 6,36
    (4,2x). Còn `r` cùng không gian đặc trưng với memory nên đọc được tự nhiên. Vậy `h`
    lấy thông tin ảnh HOÀN TOÀN qua `r`, và mismatch biến mất khỏi luồng.

    adaLN-Zero (`refs/repos/RayDiffusion/ray_diffusion/model/dit.py:105-118`) khởi tạo
    lớp điều biến bằng 0 nên `gate = 0` -> tầng là ánh xạ đồng nhất ở bước đầu, luồng
    chính đi qua nguyên vẹn.
    """

    def __init__(self, d_model=256, n_head=8, dim_feedforward=None, dropout=0.1,
                 roi_dim=768, roi_k=3):
        super().__init__()
        ff = dim_feedforward or 4 * d_model

        self.roi = RoIFeatureSampler(roi_dim, d_model, roi_k, dropout)
        self.gate = RegionGate(d_model)
        self.box_mark = BoxCoordEmbedder(d_model)          # đánh dấu `r` thuộc box nào

        self.norm_seq = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout,
                                               batch_first=True)

        self.norm_r_ca = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout,
                                                batch_first=True)

        self.norm_h_ff = nn.LayerNorm(d_model)
        self.norm_r_ff = nn.LayerNorm(d_model)
        self.ff_h = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(ff, d_model))
        self.ff_r = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(ff, d_model))

        # adaLN-Zero: shift, scale, gate cho nhánh self-attention.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 3 * d_model))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

        # Head hồi quy delta, ZERO-INIT -> tầng trả về đúng box vào ở bước 0.
        self.box_delta = nn.Linear(d_model, 4)
        nn.init.zeros_(self.box_delta.weight)
        nn.init.zeros_(self.box_delta.bias)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, h, r, memory, patch_raw, t_emb, attn_mask, valid_h=None):
        """
        x          : [B,N,4]   toạ độ cxcywh trong [0,1] — LUỒNG CHÍNH
        h          : [B,N,D]   token hình học
        r          : [B,N,D]   token ảnh
        memory     : [B,M,D]   patch đã chiếu + token text
        patch_raw  : [B,P,d_in] patch CLIP thô, cho grid_sample
        t_emb      : [B,D]
        attn_mask  : [2N,2N] bool, True = chặn
        valid_h    : [B] hoặc None
        -> (x_moi, h, r)
        """
        n = x.shape[1]

        # (1) quan sát mới tại toạ độ hiện tại.
        # `.detach()` CHẶN ĐÚNG MỘT ĐƯỜNG: grid_sample khả vi theo toạ độ lấy mẫu, nên
        # không có nó thì loss của score chảy ngược qua `r` -> `roi` -> `x` và mạng học
        # dịch box tới chỗ dễ chấm điểm. Đường mà `delta` cần thì KHÔNG đi qua đây —
        # nó đi qua attention từ `r` sang `h` — nên chặn chỗ này không làm mất tín hiệu
        # học. DiffusionDet không gặp vấn đề vì nó lấy RoI tại box của stage TRƯỚC, vốn
        # là hằng số với stage hiện tại.
        x_sample = clamp_to_valid(x, valid_h).detach()
        r = self.gate(r, self.roi(patch_raw, x_sample), t_emb)

        # (2) nối hai loại token. `box_mark` cho token ảnh biết nó thuộc box nào —
        # OminiControl (Fig 3b) đo được việc đánh dấu vị trí cho token nối thêm làm hội
        # tụ nhanh hơn hẳn so với dùng chung vị trí.
        seq = torch.cat([h, r + self.box_mark(x_sample)], dim=1)     # [B,2N,D]

        # (3) self-attention có mask + adaLN-Zero.
        shift, scale, gate = self.ada(t_emb).chunk(3, dim=-1)
        s = _modulate(self.norm_seq(seq), shift, scale)
        a, _ = self.self_attn(s, s, s, attn_mask=attn_mask, need_weights=False)
        seq = seq + gate.unsqueeze(1) * self.drop(a)
        h, r = seq[:, :n], seq[:, n:]

        # (4) chỉ `r` đọc memory; FFN riêng cho từng luồng.
        rn = self.norm_r_ca(r)
        c, _ = self.cross_attn(rn, memory, memory, need_weights=False)
        r = r + self.drop(c)
        h = h + self.ff_h(self.norm_h_ff(h))
        r = r + self.ff_r(self.norm_r_ff(r))

        # (5) cập nhật luồng chính.
        x = update_box(x, self.box_delta(h))
        return x, h, r

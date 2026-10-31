"""CSANet (BMVC 2024; tham khảo `refs/CSANet/{network,model}.py`), chỉ phần dùng thật: generator SysNet + discriminator + loss GAN.
Tên tham số như bản gốc ⇒ nạp thẳng checkpoint công bố `{'model': {'generator', 'discriminator'}}` (khoá của module không dùng —
`att.convMod2` — chỉ bị bỏ qua).

Generator: nền + ảnh vật + mask vật (ô vuông `img_size`, ảnh vật letterbox) -> VGG16-BN chung 4 kênh -> đặc trưng lưới 2/4/8 ->
CrossConvAtt (nền ↔ vật) -> [đặc trưng ; mã d_noise] -> MLP -> trans (tỉ lệ cỡ, x, y) ∈ (0, 1)³. Mã = nhiễu N(0, I) lúc sinh; lúc train
nhánh 2 dùng mã VAE của ảnh ghép thật (khôi phục trans GT). `gan_losses` = loss G / D của `train_disc_gen` bản gốc.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

__all__ = ["CSANetConfig", "SysNet", "Discriminator", "gan_losses"]

BN_MOMENTUM = 0.1
_MEAN, _STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class CSANetConfig:
    """Siêu tham số model như argparse bản gốc."""

    def __init__(self, img_size=256, d_noise=1024, d_model=512, d_k=64, d_v=64, n_heads=8, mid=512, **_):
        self.img_size, self.d_noise, self.d_model, self.d_k, self.d_v, self.n_heads = img_size, d_noise, d_model, d_k, d_v, n_heads
        self.mid = mid                   # số kênh giữa của FeatLayer / conv_cat (bản gốc cố định 512; test dùng nhỏ)


def _norm(img):
    mean = torch.tensor(_MEAN, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
    return (img - mean) / std


class LayerNorm2d(nn.Module):
    """LayerNorm theo kênh (kiểu ConvNeXt `channels_first`)."""

    def __init__(self, n, eps=1e-6):
        super().__init__()
        self.weight, self.bias, self.eps = nn.Parameter(torch.ones(n)), nn.Parameter(torch.zeros(n)), eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        x = (x - u) / torch.sqrt((x - u).pow(2).mean(1, keepdim=True) + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.a = nn.Sequential(nn.Conv2d(dim, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 11, padding=5, groups=dim))
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x, y):
        x, y = self.norm(x.permute(0, 3, 2, 1)), self.norm(y.permute(0, 3, 2, 1))
        return self.proj(self.a(x) * self.v(y))


class FeatLayer(nn.Module):
    def __init__(self, d, n_mesh, mid=512):
        super().__init__()
        self.features = nn.Sequential(*[m for cin, cout in ((d, mid), (mid, mid), (mid, d)) for m in
                                        (nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))])
        self.pool = nn.AdaptiveAvgPool2d((n_mesh, n_mesh))

    def forward(self, x):
        return self.pool(self.features(x))


class FeatExtractor(nn.Module):
    """Gộp lưới n×n của các tầng: multi ⇒ [B, C, Σ n²]."""

    def __init__(self, d, meshes, multi, mid=512):
        super().__init__()
        self.layers = nn.ModuleList([FeatLayer(d, n, mid) for n in meshes])
        self.multi = multi

    def forward(self, x):
        return torch.cat([layer(x).flatten(2) if self.multi else layer(x) for layer in self.layers], dim=2)


class CrossConvAtt(nn.Module):
    """Chú ý chéo nền ↔ vật bằng ConvMod trên key gộp hai ảnh (bản gốc dùng `convMod1` cho cả hai nhánh — giữ)."""

    def __init__(self, o):
        super().__init__()
        c, kk, vv = o.d_model, o.d_k // 2 * o.n_heads, o.d_v * o.n_heads
        self.o = o
        self.key1, self.value1, self.key2, self.value2 = nn.Conv2d(c, kk, 1), nn.Conv2d(c, vv, 1), nn.Conv2d(c, kk, 1), nn.Conv2d(c, vv, 1)
        self.gamma1, self.gamma2 = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
        self.conv_cat = nn.Sequential(nn.Conv2d(c * 2, o.mid, 3, padding=1, bias=False), nn.BatchNorm2d(o.mid, momentum=BN_MOMENTUM),
                                      nn.ReLU(), nn.Conv2d(o.mid, c, 3, padding=1, bias=False), nn.BatchNorm2d(c, momentum=BN_MOMENTUM), nn.ReLU())
        self.W_O_1, self.W_O_2 = nn.Linear(vv, c), nn.Linear(vv, c)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.LayerNorm(c)
        self.convMod1 = ConvMod(dim=o.d_k)

    def _branch(self, k, v, x, W, gamma):
        B, C, H, Wd = x.shape
        out = self.convMod1(k, v).transpose(1, 2).contiguous().view(B, -1, self.o.n_heads * self.o.d_v)
        out = gamma * W(out).view(*x.shape) + x
        return self.norm(out.view(B, H, Wd, C)).view(*x.shape)

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        nh = self.o.n_heads
        heads = lambda conv, x: conv(x).view(B, nh, -1, H * W).permute(0, 1, 3, 2)  # noqa: E731
        k = torch.cat([heads(self.key1, x1), heads(self.key2, x2)], 2).view(B, nh, -1, H * W).permute(0, 1, 3, 2)
        o1 = self._branch(k, heads(self.value1, x1), x1, self.W_O_1, self.gamma1)
        o2 = self._branch(k, heads(self.value2, x2), x2, self.W_O_2, self.gamma2)
        return self.pool(self.conv_cat(torch.cat([o1, o2], 1))).flatten(1)          # [B, d_model]


class SysNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt, self.img_size = opt, opt.img_size
        vgg = models.vgg16_bn(weights=None)
        vgg.features[0] = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.shared_net = nn.Sequential(*list(vgg.features[:34]))
        self.fg_extractor = FeatExtractor(opt.d_model, [2, 4, 8], True, opt.mid)
        self.bg_extractor = FeatExtractor(opt.d_model, [2, 4, 8], True, opt.mid)
        self.latent_extractor = FeatExtractor(opt.d_model, [1], False, opt.mid)
        self.vae_enc = nn.Module()
        self.vae_enc.fc1 = nn.Sequential(nn.Linear(opt.d_model, 1024), nn.ReLU(inplace=True))
        self.vae_enc.fc2, self.vae_enc.fc3 = nn.Linear(1024, opt.d_noise), nn.Linear(1024, opt.d_noise)
        self.regress_net = nn.Module()
        self.regress_net.regressor = nn.Sequential(nn.Linear(opt.d_model + opt.d_noise, 1024), nn.ReLU(True), nn.Dropout(0.5),
                                                   nn.Linear(1024, 1024), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(1024, 3))
        self.att = CrossConvAtt(opt)

    def encode(self, bg, fg, msk):
        """-> đặc trưng chú ý chéo [B, d_model] (không phụ thuộc mã)."""
        f = self.shared_net(torch.cat((_norm(fg), msk), 1))
        b = self.shared_net(torch.cat((_norm(bg), torch.zeros_like(msk)), 1))
        return self.att(self.fg_extractor(f).unsqueeze(-1), self.bg_extractor(b).unsqueeze(-1))

    def regress(self, attn, code):
        return torch.tanh(self.regress_net.regressor(torch.cat((attn, code), 1))) / 2.0 + 0.5

    def blend(self, bg, fg, msk, fg_bbox, trans):
        """Ghép ảnh vật vào nền theo trans bằng affine_grid (cho discriminator)."""
        inv = 1 / (trans[:, 0] + 1e-6)
        z = torch.zeros_like(inv)
        theta = torch.stack((inv, z, (1 - 2 * trans[:, 1]) * (inv - fg_bbox[:, 2] / self.img_size),
                             z, inv, (1 - 2 * trans[:, 2]) * (inv - fg_bbox[:, 3] / self.img_size)), 1).view(-1, 2, 3)
        grid = F.affine_grid(theta, fg.size(), align_corners=True)
        f, m = F.grid_sample(fg, grid, align_corners=True), F.grid_sample(msk, grid, align_corners=True)
        return m * f + (1 - m) * bg, m

    def forward(self, bg, fg, msk, noise):
        """Sinh: -> trans [B, 3] (mỗi hàng nhiễu một box)."""
        return self.regress(self.encode(bg, fg, msk), noise)


class Discriminator(nn.Module):
    """PatchGAN 4 kênh (ảnh + mask), sigmoid, TB không gian -> [B]."""

    def __init__(self, ndf=64, n_layers=3):
        super().__init__()
        seq, nf = [nn.Conv2d(4, ndf, 4, stride=2, padding=2), nn.LeakyReLU(0.2, True)], ndf
        for i in range(1, n_layers + 1):
            prev, nf = nf, min(nf * 2, 512)
            seq += [nn.Conv2d(prev, nf, 4, stride=2 if i < n_layers else 1, padding=2), nn.InstanceNorm2d(nf), nn.LeakyReLU(0.2, True)]
        seq += [nn.Conv2d(nf, 1, 4, stride=1, padding=2), nn.Sigmoid()]
        self.model = nn.Sequential(*seq)

    def forward(self, img, mask):
        return self.model(torch.cat((_norm(img), mask), 1)).mean((2, 3)).flatten()


def _bce(p, y):
    """= F.binary_cross_entropy (log kẹp >= -100) nhưng KHÔNG assert p ∈ [0, 1] trên CUDA: p NaN -> loss NaN (bước bị GradScaler bỏ)
    thay vì device assert giết cả tiến trình."""
    return -(y * torch.log(p).clamp(min=-100) + (1 - y) * torch.log1p(-p).clamp(min=-100)).mean()


def gan_losses(G, D, bg, fg, msk, fg_bbox, comp, comp_msk, trans_label, amp=False):
    """Một lượt `train_disc_gen` bản gốc (mọi mẫu nhãn dương) -> (g_loss, d_loss, thành phần). G forward trong autocast (`amp`); D + loss
    ở fp32: D kết thúc bằng sigmoid rồi BCE trên xác suất — ở fp16 sigmoid bão hoà sớm (|logit| > ~8) và đạo hàm 1/x của BCE tràn số khi
    quay về đồ thị fp16 ⇒ GradScaler bỏ bước liên tục. D nhỏ (~2,8M tham số) nên fp32 rẻ."""
    with torch.autocast(device_type=bg.device.type, dtype=torch.float16, enabled=amp):
        attn = G.encode(bg, fg, msk)
        t1 = G.regress(attn, torch.randn(attn.shape[0], G.opt.d_noise, device=bg.device, dtype=attn.dtype))
        lat = G.latent_extractor(G.shared_net(torch.cat((_norm(comp), comp_msk), 1))).flatten(1)
        h = G.vae_enc.fc1(lat)
        mu, logvar = G.vae_enc.fc2(h), G.vae_enc.fc3(h)
        t2 = G.regress(attn, mu + torch.randn_like(mu) * torch.exp(logvar / 2))
        g1, m1 = G.blend(bg, fg, msk, fg_bbox, t1)
        g2, m2 = G.blend(bg, fg, msk, fg_bbox, t2)
    g1, m1, g2, m2, t2, mu, logvar = (x.float() for x in (g1, m1, g2, m2, t2, mu, logvar))
    d1, d1_det, d2, d2_det = D(g1, m1), D(g1.detach(), m1.detach()), D(g2, m2), D(g2.detach(), m2.detach())
    d_real = D(comp.float(), comp_msk.float())
    one, zero = torch.ones_like(d_real), torch.zeros_like(d_real)
    bce = _bce
    w = torch.cat((torch.sin(t2[:, :1] * np.pi / 2), torch.cos(t2[:, :1] * np.pi / 2), torch.cos(t2[:, :1] * np.pi / 2)), 1).detach()
    parts = {"g_gan1": bce(d1, one), "g_gan2": bce(d2, one), "g_rec": ((t2 - trans_label) ** 2 * w).mean(1).mean(),
             "g_kld": (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean(),
             "d_real": bce(d_real, one), "d_fake1": bce(d1_det, zero), "d_fake2": bce(d2_det, zero)}
    g_loss = parts["g_gan1"] * 4.0 + parts["g_gan2"] + parts["g_rec"] * 50.0 + parts["g_kld"]
    d_loss = parts["d_real"] + parts["d_fake1"] * 0.25 + parts["d_fake2"]
    return g_loss, d_loss, parts

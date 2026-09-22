"""Diffu2Seg configuration.

Every number below has a reason, written down so nobody changes one blindly.
Where a value differs from the paper (arXiv 2609.06491), the reason is a number
measured on CE-130, not a preference.

Python rather than YAML because several fields are DERIVED and must never be
typed twice:
    grid_r          = canvas // vae_stride
    prompt_stride_px = prompt_stride_cells * vae_stride
A flat YAML would force the same constant to appear in two places, which is
exactly where silent drift starts.
"""

import os
from dataclasses import dataclass, field, asdict

VAE_STRIDE = 8          # SD2 VAE downsamples 8x; latent grid = canvas / 8


@dataclass
class Diffu2SegConfig:
    # ------------------------------------------------------------------
    # Image / canvas
    # ------------------------------------------------------------------
    canvas: int = 512
    # 512 is the project's ONE canonical canvas (utils/box_ops_np.py). Every
    # number already measured for A/B/C1/E1/D.1 lives in this system, so using
    # anything else would make the results incomparable for no gain.

    # ------------------------------------------------------------------
    # SD2 attention extraction
    # ------------------------------------------------------------------
    hf_model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    # ⚠️ KHÔNG PHẢI SD2 NHƯ PAPER, và đây là lý do (đo 2026-09-15):
    # cả dòng `stabilityai/stable-diffusion-2*` đã bị KHOÁ trên HuggingFace —
    # `stable-diffusion-2`, `-2-base`, `-2-1`, `-2-1-base` đều trả HTTP 401
    # "Invalid username or password" cho một repo từng công khai, đo từ BA máy
    # khác nhau (local Mac, server aiotlab, và một máy thứ ba). 401 chứ không
    # phải 404 nghĩa là repo còn tồn tại nhưng đã thành gated/private. Không
    # phải lỗi mạng của ta, và token cũng không gỡ được nếu chưa xin quyền.
    #
    # SD 1.5 KHÁC SD2 Ở ĐÂU (đo từ unet/config.json của cả hai):
    #   giống : sample_size 64 (input 512, lưới latent 64x64)
    #           block_out_channels [320, 640, 1280, 1280]
    #           up_block_types = 3x CrossAttnUpBlock2D
    #           -> tên block `up_blocks.3.attentions.{0,1,2}` VẪN ĐÚNG
    #   khác  : cross_attention_dim 768 (SD2: 1024)
    #           attention_head_dim 8   (SD2: 5)
    #
    # Cả hai khác biệt đều KHÔNG chạm vào đường ta dùng: ta hook `attn1`
    # (self-attention ảnh<->ảnh), không dùng cross-attention, và trung bình
    # trên MỌI head nên số head không quan trọng. M2N2 cũng xác nhận: hai file
    # aggregator SD1/SD2 của họ khác nhau ĐÚNG 2 chỗ — tên repo mặc định và
    # attention_resolution mặc định — toàn bộ logic hook giống hệt.
    #
    # ⚠️ ĐIỀU PHẢI ĐO LẠI: `timestep=150` là giá trị Diffuse2Seg tinh chỉnh CHO
    # SD2. Thang timestep của SD1.5 không nhất thiết đặt đặc trưng tốt nhất ở
    # cùng chỗ. Cửa chặn 0 nên quét vài giá trị t trước khi chốt.

    local_model_dir: str = "../weights/diffu2seg/stable-diffusion-v1-5"
    # ...nhưng máy chạy thật thì KHÔNG ra được HuggingFace: server trả
    #   401 Client Error / Repository Not Found / "Invalid username or password"
    # cho một repo CÔNG KHAI, trong khi HF_TOKEN rỗng và không có file token nào
    # (đo 2026-09-15). Không có credential nào để mà sai -> chặn ở tầng mạng.
    #
    # Nên SD2 tải ở local rồi đưa lên server theo đúng quy ước `weights/` của dự
    # án (zip thủ công, không scp/rsync). Đường dẫn tương đối so với thư mục
    # diffu2seg/, khớp với weights/ nằm cạnh nó.
    #
    # `model_source` dưới đây tự chọn: có thư mục local thì dùng, không thì rơi
    # về tên repo. Không cần sửa config khi đổi máy.
    timesteps: tuple = (150,)
    # Paper uses t=150 (following M2N2, which used 100). STAGE 1 USES EXACTLY
    # ONE TIMESTEP: blending two (the paper's w1=0.85 / w2=0.15) is a SECOND
    # variable, and the project rule is one variable per step.

    timestep_weights: tuple = (1.0,)

    w_down_0: float = 0.0
    w_down_1: float = 0.0
    w_up_0: float = 0.5
    w_up_1: float = 0.5
    w_up_2: float = 0.0
    # M2N2's measured optimum (docs/old/ROUND_1_ARCHIVE.md (phần 05), and their Table 1 on DAVIS:
    # up_0 alone 6.90, up_1 alone 7.10, both at 0.5 -> 6.72 NoC90; the two
    # down blocks are far worse at 15.25 / 13.18). These are the two highest-
    # resolution decoder blocks, which is also what Diffuse2Seg uses.

    tau_att: float = 0.55
    # Paper value. Sharpens the affinity before propagation.

    prompt_text: str = ""
    # Empty prompt = class-agnostic, as the paper does ("we omit the text prompt
    # and only pass null text embeddings"). CE-130 ships a usable caption per
    # image (`class_based_caption`, e.g. "pill"), and the wiring accepts it --
    # but turning it on is a THIRD variable. Not in stage 1.

    unet_dtype: str = "float16"
    # UNet in fp16 to keep the single denoise step cheap.

    math_dtype: str = "float32"
    # ...but the affinity and the whole p-Laplacian stay fp32. g**(p-2) with
    # p-2 = -0.4 amplifies small values, and fp16's eps ~6e-8 would underflow
    # straight into inf. M2N2 also calls .float() inside its callback.
    # (Unrelated to the project's no-AMP rule, which is about training
    # DiffusionDet -- stated here so nobody conflates the two.)

    # ------------------------------------------------------------------
    # p-Laplacian propagation (Algorithm 1)
    # ------------------------------------------------------------------
    p: float = 1.6
    # Paper value. GATE 1 sweeps {2.0, 1.8, 1.6, 1.4}: at p=2 the exponent
    # g**0 = 1 collapses gamma to 2*A and the whole algorithm becomes ordinary
    # linear diffusion -- one line. If p<2 buys nothing measurable on CE-130,
    # compute_g is wasted work and should be deleted. That is the gate.

    lam: float = 1e-5
    # Paper value, kept so the comparison to the paper stays honest. NOTE it is
    # a weak anchor: on a synthetic graph the solution decayed toward zero as
    # iterations grew, so tau_prop below is NOT a minor knob. If gate 1 returns
    # GREY, lam is the next thing to sweep -- after p, never together with it.

    tau_prop: float = 1e-4
    max_iter: int = 200
    # ⚠️ max_iter IS NOT A PAPER VALUE -- the paper gives tau_prop and no cap.
    # A cap is mandatory or one non-converging image hangs the job. 200 was set
    # for CE-130 at r=64; config/paper.py raises it to 1000, because at r=140 a
    # prompt has to travel over five times as many tokens to cover the same
    # fraction of the image. n_iter is dumped per image so the cap is visible
    # rather than assumed.

    g_eps: float = 1e-8
    # NOT IN THE PAPER, and mandatory. g is exactly 0 on flat regions (common
    # on CE-130's plain backgrounds); with p-2 = -0.4, 0**(-0.4) = inf, which
    # turns f into NaN on the next line.

    # ------------------------------------------------------------------
    # Prompt grid
    # ------------------------------------------------------------------
    prompt_stride_cells: int = 3
    # Paper uses 6. Measured on 200 val images (canvas 512, r=64):
    #   s=2 -> 1024 prompts, 99.0 % of boxes get >=1 prompt
    #   s=3 ->  441 prompts, 96.6 %          <- chosen
    #   s=4 ->  256 prompts, 91.5 %
    # The median short side of a CE-130 box is 4.65 cells and the median
    # SMALLEST box per image is 2.24 cells, so the paper's s=6 would step over
    # small objects entirely. s=2 buys +2.4 % for 2.3x the cost -> later sweep.

    min_valid_frac: float = 1.0
    # Fraction of a prompt cell that must lie inside the real image. 1.0 = drop
    # any prompt touching padding. Padding is ~29 % of the canvas at the median
    # aspect ratio (1.41) because every CE-130 image is 384 tall and up to 1918
    # wide. A prompt seeded on flat CLIP-mean grey propagates over the padding
    # and returns one huge garbage mask. The paper has no such notion: its
    # images are square.

    # ------------------------------------------------------------------
    # Mask extraction (STAGE 1: a single level)
    # ------------------------------------------------------------------
    mask_threshold_mode: str = "quantile"
    mask_quantile: float = 0.90
    # Take the top 10 % of each propagated map. Order-of-magnitude estimate:
    # ~21 objects/image (val median) x (4.65 cells)^2 ~ 450 of 4096 cells ~ 11 %
    # foreground. THIS IS THE WEAKEST NUMBER IN THIS FILE -- sweep
    # {0.80, 0.85, 0.90, 0.95} right after p.

    mask_rel_floor: float = 0.05
    # A quantile is RELATIVE and therefore always selects ~10 % of cells, even
    # from a map with no signal at all. Measured: a prompt on background
    # propagated to max 0.0000 while its own q90 was 1.5e-05, so it still
    # produced a 1x1 box -- two objects came out as 58 boxes. This floor asks
    # the absolute question too: is anything here at least 5 % of the strongest
    # response in the image? Prompts that land on nothing now return nothing.

    min_component_cells: int = 1
    # No size filter in stage 1. p10 of the smallest box per image is 0.89
    # cells, so filtering >=2 would throw away the small objects before they
    # can even be measured.

    connectivity: int = 1
    # 4-connected, matching M2N2's 4-neighbour flood fill.
    # scipy.ndimage.label's default structure is exactly this.

    # ------------------------------------------------------------------
    # Box extraction
    # ------------------------------------------------------------------
    dedup_iou: float = 0.70
    # ~441 prompts over ~21 objects means ~21 near-identical boxes per object.
    # Tighter than CE-LocModel's nms_iou=0.5 on purpose: CE-130 objects of the
    # same class genuinely crowd and overlap, and 0.5 would merge neighbours.

    min_box_cells: float = 0.5
    max_box_area_frac: float = 0.25
    # Upper bound is a TRIPWIRE, not a filter for accuracy: a box covering >25 %
    # of the canvas on CE-130 (median object area 0.33 % of the image) almost
    # always means a prompt escaped into the padding. The count of boxes it
    # removes is reported; a large count means the padding logic is broken.

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    stage: int = 1
    seed: int = 0

    # ------------------------------------------------------------------
    # STAGE 2 -- Algorithm 2 of the paper (mask merging + NMS)
    # ------------------------------------------------------------------
    n_levels: int = 6
    # §A.1: "L = 6 log-spaced thresholds". Their sweep (Fig. 6g) shows AR
    # rising to 6 and flattening; §A.1 concludes "we therefore keep 6 levels".

    kl_h_min: float = 0.186
    kl_h_max: float = 2.99
    # §A.1: "six thresholds in the range [0.186, 2.99] that are log-spaced to
    # emphasize merges at lower thresholds". LOG-spaced, not linear: small h
    # gives many clusters (small objects), large h gives few (whole objects),
    # and the interesting structure is at the low end.

    nms_iou: float = 0.9
    # §A.1: "tau_IoU = 0.9, which worked best to preserve recall". DELIBERATELY
    # PERMISSIVE -- the paper measured that 0.5 "reduces recall significantly by
    # 3.1 p.p. in mAR", because adjacent granularity levels legitimately produce
    # nested masks (a chair and its seat) that a strict NMS would delete.

    min_area_px: int = 100
    # §A.1: "We remove all masks containing fewer than 100 pixels to remove
    # noise". In ORIGINAL IMAGE pixels, applied after upsampling.

    max_masks: int = 1000
    # §3.4: "at most Nmax complementary masks per image [...] set to 1000".
    # The same 1000 that AR_1000 is named after.

    kl_eps: float = 1e-12
    # NOT IN THE PAPER. KL divergence takes log(p/q); a propagated map is zero
    # over most of the image, and log(0) = -inf poisons the whole distance
    # matrix. Same class of omission as g_eps.

    keep_level_masks: bool = False
    # CHỈ ĐỂ VẼ, không ảnh hưởng phép tính nào. Bật thì merge_maps_to_masks trả
    # thêm info["level_masks"] = mask của TỪNG mức granularity, chưa qua NMS.
    # Cần vì mỗi mức là một PHÂN HOẠCH riêng; vẽ cả 6 mức chồng lên nhau (như
    # bản visualize đầu tiên) làm mức thô đè mức mịn và ảnh thành loang lổ,
    # trong khi Hình 1 của paper vẽ mỗi panel MỘT phân hoạch.
    # Mặc định TẮT: giữ thêm ~745 mask (H, W) bool mỗi ảnh, đường chạy đo AR
    # không cần.

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------
    @property
    def peak_attention_gb(self) -> float:
        """Bộ nhớ ĐỈNH của bước trích attention — KHÔNG phải cỡ của A.

        A là (N, N) fp32 = 1,54 GB ở r=140, và tôi đã tính NHẦM đó là chi phí
        chính. Thực tế chi phí nằm ở tensor TRUNG GIAN bên trong hook:

            attn_weight (1, n_head, N, N) fp16   = 6,15 GB ở r=140, 8 head
            + bộ đệm fp32 (N, N) trong callback  = 1,54 GB
            + một head fp32 tạm (N, N)           = 1,54 GB

        Bản gốc chép từ M2N2 dựng `x.float()` trên CẢ tensor (B, H, N, N) —
        12,3 GB — cộng ba bản attn_weight chồng nhau, tổng ~31,5 GB. OOM trên
        A30 24 GB ngay ảnh đầu (đo 2026-09-15: "Tried to allocate 11.45 GiB").
        Ở r=64 của M2N2 cùng đoạn code chỉ tốn ~0,8 GB nên không ai thấy.

        Sau khi sửa (in-place + cộng dồn từng head): ~9,2 GB, cộng SD1.5 fp16
        ~1,9 GB là ~11,2 GB. Vừa A30 với biên rộng.

        ⚠️ n_head = 8 cho SD 1.5, 5 cho SD2. Công thức dùng 8 (trường hợp xấu).
        """
        n = self.n_tokens
        return (n * n * 8 * 2 + n * n * 4 * 2) / 1e9

    @property
    def grid_r(self) -> int:
        """Latent grid side. canvas=512 -> r=64 -> N = 4096 tokens.

        The affinity is N x N: 0.07 GB fp32 here, versus 1.54 GB at the paper's
        r=140. 22x cheaper, and it lines up exactly with the project's 512
        canvas.

        THE COST, which must be stated whenever results are read: one cell is
        8 canvas px, and p10 of the smallest box per image is 0.89 cells. Around
        10 % of images have their smallest object below one cell -- those are
        unrecoverable at this resolution, no matter what p does. If gate 1 comes
        back GREY, r=96 (0.34 GB) is the next variable to sweep -- after p.
        """
        assert self.canvas % VAE_STRIDE == 0, "canvas must be a multiple of 8"
        return self.canvas // VAE_STRIDE

    @property
    def model_source(self) -> str:
        """Đường dẫn local nếu có, ngược lại là tên repo HF.

        Trả về local_model_dir khi thư mục đó tồn tại VÀ có model_index.json
        (file mà StableDiffusionImg2ImgPipeline.from_pretrained đọc đầu tiên) --
        chỉ kiểm tra thư mục có tồn tại là chưa đủ: một thư mục rỗng do giải nén
        hỏng sẽ lọt qua rồi mới chết ở from_pretrained với thông báo khó hiểu.
        """
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = self.local_model_dir
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(base, path))
        if os.path.isfile(os.path.join(path, "model_index.json")):
            return path
        return self.hf_model_id

    @property
    def n_tokens(self) -> int:
        return self.grid_r * self.grid_r

    @property
    def prompt_stride_px(self) -> int:
        return self.prompt_stride_cells * VAE_STRIDE

    def validate(self):
        assert len(self.timesteps) == len(self.timestep_weights), \
            "timesteps and timestep_weights must have equal length"
        assert abs(sum(self.timestep_weights) - 1.0) < 1e-9, \
            f"timestep_weights must sum to 1, got {sum(self.timestep_weights)}"
        ws = [self.w_down_0, self.w_down_1, self.w_up_0, self.w_up_1, self.w_up_2]
        assert abs(sum(ws) - 1.0) < 1e-9, f"block weights must sum to 1, got {sum(ws)}"
        assert 0 < self.p <= 2.0, "p must be in (0, 2]"
        assert self.g_eps > 0, "g_eps must be > 0 or g**(p-2) can be inf"
        assert self.max_iter > 0
        assert 1 <= self.prompt_stride_cells < self.grid_r
        assert 0.0 < self.mask_quantile < 1.0
        assert self.math_dtype == "float32", \
            "p-Laplacian must run in fp32: g**(p-2) underflows in fp16"
        if self.stage >= 2:
            assert 0 < self.kl_h_min < self.kl_h_max, \
                f"need 0 < kl_h_min < kl_h_max, got {self.kl_h_min}, {self.kl_h_max}"
            assert self.n_levels >= 1
            assert 0.0 < self.nms_iou <= 1.0
            assert self.min_area_px >= 0
            assert self.max_masks >= 1
            assert self.kl_eps > 0, "log(0) in the KL distance without kl_eps"
        return self

    def to_dict(self):
        d = asdict(self)
        d.update(grid_r=self.grid_r, n_tokens=self.n_tokens,
                 prompt_stride_px=self.prompt_stride_px,
                 model_source=self.model_source,
                 peak_attention_gb=round(self.peak_attention_gb, 2))
        return d

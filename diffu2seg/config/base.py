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
    hf_model_id: str = "stabilityai/stable-diffusion-2"
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
    # M2N2's measured optimum (docs/05-nguon.md, and their Table 1 on DAVIS:
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
    # Paper gives tau_prop but no iteration cap. A cap is mandatory: without it
    # one non-converging image hangs the whole job. n_iter is dumped per image
    # so the gate can check convergence instead of assuming it.

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
    # Stage 2 only (unused while stage == 1)
    # ------------------------------------------------------------------
    n_levels: int = 6
    kl_thresholds: tuple = ()
    nms_iou: float = 0.70
    level_priority: str = "finest_first"

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------
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
        return self

    def to_dict(self):
        d = asdict(self)
        d.update(grid_r=self.grid_r, n_tokens=self.n_tokens,
                 prompt_stride_px=self.prompt_stride_px)
        return d

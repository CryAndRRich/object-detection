# DiffuGroundingDINO: baseline config plus the diffusion process over reference points.
#
# A delta on cfg_odvg.py rather than a copy, so the two cannot silently drift apart
# and an A/B really is one flag's worth of difference.

_base_ = "cfg_odvg.py"

use_diffusion = True

# ---------------------------------------------------------------- schedule
# T=1000 follows the released DiffuDINO code. The paper's §3.2 argues for T=100 on
# the grounds that the latent is only 4-dimensional; worth an ablation, since
# nothing else needs to change.
diff_num_timesteps = 1000

# Decoder evaluations at inference. 3 is the optimum in the DiffuDETR ablation
# (Tables 6/7: 51.95 AP at 3 vs 51.68 at 1 and 51.49 at 10), *not* DiffusionDet's
# SAMPLE_STEP=1. Costs ~+17% GFLOPs over one step because only the decoder repeats.
diff_sampling_timesteps = 3

# Latent rescaling. 2.0 matches both DiffusionDet's SNR_SCALE default and
# DiffuDINO's self.scale.
diff_snr_scale = 2.0

# 0.0 makes DDIM deterministic, matching the released DiffuDINO (its
# ddim_sampling_eta is hardcoded to 0.0; the stochastic-sampler term is dead code
# in their released version). Set to 1.0 for a fully stochastic sampler instead.
diff_ddim_eta = 0.0

# cosine / linear / sqrt. Cosine wins in the DiffuDETR ablation (Table 5).
diff_schedule = "cosine"

# ---------------------------------------------------------------- conditioning
# "film": per-layer scale-shift on the query, what the released DiffuDINO does.
# "add": add a projection of t to query_pos, closer to the paper's eq. 3.
diff_time_inject = "film"

# "triple" is what the released DiffuDINO actually does: three independently
# parameterised FiLM blocks per layer, before self-attention, between
# self-attention and the cross-attentions, and before the FFN. "post_sa" is the
# cheaper single-block approximation from the paper's eq. 3 (MSDA(SA(q) + t));
# "pre_layer" conditions the query before the whole layer instead.
diff_time_inject_point = "triple"
diff_time_hidden_mult = 4

# DiffuDINO's TimeStepBlock returns `x + h`, which is 2*x at initialisation. That is
# harmless when training from scratch (their setting) and destructive when
# finetuning from groundingdino_swint_ogc.pth (ours), so the default is the
# identity-at-init form. Flip to True to reproduce them literally.
diff_film_residual = False
diff_time_share_layers = False

# ---------------------------------------------------------------- loss
# Weight the box/classification loss by w(t): a sample noised almost to nothing
# should not be graded as if its boxes were recoverable.
diff_loss_t_weighting = True
diff_loss_weight_mode = "diffudino"  # "diffudino" | "vlb" | "none"
# Rescale w(t) to mean 1. Without it the raw DiffuDINO weights average ~0.2, which
# quietly divides the box loss by ~5 and makes bbox_loss_coef=5.0 mean something
# different than it does in the baseline.
diff_normalize_loss_weight = True

# Filler boxes for queries with no ground truth. "center" is the released
# DiffuDINO's actual default (a constant [0.5,0.5,0.5,0.5] box, its noisy_gt=False
# path); "normal" is DiffusionDet's N(0.5, 1/6) jitter; "sigmoid_normal" is
# DiffuDINO's other variant.
diff_pad_mode = "center"

# ---------------------------------------------------------------- warm-up
# For the first N steps the pretrained towers are frozen while the freshly
# initialised timestep modules settle. The freeze is recomputed from the global step
# each iteration, so it survives a resume with no extra state.
diff_warmup_iters = 2000
diff_warmup_freeze_keywords = ["backbone.0", "bert"]

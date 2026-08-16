# Baseline GroundingDINO on ODVG data -- no diffusion.
#
# This is the A/B reference: identical to config/cfg_odvg_diffusion.py except that
# `use_diffusion` is off. Values match Open-GroundingDino's cfg_odvg.py so that a
# run here is comparable with their published COCO finetune (57.3 mAP).

# ---------------------------------------------------------------- data aug
data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
data_aug_max_size = 1333
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]
data_aug_scale_overlap = None

# ---------------------------------------------------------------- model
modelname = "diffu_groundingdino"
backbone = "swin_T_224_1k"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
dilation = False

enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
transformer_activation = "relu"

two_stage_type = "standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
dec_pred_bbox_embed_share = True
dec_pred_class_embed_share = True
embed_init_tgt = True

# text
text_encoder_type = "bert-base-uncased"  # or a local dir, e.g. ../weights/diffu_grounding_dino/bert-base-uncased
max_text_len = 256
sub_sentence_present = True
use_text_enhancer = True
use_fusion_layer = True
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
max_labels = 50  # categories per prompt: positives plus sampled negatives

# gradient checkpointing -- both on to keep memory headroom on a single 24GB GPU
use_checkpoint = True
use_transformer_ckpt = True

# ---------------------------------------------------------------- optim
batch_size = 4
lr = 0.0001
lr_backbone = 1e-05
lr_backbone_names = ["backbone.0", "bert"]
lr_linear_proj_mult = 1e-05  # absolute lr for these groups, despite the name
lr_linear_proj_names = ["ref_point_head", "sampling_offsets"]
weight_decay = 0.0001
param_dict_type = "ddetr_in_mmdet"
freeze_keywords = ["bert"]  # e.g. ["backbone.0", "bert"] to freeze both towers

epochs = 15
lr_drop = 4
lr_drop_list = [4, 8]
onecyclelr = False
multi_step_lr = False
save_checkpoint_interval = 1
clip_max_norm = 0.1

# ---------------------------------------------------------------- loss
aux_loss = True
matcher_type = "HungarianMatcher"
set_cost_class = 1.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 2.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25
focal_gamma = 2.0

# ---------------------------------------------------------------- eval
num_select = 300
nms_iou_threshold = -1
use_coco_eval = True
label_list = []  # used only when use_coco_eval is False

# ---------------------------------------------------------------- misc
use_ema = False
ema_decay = 0.9997
ema_epoch = 0
debug_nan = False

# Diffusion off. Kept here so both configs expose the same field set and code paths
# never have to guess whether the key exists.
use_diffusion = False

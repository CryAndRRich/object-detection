"""
học denoising lora mới
"""
print("run train_fsc_remove_v3.py")
import sys,os
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
import argparse
import copy
import logging
import math
import os
from contextlib import contextmanager
import functools
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from packaging import version
from peft import LoraConfig
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from src.hook import save_model_hook,load_model_hook
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
)
from src.UniCombineTransformer2DModel import UniCombineTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from src.fsc_remove_dataloader import get_dataset,prepare_dataset,collate_fn
if is_wandb_available():
    pass
from src.text_encoder import encode_prompt
from datetime import datetime
import torchvision
import numpy as np
from PIL import Image
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

logger = get_logger(__name__, log_level="INFO")


@contextmanager
def preserve_requires_grad(model):
    # 备份 requires_grad 状态
    requires_grad_backup = {name: param.requires_grad for name, param in model.named_parameters()}
    yield
    # 恢复 requires_grad 状态
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad_backup[name]
def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two

def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)

def decode_latents(latents: torch.Tensor, vae: torch.nn.Module) -> torch.Tensor:
    """Decode VAE latents to pixel images in [-1, 1] range."""
    latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    images = vae.decode(latents.to(vae.dtype)).sample
    return images.float().clamp(-1., 1.)

class ImageLogger:
    """
    Saves decoded sample images to disk during training at a fixed step interval.
    Adapted from the PyTorch Lightning ImageLogger in main.py for use with Accelerate.

    At every `log_every_n_steps` global steps (on the main process only) it saves a PNG
    grid to <work_dir>/images/gs-{step:06d}.png with columns:
        [condition_0, condition_1, ..., ground-truth target, model prediction (x0)]

    The model prediction is recovered from the velocity output via:
        x0_pred = noisy_input - sigma * model_pred_velocity
    and then decoded through the VAE back to pixel space.
    """

    def __init__(self, work_dir: str, log_every_n_steps: int = 500, max_images: int = 4):
        self.image_dir = os.path.join(work_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        self.log_every_n_steps = log_every_n_steps
        self.max_images = max_images

    def should_log(self, global_step: int) -> bool:
        return self.log_every_n_steps > 0 and global_step % self.log_every_n_steps == 0

    @torch.no_grad()
    def log(
        self,
        global_step: int,
        batch: dict,
        vae: torch.nn.Module,
        weight_dtype,
        # --- prediction inputs ---
        model_pred: torch.Tensor,       # velocity pred [B, C, H, W], unpacked
        noisy_model_input: torch.Tensor,# noisy latent  [B, C, H, W]
        sigmas: torch.Tensor,           # sigmas used for this batch [B,1,1,1]
    ):
        """
        Build and save a grid with columns:
            condition images... | ground-truth | x0 prediction
        Each row is one sample from the batch (up to max_images).
        """
        n = min(self.max_images, batch["pixel_values"].shape[0])

        panels = []
        panel_labels = []

        # --- condition images (subject / fill / …) ---
        # batch["condition_latents"] holds raw pixels [B, num_conds, C, H, W]
        if "condition_latents" in batch:
            cond_imgs = batch["condition_latents"][:n]
            for ci in range(cond_imgs.shape[1]):
                c = cond_imgs[:, ci].float().clamp(-1., 1.)
                panels.append(c)
                cond_type = (
                    batch["condition_types"][0][ci]
                    if "condition_types" in batch
                    else f"cond{ci}"
                )
                panel_labels.append(cond_type)

        # --- ground-truth target (already pixels in [-1, 1]) ---
        gt_pixels = batch["pixel_values"][:n].float().clamp(-1., 1.)
        panels.append(gt_pixels)
        panel_labels.append("ground-truth")

        # --- model prediction: recover x0 from velocity, then decode ---
        # Flux uses flow-matching: noisy = (1-σ)*x0 + σ*noise
        # model predicts velocity v = noise - x0, so:
        #   x0_pred = noisy_model_input - σ * model_pred
        x0_pred_latent = noisy_model_input[:n] - sigmas[:n] * model_pred[:n]
        pred_pixels = decode_latents(x0_pred_latent.float(), vae)  # [-1, 1]
        panels.append(pred_pixels)
        panel_labels.append("prediction")

        # --- build grid: rows = samples, cols = panel types ---
        rows = []
        for i in range(n):
            for p in panels:
                img = p[i].cpu()                      # [C, H, W] in [-1, 1]
                img = (img + 1.0) / 2.0               # to [0, 1]
                rows.append(img)

        ncols = len(panels)
        grid = torchvision.utils.make_grid(rows, nrow=ncols, normalize=False, padding=4)
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        filename = f"gs-{global_step:06d}.png"
        save_path = os.path.join(self.image_dir, filename)
        Image.fromarray(grid_np).save(save_path)
        logger.info(
            f"[ImageLogger] Saved grid → {save_path}  "
            f"(cols per row: {panel_labels})"
        )

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="training script.")
    parser.add_argument( "--pretrained_model_name_or_path",type=str,default="ckpt/FLUX.1-schnell")
    parser.add_argument("--transformer",type=str,default="ckpt/FLUX.1-schnell",)
    parser.add_argument("--work_dir",type=str,default="output_remove/train_result_v3",)
    parser.add_argument("--output_denoising_lora",type=str,default="fill_remove_subject",)
    parser.add_argument("--pretrained_condition_lora_dir",type=str,default="ckpt/Condition_LoRA",)
    parser.add_argument("--training_adapter",type=str,default="ckpt/FLUX.1-schnell-training-adapter",)
    parser.add_argument("--checkpointing_steps",type=int,default=200,)
    parser.add_argument("--resume_from_checkpoint",type=str,default=None,)
    parser.add_argument("--rank",type=int,default=4,help="The dimension of the LoRA rank.")

    parser.add_argument("--dataset_name",type=str,default=[
            "./data/CE130_V2",
        ],
    )
    parser.add_argument("--condition_types",type=str,nargs='+',default=["subject","fill"],)

    parser.add_argument("--max_sequence_length",type=int,default=512,help="Maximum sequence length to use with with the T5 text encoder")
    parser.add_argument("--mixed_precision",type=str,default="bf16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--cache_dir",type=str,default="cache",)
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--resolution",type=int,default=512,)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=30000,)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=2)

    parser.add_argument("--learning_rate",type=float,default=1e-4)
    parser.add_argument("--scale_lr",action="store_true",default=False,)
    parser.add_argument("--lr_scheduler",type=str,default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500,)
    parser.add_argument("--weighting_scheme",type=str,default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument("--dataloader_num_workers",type=int,default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_xformers_memory_efficient_attention", default=False)
    parser.add_argument("--log_image_steps", type=int, default=200,
                        help="Save sample images every N global steps (0 to disable).")
    parser.add_argument("--log_max_images", type=int, default=4,
                        help="Max number of images to log per logging step.")


    args = parser.parse_args()
    args.revision = None
    args.variant = None
    args.work_dir = os.path.join(args.work_dir,datetime.now().strftime('%y_%m_%d-%H:%M'))
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    os.makedirs(os.path.join(args.work_dir, "ckpt"), exist_ok=True)
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",          # <-- add this
        project_dir=args.work_dir,       # <-- and this (tells it where to write)

    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.work_dir, exist_ok=True)

    image_logger = ImageLogger(
        work_dir=args.work_dir,
        log_every_n_steps=args.log_image_steps,
        max_images=args.log_max_images,
    ) if accelerator.is_main_process else None

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder"
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    text_encoder_one = text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two = text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    ).to(accelerator.device, dtype=weight_dtype)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)


    transformer = UniCombineTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant
    ).to(accelerator.device, dtype=weight_dtype)
    # load lora !!!!!
    lora_names = args.condition_types
    for condition_type in lora_names:
        transformer.load_lora_adapter(f"{args.pretrained_condition_lora_dir}/{condition_type}.safetensors", adapter_name=condition_type)

    transformer.load_lora_adapter(f"{args.training_adapter}/pytorch_lora_weights.safetensors", adapter_name="schnell_assistant")

    logger.info("All models loaded successfully")
    # freeze parameters of models to save more memory
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    logger.info("All models keeps requires_grad = False")

    single_transformer_blocks_lora = [
        f"single_transformer_blocks.{i}.proj_out"
        for i in range(len(transformer.single_transformer_blocks))
    ] + [
        f"single_transformer_blocks.{i}.proj_mlp"
        for i in range(len(transformer.single_transformer_blocks))
    ]

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=[
            "x_embedder",
            "norm1.linear",
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "ff.net.2",
            "norm.linear",
        ]+single_transformer_blocks_lora,
    )
    transformer.add_adapter(transformer_lora_config,adapter_name=args.output_denoising_lora)
    logger.info(f"Trainable lora: {args.output_denoising_lora} is loaded successfully")
    # hook
    accelerator.register_save_state_pre_hook(functools.partial(save_model_hook,wanted_model=transformer,accelerator=accelerator,adapter_names=[args.output_denoising_lora]))
    accelerator.register_load_state_pre_hook(functools.partial(load_model_hook,wanted_model=transformer,accelerator=accelerator,adapter_names=[args.output_denoising_lora]))
    logger.info("Hooks for save and load is ok.")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            # import xformers

            # xformers_version = version.parse(xformers.__version__)
            # if xformers_version == version.parse("0.0.16"):
            #     logger.warning(
            #         "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            #     )
            transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")


    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(transformer, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_lora_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    logger.info("Optimizer initialized successfully.")

    # Preprocessing the datasets.
    train_dataset = get_dataset(args)
    train_dataset = prepare_dataset(train_dataset, vae_scale_factor, accelerator, args)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    logger.info("Training dataset and Dataloader initialized successfully.")

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            text_ids = text_ids.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds, text_ids


    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )
    logger.info(f"lr_scheduler:{args.lr_scheduler} initialized successfully.")

    with preserve_requires_grad(transformer):
        transformer.set_adapters([i for i in lora_names] + [args.output_denoising_lora] + ['schnell_assistant'])
    logger.info(f"Set Adapters:{[i for i in lora_names] + [args.output_denoising_lora] + ['schnell_assistant']}")

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    def sanitize_config(cfg: dict) -> dict:
        """Keep only TensorBoard-safe scalar types."""
        safe = {}
        for k, v in cfg.items():
            if isinstance(v, (int, float, str, bool)):
                safe[k] = v
            elif isinstance(v, (list, tuple)):
                safe[k] = str(v)   # convert list → string representation
            elif v is None:
                safe[k] = "None"
            else:
                safe[k] = str(v)
        return safe
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("UniCombine", config=sanitize_config(vars(args)))


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.work_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.work_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                prompts = batch["descriptions"]
                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                    prompts, text_encoders, tokenizers
                )
                latent_image = encode_images(pixels=batch["pixel_values"],vae=vae,weight_dtype=weight_dtype)
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    latent_image.shape[0],
                    latent_image.shape[2] // 2,
                    latent_image.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                
                condition_latents = list(torch.unbind(batch["condition_latents"], dim=1))
                condition_ids = []
                condition_types = batch["condition_types"][0]
                for i,images_per_condition in enumerate(condition_latents):
                    images_per_condition = encode_images(pixels=images_per_condition,vae=vae,weight_dtype=weight_dtype)
                    cond_ids = FluxPipeline._prepare_latent_image_ids(
                        images_per_condition.shape[0],
                        images_per_condition.shape[2] // 2,
                        images_per_condition.shape[3] // 2,
                        accelerator.device,
                        weight_dtype,
                    )
                    if condition_types[i] == "subject":
                        cond_ids[:, 2] += images_per_condition.shape[2] // 2
                    condition_ids.append(cond_ids)
                    condition_latents[i] = images_per_condition

                noise = torch.randn_like(latent_image)
                bsz = latent_image.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)

                sigmas = get_sigmas(timesteps, n_dim=latent_image.ndim, dtype=latent_image.dtype)
                noisy_model_input = (1.0 - sigmas) * latent_image + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=latent_image.shape[0],
                    num_channels_latents=latent_image.shape[1],
                    height=latent_image.shape[2],
                    width=latent_image.shape[3],
                )
                for i, images_per_condition in enumerate(condition_latents):
                    condition_latents[i] = FluxPipeline._pack_latents(
                        images_per_condition,
                        batch_size=latent_image.shape[0],
                        num_channels_latents=latent_image.shape[1],
                        height=latent_image.shape[2],
                        width=latent_image.shape[3],
                    )

                if accelerator.unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(latent_image.shape[0])
                else:
                    guidance = None

            # stash for ImageLogger (populated inside accumulate block below)
            _log_model_pred = None
            _log_noisy_input = noisy_model_input
            _log_sigmas = sigmas
            # Bug fix: stash raw pixel condition images BEFORE they are encoded+packed below
            _log_condition_pixels = batch["condition_latents"].clone()
            with accelerator.accumulate(transformer):
                model_pred = transformer(
                    model_config={},
                    condition_latents=condition_latents,
                    condition_ids=condition_ids,
                    condition_type_ids=None,
                    condition_types = condition_types,
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[2] * vae_scale_factor,
                    width=noisy_model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                _log_model_pred = model_pred.detach()
                
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - latent_image

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)

                _grad_norm = 0.0
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    # Bug fix: capture grad_norm here (before step), not again in the log dict
                    _grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm).item()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Bug fix: log once per optimizer step (inside sync_gradients), using
                # the grad_norm captured before optimizer.step() above.
                logs = {
                    "train/loss": loss.detach().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                    "train/grad_norm": _grad_norm,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.work_dir, "ckpt", f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # --- ImageLogger: save decoded sample grid ---
                    if image_logger is not None and image_logger.should_log(global_step):
                        if _log_model_pred is not None:
                            # Bug fix: pass the raw pixel conditions (stashed before encoding)
                            log_batch = {**batch, "condition_latents": _log_condition_pixels}
                            image_logger.log(
                                global_step=global_step,
                                batch=log_batch,
                                vae=vae,
                                weight_dtype=weight_dtype,
                                model_pred=_log_model_pred,
                                noisy_model_input=_log_noisy_input,
                                sigmas=_log_sigmas,
                            )

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
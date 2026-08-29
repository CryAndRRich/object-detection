import torch
import torch.nn as nn
import torch.nn.functional as TF
from models.vision_encoder import SpatialVisualEncoder
from models.text_encoder import CLIPTextEncoder
from models.noise_pred_net import ConditionalUnet1D
from models.transformer_noise_net import TransformerNoisePredNet
from utils.matcher import hungarian_match, generalized_box_iou, _cxcywh_to_xyxy

class ObjectPlacementPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # --- CONFIG LOADING ---
        vis_dim = cfg['vision_encoder']['output_dim']
        text_dim = cfg['text_encoder']['output_dim']
        cond_dim = vis_dim + text_dim

        self.noise_net_type = cfg['noise_net'].get('type', 'cnn')

        # Variant (c) — sinh N box cùng lúc (horizon > 1) thay vì 1 box/lần.
        # num_proposals=1 (default) reproduces every existing config's behavior
        # exactly (horizon=1, no matching, plain epsilon-MSE loss below).
        self.num_proposals = cfg['noise_net'].get('num_proposals', 1)
        if self.noise_net_type == 'cnn' and not (
            self.num_proposals == 1 or (self.num_proposals >= 8 and self.num_proposals % 4 == 0)
        ):
            # ConditionalUnet1D halves the horizon axis twice (stride-2 Conv1d)
            # and rebuilds it with stride-2 ConvTranspose1d, concatenating a skip
            # at each level. Verified empirically over N=1..140 with the default
            # down_dims=[64,128,256]: exactly N == 1 and N >= 8 with N % 4 == 0
            # both run AND return N boxes. Everything else either raises a size
            # mismatch or — far worse — SILENTLY returns a different box count
            # (N=2 -> 1, N=7 -> 8, N=103 -> 104), which would misalign every
            # prediction against its target without any error.
            raise ValueError(
                f"noise_net.num_proposals={self.num_proposals} is not a valid horizon for the "
                f"CNN ConditionalUnet1D: it needs num_proposals == 1, or num_proposals >= 8 and "
                f"divisible by 4 (the box axis is halved twice and rebuilt through skip "
                f"connections). Note N=4 is NOT valid despite being divisible by 4."
            )

        # 1. Encoders
        self.vision_encoder = SpatialVisualEncoder(
            output_dim=vis_dim,
            model_name=cfg['vision_encoder'].get('model_name', 'resnet18'),
            pretrained=cfg['vision_encoder'].get('pretrained', True),
        )

        self.text_encoder = CLIPTextEncoder(
            model_name=cfg['text_encoder']['model_name'],
            output_dim=text_dim,
            freeze_backbone=cfg['text_encoder']['freeze']
        )

        # 2. Noise prediction network
        if self.noise_net_type == 'cnn':
            self.noise_net = ConditionalUnet1D(
                input_dim=cfg['noise_net']['input_dim'],
                global_cond_dim=cond_dim,
                down_dims=cfg['noise_net']['down_dims'],
                kernel_size=cfg['noise_net']['kernel_size'],
                n_groups=cfg['noise_net']['n_groups']
            )
        elif self.noise_net_type == 'transformer':
            t_cfg = cfg['noise_net'].get('transformer', {})
            self.noise_net = TransformerNoisePredNet(
                input_dim=cfg['noise_net']['input_dim'],
                cond_dim=cond_dim,
                horizon=self.num_proposals,
                n_layer=t_cfg.get('n_layer', 8),
                n_head=t_cfg.get('n_head', 8),
                n_emb=t_cfg.get('n_emb', 256),
                p_drop_emb=t_cfg.get('p_drop_emb', 0.1),
                p_drop_attn=t_cfg.get('p_drop_attn', 0.3),
                n_cond_layers=t_cfg.get('n_cond_layers', 0),
            )
        else:
            raise ValueError(f"Unknown noise_net.type={self.noise_net_type!r}; expected 'cnn' or 'transformer'")

        # 3. NOISE SCHEDULER SETUP (DDPM)
        # This was missing in the previous version!
        diff_cfg = cfg.get('diffusion', {})
        self.num_timesteps = diff_cfg.get('num_timesteps', 100)
        # These were read from the config file's own keys rather than hardcoded:
        # default.yaml has always carried beta_start/beta_end, but the original
        # setup_noise_schedule ignored them, so editing the yaml silently did nothing.
        self.beta_start = float(diff_cfg.get('beta_start', 0.0001))
        self.beta_end = float(diff_cfg.get('beta_end', 0.02))
        # Variant (c) only: DiffusionDet's SNR_SCALE — x_start is scaled up
        # before q_sample so the box coordinates dominate the fixed noise
        # schedule (detector.py: `x_start = (x_start * 2 - 1) * self.scale`,
        # default scale=2.0). Read for every variant but only ever USED by the
        # multi-box paths (compute_loss_multibox / sample_boxes_multibox), so
        # variants (a)/(b) are bit-for-bit unaffected by its value.
        self.snr_scale = float(diff_cfg.get('snr_scale', 2.0))
        self.box_loss_type = diff_cfg.get('loss', 'l1_giou')
        self.setup_noise_schedule()
        print(
            f"Diffusion noise schedule set up with {self.num_timesteps} timesteps "
            f"(beta {self.beta_start} -> {self.beta_end}, "
            f"alpha_bar[-1]={self.alphas_cumprod[-1]:.6f})."
        )

    def setup_noise_schedule(self):
        """
        Defines the linear beta schedule and pre-calculates alpha_bar.
        """
        beta_start = self.beta_start
        beta_end = self.beta_end

        # 1. Betas (Linear Schedule)
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps)
        
        # 2. Alphas = 1 - Betas
        alphas = 1.0 - betas
        
        # 3. Alpha Cumulative Product (Alpha Bar)
        # This represents how much "signal" remains at step t
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register as a buffer so it moves to GPU automatically with the model
        # but is NOT treated as a trainable parameter.
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def get_alpha_bar(self, t):
        """
        Retrieves alpha_bar for a batch of timesteps t.
        Returns shape [Batch, 1] for broadcasting.
        """
        # Gather values at indices t
        # alpha_cumprod is shape [100], t is shape [B]
        acc_alpha = self.alphas_cumprod[t] 
        
        # Reshape to [B, 1] so we can multiply with BBox [B, 4]
        return acc_alpha.unsqueeze(-1)

    def predict_noise(self, noisy_bbox, t, global_cond):
        """
        Unified interface over both noise_net architectures, so callers
        (inference.py, test_mul_box.py) don't need to know which one is active.

        noisy_bbox: [B, 4] (single box, horizon=1) or [B, N, 4] (variant (c),
                    horizon=N=self.num_proposals — all N boxes denoised in one
                    forward pass, sharing one condition token per image).
        t: [B] timestep (one per IMAGE, not per box — matches DiffusionDet's
           `prepare_diffusion_concat`, which draws a single t per image and
           applies it to every proposal in that image).
        global_cond: [B, cond_dim]
        returns: predicted noise, same shape as noisy_bbox
        """
        squeeze_back = noisy_bbox.dim() == 2
        sample = noisy_bbox.unsqueeze(1) if squeeze_back else noisy_bbox  # [B, N, 4]
        if self.noise_net_type == 'cnn':
            noise_pred = self.noise_net(sample=sample, timestep=t, global_cond=global_cond)
        else:  # 'transformer'
            cond = global_cond.unsqueeze(1)  # [B, 1, cond_dim], one token
            noise_pred = self.noise_net(sample=sample, timestep=t, cond=cond)
        return noise_pred.squeeze(1) if squeeze_back else noise_pred

    def compute_loss(self, rgb, density, text, gt_bbox):
        # gt_bbox shape: [Batch, 4]
        
        # 1. Encode Conditions
        vis_emb = self.vision_encoder(rgb, density) # [B, 128]
        text_emb = self.text_encoder(text)          # [B, 128]
        global_cond = torch.cat([vis_emb, text_emb], dim=-1) # [B, 256]
        
        # 2. Sample Noise and Timestep
        B = rgb.shape[0]
        # Random timestep for each item in batch
        t = torch.randint(0, self.num_timesteps, (B, ), device=rgb.device)
        noise = torch.randn_like(gt_bbox)
        
        # 3. Add Noise (Forward Diffusion)
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
        alpha_bar_t = self.get_alpha_bar(t)
        noisy_bbox = torch.sqrt(alpha_bar_t) * gt_bbox + torch.sqrt(1 - alpha_bar_t) * noise
        
        # 4. Predict Noise
        noise_pred = self.predict_noise(noisy_bbox, t, global_cond)

        # 5. Loss (MSE between predicted noise and actual noise)
        return torch.nn.functional.mse_loss(noise_pred, noise)

    def compute_loss_multibox(self, rgb, density, text, gt_boxes, box_mask):
        """
        Variant (c) loss — N boxes denoised jointly per image, matched against
        that image's real (non-padding) boxes with Hungarian assignment before
        the loss is computed, following DiffusionDet's prepare_diffusion_concat
        + SetCriterionDynamicK.loss_boxes (see utils/matcher.py for exactly
        which parts are cross-checked vs. reused as-is vs. deliberately not
        ported — no class head here, so plain 1-to-1 Hungarian, not SimOTA).

        gt_boxes: [B, N, 4] normalized [-1,1] cxcywh (N = self.num_proposals),
                  real boxes first per data/dataset.py's _build_multi_box_target
        box_mask: [B, N] bool, True for real (non-padding) boxes
        """
        B, N, _ = gt_boxes.shape
        device = rgb.device

        # 1. Encode Conditions (identical to compute_loss)
        vis_emb = self.vision_encoder(rgb, density)
        text_emb = self.text_encoder(text)
        global_cond = torch.cat([vis_emb, text_emb], dim=-1)

        # 2. Forward diffusion — one t per IMAGE (DiffusionDet: `t = randint(0,
        # T, (1,))` per image, applied to all its proposals), SNR-scaled.
        t = torch.randint(0, self.num_timesteps, (B,), device=device)
        x_start = gt_boxes * self.snr_scale
        noise = torch.randn_like(x_start)
        alpha_bar_t = self.get_alpha_bar(t).unsqueeze(-1)  # [B,1,1], broadcasts over [B,N,4]
        noisy_boxes = torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise

        # 3. Predict noise for all N boxes in one forward pass, then recover
        # the model's x0 estimate (needed for the L1/GIoU box loss and for
        # matching, since a random unmatched noise target has no geometric
        # meaning — DiffusionDet's own loss_boxes operates on x0, not epsilon).
        noise_pred = self.predict_noise(noisy_boxes, t, global_cond)
        pred_x_start = (noisy_boxes - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        # Clamp exactly as DiffusionDet does at detector.py:181. Without it the
        # x0 reconstruction is numerically explosive at high t: the 1/sqrt(a_bar)
        # factor is 157x at t=999 (a_bar=4e-05) and >10x on 33% of the schedule,
        # so an untrained/imperfect epsilon turns [-1,1] boxes into values in the
        # hundreds (measured: +-301 at t=999). That poisons both the Hungarian
        # cost matrix and the l1_giou loss, which assume box-scale coordinates.
        pred_x_start = torch.clamp(pred_x_start, min=-self.snr_scale, max=self.snr_scale)
        pred_boxes = pred_x_start / self.snr_scale  # back to [-1,1] cxcywh space

        # 4. Hungarian match per image (real boxes only — padding has no target).
        total_loss = pred_boxes.sum() * 0.0  # keeps autograd graph if a batch has 0 valid images
        n_valid_images = 0
        for b in range(B):
            n_gt = int(box_mask[b].sum().item())
            if n_gt == 0:
                continue
            gt_b = gt_boxes[b, box_mask[b]]       # [n_gt, 4]
            pred_b = pred_boxes[b]                # [N, 4] — match against ALL N predictions,
                                                   # not just the padding-aligned ones: the
                                                   # model doesn't see box_mask, so prediction
                                                   # slot i has no reason to line up with GT slot i.
            pred_idx, gt_idx = hungarian_match(pred_b.detach(), gt_b.detach())
            if pred_idx.numel() == 0:
                continue
            matched_pred = pred_b[pred_idx.to(device)]
            matched_gt = gt_b[gt_idx.to(device)]

            if self.box_loss_type == 'x0_mse':
                # MSE in x0 (box) space on the matched pairs — the closest
                # analogue of the single-box path's criterion that is still
                # coherent under set matching.
                #
                # NOTE: an epsilon-space MSE is deliberately NOT offered here.
                # `noise[b, k]` is the noise added to GT SLOT k, so after the
                # matcher reassigns prediction slot p to a different GT g, there
                # is no epsilon that both belongs to slot p and corresponds to
                # target g — supervising toward noise[b, p] would train the model
                # against a padding box's noise while claiming to supervise GT g.
                # DiffusionDet has the same structure and likewise never uses an
                # epsilon loss: SetCriterionDynamicK only has loss_boxes (L1 +
                # GIoU on x0) and loss_labels (loss.py:159-201).
                loss_b = TF.mse_loss(matched_pred, matched_gt)
            elif self.box_loss_type == 'l1_giou':
                loss_l1 = TF.l1_loss(matched_pred, matched_gt, reduction='mean')
                giou = torch.diag(generalized_box_iou(
                    _cxcywh_to_xyxy(matched_pred), _cxcywh_to_xyxy(matched_gt)
                ))
                loss_giou = (1.0 - giou).mean()
                loss_b = loss_l1 + loss_giou
            else:
                raise ValueError(f"Unknown diffusion.loss={self.box_loss_type!r}; "
                                  f"expected 'l1_giou' or 'x0_mse'")

            total_loss = total_loss + loss_b
            n_valid_images += 1

        if n_valid_images == 0:
            return total_loss  # zero, but keeps the graph so .backward() doesn't error
        return total_loss / n_valid_images
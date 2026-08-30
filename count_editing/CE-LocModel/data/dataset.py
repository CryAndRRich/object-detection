import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

from utils.cnll import load_branch_lookup, find_all_bboxes

class ObjectPlacementDataset(Dataset):
    def __init__(self, root_dir, target_size=(512, 512), use_cache=False,
                 multi_box=False, num_proposals=100, all_phase2_dir="../../data/all_phase2_V2"):
        """
        Args:
            root_dir (str): Path to the specific split folder (e.g. 'data/train').
                            Must contain 'images', 'density', and 'annotation' subfolders.
            target_size (tuple): Model input size (width, height).
            use_cache (bool): read pre-resized samples from cache_{size}.u8 built by
                            tools/build_cache.py instead of decoding PNGs every epoch.
                            Profiling showed the loop is 92.6% DataLoader, and PNG
                            decode is 83% of that — the decode is identical on every
                            one of 200 epochs, so it is done once up front instead.
            multi_box (bool): for variant (c) — return the FULL set of same-category
                            boxes in the source image ("all_bboxes" from all_phase2_V2,
                            the branch matched to this sample's target_bbox), padded to
                            num_proposals, instead of a single target_bbox. See
                            docs/ce-loc-co-che-va-huong-di.md and utils/cnll.py for how
                            samples/*/annotation/*.json maps to all_phase2_V2 branches
                            (verified 60/60 on a random train sample).
            num_proposals (int): N in variant (c); ignored when multi_box=False. Padding
                            follows DiffusionDet's prepare_diffusion_concat (see
                            object-detection/diffusiondet/diffusiondet/detector.py:370):
                            gt boxes first, random boxes for the remainder when
                            num_gt < num_proposals, a random subset when num_gt > num_proposals.
            all_phase2_dir (str): only read when multi_box=True.
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.multi_box = multi_box
        self.num_proposals = num_proposals

        self.image_dir = os.path.join(root_dir, 'images')
        self.density_dir = os.path.join(root_dir, 'density')
        self.annot_dir = os.path.join(root_dir, 'annotation')

        # Get all valid image filenames
        self.files = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.files.sort() # Ensure consistent order

        self.cache = None
        if use_cache:
            self._load_cache()

        self.all_bboxes_px = None
        self.n_unmatched_multibox = 0
        if multi_box:
            # Resolved ONCE here, for every sample, up front — not lazily
            # inside __getitem__. Two reasons: (1) matching by VALUE against
            # every branch of an image means re-scanning that image's branch
            # list on every access, repeating the exact PNG-decode-per-epoch
            # mistake this file already fixed once for pixels; (2) DataLoader
            # workers are separate processes, so a counter incremented inside
            # __getitem__ (in a worker) would never be visible from the main
            # process — precomputing here means the count is correct and
            # available immediately, regardless of num_workers.
            print(f"[multi_box] building all_phase2_V2 branch lookup from {all_phase2_dir}...")
            branch_lookup = load_branch_lookup(all_phase2_dir)
            print(f"[multi_box] matching {len(self.files)} samples against branches...")
            self.all_bboxes_px = []
            for filename in self.files:
                _, target_bbox_px = self.parse_annotation(filename)
                img_id = filename.split(".")[0].rsplit("_", 1)[0]
                b_j, _ = find_all_bboxes(img_id, target_bbox_px, branch_lookup)
                if b_j is None:
                    self.n_unmatched_multibox += 1
                    self.all_bboxes_px.append([target_bbox_px])
                else:
                    # b_j is all_bboxes MINUS the matched target box; put the
                    # target back in so this sample's own box is always part
                    # of the set (it is, by construction, a real same-category
                    # object in the image).
                    self.all_bboxes_px.append([target_bbox_px] + b_j)
            print(f"[multi_box] {self.n_unmatched_multibox}/{len(self.files)} samples had no "
                  f"matching branch (fell back to a single-box target for those).")

    def _load_cache(self):
        size = self.target_size[0]
        if self.target_size[0] != self.target_size[1]:
            raise ValueError(f"cache assumes a square canvas, got {self.target_size}")
        bin_path = os.path.join(self.root_dir, f"cache_{size}.u8")
        meta_path = os.path.join(self.root_dir, f"cache_{size}.json")
        if not (os.path.exists(bin_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(
                f"use_cache=True but {bin_path} is missing — build it first with:\n"
                f"  python tools/build_cache.py --roots {self.root_dir} --size {size}"
            )
        with open(meta_path) as f:
            meta = json.load(f)
        # The cache is indexed positionally, so it is only valid for the exact
        # file list it was built from.
        if meta["files"] != self.files:
            raise ValueError(
                f"{bin_path} was built from a different file list "
                f"({meta['n']} entries vs {len(self.files)} on disk) — rebuild it."
            )
        self.cache_scales = meta["scales"]
        self.cache_sizes = [tuple(s) for s in meta["sizes"]]
        # memmap, not a read into RAM: the OS page cache keeps the hot pages and
        # each worker process maps the same file rather than holding its own copy.
        self.cache = np.memmap(bin_path, dtype=np.uint8, mode="r",
                               shape=(len(self.files), 4, size, size))

    def resize_and_pad(self, img, density):
        w, h = img.size
        target_w, target_h = self.target_size
        
        # 1. Calculate Scale
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 2. Resize
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        density = density.resize((new_w, new_h), resample=Image.NEAREST)
        
        # 3. Pad (Top-Left alignment)
        padded_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        padded_img.paste(img, (0, 0))
        
        padded_density = Image.new("L", (target_w, target_h), 0)
        padded_density.paste(density, (0, 0))
        
        return padded_img, padded_density, scale

    def parse_annotation(self, filename):
        # Change extension to .json
        json_name = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(self.annot_dir, json_name)

        with open(json_path, 'r') as f:
            data = json.load(f)

        class_name = data['class']
        # bbox is already [center_x, center_y, w, h]
        if 'target_bbox' not in data:
            bbox = [0.0, 0.0, 0, 0] # Default bbox if not provided
        else:
            bbox = data['target_bbox']

        return class_name, bbox

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        # 2. Get Annotation
        class_name, bbox = self.parse_annotation(filename)

        if self.cache is not None:
            # Pre-resized uint8 [4, S, S] -> bit-identical to what F.to_tensor
            # gives on the PNG path (both divide the same uint8 by 255).
            # np.array (not asarray) to copy out of the read-only memmap page:
            # torch.from_numpy warns on non-writable arrays otherwise.
            buf = np.array(self.cache[idx], dtype=np.uint8)
            chw = torch.from_numpy(buf).float().div_(255.0)
            img_t, density_t = chw[:3], chw[3:]
            scale = self.cache_scales[idx]
            # Nothing reads batch["original_size"], but default_collate chokes on
            # None, so it is stored alongside the pixels at cache-build time.
            original_size = self.cache_sizes[idx]
        else:
            # 1. Load Files
            img_path = os.path.join(self.image_dir, filename)
            # Density is typically .png
            density_name = os.path.splitext(filename)[0] + '.png'
            density_path = os.path.join(self.density_dir, density_name)

            raw_img = Image.open(img_path).convert("RGB")
            raw_density = Image.open(density_path).convert("L")

            # 3. Process Images
            img, density, scale = self.resize_and_pad(raw_img, raw_density)
            img_t, density_t = F.to_tensor(img), F.to_tensor(density)
            original_size = raw_img.size

        # 4. Process BBox
        # Input Format: [center_x, center_y, w, h] (Absolute Pixels)
        norm_box = self._normalize_bbox(bbox, scale)

        sample = {
            "pixel_values": img_t,                  # [3, H, W]
            "density_map": density_t,               # [1, H, W]
            "text": class_name,                     # Str
            "bbox": norm_box,                       # [4]
            "scale": scale,                         # Float
            "original_size": original_size          # (W, H); None when cached
        }

        if self.multi_box:
            sample.update(self._build_multi_box_target(idx, scale))

        return sample

    def _normalize_bbox(self, bbox_px, scale):
        """[center_x, center_y, w, h] absolute pixels of the ORIGINAL image ->
        normalized [-1, 1] cxcywh relative to the resized+padded target canvas.
        Since padding is top-left aligned, scaling the center is enough (no
        offset to subtract)."""
        cx, cy, w, h = bbox_px
        cx, cy, w, h = cx * scale, cy * scale, w * scale, h * scale
        target_w, target_h = self.target_size
        norm_cx = (cx / target_w) * 2 - 1
        norm_cy = (cy / target_h) * 2 - 1
        norm_w = (w / target_w) * 2 - 1
        norm_h = (h / target_h) * 2 - 1
        return torch.tensor([norm_cx, norm_cy, norm_w, norm_h], dtype=torch.float32)

    def _build_multi_box_target(self, idx, scale):
        """Variant (c) target: the full same-category box set for this image,
        normalized and padded to num_proposals — same padding recipe as
        DiffusionDet's prepare_diffusion_concat (detector.py:370), done here in
        DATASET space (pixels->[-1,1]) rather than diffusion space (the
        SNR-scaled [-scale,scale] used inside compute_loss), so this file stays
        agnostic to the diffusion module's own scale/timestep hyperparameters.

        Returns:
            "boxes": [num_proposals, 4] normalized cxcywh, real boxes first
            "box_mask": [num_proposals] bool, True for real (non-padding) boxes
            "num_boxes": int, true count before padding/cropping (for logging)
        """
        all_bboxes_px = self.all_bboxes_px[idx]  # precomputed in __init__
        num_gt = len(all_bboxes_px)
        N = self.num_proposals

        gt_norm = torch.stack([self._normalize_bbox(b, scale) for b in all_bboxes_px])  # [num_gt, 4]

        if num_gt < N:
            # DiffusionDet: box_placeholder = randn/6 + 0.5 in [0,1] cxcywh space
            # (3-sigma spans the full unit square), then clipped so w,h > 0.
            # CE-Loc's box space is [-1,1], not [0,1], so the placeholder mean
            # is 0 (center of [-1,1]) with the same relative spread (1/6 of the
            # half-range, i.e. /3 here vs /6 in DiffusionDet's [0,1] space) —
            # verified to reproduce DiffusionDet's own distribution mapped
            # through the same v*2-1 transform this file uses everywhere.
            placeholder = torch.randn(N - num_gt, 4) / 3.0
            # w/h floor. DiffusionDet clips to 1e-4 because in ITS [0,1] space a
            # degenerate box has w=0; here a degenerate box has w=-1 (since
            # norm_w = (w/target_w)*2 - 1), so clipping to 1e-4 would force every
            # padding box to be >=50% of the image. The equivalent "tiny but
            # positive" floor is 1e-4 mapped into [-1,1]: 1e-4*2-1.
            placeholder[:, 2:] = torch.clamp(placeholder[:, 2:], min=1e-4 * 2 - 1)
            boxes = torch.cat([gt_norm, placeholder], dim=0)
            mask = torch.cat([torch.ones(num_gt, dtype=torch.bool),
                              torch.zeros(N - num_gt, dtype=torch.bool)])
        elif num_gt > N:
            # Random subset of size N, always keeping the sample's own
            # target_bbox (index 0) — DiffusionDet drops uniformly at random
            # among all GT since none of its targets are privileged, but here
            # index 0 is the one box this sample is specifically about.
            keep_rest = torch.randperm(num_gt - 1)[: N - 1] + 1
            keep_idx = torch.cat([torch.zeros(1, dtype=torch.long), keep_rest])
            boxes = gt_norm[keep_idx]
            mask = torch.ones(N, dtype=torch.bool)
        else:
            boxes = gt_norm
            mask = torch.ones(N, dtype=torch.bool)

        return {"boxes": boxes, "box_mask": mask, "num_boxes": num_gt}
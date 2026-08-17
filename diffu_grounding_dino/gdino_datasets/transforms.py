"""Detection transforms that keep boxes in sync with the image.

Boxes arrive as absolute ``xyxy`` and stay that way through every geometric
transform; ``Normalize`` is the last step and converts them to the normalized
``cxcywh`` the model and loss expect. Anything that inserts a geometric transform
*after* ``Normalize`` will silently corrupt the targets.

Cropping drops boxes that end up with zero area, and drops their labels with them
-- a box list and a label list that disagree in length is the most common way this
kind of pipeline breaks.
"""

import random
from typing import List, Optional, Sequence, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh

# Fields that must be filtered together with ``boxes``.
PER_BOX_FIELDS = ("labels", "area", "iscrowd")


def crop(image, target, region: Tuple[int, int, int, int]):
    """Crop to ``region = (top, left, height, width)`` and fix up the targets."""
    cropped_image = F.crop(image, *region)
    target = target.copy()
    top, left, height, width = region
    target["size"] = torch.tensor([height, width])

    fields = [f for f in PER_BOX_FIELDS if f in target]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([width, height], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([left, top, left, top], dtype=boxes.dtype)
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size).clamp(min=0)
        target["area"] = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        if "boxes" not in fields:
            fields.append("boxes")
        if "area" not in fields:
            fields.append("area")

        keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped = F.hflip(image)
    width = image.size[0] if hasattr(image, "size") else image.shape[-1]

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        # Mirror and swap x1/x2 so the box stays well-ordered.
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([width, 0, width, 0])
        target["boxes"] = boxes
    return flipped, target


def _resized_size(image_size, size, max_size=None) -> Tuple[int, int]:
    """Target ``(h, w)`` for "resize the short side to ``size``"."""
    w, h = image_size
    if isinstance(size, (list, tuple)):
        return size[::-1]

    if max_size is not None:
        min_original, max_original = float(min(w, h)), float(max(w, h))
        if max_original / min_original * size > max_size:
            size = int(round(max_size * min_original / max_original))

    if (w <= h and w == size) or (h <= w and h == size):
        return h, w
    if w < h:
        return int(size * h / w), size
    return size, int(size * w / h)


def resize(image, target, size, max_size=None):
    new_h, new_w = _resized_size(image.size, size, max_size)
    rescaled = F.resize(image, (new_h, new_w))

    if target is None:
        return rescaled, None

    ratio_w = rescaled.size[0] / image.size[0]
    ratio_h = rescaled.size[1] / image.size[1]

    target = target.copy()
    if "boxes" in target:
        target["boxes"] = target["boxes"] * torch.as_tensor([ratio_w, ratio_h, ratio_w, ratio_h])
    if "area" in target:
        target["area"] = target["area"] * (ratio_w * ratio_h)
    target["size"] = torch.tensor([new_h, new_w])
    return rescaled, target


def pad(image, target, padding: Tuple[int, int]):
    """Pad right/bottom. Boxes are unchanged; only the recorded size grows."""
    padded = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded, None
    target = target.copy()
    target["size"] = torch.tensor(padded.size[::-1])
    return padded, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return crop(img, target, T.RandomCrop.get_params(img, self.size))


class RandomSizeCrop:
    """Crop a random window whose sides lie in ``[min_size, max_size]``."""

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, target):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        return crop(img, target, T.RandomCrop.get_params(img, [h, w]))


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        top = int(round((image_height - crop_height) / 2.0))
        left = int(round((image_width - crop_width) / 2.0))
        return crop(img, target, (top, left, crop_height, crop_width))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize:
    def __init__(self, sizes: Sequence, max_size: Optional[int] = None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        return resize(img, target, random.choice(self.sizes), self.max_size)


class ResizeDebug:
    """Fixed-size resize, for FLOP counting where shapes must be constant."""

    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        return resize(img, target, self.size)


class RandomPad:
    def __init__(self, max_pad: int):
        self.max_pad = max_pad

    def __call__(self, img, target):
        return pad(img, target, (random.randint(0, self.max_pad), random.randint(0, self.max_pad)))


class RandomSelect:
    """Apply one of two transforms, with probability ``p`` for the first."""

    def __init__(self, transforms1, transforms2, p: float = 0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        chosen = self.transforms1 if random.random() < self.p else self.transforms2
        return chosen(img, target)


class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing:
    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize:
    """Normalize pixels and convert boxes to normalized ``cxcywh``.

    Must be the last geometric-sensitive step in a pipeline.
    """

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None

        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = box_xyxy_to_cxcywh(target["boxes"])
            target["boxes"] = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

    def __repr__(self):
        body = "\n".join(f"    {t}" for t in self.transforms)
        return f"{type(self).__name__}(\n{body}\n)"


__all__ = [
    "CenterCrop",
    "Compose",
    "Normalize",
    "RandomCrop",
    "RandomErasing",
    "RandomHorizontalFlip",
    "RandomPad",
    "RandomResize",
    "RandomSelect",
    "RandomSizeCrop",
    "ResizeDebug",
    "ToTensor",
    "crop",
    "hflip",
    "pad",
    "resize",
]

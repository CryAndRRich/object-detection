import os
import json 
import random
from accelerate.logging import get_logger
import torch
logger = get_logger(__name__)
from PIL import Image
from .condition import Condition
from diffusers.image_processor import VaeImageProcessor
from datasets import Dataset

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def get_dataset(args):
    root_dir = args.dataset_name[0] if isinstance(args.dataset_name, list) else args.dataset_name

    image_dir = os.path.join(root_dir, 'inpaint/train')
    inpaint_dir = os.path.join(root_dir, 'images/train')
    box_dir = os.path.join(root_dir, 'box_json/train')

    filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    data_list = []
    for name in filenames:
        image_path = os.path.join(image_dir, f"{name}.png")
        inpaint_path = os.path.join(inpaint_dir, f"{name}.png")
        box_path = os.path.join(box_dir, f"{name}.json")

        if os.path.exists(inpaint_path) and os.path.exists(box_path):
            data_list.append({
                "image_path": image_path,
                "inpaint_path": inpaint_path,
                "box_path": box_path
            })
        else:
            logger.warning(f"Missing inpaint or box file for {name}. Skipping.")

    dataset = Dataset.from_list(data_list)
    logger.info(f"Loaded {len(dataset)} valid samples from local dataset.")
    return dataset


def crop_and_adjust_bbox(input_image, gt_image, x1, y1, x2, y2, resolution):
    """Smart square crop căn giữa bbox, trả về ảnh đã crop và bbox đã scale."""
    orig_width, orig_height = gt_image.size
    crop_size = int(min(resolution, orig_width, orig_height))

    # Tính crop coords căn giữa bbox
    crop_x1 = int((x1 + x2) / 2 - crop_size / 2)
    crop_y1 = int((y1 + y2) / 2 - crop_size / 2)
    crop_x1 = max(0, min(crop_x1, orig_width - crop_size))
    crop_y1 = max(0, min(crop_y1, orig_height - crop_size))
    crop_x2, crop_y2 = crop_x1 + crop_size, crop_y1 + crop_size

    input_image = input_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    gt_image = gt_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # Dịch chuyển và clip bbox về không gian ảnh mới
    new_x1 = max(0, min(crop_size, x1 - crop_x1))
    new_y1 = max(0, min(crop_size, y1 - crop_y1))
    new_x2 = max(0, min(crop_size, x2 - crop_x1))
    new_y2 = max(0, min(crop_size, y2 - crop_y1))

    # Scale bbox lên resolution của model
    scale = resolution / crop_size
    scaled_bbox = [int(new_x1 * scale), int(new_y1 * scale), int(new_x2 * scale), int(new_y2 * scale)]

    return input_image, gt_image, (new_x1, new_y1, new_x2, new_y2), scaled_bbox



def preprocess_conditions(conditions, image_processor, resolution):
    """Chuyển danh sách Condition thành tensor stack và danh sách condition_type."""
    tensors, types = [], []
    for cond in conditions:
        tensors.append(image_processor.preprocess(cond.condition, width=resolution, height=resolution).squeeze(0))
        types.append(cond.condition_type)
    return torch.stack(tensors, dim=0), types


def preprocess(examples, image_processor, args):
    pixel_values, condition_latents, condition_types, bboxes, descriptions = [], [], [], [], []

    for img_p, inpaint_p, box_p in zip(examples["image_path"], examples["inpaint_path"], examples["box_path"]):
        # Load ảnh và metadata
        input_image = Image.open(img_p).convert("RGB")
        gt_image = Image.open(inpaint_p).convert("RGB")
        with open(box_p, 'r', encoding='utf-8') as f:
            box_data = json.load(f)
        x1, y1, x2, y2 = box_data["loc_box"]
        class_name = box_data["class_name"]

        # Crop và điều chỉnh bbox
        input_image, gt_image, local_bbox, scaled_bbox = crop_and_adjust_bbox(
            input_image, gt_image, x1, y1, x2, y2, args.resolution
        )
        input_image.paste((0,0,0), local_bbox)
        # Ground truth tensor
        pixel_values.append(image_processor.preprocess(gt_image, width=args.resolution, height=args.resolution).squeeze(0))

        subject_image = Image.new("RGB", (512, 512), (127, 127, 127))

        # Conditions
        conditions = []
        for condition_type in args.condition_types:
            if condition_type == "subject":
                conditions.append(Condition("subject", condition=subject_image))
            elif condition_type == "fill":
                conditions.append(Condition("fill", condition=input_image))
            else:
                raise ValueError(f"Only support subject and fill for local folders currently. Got {condition_type}")

        cond_tensors, cond_types = preprocess_conditions(conditions, image_processor, args.resolution)
        condition_latents.append(cond_tensors)
        condition_types.append(cond_types)
        bboxes.append(scaled_bbox)
        descriptions.append({
            "description_0": f"remove a single instance of {class_name} from the scene, preserving the position, appearance, and count of all remaining {class_name}s, do not introduce any additional {class_name}, and inpaint the region to blend naturally with nearby content",
            "item": "empty background"
        })

    examples["pixel_values"] = pixel_values
    examples["condition_latents"] = condition_latents
    examples["condition_types"] = condition_types
    examples["bbox"] = bboxes
    examples["description"] = descriptions
    return examples


def prepare_dataset(dataset, vae_scale_factor, accelerator, args):
    image_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor * 2,
        do_resize=True,
        do_convert_rgb=True
    )

    with accelerator.main_process_first():
        dataset = dataset.with_transform(lambda examples: preprocess(examples, image_processor, args))

    return dataset

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    condition_latents = torch.stack([example["condition_latents"] for example in examples])
    condition_latents = condition_latents.to(memory_format=torch.contiguous_format).float()
    bboxes= [example["bbox"] for example in examples]
    condition_types= [example["condition_types"] for example in examples]
    descriptions = [example["description"]["description_0"] for example in examples]
    items = [example["description"]["item"] for example in examples]
    return {"pixel_values": pixel_values, "condition_latents": condition_latents,
            "condition_types":condition_types,"descriptions": descriptions, "bboxes": bboxes,"items":items}


# =====================================================================
# CÁC HÀM VISUALIZE DÀNH CHO SERVER
# =====================================================================

def unnormalize_tensor_to_pil(tensor_data):
    """Hàm phụ trợ chuyển tensor ảnh [-1, 1] về PIL Image [0, 255]"""
    if not isinstance(tensor_data, torch.Tensor):
        tensor_data = torch.tensor(tensor_data)
        
    img = tensor_data.clone().detach().cpu()
    img = (img / 2 + 0.5).clamp(0, 1) 
    img = img.permute(1, 2, 0).numpy() 
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def visualize_input(dataset, index=0, save_dir="debug_outputs"):
    """
    Trực quan hóa một sample từ dataset và lưu ra file thay vì hiển thị trên UI.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    sample = dataset[index]
    
    # 1. Trích xuất Ground Truth
    gt_img = unnormalize_tensor_to_pil(sample["pixel_values"])
    
    # 2. Trích xuất Bbox và Description
    bbox = sample["bbox"] 
    desc = sample["description"]["description_0"]
    
    # 3. Trích xuất Conditions
    cond_tensors = sample["condition_latents"]
    cond_types = sample["condition_types"]
    
    num_plots = 2 + len(cond_types)
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 5))
    
    # Plot 1: Ground Truth
    axes[0].imshow(gt_img)
    axes[0].set_title(f"Ground Truth\nDesc: {desc}")
    axes[0].axis("off")
    
    # Plot 2: GT + Bounding Box
    axes[1].imshow(gt_img)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], 
                             linewidth=2, edgecolor='red', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].set_title("GT + Scaled BBox (Red)")
    axes[1].axis("off")
    
    # Plot 3+: Các Conditions
    for i, (cond_tensor, cond_type) in enumerate(zip(cond_tensors, cond_types)):
        cond_img = unnormalize_tensor_to_pil(cond_tensor)
        ax = axes[2 + i]
        ax.imshow(cond_img)
        ax.set_title(f"Condition: {cond_type.upper()}")
        ax.axis("off")
        
        if cond_type == "fill":
            rect_fill = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], 
                                          linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
            ax.add_patch(rect_fill)
            ax.set_title(f"Condition: {cond_type.upper()}\n+ BBox (Lime)")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"sample_debug_{index}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig) 
    
    print(f"[Debug] Đã lưu ảnh trực quan hóa tại: {save_path}")


# =====================================================================
# MAIN THỰC THI (TESTING SCRIPT)
# =====================================================================
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # 1. Khởi tạo Args giả lập để test
    class DummyArgs:
        def __init__(self):
            self.dataset_name = "path/to/CEGEN_train" 
            self.resolution = 512
            self.condition_types = ["subject", "fill"]
            self.pg = 0.5
            self.pa = 0.5

    args = DummyArgs()

    # 2. Khởi tạo Accelerator giả lập
    class DummyContextManager:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_value, traceback): pass

    class DummyAccelerator:
        def main_process_first(self):
            return DummyContextManager()

    accelerator = DummyAccelerator()
    vae_scale_factor = 8 # Thông số tiêu chuẩn cho Stable Diffusion VAE

    print(f"Đang tiến hành load dataset từ: {args.dataset_name}...")
    dataset = get_dataset(args)

    if len(dataset) > 0:
        print("Đang xử lý dataset qua pipeline preprocess...")
        processed_dataset = prepare_dataset(dataset, vae_scale_factor, accelerator, args)
        
        # 3. Chạy vòng lặp visualize một số ảnh (ví dụ 5 ảnh đầu tiên)
        num_samples_to_check = min(20, len(processed_dataset))
        print(f"Bắt đầu xuất {num_samples_to_check} ảnh debug...")
        
        for i in range(num_samples_to_check):
            visualize_input(processed_dataset, index=i, save_dir="debug_outputs_rm")
            
        print("Hoàn tất! Vui lòng kiểm tra thư mục 'debug_outputs_rm'.")
    else:
        print("Không tìm thấy dữ liệu hợp lệ. Hãy kiểm tra lại đường dẫn dataset và cấu trúc thư mục (images, inpaint, box_json).")
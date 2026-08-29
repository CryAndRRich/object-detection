import os,sys
from tqdm import tqdm
import ipdb
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
import torch
from src.condition import Condition
from PIL import Image, ImageDraw
from src.UniCombineTransformer2DModel import UniCombineTransformer2DModel
from src.UniCombinePipeline import UniCombinePipeline
from accelerate.utils import set_seed
import json
import argparse
import cv2
import numpy as np
from datetime import datetime
weight_dtype = torch.bfloat16
device = torch.device("cuda:0")

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="inference script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,default="ckpt/FLUX.1-schnell",)
    parser.add_argument("--transformer",type=str,default="ckpt/FLUX.1-schnell/transformer",)
    parser.add_argument("--condition_types", type=str, nargs='+', default=["fill","subject"],)
    parser.add_argument("--condition_lora_dir",type=str,default="./ckpt/CE-Gen_ckpt/ConditionRemove",)
    parser.add_argument("--denoising_lora_dir",type=str,default="./ckpt/CE-Gen_ckpt/DenoisingRemove",)
    parser.add_argument("--denoising_lora_name",type=str,default="subject_fill_remove_union",)
    parser.add_argument("--denoising_lora_weight",type=float,default=1.0,)
    parser.add_argument("--work_dir",type=str,default="./output_remove",)
    parser.add_argument("--test_dir",type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution",type=int,default=512,)
    parser.add_argument("--exam_size",type=int,default=224,)
    args = parser.parse_args()
    args.revision = None
    args.variant = None
    
    return args

def convert_image(image):
    return image if isinstance(image, Image.Image) else Image.open(image)

def get_inputs(test_dir):
    print("get inputs")
    inputs = []
    img_test_dir = os.path.join(test_dir, "img")
    for img in os.listdir(img_test_dir):
        img_folder = img.split('.')[0]
        # get img_path
        img_path = os.path.join(test_dir, "img", img)

        # get prompt, bbox
        anno_path = os.path.join(test_dir, 'anno', img.replace('png', 'json'))
        with open(anno_path, 'r') as f:
            anno = json.load(f)
        
        ## get location bbox
        loc_bbox = anno["inpainted_bboxes"][1]
        ## get prompt
        class_name = anno["class_based_caption"]
        prompt = f"remove a single instance of {class_name} from the scene, preserving the position, appearance, and count of all remaining {class_name}s, do not introduce any additional {class_name}, and inpaint the region to blend naturally with nearby content"
    
        inputs.append([
            img_folder,
            prompt,
            img_path,
            loc_bbox,
        ])

    return inputs

    
def get_background(bg_path, loc_bbox):
    """
    Crop vùng xung quanh loc_bbox với kích thước vuông max là 256, 
    sau đó resize ảnh crop lên 512x512 và cập nhật lại loc_bbox tương ứng.
    """
    img = Image.open(bg_path).convert("RGB")
    img_w, img_h = img.size
    
    x1, y1, x2, y2 = loc_bbox

    # Kích thước hình vuông cần crop
    crop_size = min(512, min(img_w, img_h))

    # Lấy tâm của loc_bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Tính toán tọa độ crop ban đầu (đăt bbox vào giữa)
    crop_x1 = int(cx - crop_size / 2)
    crop_y1 = int(cy - crop_size / 2)
    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size

    # Dịch chuyển crop box nếu bị tràn ra ngoài biên ảnh
    # Giữ nguyên kích thước crop_size
    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0

    if crop_x2 > img_w:
        crop_x1 -= (crop_x2 - img_w)  # Dịch sang trái
        crop_x2 = img_w
    if crop_y2 > img_h:
        crop_y1 -= (crop_y2 - img_h)  # Dịch lên trên
        crop_y2 = img_h

    # Ép kiểu và clamp cẩn thận lại lần cuối
    crop_x1 = max(0, int(crop_x1))
    crop_y1 = max(0, int(crop_y1))
    crop_x2 = min(img_w, int(crop_x2))
    crop_y2 = min(img_h, int(crop_y2))

    crop_box = [crop_x1, crop_y1, crop_x2, crop_y2]
    
    # Cắt ảnh background
    cropped_img = img.crop(crop_box)

    # Tính kích thước thực tế của vùng crop
    actual_crop_w = crop_x2 - crop_x1
    actual_crop_h = crop_y2 - crop_y1

    # Tính toán tỉ lệ scale lên 512
    # Tính riêng cho cả x và y để phòng hờ trường hợp ảnh gốc quá nhỏ khiến bbox bị clamp lệch
    scale_x = 512.0 / actual_crop_w
    scale_y = 512.0 / actual_crop_h

    # Resize ảnh crop lên 512x512
    cropped_img = cropped_img.resize((512, 512), Image.Resampling.LANCZOS)

    # Cập nhật tọa độ loc_bbox mới: Tọa độ tương đối so với ảnh đã crop x Tỉ lệ scale
    new_loc_bbox = [
        int((x1 - crop_x1) * scale_x),
        int((y1 - crop_y1) * scale_y),
        int((x2 - crop_x1) * scale_x),
        int((y2 - crop_y1) * scale_y)
    ]

    return img, cropped_img, new_loc_bbox, crop_box

from PIL import Image

def get_conditions(img_path, loc_bbox):
    conditions = []
    
    # subject
    subject= Image.new("RGB", (512, 512), (127, 127, 127))        
    conditions.append(Condition("subject", raw_img=convert_image(subject))) 
    # fill
    original_img, fill, new_loc_bbox, crop_box = get_background(img_path, loc_bbox)
    fill.paste((0,0,0), new_loc_bbox)
    conditions.append(Condition("fill", raw_img=convert_image(fill)))

    # mask (for calculate metric)
    mask = create_mask_from_bbox(original_img.size, loc_bbox)
    return conditions, new_loc_bbox, mask, original_img, crop_box

def create_mask_from_bbox(image_size, bbox):
    """
    Create mask image from bbox (xyxy)
    
    Args:
        image_size: (width, height)
        bbox: (x1, y1, x2, y2)
    
    Returns:
        PIL Image mask (mode 'L')
    """
    width, height = image_size
    x1, y1, x2, y2 = bbox

    # Create black mask
    mask = Image.new('L', (width, height), 0)

    # Draw white rectangle for bbox
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=255)

    return mask

def inference(args):
    if args.seed is not None:
        set_seed(args.seed)
    
    # model
    transformer = UniCombineTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path=args.transformer,
    ).to(device = device, dtype=weight_dtype)

    for condition_type in args.condition_types:
        transformer.load_lora_adapter(f"{args.condition_lora_dir}/{condition_type}.safetensors", adapter_name=condition_type)

    pipe = UniCombinePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype = weight_dtype,
        transformer = None
    )
    pipe.transformer = transformer

    pipe.transformer.load_lora_adapter(os.path.join(args.denoising_lora_dir,args.denoising_lora_name),adapter_name=args.denoising_lora_name, use_safetensors=True)
    pipe.transformer.set_adapters([i for i in args.condition_types] + [args.denoising_lora_name],[1.0,1.0,args.denoising_lora_weight])
   
    pipe = pipe.to(device)

    output_dir = args.work_dir
    os.makedirs(os.path.join(output_dir, "grid"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)
    # infer
    inputs = get_inputs(args.test_dir)
    for img_name, prompt, img_path, loc_bbox in tqdm(inputs):
        conditions, new_loc_bbox, mask, original_img, crop_box = get_conditions(img_path, loc_bbox)
        result_img = pipe(
            prompt=prompt,
            conditions=conditions,
            height=512,
            width=512,
            num_inference_steps=8,
            max_sequence_length=512,
            model_config = {},
        ).images[0]
    
        # save grid
        concat_image = Image.new("RGB", (512 + len(args.condition_types)*512, 512))
        for j, cond_type in enumerate(args.condition_types):
            cond_image = conditions[j].condition
            if cond_type == "fill":
                cond_image = cv2.rectangle(np.array(cond_image), new_loc_bbox[:2], new_loc_bbox[2:], color=(128, 128, 128),thickness=-1)
                cond_image = Image.fromarray(cv2.rectangle(cond_image, new_loc_bbox[:2], new_loc_bbox[2:], color=(255, 215, 0), thickness=2))
            concat_image.paste(cond_image, (j * 512, 0))
        concat_image.paste(result_img, (j * 512 + 512, 0))
        concat_image.save(os.path.join(output_dir, "grid", f"{img_name}.png"))

        # ==========================================
        # SAVE RESULT IMG IN ORIGIN FORM (BLUR BLEND)
        # ==========================================
        
        # Dán ảnh inpaint trở lại ảnh gốc
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        
        ### 1. Resize ảnh kết quả (512x512) về lại kích thước của vùng crop ban đầu
        result_img_resized = result_img.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
        
        ### 2. Tạo bản sao của ảnh gốc và dán cứng (hard paste) vào một biến tạm
        temp_img = original_img.copy()
        temp_img.paste(result_img_resized, (crop_x1, crop_y1))
        
        ### 3. Sử dụng Gaussian Blur Blend để làm mượt viền (Feathering)
        orig_image_np = np.array(original_img).astype(np.float32)
        pasted_image_np = np.array(temp_img).astype(np.float32)
        
        # Tạo mask trống cùng kích thước với ảnh gốc
        mask_np = np.zeros((original_img.size[1], original_img.size[0]), dtype=np.float32)
        
        # --- LOGIC TỰ ĐỘNG TÍNH PAD VÀ KERNEL SIZE ---
        
        # Tự động tính pad bằng 5% (0.05) chiều ngắn nhất của vùng crop
        pad_ratio = 0.05
        pad = int(min(crop_w, crop_h) * pad_ratio)
        
        # Đảm bảo pad không nhỏ hơn 3 pixel (tránh việc không có hiệu ứng)
        pad = max(3, pad)
        
        pad_y = min(pad, crop_h // 2 - 1)
        pad_x = min(pad, crop_w // 2 - 1)
        
        # Tự động tính kernel_size để đủ bao phủ vùng pad (gấp khoảng 2.5 lần pad)
        calculated_kernel = int(pad * 2.5)
        
        # Bắt buộc kernel_size phải là SỐ LẺ
        kernel_size = calculated_kernel if calculated_kernel % 2 != 0 else calculated_kernel + 1
        kernel_size = max(3, kernel_size) # Đảm bảo kernel tối thiểu là 3
        
        # ---------------------------------------------
        
        mask_np[crop_y1 + pad_y : crop_y2 - pad_y, crop_x1 + pad_x : crop_x2 - pad_x] = 1.0
        
        # Blur mask với kernel_size tự động
        mask_blurred = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), 0)
        
        # Mở rộng chiều mask (H, W) -> (H, W, 1) để nhân được với ảnh màu RGB
        mask_blurred = mask_blurred[:, :, np.newaxis]
        
        # Áp dụng blend
        image_blended = orig_image_np * (1 - mask_blurred) + pasted_image_np * mask_blurred
        
        ### 4. Lưu ảnh hoàn chỉnh
        # Clip giá trị để đảm bảo nằm trong khoảng 0-255 và chuyển về uint8
        final_img = Image.fromarray(np.clip(image_blended, 0, 255).astype(np.uint8))
        final_img.save(os.path.join(output_dir, "img", f"{img_name}.png"))

        # save mask
        mask.save(os.path.join(output_dir, "mask",f"{img_name}.png"))

if __name__ == "__main__":
    args = parse_args()
    inference(args)

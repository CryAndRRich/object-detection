import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.transforms import functional as F
from utils.visualization import visualize_result
import os

# Import your model wrapper
from models.diffusion_module import ObjectPlacementPolicy

# 1. Setup & Configuration
# device = torch.device("cpu") #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Config
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

model_cfg = load_config("checkpoints/best_ckpt/model_config_final.yaml")
train_cfg = load_config("checkpoints/best_ckpt/train_config_final.yaml") # Needed if you used specific diffusion steps

# 2. Load Weights (Crucial Step)
# This path should come from train_cfg or a specific run argument
checkpoint_path = "checkpoints/best_ckpt/best_model.pth" 

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ### -----------------------------------------------------------------------------------------------
    if 'args' in checkpoint and 'num_steps' in checkpoint['args']:
        trained_steps = checkpoint['args']['num_steps']
        if trained_steps is not None:
            print(f"Detected training steps from checkpoint: {trained_steps}")
            # Inject into model config so buffers initialize correctly
            if 'diffusion' not in model_cfg: model_cfg['diffusion'] = {}
            model_cfg['diffusion']['num_timesteps'] = trained_steps
            # Also helpful to set a variable for the sampling loop later
            num_sampling_steps = 50 # Or use trained_steps for full quality
    else:
        print("Warning: Could not detect num_steps in checkpoint. Assuming default (100).")
    ### -----------------------------------------------------------------------------------------------
    # 3. Initialize Model WITH Config
    print("Initializing model with configuration...")
    # !!! THIS IS THE FIX !!!
    model = ObjectPlacementPolicy(cfg=model_cfg) 
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    # 3. Initialize Model WITH Config
    print("Initializing model with configuration...")
    # !!! THIS IS THE FIX !!!
    model = ObjectPlacementPolicy(cfg=model_cfg) 
    print("Warning: No checkpoint found, using random weights (output will be noise)")

model.to(device)
model.eval()

# 3. Preprocessing Function (Resize & Pad)
def preprocess(image_path, density_path, target_size=512):
    # Load images
    img = Image.open(image_path).convert("RGB")
    density = Image.open(density_path).convert("L") # Grayscale density

    # Resize and Pad logic (Same as Dataset)
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
    density_resized = density.resize((new_w, new_h), resample=Image.NEAREST)
    
    # Create padded canvas
    padded_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    padded_img.paste(img_resized, (0, 0))
    
    padded_density = Image.new("L", (target_size, target_size), 0)
    padded_density.paste(density_resized, (0, 0))
    
    # To Tensor
    img_t = F.to_tensor(padded_img).unsqueeze(0).to(device)       # [1, 3, 512, 512]
    density_t = F.to_tensor(padded_density).unsqueeze(0).to(device) # [1, 1, 512, 512]
    
    return img_t, density_t, scale, (w, h)

# 4. The Sampling Loop (Reverse Diffusion)
@torch.no_grad()
def sample_location(model, img_t, density_t, text_prompt, num_steps=100):
    """
    Runs the reverse diffusion process to generate a bounding box.
    """
    # A. Encode Conditions
    # Vision
    vis_emb = model.vision_encoder(img_t, density_t)
    
    # Text (List of strings)
    text_emb = model.text_encoder([text_prompt]) 
    
    # Combine
    global_cond = torch.cat([vis_emb, text_emb], dim=-1) # [1, 256]

    # B. Start from Pure Noise
    # Shape: [Batch, 4] for (x, y, w, h)
    box = torch.randn((1, 4), device=device)
    
    # C. DDPM Reverse Loop
    # (Simplified: In production, use a scheduler like DDPMScheduler from diffusers)
    print(f"Sampling for '{text_prompt}'...")
    
    for t in reversed(range(num_steps)):
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        
        # Add Horizon dim for U-Net: [1, 4] -> [1, 1, 4]
        box_in = box.unsqueeze(1)
        
        # Predict noise
        noise_pred = model.noise_net(box_in, t_batch, global_cond=global_cond)
        noise_pred = noise_pred.squeeze(1) # Remove horizon
        
        # Update box (x_{t-1} = (x_t - beta * noise) / alpha ...)
        # This is a mock update. YOU MUST USE REAL ALPHAS/BETAS HERE.
        # For this example, we just subtract noise scaled by step size
        step_size = 1.0 / num_steps
        box = box - (step_size * noise_pred)
        
    return box.cpu().squeeze().numpy() # [x, y, w, h] (Normalized -1 to 1)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Dummy paths - replace with your real files
    img_path = "examples/images/2.png"  
    density_path = "examples/density/2.png"
    prompt = "sea shell"
    
    # Create dummy files if they don't exist for testing
    import os
    if not os.path.exists("data"): os.makedirs("data")
    if not os.path.exists(img_path): Image.new('RGB', (640, 480), color='white').save(img_path)
    if not os.path.exists(density_path): Image.new('L', (640, 480), color='black').save(density_path)

    for idx in range(15, 30):  # Run multiple times to see variability
        # 1. Preprocess
        img_t, density_t, scale, orig_size = preprocess(img_path, density_path)
        
        # 2. Inference
        pred_box = sample_location(model, img_t, density_t, prompt, num_steps=num_sampling_steps)
        
        # 3. Visualize
        print(f"Predicted Normalized Box: {pred_box}")
        # visualize_result(img_path, pred_box, scale, orig_size)
        # Assuming your config or code defines the input size as 512x512
        # import pdb; pdb.set_trace()
        visualize_result(
            image_path=img_path, 
            pred_box=pred_box, 
            scale=scale, 
            save_path=f"examples/outputs/result_2_{idx}.jpg",
            original_size=orig_size # Pass as tuple
        )

        # image 930, 7065
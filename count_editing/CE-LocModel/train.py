import os
import yaml
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Project Imports
from models.diffusion_module import ObjectPlacementPolicy
from data.dataset import ObjectPlacementDataset

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# --- NEW FUNCTION TO SAVE LOSS ---
def save_training_history(save_dir, loss_history):
    """
    Saves loss data to JSON and plots a learning curve.
    """
    # 1. Save Raw Data
    json_path = os.path.join(save_dir, "loss_history.json")
    with open(json_path, 'w') as f:
        json.dump(loss_history, f, indent=4)
        
    # 2. Save Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()
    
    plot_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(plot_path)
    plt.close()
    # print(f"History saved to {json_path} and {plot_path}")

def train():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Load Configs
    train_cfg = load_config("config/default.yaml")
    model_cfg = load_config("config/model_config.yaml")
    
    # 2. Data
    # Assuming 'samples' folder is in current directory or specified path
    data_root = train_cfg['training']['data']['train_path']
    train_dataset = ObjectPlacementDataset(data_root)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_cfg['training']['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Found {len(train_dataset)} training samples.")

    # 3. Model
    model = ObjectPlacementPolicy(model_cfg)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=float(train_cfg['training']['learning_rate']))
    
    # 4. Loop
    num_epochs = train_cfg['training']['num_epochs']
    save_dir = train_cfg['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    best_loss = float('inf')
    loss_history = []  # <--- Initialize history list
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Move data to GPU
            rgb = batch['pixel_values'].to(device)
            density = batch['density_map'].to(device)
            gt_bbox = batch['bbox'].to(device)
            text = batch['text'] # List of strings, handled by tokenizer automatically
            
            optimizer.zero_grad()
            
            # Forward + Loss
            loss = model.compute_loss(rgb, density, text, gt_bbox)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss) # <--- Record loss

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        # --- SAVE HISTORY ---
        # We save every epoch so you can check progress while it runs
        save_training_history(save_dir, loss_history)
        
        # Save Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(save_dir, "best_model.pth"))
            print(f"  -> Saved Best Model (Loss: {best_loss:.4f})")
        
        # Optional: Save latest checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
             torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_dir, "last_model.pth"))
            
    print("Training Complete.")

if __name__ == "__main__":
    train()
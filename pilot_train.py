import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import time
import matplotlib.pyplot as plt # <--- NEW: For plotting
import sys

# --- IMPORTS ---
# Ensure you are in the correct directory!
try:
    from models.ClearVision import ClearVisionGenerator, Discriminator
    from utils.dataset import TurbidDataset
    from utils.losses import generator_loss, PerceptualLoss
except ImportError:
    print("❌ Error: Could not import project modules.")
    sys.exit(1)

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4            
NUM_EPOCHS = 20           
LR = 0.0002               

# --- GOOGLE DRIVE PATHS (CRITICAL) ---
# We save results to DRIVE so they survive if Colab disconnects
# Make sure you mount drive first!
DRIVE_ROOT = "/content/drive/MyDrive/ClearVision_Experiment"
SAVE_IMG_DIR = os.path.join(DRIVE_ROOT, "images")
CHECKPOINT_DIR = os.path.join(DRIVE_ROOT, "checkpoints")

# DATA PATHS (Local Colab is faster for reading data)
TURBID_PATH = "/content/dataset/Sorted-UIEB/Raw"      
CLEAR_PATH = "/content/dataset/Sorted-UIEB/GT" 
DEPTH_PATH = "/content/dataset/Sorted-UIEB/depths"

def run_pilot():
    print(f"🚀 Starting Robust Pilot Run on {DEVICE}...")
    
    # 1. Create Folders in Drive
    os.makedirs(SAVE_IMG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"💾 Results will be saved to: {DRIVE_ROOT}")

    # 2. Init Models
    generator = ClearVisionGenerator(ngf=64).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
    
    scaler = torch.amp.GradScaler('cuda') 

    print("⏳ Loading VGG for perceptual loss...")
    from torchvision.models import vgg19
    vgg = vgg19(weights='DEFAULT').features.to(DEVICE).eval()
    perceptual_fn = PerceptualLoss(vgg).to(DEVICE)
    
    lambdas = {"adv": 0.1, "pixel": 10.0, "color": 5.0, "edge": 1.0, "perc": 1.0, "depth": 1.0}

    # 3. Data Loading
    print("📂 Loading Dataset...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    dataset = TurbidDataset(TURBID_PATH, CLEAR_PATH, DEPTH_PATH, transform=transform, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    # --- METRICS STORAGE ---
    g_losses = [] # Store average loss per epoch
    d_losses = []
    
    # 4. Training Loop
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        generator.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        loop = tqdm(enumerate(loader), total=len(loader), leave=False)
        
        for i, (turbid, clear, depth) in loop:
            turbid, clear, depth = turbid.to(DEVICE), clear.to(DEVICE), depth.to(DEVICE)
            
            # --- Train G ---
            opt_g.zero_grad()
            with torch.amp.autocast('cuda'):
                fake_clear = generator(turbid)
                loss_G_total, loss_dict = generator_loss(
                    discriminator, clear, fake_clear, turbid, 
                    depth=depth, perceptual_fn=perceptual_fn, lambdas=lambdas
                )

            scaler.scale(loss_G_total).backward()
            scaler.step(opt_g)
            scaler.update()
                
            # --- Train D ---
            opt_d.zero_grad()
            with torch.amp.autocast('cuda'):
                pred_real = discriminator(clear, turbid)
                pred_fake = discriminator(fake_clear.detach(), turbid)
                loss_D = 0.5 * (torch.mean((pred_real - 1)**2) + torch.mean(pred_fake**2))
            
            scaler.scale(loss_D).backward()
            scaler.step(opt_d)
            scaler.update()

            # Accumulate
            epoch_g_loss += loss_G_total.item()
            epoch_d_loss += loss_D.item()
            
            loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

        # --- END OF EPOCH TASKS ---
        avg_g = epoch_g_loss / len(loader)
        avg_d = epoch_d_loss / len(loader)
        
        # 1. Update List
        g_losses.append(avg_g)
        d_losses.append(avg_d)
        
        print(f"✅ Epoch {epoch+1} Done. Avg G Loss: {avg_g:.4f}")

        # 2. PLOT & SAVE GRAPH (The Graph You Wanted!)
        plt.figure(figsize=(10, 5))
        plt.plot(g_losses, label="Generator Loss", color="blue")
        plt.plot(d_losses, label="Discriminator Loss", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("ClearVision Training Progress")
        plt.savefig(os.path.join(DRIVE_ROOT, "loss_graph.png")) # Overwrites the file each time
        plt.close()

        # 3. SAVE CHECKPOINT (The Safety Net)
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": generator.state_dict(),
            "optimizer_state_dict": opt_g.state_dict(),
            "loss": avg_g,
        }
        # Save "latest" (overwrites) and specific epoch
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth"))
        if (epoch + 1) % 5 == 0: # Also save every 5 epochs permanently
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"ckpt_epoch_{epoch+1}.pth"))
        
        # 4. Save Visual Sample
        with torch.no_grad():
            generator.eval()
            sample = torch.cat((turbid[0], fake_clear[0], clear[0]), dim=2) 
            save_image(sample * 0.5 + 0.5, os.path.join(SAVE_IMG_DIR, f"epoch_{epoch+1}.png"))

    print(f"\n🎉 Finished! Check your Google Drive folder: {DRIVE_ROOT}")

if __name__ == "__main__":
    run_pilot()
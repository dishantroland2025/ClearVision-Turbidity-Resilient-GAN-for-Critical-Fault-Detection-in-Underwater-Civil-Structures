import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import time
import traceback
import sys

# --- IMPORTS ---
# Make sure you are in the correct directory! (e.g. /content/ClearVision...)
try:
    from models.ClearVision import ClearVisionGenerator, Discriminator
    from utils.dataset import TurbidDataset
    from utils.losses import generator_loss, PerceptualLoss
except ImportError:
    print("❌ Error: Could not import project modules.")
    print("Make sure you are in the root folder of your GitHub repo (where models/ and utils/ are).")
    print("Try running: %cd /content/ClearVision-Turbidity-Resilient-GAN (or your repo name)")
    sys.exit(1)

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4            
NUM_EPOCHS = 20           
LR = 0.0002               
SAVE_IMG_DIR = "pilot_results"

# --- PATHS (COLAB SPECIFIC) ---
# Check your folder names in the file explorer to be sure!
# Assuming structure: /content/dataset/Sorted/[raw, ground_truth, depth]
TURBID_PATH = "/content/dataset/Sorted-UIEB/Raw"      
CLEAR_PATH = "/content/dataset/Sorted-UIEB/GT" 
DEPTH_PATH = "/content/dataset/Sorted-UIEB/depths"

def check_paths():
    """Verifies datasets exist before starting"""
    if not os.path.exists(TURBID_PATH):
        raise FileNotFoundError(f"❌ Cannot find Turbid images at: {TURBID_PATH}\n   -> Did you unzip the dataset to /content/dataset?")
    if not os.path.exists(CLEAR_PATH):
        raise FileNotFoundError(f"❌ Cannot find Ground Truth at: {CLEAR_PATH}")
    print(f"✅ Data paths verified. Found {len(os.listdir(TURBID_PATH))} images.")

def run_pilot():
    print(f"🚀 Starting Pilot Run on {DEVICE}...")
    check_paths()
    os.makedirs(SAVE_IMG_DIR, exist_ok=True)

    # 1. Init Models
    generator = ClearVisionGenerator(ngf=64).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    opt_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
    
    scaler = torch.amp.GradScaler('cuda') 

    print("⏳ Loading VGG for perceptual loss...")
    from torchvision.models import vgg19
    vgg = vgg19(weights='DEFAULT').features.to(DEVICE).eval()
    perceptual_fn = PerceptualLoss(vgg).to(DEVICE)
    
    # Standard Weights
    lambdas = {"adv": 0.1, "pixel": 10.0, "color": 5.0, "edge": 1.0, "perc": 1.0, "depth": 1.0}

    # 2. Data Loading
    print("📂 Loading Dataset...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    dataset = TurbidDataset(TURBID_PATH, CLEAR_PATH, DEPTH_PATH, transform=transform, augment=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    # 3. Training Loop
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        generator.train()
        epoch_g_loss = 0.0
        
        loop = tqdm(enumerate(loader), total=len(loader), leave=False)
        
        for i, (turbid, clear, depth) in loop:
            turbid, clear, depth = turbid.to(DEVICE), clear.to(DEVICE), depth.to(DEVICE)
            
            # --- Train Generator ---
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
                
            # --- Train Discriminator ---
            opt_d.zero_grad()
            with torch.amp.autocast('cuda'):
                pred_real = discriminator(clear, turbid)
                pred_fake = discriminator(fake_clear.detach(), turbid)
                loss_D = 0.5 * (torch.mean((pred_real - 1)**2) + torch.mean(pred_fake**2))
            
            scaler.scale(loss_D).backward()
            scaler.step(opt_d)
            scaler.update()

            epoch_g_loss += loss_G_total.item()
            loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")

            # --- PRINT LOGS FOR PROFESSOR (Every 50 batches) ---
            if i % 50 == 0:
                print(f"\n[Epoch {epoch+1}][Batch {i}] Loss_G: {loss_G_total.item():.4f} | Loss_D: {loss_D.item():.4f}")

        # --- Visual Check ---
        avg_g_loss = epoch_g_loss / len(loader)
        print(f"✅ Epoch {epoch+1} Done. Avg Loss: {avg_g_loss:.4f}")
        
        with torch.no_grad():
            generator.eval()
            sample = torch.cat((turbid[0], fake_clear[0], clear[0]), dim=2) 
            save_image(sample * 0.5 + 0.5, f"{SAVE_IMG_DIR}/epoch_{epoch+1}.png")

    print(f"\n🎉 Finished in {(time.time() - start_time)/60:.1f} min.")
    print(f"Check the folder: {os.path.abspath(SAVE_IMG_DIR)}")

if __name__ == "__main__":
    try:
        run_pilot()
    except Exception as e:
        traceback.print_exc()
import torch
import torch.nn as nn
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import cv2

# --- Helper: Tensor Processing ---
def preprocess_batch(img_batch):
    """
    Converts input (Tensor or Numpy) into a list of [H, W, C] numpy arrays [0, 255].
    Handles Denormalization from [-1, 1] to [0, 255].
    """
    # 1. Handle PyTorch Tensors
    if isinstance(img_batch, torch.Tensor):
        # Detach from graph, move to CPU, convert to numpy
        img_batch = img_batch.detach().cpu().numpy()

    # 2. Handle Dimensions
    # If single image (C, H, W) -> add batch dim -> (1, C, H, W)
    if img_batch.ndim == 3:
        img_batch = img_batch[np.newaxis, ...]

    # 3. Process the batch
    processed_imgs = []
    for img in img_batch:
        # Check if image is Channel-First (C, H, W) typically 3x256x256
        if img.shape[0] == 3 or img.shape[0] == 1:
            img = np.transpose(img, (1, 2, 0)) # Convert to (H, W, C)

        # 4. Denormalize: Assume GAN output is [-1, 1], convert to [0, 255]
        # Only denormalize if values are small (float range)
        if img.max() <= 1.1: 
            img = (img * 0.5 + 0.5) * 255.0
        
        # Clip and cast
        img = np.clip(img, 0, 255).astype(np.uint8)
        processed_imgs.append(img)
        
    return processed_imgs

# --- 1. Standard Reference Metrics (PSNR, SSIM, LPIPS) ---

def calculate_psnr(img1, img2):
    """Calculates average PSNR for a batch."""
    imgs1 = preprocess_batch(img1)
    imgs2 = preprocess_batch(img2)
    
    scores = []
    for i in range(len(imgs1)):
        # data_range=255 is valid because preprocess_batch casts to uint8
        score = peak_signal_noise_ratio(imgs1[i], imgs2[i], data_range=255)
        scores.append(score)
        
    return np.mean(scores)

def calculate_ssim(img1, img2):
    """Calculates average SSIM for a batch."""
    imgs1 = preprocess_batch(img1)
    imgs2 = preprocess_batch(img2)
    
    scores = []
    for i in range(len(imgs1)):
        # channel_axis=2 handles RGB (H, W, C)
        score = structural_similarity(
            imgs1[i], 
            imgs2[i], 
            data_range=255, 
            channel_axis=2 
        )
        scores.append(score)
        
    return np.mean(scores)

# LPIPS requires a model download, initialized once to save time
class LPIPSMetric:
    def __init__(self, device='cuda'):
        self.device = device
        # AlexNet is the standard backbone for LPIPS comparison
        self.loss_fn = lpips.LPIPS(net='alex').to(device).eval()

    def calculate(self, img1, img2):
        """
        Inputs: PyTorch Tensors or Numpy.
        Calculates LPIPS distance.
        """
        # Ensure inputs are tensors on the correct device
        if not isinstance(img1, torch.Tensor):
            t1 = torch.from_numpy(img1).float().to(self.device)
        else:
            t1 = img1.to(self.device)

        if not isinstance(img2, torch.Tensor):
            t2 = torch.from_numpy(img2).float().to(self.device)
        else:
            t2 = img2.to(self.device)
            
        # LPIPS expects inputs in [-1, 1].
        # If input is [0, 1] (Sigmoid), normalize to [-1, 1]
        if t1.max() <= 1.0 and t1.min() >= 0.0:
            t1 = t1 * 2.0 - 1.0
            t2 = t2 * 2.0 - 1.0
        
        # If input is [0, 255], normalize to [-1, 1]
        elif t1.max() > 1.1:
            t1 = (t1 / 127.5) - 1.0
            t2 = (t2 / 127.5) - 1.0

        with torch.no_grad():
            dist = self.loss_fn(t1, t2)
            
        return dist.mean().item()


# --- 2. Underwater Specific Metric (UIQM) ---

def calculate_uiqm(img_input):
    """
    Underwater Image Quality Measure (UIQM).
    Handles Batch inputs by averaging scores.
    """
    # Preprocess converts to [0, 255] numpy HWC list
    imgs = preprocess_batch(img_input)
    
    c1, c2, c3 = 0.0282, 0.2953, 3.5753 
    scores = []
    
    for img in imgs:
        uicm = _uicm(img)
        uism = _uism(img)
        uiconm = _uiconm(img)
        scores.append(c1 * uicm + c2 * uism + c3 * uiconm)
    
    return np.mean(scores)

def _uicm(img):
    """Underwater Image Colorfulness Measure"""
    # Cast to float for calculations
    img = img.astype(np.float32)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    
    mu_rg, sig_rg = np.mean(rg), np.std(rg)
    mu_yb, sig_yb = np.mean(yb), np.std(yb)
    
    return -0.0268 * np.sqrt(mu_rg**2 + mu_yb**2) + 0.1586 * np.sqrt(sig_rg**2 + sig_yb**2)

def _uism(img):
    """Underwater Image Sharpness Measure"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    edge_map = np.sqrt(gx**2 + gy**2)
    return np.mean(edge_map)

def _uiconm(img):
    """Underwater Image Contrast Measure (LogAMEE)"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    h, w = gray.shape
    k = 32
    entropy_vals = []
    
    for i in range(0, h-k, k):
        for j in range(0, w-k, k):
            block = gray[i:i+k, j:j+k]
            max_val = np.max(block)
            min_val = np.min(block)
            
            # Avoid division by zero
            denom = max_val + min_val
            if denom > 0 and max_val > min_val:
                val = (max_val - min_val) / denom
                entropy_vals.append(val)
    
    if len(entropy_vals) == 0: return 0.0
    return np.mean(entropy_vals)
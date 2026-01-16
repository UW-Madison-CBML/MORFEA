import numpy as np
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from dataset_ivf_embryo import IVFEmbryoDataset
from raffael_model import ConvLSTMAutoencoder
from huggingface_hub import login, HfApi
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from dataset_ivf import IVFSequenceDataset
import torch.nn.functional as F

import math

class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def push(self, x):
        """Add a new value and update statistics."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self):
        """Returns sample variance (unbiased). Use self.m2 / self.n for population."""
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std_dev(self):
        """Returns sample standard deviation."""
        return math.sqrt(self.variance)
def ssim(img1, img2, kernel_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    """
    Single-scale SSIM
    Args:
        img1, img2: (B, C, H, W)
    """
    kernel = gaussian_kernel(kernel_size, sigma).to(img1.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
    
    mu1 = F.conv2d(img1, kernel, padding=kernel_size//2)
    mu2 = F.conv2d(img2, kernel, padding=kernel_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=kernel_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=kernel_size//2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def ms_ssim(img1, img2, kernel_size=11, sigma=1.5, weights=None, levels=5):
    """
    Multi-Scale SSIM (MS-SSIM)
    Args:
        img1, img2: (B, C, H, W)
        weights: weights for each scale, default [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        levels: number of scales
    """
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], 
                              device=img1.device)
    
    # Ensure weight count matches
    weights = weights[:levels]
    weights = weights / weights.sum()
    
    mcs_list = []
    ssim_val = None
    
    for i in range(levels):
        if i == levels - 1:
            # Last layer computes SSIM
            ssim_val = ssim(img1, img2, kernel_size, sigma)
        else:
            # Other layers compute contrast
            kernel = gaussian_kernel(kernel_size, sigma).to(img1.device)
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            
            mu1 = F.conv2d(img1, kernel, padding=kernel_size//2)
            mu2 = F.conv2d(img2, kernel, padding=kernel_size//2)
            
            sigma1_sq = F.conv2d(img1 * img1, kernel, padding=kernel_size//2) - mu1 ** 2
            sigma2_sq = F.conv2d(img2 * img2, kernel, padding=kernel_size//2) - mu2 ** 2
            sigma12 = F.conv2d(img1 * img2, kernel, padding=kernel_size//2) - mu1 * mu2
            
            C2 = 0.03 ** 2
            mcs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
            mcs_list.append(mcs.mean())
        
        # Downsample to next level
        if i < levels - 1:
            img1 = F.avg_pool2d(img1, 2)
            img2 = F.avg_pool2d(img2, 2)
    
    # Combine all scales
    ms_ssim_val = ssim_val
    for i, mcs in enumerate(mcs_list):
        ms_ssim_val = ms_ssim_val ** weights[i] * mcs ** weights[i]
    
    return ms_ssim_val


VAL_EMBRYOS = ["CZ594-5","CJ261-10","RL747-8","TM272-9","LFA766-1","GT353-3","LGA881-2-5","LBE649-3","TH481-5","LTA908-2","BS648-7","GS955-7","HA1040-4","CM892-5","FC048-6","GC702-6","DI358-3","MM912-4","RK787-3","GSS052-2","OJ319-5","DML373-2","PS292-4","TM294-2","KT573-4","DJC641-4","FE14-020","LD400-1","MV930-2","MDCH869-4","AS662-2","LH1169-8","GA664-1","PMDPI029-1-3","DV116-3","FV709-11","GM456-3","RA361-4","LM844-1","DL020-3","VM570-4","MC833-6","LV613-2","ZS435-5","RM126-7","BK428-2","LS93-8","GS490-7","GF976-4","PMDPI029-1-11","DRL1048-1","BS294-7","CA658-12","RO793-2","GJ191-1","CC007-2","SL313-11","RC545-2-8","OJ319-9","PA289-8","TK319-10","SM686-7","KJ1077-3","BE645-10","BC167-4","VC581-1","FM162-6","PC758-2","HC459-6","DE069-10","GC340-3","BS596-5","PE256-2","LBE857-1","PH783-3","LS1045-4","CC455-3","DL617-6","BS1086-1","CK601-4","DA309-5","LTE064-1","KF460-4","LP181-1","GS349-4","LC47-8","GS205-6","EH309-8","BS1033-2","LL854-1","DHDPI042-6","BN356-6","PA145-2","GC340-1","MM334-5","AG274-2","BA518-7","BC973-4","BA1195-9","AM33-2","AB91-1","AB028-6","BC167-4","AL884-2","AM685-3"]
def main(model_name):
    if(model_name == None or len(model_name) == 0):
        raise ValueError("bad model name")

    login(os.getenv("HF_KEY"))
    api = HfApi()
    model = None
    model_loaded = False

    # Try specific model name first
    print(f"Attempting to load model: {model_name}")
    model = ConvLSTMAutoencoder.from_pretrained("JensLundsgaard/"+model_name)

    

    model = model.to(DEVICE)

    df = pd.read_csv(os.path.abspath("index.csv"))
    mask = df["cell_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    val_df = df[mask]
    val_dataset = IVFSequenceDataset(val_df, resize=128, norm="minmax01")

    loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,  # No shuffle for validation
        num_workers=4,
        pin_memory=True,
        drop_last=False  # Don't drop last for validation
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(loader) * 10)


    # Comprehensive validation with multiple metrics
    val_metrics = {
        'mse': 0.0,
        'l1': 0.0,
        'ms_ssim_value': 0.0,
        'ms_ssim_loss': 0.0,
        'temporal_smoothness': 0.0
    }
    mse = RunningStats()
    l1 = RunningStats()
    ssim = RunningStats()
    temp = RunningStats()

    model.eval()  
    with torch.no_grad():
        for embryo_vol, _, _ in val_loader:
            embryo_vol = embryo_vol.to(DEVICE)
            val_recon, val_lat = model(embryo_vol)

            B, T, C, H, W = embryo_vol.shape

            # MSE
            mse.push(F.mse_loss(val_recon, embryo_vol).item())

            # L1
            l1.push(F.l1_loss(val_recon, embryo_vol).item())

            # MS-SSIM
            val_recon_flat = val_recon.view(B * T, C, H, W)
            embryo_vol_flat = embryo_vol.view(B * T, C, H, W)
            ms_ssim_val = ms_ssim(val_recon_flat, embryo_vol_flat)
            ssim.push((1 - ms_ssim_val).item())

            # Temporal smoothness of latents
            # val_lat is (B, T, latent_size)
            if T > 1:
                lat_diff = torch.diff(val_lat, dim=1)  # (B, T-1, latent_size)
                temporal_smooth = lat_diff.norm(dim=-1).mean()  # Average L2 norm of differences
                temp.push(temporal_smooth.item())

    print("MSE: " + str(mse.mean) + " +- " + str(mse.std_dev))
    print("L1: " + str(l1.mean) + " +- " + str(l1.std_dev))
    print("SSIM: " + str(ssim.mean) + " +- " + str(ssim.std_dev))
    print("Temp: " + str(temp.mean) + " +- " + str(temp.std_dev))

if __name__ == "__main__":
    import sys
    if(len(sys.argv) < 2):
        raise ValueError("no model specified")
    main(sys.argv[1])


     




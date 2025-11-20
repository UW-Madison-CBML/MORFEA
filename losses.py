"""
High-Quality Loss Functions
- MS-SSIM Loss (Multi-Scale Structural Similarity)
- L1 Loss
- Combined Reconstruction Loss
- Classification Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(size=11, sigma=1.5):
    """Generate Gaussian kernel for SSIM"""
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)


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


def reconstruction_loss(x_rec, x_true, l1_weight=0.5, ms_ssim_weight=0.5):
    """
    Combined reconstruction loss: L1 + MS-SSIM
    Args:
        x_rec: (B, T, 1, H, W) - reconstructed video
        x_true: (B, T, 1, H, W) - original video
        l1_weight: L1 loss weight
        ms_ssim_weight: MS-SSIM loss weight
    """
    B, T, C, H, W = x_rec.shape
    
    # Flatten temporal dimension for MS-SSIM computation
    x_rec_flat = x_rec.view(B * T, C, H, W)  # (B*T, 1, 128, 128)
    x_true_flat = x_true.view(B * T, C, H, W)  # (B*T, 1, 128, 128)
    
    # L1 Loss
    l1_loss = F.l1_loss(x_rec, x_true)
    
    # MS-SSIM Loss
    ms_ssim_val = ms_ssim(x_rec_flat, x_true_flat)
    ms_ssim_loss = 1 - ms_ssim_val
    
    # Combined loss
    total_loss = l1_weight * l1_loss + ms_ssim_weight * ms_ssim_loss
    
    return total_loss, {
        "l1_loss": l1_loss.item(),
        "ms_ssim_loss": ms_ssim_loss.item(),
        "ms_ssim_value": ms_ssim_val.item()
    }


def temporal_smoothness_loss(z_seq, weight=0.1):
    """
    Temporal smoothness loss: encourages similar latents for adjacent timesteps
    Args:
        z_seq: (B, T, C, H, W) - latent sequence
        weight: loss weight
    """
    if z_seq.size(1) < 2:
        return torch.tensor(0.0, device=z_seq.device)
    
    # Compute difference between adjacent timesteps
    diff = z_seq[:, 1:] - z_seq[:, :-1]  # (B, T-1, C, H, W)
    smooth_loss = (diff ** 2).mean()
    
    return weight * smooth_loss


def classification_loss(logits, labels, criterion=None):
    """
    Classification loss
    Args:
        logits: (B, num_classes) - classification logits
        labels: (B,) - ground truth labels
        criterion: loss function, default CrossEntropyLoss
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    return criterion(logits, labels)


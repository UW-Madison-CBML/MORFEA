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


def gaussian_window(window_size, sigma, channels, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window = window_2d.expand(channels, 1, window_size, window_size)
    return window


def _ssim_and_mcs(img1, img2, window_size=11, sigma=1.5, data_range=1.0, size_average=True):
    """
    Compute both SSIM and MCS (contrast-structure) maps in the standard decomposition.
    Returns:
        ssim_val: scalar (or per-sample if size_average=False)
        mcs_val:  scalar (or per-sample if size_average=False)
    """
    assert img1.shape == img2.shape
    B, C, H, W = img1.shape
    device = img1.device

    window = gaussian_window(window_size, sigma, C, device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1_mu2

    # Standard constants scaled by data range
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Luminance term
    l = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1 + 1e-12)
    # Contrast-structure term (often called "cs" or "mcs")
    cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2 + 1e-12)

    ssim_map = l * cs
    mcs_map = cs  # standard MS-SSIM uses cs for scales 1..M-1

    if size_average:
        return ssim_map.mean(), mcs_map.mean()
    else:
        # per-sample
        return ssim_map.mean(dim=[1, 2, 3]), mcs_map.mean(dim=[1, 2, 3])


def ms_ssim(
    img1,
    img2,
    window_size=11,
    sigma=1.5,
    data_range=1.0,
    weights=None,
    levels=5,
    size_average=True
):
    """
    Standard MS-SSIM:
        MS-SSIM = (SSIM_M)^{w_M} * Π_{j=1}^{M-1} (MCS_j)^{w_j}

    Args:
        img1, img2: (B, C, H, W) in [0, data_range]
        weights: length==levels, default is the common 5-scale weights
        levels: number of scales M
    """
    assert img1.shape == img2.shape
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=img1.device)
    else:
        weights = torch.as_tensor(weights, device=img1.device, dtype=torch.float32)

    weights = weights[:levels]
    weights = weights / weights.sum()  # normalized weights (optional but fine)

    mcs_vals = []
    ssim_val = None

    x1, x2 = img1, img2
    for j in range(levels):
        ssim_j, mcs_j = _ssim_and_mcs(
            x1, x2, window_size=window_size, sigma=sigma, data_range=data_range, size_average=size_average
        )

        if j < levels - 1:
            mcs_vals.append(mcs_j)
            x1 = F.avg_pool2d(x1, kernel_size=2, stride=2)
            x2 = F.avg_pool2d(x2, kernel_size=2, stride=2)
        else:
            ssim_val = ssim_j

    # Combine exactly once (no iterative re-exponentiation)
    # MS-SSIM = Π_{j=1}^{M-1} mcs_j^{w_j} * ssim_M^{w_M}
    out = ssim_val.pow(weights[levels - 1])
    for j, mcs_j in enumerate(mcs_vals):
        out = out * mcs_j.pow(weights[j])

    return out


def reconstruction_loss(x_rec, x_true, l1_weight=0.5, ms_ssim_weight=0.5,
                        window_size=11, sigma=1.5, data_range=1.0, levels=5, weights=None):
    """
    Combined reconstruction loss: L1 + MS-SSIM
    Args:
        x_rec, x_true: (B, T, C, H, W)
    """
    assert x_rec.shape == x_true.shape
    B, T, C, H, W = x_rec.shape

    # Flatten temporal dimension for MS-SSIM computation
    x_rec_flat = x_rec.reshape(B * T, C, H, W)
    x_true_flat = x_true.reshape(B * T, C, H, W)

    l1 = F.l1_loss(x_rec, x_true)

    ms_val = ms_ssim(
        x_rec_flat, x_true_flat,
        window_size=window_size, sigma=sigma, data_range=data_range,
        weights=weights, levels=levels, size_average=True
    )
    ms_loss = 1.0 - ms_val

    total = l1_weight * l1 + ms_ssim_weight * ms_loss

    return total, {
        "l1_loss": float(l1.detach().cpu()),
        "ms_ssim_loss": float(ms_loss.detach().cpu()),
        "ms_ssim_value": float(ms_val.detach().cpu()),
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


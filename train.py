import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import math
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from model import Model
from raffael_model import ConvLSTMAutoencoder
import sys
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from torch.utils.data import DataLoader
from dataset_ivf import IVFSequenceDataset
from tqdm import tqdm
from datetime import datetime
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
batch_size = 50
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from huggingface_hub import HfApi
import wandb
import gc
gc.collect()
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import time

from huggingface_hub import login
import shutil
import hashlib
import json
from torchsummary import summary
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

VAL_EMBRYOS = []#"RS363-7", "CZ594-5","CJ261-10","RL747-8","TM272-9","LFA766-1","GT353-3","LGA881-2-5","LBE649-3","TH481-5","LTA908-2","BS648-7","GS955-7","HA1040-4","CM892-5","FC048-6","GC702-6","DI358-3","MM912-4","RK787-3","GSS052-2","OJ319-5","DML373-2","PS292-4","TM294-2","KT573-4","DJC641-4","FE14-020","LD400-1","MV930-2","MDCH869-4","AS662-2","LH1169-8","GA664-1","PMDPI029-1-3","DV116-3","FV709-11","GM456-3","RA361-4","LM844-1","DL020-3","VM570-4","MC833-6","LV613-2","ZS435-5","RM126-7","BK428-2","LS93-8","GS490-7","GF976-4","PMDPI029-1-11","DRL1048-1","BS294-7","CA658-12","RO793-2","GJ191-1","CC007-2","SL313-11","RC545-2-8","OJ319-9","PA289-8","TK319-10","SM686-7","KJ1077-3","BE645-10","BC167-4","VC581-1","FM162-6","PC758-2","HC459-6","DE069-10","GC340-3","BS596-5","PE256-2","LBE857-1","PH783-3","LS1045-4","CC455-3","DL617-6","BS1086-1","CK601-4","DA309-5","LTE064-1","KF460-4","LP181-1","GS349-4","LC47-8","GS205-6","EH309-8","BS1033-2","LL854-1","DHDPI042-6","BN356-6","PA145-2","GC340-1","MM334-5","AG274-2","BA518-7","BC973-4","BA1195-9","AM33-2","AB91-1","AB028-6","BC167-4","AL884-2","AM685-3"]
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


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Single GPU fallback
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank
def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def generate_repo_name(mode, config_dict, file_paths, date_str):
    """
    Generate a unique, deterministic repository name based on configuration and code.

    Args:
        mode: Training mode (e.g., "convlstm", "convlstm_latent_split")
        config_dict: Dictionary of all configuration parameters
        file_paths: List of file paths to hash
        date_str: Date string (YYYY-MM-DD)

    Returns:
        str: Repository name (max 96 chars)
    """
    # Create hash input from config
    config_str = json.dumps(config_dict, sort_keys=True)

    # Hash all file contents
    file_hasher = hashlib.sha256()
    for file_path in file_paths:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                file_hasher.update(f.read())
        else:
            # If file doesn't exist, add its name to the hash anyway
            file_hasher.update(file_path.encode())

    # Combine everything into final hash
    combined_hasher = hashlib.sha256()
    combined_hasher.update(config_str.encode())
    combined_hasher.update(file_hasher.digest())
    combined_hasher.update(date_str.encode())

    # Get short hash (first 8 characters is enough for uniqueness)
    short_hash = combined_hasher.hexdigest()[:8]

    # Build repo name: embryo-{mode}-{hash}-{date}
    # Example: embryo-convlstm-a3f2b1c9-2025-12-21
    repo_name = f"embryo-{mode}-{short_hash}-{date_str}"

    # Ensure it's under 96 characters
    if len(repo_name) > 96:
        # Truncate mode if needed
        max_mode_len = 96 - len(f"embryo--{short_hash}-{date_str}")
        truncated_mode = mode[:max_mode_len]
        repo_name = f"embryo-{truncated_mode}-{short_hash}-{date_str}"

    return repo_name

def save_and_push_model(model, repo_name, required_files, model_config=None):
    """
    Save model and push it along with required training files to HuggingFace Hub

    Args:
        model: The model to save
        repo_name: Repository name on HuggingFace Hub
        required_files: List of file paths to include in the repo
        model_config: Optional dictionary with model configuration to save as config.json
    """
    # Create temporary directory for the repo
    os.makedirs(repo_name, exist_ok=True)

    # Save the model weights
    try:
        model.save_pretrained(repo_name)
        print(f"Saved model using save_pretrained")
    except Exception as e:
        # If save_pretrained fails, just save the state dict
        print(f"save_pretrained failed ({e}), saving state dict only")
        torch.save(model.state_dict(), os.path.join(repo_name, "pytorch_model.bin"))

    # Save custom config.json with all ablation parameters
    if model_config is not None:
        config_path = os.path.join(repo_name, "config.json")
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        print(f"Saved config.json with ablation parameters")

    # Copy all required files
    for file_path in required_files:
        if os.path.exists(file_path):
            shutil.copy2(file_path, repo_name)
            print(f"Added {file_path} to repository")
        else:
            print(f"Warning: {file_path} not found, skipping")

    # Push model to hub (this uploads model weights and config)
    try:
        model.push_to_hub(repo_name)
        print(f"Pushed model weights to {repo_name}")
    except Exception as e:
        print(f"Warning: push_to_hub failed ({e}), will upload manually")

    # Upload all files using HfApi (including config.json)
    api = HfApi()

    # Upload config.json first if it exists
    config_file = os.path.join(repo_name, "config.json")
    if os.path.exists(config_file):
        try:
            api.upload_file(
                path_or_fileobj=config_file,
                path_in_repo="config.json",
                repo_id=f"JensLundsgaard/{repo_name}",
                repo_type="model"
            )
            print(f"Uploaded config.json to HuggingFace Hub")
        except Exception as e:
            print(f"Warning: Failed to upload config.json: {e}")

    
    # Upload additional required files
    for file_path in required_files:
        local_file = os.path.join(repo_name, os.path.basename(file_path))
        if os.path.exists(local_file):
            try:
                api.upload_file(
                    path_or_fileobj=local_file,
                    path_in_repo=os.path.basename(file_path),
                    repo_id=f"JensLundsgaard/{repo_name}",
                    repo_type="model"
                )
                print(f"Uploaded {file_path} to HuggingFace Hub")
            except Exception as e:
                print(f"Warning: Failed to upload {file_path}: {e}")
        else:
            print(f"Warning: {local_file} not found, skipping upload")

    print(f"Successfully pushed all files to {repo_name}")
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
    if weights is None:
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
                              device=img1.device)[:levels]
    
    kernel = gaussian_kernel(kernel_size, sigma).to(img1.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(img1.shape[1], 1, 1, 1)
    
    mcs_list = []
    
    for i in range(levels):
        if i == levels - 1:
            ssim_val = ssim(img1, img2, kernel_size, sigma)
        else:
            # Compute CS (contrast-structure) only
            mu1 = F.conv2d(img1, kernel, padding=kernel_size//2, groups=img1.shape[1])
            mu2 = F.conv2d(img2, kernel, padding=kernel_size//2, groups=img1.shape[1])
            
            sigma1_sq = F.conv2d(img1**2, kernel, padding=kernel_size//2, groups=img1.shape[1]) - mu1**2
            sigma2_sq = F.conv2d(img2**2, kernel, padding=kernel_size//2, groups=img1.shape[1]) - mu2**2
            sigma12 = F.conv2d(img1*img2, kernel, padding=kernel_size//2, groups=img1.shape[1]) - mu1*mu2
            
            C2 = 0.03**2
            cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
            mcs_list.append(cs.mean())
            
            img1 = F.avg_pool2d(img1, 2)
            img2 = F.avg_pool2d(img2, 2)
    
    # Correct combination
    ms_ssim_val = torch.prod(torch.stack([mcs ** w for mcs, w in zip(mcs_list, weights[:-1])]))
    ms_ssim_val = (ssim_val ** weights[-1]) * ms_ssim_val
    
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


def train_convlstm(
    loss_type="l1",
    ms_ssim_weight=0.5,
    rec_weight=0.5,
    temporal_weight=0.1,
    dropout_rate=0.1,
    use_convlstm=True,
    use_residual=True,
    use_batchnorm=True,
    model_name="", 
    latent_size = 4096
):
    gc.collect()
    """Training ConvLSTM Autoencoder with configurable loss (single GPU)

    Args:
        loss_type: "l1" or "mse" - type of reconstruction loss to use with MS-SSIM
        ms_ssim_weight: float - weight for MS-SSIM loss (0 to disable)
        rec_weight: float - weight for reconstruction loss L1/MSE (0 to disable)
        temporal_weight: float - weight for temporal smoothness loss (0 to disable)
        dropout_rate: float - dropout rate (0 to disable)
        use_convlstm: bool - whether to use ConvLSTM (False = no temporal modeling)
        use_residual: bool - whether to use residual connections
        use_batchnorm: bool - whether to use batch normalization
    """
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VAL_EMBRYOS = pd.read_csv("embryo_dataset_grades.csv").rename(columns={"video_name":"embryo_id"}).dropna(subset=["ICM"])["embryo_id"].astype(str).tolist()
    # Build loss description for logging
    loss_components = []
    if ms_ssim_weight > 0:
        loss_components.append(f"MS-SSIM({ms_ssim_weight})")
    if rec_weight > 0:
        loss_components.append(f"{loss_type.upper()}({rec_weight})")
    if temporal_weight > 0:
        loss_components.append(f"Temporal({temporal_weight})")
    loss_description = " + ".join(loss_components) if loss_components else "None"

    # Build model description for logging
    model_features = []
    if use_convlstm:
        model_features.append("ConvLSTM")
    if use_residual:
        model_features.append("Residual")
    if use_batchnorm:
        model_features.append("BatchNorm")
    if dropout_rate > 0:
        model_features.append(f"Dropout({dropout_rate})")
    model_description = "+".join(model_features) if model_features else "Baseline"
    date_label = datetime.now().strftime("%Y-%m-%d")

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
        name=model_name +"-" + date_label,
        config={
            "learning_rate": 0.02,
            "architecture": "ConvLSTM Autoencoder",
            "model_features": model_description,
            "dataset": "https://zenodo.org/records/7912264",
            "epochs": 10,
            "train_split": 0.85,
            "val_split": 0.15,
            "loss": loss_description,
            "loss_type": loss_type,
            "ms_ssim_weight": ms_ssim_weight,
            "rec_weight": rec_weight,
            "temporal_weight": temporal_weight,
            "dropout_rate": dropout_rate,
            "use_convlstm": use_convlstm,
            "use_residual": use_residual,
            "use_batchnorm": use_batchnorm,
            "latent_size": latent_size,
            "seq_len": 50,
            "image_size": 128,
            "distributed": False,
        },
    )

    login(os.getenv("HF_KEY"))
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    print(DEVICE)
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY - Training Configuration")
    print(f"{'='*60}")
    print(f"\nLoss Configuration:")
    print(f"  Base Loss Type: {loss_type.upper()}")
    print(f"  MS-SSIM Weight: {ms_ssim_weight} {'(DISABLED)' if ms_ssim_weight == 0 else ''}")
    print(f"  Reconstruction Weight: {rec_weight} {'(DISABLED)' if rec_weight == 0 else ''}")
    print(f"  Temporal Smoothness Weight: {temporal_weight} {'(DISABLED)' if temporal_weight == 0 else ''}")
    print(f"  Combined Loss: {loss_description}")
    print(f"\nModel Architecture Configuration:")
    print(f"  ConvLSTM: {'ENABLED' if use_convlstm else 'DISABLED'}")
    print(f"  Residual Connections: {'ENABLED' if use_residual else 'DISABLED'}")
    print(f"  Batch Normalization: {'ENABLED' if use_batchnorm else 'DISABLED'}")
    print(f"  Dropout Rate: {dropout_rate} {'(DISABLED)' if dropout_rate == 0 else ''}")
    print(f"  Model Features: {model_description}")
    print(f"{'='*60}\n")

    # Save detailed training configuration
    config_content = f"""ConvLSTM Autoencoder Training Configuration (ABLATION)
================================================================================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

ABLATION STUDY CONFIGURATION
================================================================================

"""

    with open("training_config_detailed.txt", "w") as f:
        f.write(config_content)

    print("Configuration saved to training_config_detailed.txt")

    model = ConvLSTMAutoencoder(
        None,
        seq_len=50,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        latent_size=latent_size,
        use_classifier=False,
        num_classes=2,
        use_latent_split=False,
        # Ablation parameters
        dropout_rate=dropout_rate,
        use_convlstm=use_convlstm,
        use_residual=use_residual,
        use_batchnorm=use_batchnorm
    )

    model = model.to(DEVICE)
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}"
    )
    run.log({"train_params":trainable_params})
    run.log({"params":all_params})
    learning_rate = 2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    df = pd.read_csv(os.path.abspath("index.csv"))
    mask = df["cell_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    val_df = df[mask]
    train_df = df[~mask]
    train_dataset = IVFSequenceDataset(train_df, resize=128, norm="minmax01")
    val_dataset = IVFSequenceDataset(val_df, resize=128, norm="minmax01")
    print("val size: ", str(len(val_df) / len(df)))

    #generator = torch.Generator().manual_seed(42)
    #train_dataset, val_dataset = torch.utils.data.random_split(ds, [train_size, val_size], generator=generator)

    # Create DataLoaders
    loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,  # No shuffle for validation
        num_workers=16,
        pin_memory=True,
        drop_last=False  # Don't drop last for validation
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(loader) * 10)

    for epoch in range(10):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        total = 0.0
        count = 0
        start_time = time.perf_counter()
        end_time = time.perf_counter()
        for index, (embryo_vol, _, _) in enumerate(pbar):
            optimizer.zero_grad()

            embryo_vol = embryo_vol.to(DEVICE)  # (1, T, 1, 500, 500)

            # Forward pass - returns (reconstruction, latent_seq)
            embryo_recon, embryo_lat = model(embryo_vol)
            if(index % 47 == 0):
                vol_img = embryo_vol[0, -1, 0].cpu().detach().numpy()
                recon_img = embryo_recon[0, -1, 0].cpu().detach().numpy()

                vol_img = (vol_img * 255).astype(np.uint8)
                recon_img = (recon_img * 255).astype(np.uint8)
                comparison = np.concatenate((vol_img, recon_img), axis=1)
     
                images = wandb.Image(comparison, caption="Embryo vs Recon comparison")
                run.log({"reconstruction": images})
                traj = embryo_lat.cpu().detach().numpy()[0]
                dist_matrix = distance_matrix(traj, traj)
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(dist_matrix, cmap='viridis')

                ax.set_xlabel("Time Index")
                ax.set_ylabel("Time Index")
                plt.colorbar(im, ax=ax)
                wandb.log({"temp_smoothness": wandb.Image(fig)})

                plt.close(fig)
            # Reconstruction loss using MS-SSIM + L1 or MSE (with configurable weights)
            if loss_type == "l1":
                rec_loss, rec_metrics = reconstruction_loss(
                    embryo_recon, embryo_vol, l1_weight=rec_weight, ms_ssim_weight=ms_ssim_weight
                )
            elif loss_type == "mse":
                # MS-SSIM + MSE loss
                B, T, C, H, W = embryo_recon.shape
                x_rec_flat = embryo_recon.view(B * T, C, H, W)
                x_true_flat = embryo_vol.view(B * T, C, H, W)

                mse_loss = F.mse_loss(embryo_recon, embryo_vol)
                ms_ssim_val = ms_ssim(x_rec_flat, x_true_flat)
                ms_ssim_loss = 1 - ms_ssim_val

                rec_loss = rec_weight * mse_loss + ms_ssim_weight * ms_ssim_loss
                rec_metrics = {
                    "mse_loss": mse_loss.item(),
                    "ms_ssim_loss": ms_ssim_loss.item(),
                    "ms_ssim_value": ms_ssim_val.item()
                }
            else:
                raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'l1' or 'mse'")

            if temporal_weight > 0:
                smooth_loss = temporal_smoothness_loss(embryo_lat, weight=temporal_weight)
                loss = rec_loss + smooth_loss
            else:
                smooth_loss = torch.tensor(0.0, device=DEVICE)
                loss = rec_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf detected, skipping batch")
                continue

            loss.backward()
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            if total_norm > 100:
                print(f"Warning: Large gradient norm: {total_norm:.2f}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scheduler.step()
            optimizer.step()
            end_time = time.perf_counter()

            
            total += loss.item()
            count += 1

            if (index % 50 == 0) and run is not None:
                log_dict = {
                    "step": epoch * len(loader) + index,
                    "loss": loss.item(),
                    "rec_loss": rec_loss.item(),
                    "smooth_loss": smooth_loss.item(),
                    "ms_ssim": rec_metrics["ms_ssim_value"],
                    "lr": scheduler.get_last_lr()[0]
                }

                # Add loss-specific metrics
                if loss_type == "l1":
                    log_dict["l1_loss"] = rec_metrics["l1_loss"]
                elif loss_type == "mse":
                    log_dict["mse_loss"] = rec_metrics["mse_loss"]

                run.log(log_dict)

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    rec=f"{rec_loss.item():.4f}",
                    smooth=f"{smooth_loss.item():.4f}"
                )


        duration = end_time - start_time
        print(f"Duration: {duration}")
        run.log({"epoch_time":duration})
        avg_loss = total/max(1, count)
        run.log({"avg_loss": avg_loss})
        print(f"epoch {epoch} avg loss={avg_loss:.4f}")

        # Save the state dict
        torch.save(model.state_dict(), "convlstm_model_weights.pth")

        # Generate unique repo name based on config and code
        date_label = datetime.now().strftime("%Y-%m-%d")

        # Collect all config for hashing
        config_for_hash = {
            "mode": "convlstm",
            "loss_type": loss_type,
            "ms_ssim_weight": ms_ssim_weight,
            "rec_weight": rec_weight,
            "temporal_weight": temporal_weight,
            "dropout_rate": dropout_rate,
            "use_convlstm": use_convlstm,
            "use_residual": use_residual,
            "use_batchnorm": use_batchnorm,
            "learning_rate": 2e-4,
            "encoder_hidden_dim": 256,
            "encoder_layers": 2,
            "decoder_hidden_dim": 128,
            "decoder_layers": 2,
            "latent_size": latent_size,
            "seq_len": 50,
            "image_size": 128,
        }

        # Required files for ConvLSTM model
        required_files = [
            "train.py",
            "raffael_model.py",
            "raffael_losses.py",
            "raffael_conv_lstm.py",
            "dataset_ivf.py",
            "train_model.sh",
            "training_config.txt",
            "training_config_detailed.txt",
        ]

        # Generate unique repo name
        repo_name = generate_repo_name("convlstm", config_for_hash, required_files, date_label)

        # Create comprehensive config for HuggingFace
        hf_config = {
            "model_type": "ConvLSTMAutoencoder",
            "architecture": "ConvLSTM Autoencoder",
            # Model architecture parameters
            "seq_len": 50,
            "input_channels": 1,
            "encoder_hidden_dim": 256,
            "encoder_layers": 2,
            "decoder_hidden_dim": 128,
            "decoder_layers": 2,
            "latent_size": latent_size,
            "use_classifier": False,
            "num_classes": 2,
            "use_latent_split": False,
            "image_size": 128,
            # Ablation parameters
            "dropout_rate": dropout_rate,
            "use_convlstm": use_convlstm,
            "use_residual": use_residual,
            "use_batchnorm": use_batchnorm,
            # Loss configuration
            "loss_type": loss_type,
            "ms_ssim_weight": ms_ssim_weight,
            "rec_weight": rec_weight,
            "temporal_weight": temporal_weight,
            "loss_description": loss_description,
            # Training configuration
            "learning_rate": 2e-4,
            "weight_decay": 1e-5,
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "batch_size": 1,
            "epochs": 10,
            "gradient_clip": 5.0,
            # Dataset
            "dataset": "https://zenodo.org/records/7912264",
            "resize": 128,
            "normalization": "minmax01",
            # Reproducibility
            "repo_name": repo_name,
            "date": date_label,
            "hash": repo_name.split("-")[-2] if "-" in repo_name else "",
        }

        save_and_push_model(model, model_name +"-"+ date_label, required_files, model_config=hf_config)
        val_metrics = {
            'mse': RunningStats(),
            'l1': RunningStats(),
            'ssim': RunningStats(),
            'temp': RunningStats()
        }
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for embryo_vol, _, _ in val_loader:
                embryo_vol = embryo_vol.to(DEVICE)  # (1, T, 1, H, W)
                val_recon, val_lat = model(embryo_vol)
                B, T, C, H, W = embryo_vol.shape

                # MSE
                val_metrics['mse'].push(F.mse_loss(val_recon, embryo_vol).item())

                # L1
                val_metrics['l1'].push(F.l1_loss(val_recon, embryo_vol).item())

                # MS-SSIM
                val_recon_flat = val_recon.view(B * T, C, H, W)
                embryo_vol_flat = embryo_vol.view(B * T, C, H, W)
                ms_ssim_val = ms_ssim(val_recon_flat, embryo_vol_flat)
                val_metrics['ssim'].push((1 - ms_ssim_val).item())

                # Temporal smoothness of latents
                # val_lat is (B, T, latent_size)
                if T > 1:
                    val_metrics['temp'].push(temporal_smoothness_loss(val_lat).item())
        # Log to wandb with val_ prefix
        val_log_dict = {
            f"val_{key}": value.mean for key, value in val_metrics.items()
        }
        val_log_std_dict = {
            f"val_{key}_std": value.std_dev for key, value in val_metrics.items()
        }

        run.log(val_log_dict)
        run.log(val_log_std_dict)
        

    run.finish()
    gc.collect()
    torch.cuda.empty_cache()


def train_convlstm_latent_split(
    loss_type="l1",
    ms_ssim_weight=0.5,
    rec_weight=0.5,
    temporal_weight=0.1,
    dropout_rate=0.1,
    use_convlstm=True,
    use_residual=True,
    use_batchnorm=True,
    model_name ="",
    latent_size = 4096
):
    gc.collect()
    """Training ConvLSTM Autoencoder with LATENT SPLIT enabled (single GPU)

    Args:
        loss_type: "l1" or "mse" - type of reconstruction loss to use with MS-SSIM
        ms_ssim_weight: float - weight for MS-SSIM loss (0 to disable)
        rec_weight: float - weight for reconstruction loss L1/MSE (0 to disable)
        temporal_weight: float - weight for temporal smoothness loss (0 to disable)
        dropout_rate: float - dropout rate (0 to disable)
        use_convlstm: bool - whether to use ConvLSTM (False = no temporal modeling)
        use_residual: bool - whether to use residual connections
        use_batchnorm: bool - whether to use batch normalization
    """
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build loss description for logging
    loss_components = []
    if ms_ssim_weight > 0:
        loss_components.append(f"MS-SSIM({ms_ssim_weight})")
    if rec_weight > 0:
        loss_components.append(f"{loss_type.upper()}({rec_weight})")
    if temporal_weight > 0:
        loss_components.append(f"Temporal({temporal_weight})")
    loss_description = " + ".join(loss_components) if loss_components else "None"

    # Build model description for logging
    model_features = []
    if use_convlstm:
        model_features.append("ConvLSTM")
    if use_residual:
        model_features.append("Residual")
    if use_batchnorm:
        model_features.append("BatchNorm")
    if dropout_rate > 0:
        model_features.append(f"Dropout({dropout_rate})")
    model_description = "+".join(model_features) if model_features else "Baseline"
    date_label = datetime.now().strftime("%Y-%m-%d")

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
        name=model_name + "-" + date_label,
        config={
            "learning_rate": 0.02,
            "architecture": "ConvLSTM Autoencoder with Latent Split",
            "model_features": model_description,
            "dataset": "https://zenodo.org/records/7912264",
            "epochs": 10,
            "train_split": 0.85,
            "val_split": 0.15,
            "loss": loss_description,
            "loss_type": loss_type,
            "ms_ssim_weight": ms_ssim_weight,
            "rec_weight": rec_weight,
            "temporal_weight": temporal_weight,
            "dropout_rate": dropout_rate,
            "use_convlstm": use_convlstm,
            "use_residual": use_residual,
            "use_batchnorm": use_batchnorm,
            "latent_size": latent_size,
            "latent_split": True,
            "embryo_latent_size": latent_size//2,
            "empty_latent_size": latent_size//2,
            "seq_len": 50,
            "image_size": 128,
            "distributed": False,
        },
    )

    login(os.getenv("HF_KEY"))
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    print(DEVICE)
    print(f"\n{'='*60}")
    print(f"ABLATION STUDY - Training Configuration")
    print(f"{'='*60}")
    print(f"\nLoss Configuration:")
    print(f"  Base Loss Type: {loss_type.upper()}")
    print(f"  MS-SSIM Weight: {ms_ssim_weight} {'(DISABLED)' if ms_ssim_weight == 0 else ''}")
    print(f"  Reconstruction Weight: {rec_weight} {'(DISABLED)' if rec_weight == 0 else ''}")
    print(f"  Temporal Smoothness Weight: {temporal_weight} {'(DISABLED)' if temporal_weight == 0 else ''}")
    print(f"  Combined Loss: {loss_description}")
    print(f"\nModel Architecture Configuration:")
    print(f"  ConvLSTM: {'ENABLED' if use_convlstm else 'DISABLED'}")
    print(f"  Residual Connections: {'ENABLED' if use_residual else 'DISABLED'}")
    print(f"  Batch Normalization: {'ENABLED' if use_batchnorm else 'DISABLED'}")
    print(f"  Dropout Rate: {dropout_rate} {'(DISABLED)' if dropout_rate == 0 else ''}")
    print(f"  Model Features: {model_description}")
    print(f"\nLatent Configuration:")
    print(f"  Latent Split: ENABLED (2048 for empty, 2048 for embryo)")
    print(f"{'='*60}\n")

    # Save detailed training configuration
    config_content = f"""ConvLSTM Autoencoder Training Configuration (LATENT SPLIT + ABLATION)
================================================================================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

ABLATION STUDY CONFIGURATION
================================================================================

"""

    with open("training_config_latent_split.txt", "w") as f:
        f.write(config_content)

    print("Configuration saved to training_config_latent_split.txt")

    # Create model with LATENT SPLIT and ABLATION parameters
    """model = ConvLSTMAutoencoder(
        None,
        seq_len=50,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        latent_size=latent_size,
        use_classifier=False,
        num_classes=2,
        use_latent_split=True,  # ENABLE LATENT SPLIT
        # Ablation parameters
        dropout_rate=dropout_rate,
        use_convlstm=use_convlstm,
        use_residual=use_residual,
        use_batchnorm=use_batchnorm
    """
    model = ConvLSTMAutoencoder(
        seq_len=50,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        latent_size=4096,
        use_classifier=True,
        num_classes=2
    )
    model = model.to(DEVICE)

    learning_rate = 2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    df = pd.read_csv(os.path.abspath("index.csv"))
    mask = df["cell_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    val_df = df[mask]
    train_df = df[~mask]
    train_dataset = IVFSequenceDataset(train_df, resize=128, norm="minmax01")
    val_dataset = IVFSequenceDataset(val_df, resize=128, norm="minmax01")
    print("val size: ", str(len(val_df) / len(df)))

    #generator = torch.Generator().manual_seed(42)
    #train_dataset, val_dataset = torch.utils.data.random_split(ds, [train_size, val_size], generator=generator)

    # Create DataLoaders
    loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,  # No shuffle for validation
        num_workers=4,
        pin_memory=True,
        drop_last=False  # Don't drop last for validation
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(loader) * 10)

    for epoch in range(10):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        total = 0.0
        count = 0

        for index, (embryo_vol, empty_well_vol, _) in enumerate(pbar):
            optimizer.zero_grad()

            # embryo_vol and empty_well_vol are (1, T, 1, H, W)
            embryo_vol = embryo_vol.to(DEVICE)
            empty_well_vol = empty_well_vol.to(DEVICE)

            # Forward pass for embryo (uses second half of latent: 2048:4096)
            embryo_recon, embryo_lat = model(embryo_vol, empty_well=False)

            # Forward pass for empty well (uses first half of latent: 0:2048)
            empty_recon, empty_lat = model(empty_well_vol, empty_well=True)
            if(index % 47 == 0):
                vol_img = embryo_vol[0, -1, 0].cpu().detach().numpy()
                recon_img = embryo_recon[0, -1, 0].cpu().detach().numpy()

                vol_img = (vol_img * 255).astype(np.uint8)
                recon_img = (recon_img * 255).astype(np.uint8)
                comparison = np.concatenate((vol_img, recon_img), axis=1)
     
                images = wandb.Image(comparison, caption="Embryo vs Recon comparison")
                run.log({"reconstruction": images})
            # Reconstruction loss for embryo (with configurable weights)
            if loss_type == "l1":
                rec_loss_embryo, rec_metrics_embryo = convlstm_reconstruction_loss(
                    embryo_recon, embryo_vol, l1_weight=rec_weight, ms_ssim_weight=ms_ssim_weight
                )
            elif loss_type == "mse":
                B, T, C, H, W = embryo_recon.shape
                x_rec_flat = embryo_recon.view(B * T, C, H, W)
                x_true_flat = embryo_vol.view(B * T, C, H, W)

                mse_loss = F.mse_loss(embryo_recon, embryo_vol)
                ms_ssim_val = ms_ssim(x_rec_flat, x_true_flat)
                ms_ssim_loss = 1 - ms_ssim_val

                rec_loss_embryo = rec_weight * mse_loss + ms_ssim_weight * ms_ssim_loss
                rec_metrics_embryo = {
                    "mse_loss": mse_loss.item(),
                    "ms_ssim_loss": ms_ssim_loss.item(),
                    "ms_ssim_value": ms_ssim_val.item()
                }
            else:
                raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'l1' or 'mse'")

            # Reconstruction loss for empty well (with configurable weights)
            if loss_type == "l1":
                rec_loss_empty, rec_metrics_empty = convlstm_reconstruction_loss(
                    empty_recon, empty_well_vol, l1_weight=rec_weight, ms_ssim_weight=ms_ssim_weight
                )
            elif loss_type == "mse":
                B, T, C, H, W = empty_recon.shape
                x_rec_flat = empty_recon.view(B * T, C, H, W)
                x_true_flat = empty_well_vol.view(B * T, C, H, W)

                mse_loss = F.mse_loss(empty_recon, empty_well_vol)
                ms_ssim_val = ms_ssim(x_rec_flat, x_true_flat)
                ms_ssim_loss = 1 - ms_ssim_val

                rec_loss_empty = rec_weight * mse_loss + ms_ssim_weight * ms_ssim_loss
                rec_metrics_empty = {
                    "mse_loss": mse_loss.item(),
                    "ms_ssim_loss": ms_ssim_loss.item(),
                    "ms_ssim_value": ms_ssim_val.item()
                }

            # Total reconstruction loss
            rec_loss = rec_loss_embryo + rec_loss_empty

            # Temporal smoothness loss (with configurable weight)
            if temporal_weight > 0:
                smooth_loss_embryo = temporal_smoothness_loss(embryo_lat, weight=temporal_weight)
                smooth_loss_empty = temporal_smoothness_loss(empty_lat, weight=temporal_weight)
                smooth_loss = smooth_loss_embryo + smooth_loss_empty
                loss = rec_loss + smooth_loss
            else:
                smooth_loss = torch.tensor(0.0, device=DEVICE)
                loss = rec_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf detected, skipping batch")
                continue

            loss.backward()
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            if total_norm > 100:
                print(f"Warning: Large gradient norm: {total_norm:.2f}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scheduler.step()
            optimizer.step()
            total += loss.item()
            count += 1

            if (index % 50 == 0) and run is not None:
                log_dict = {
                    "step": epoch * len(loader) + index,
                    "loss": loss.item(),
                    "rec_loss": rec_loss.item(),
                    "rec_loss_embryo": rec_loss_embryo.item(),
                    "rec_loss_empty": rec_loss_empty.item(),
                    "smooth_loss": smooth_loss.item(),
                    "ms_ssim_embryo": rec_metrics_embryo["ms_ssim_value"],
                    "ms_ssim_empty": rec_metrics_empty["ms_ssim_value"],
                    "lr": scheduler.get_last_lr()[0]
                }

                # Add loss-specific metrics
                if loss_type == "l1":
                    log_dict["l1_loss_embryo"] = rec_metrics_embryo["l1_loss"]
                    log_dict["l1_loss_empty"] = rec_metrics_empty["l1_loss"]
                elif loss_type == "mse":
                    log_dict["mse_loss_embryo"] = rec_metrics_embryo["mse_loss"]
                    log_dict["mse_loss_empty"] = rec_metrics_empty["mse_loss"]

                run.log(log_dict)

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    rec_e=f"{rec_loss_embryo.item():.4f}",
                    rec_empty=f"{rec_loss_empty.item():.4f}",
                    smooth=f"{smooth_loss.item():.4f}"
                )

        avg_loss = total/max(1, count)
        run.log({"avg_loss": avg_loss})
        print(f"epoch {epoch} avg loss={avg_loss:.4f}")

        # Save the state dict
        torch.save(model.state_dict(), "convlstm_latent_split_weights.pth")

        # Generate unique repo name based on config and code
        date_label = datetime.now().strftime("%Y-%m-%d")

        # Collect all config for hashing
        config_for_hash = {
            "mode": "convlstm_latent_split",
            "loss_type": loss_type,
            "ms_ssim_weight": ms_ssim_weight,
            "rec_weight": rec_weight,
            "temporal_weight": temporal_weight,
            "dropout_rate": dropout_rate,
            "use_convlstm": use_convlstm,
            "use_residual": use_residual,
            "use_batchnorm": use_batchnorm,
            "use_latent_split": True,
            "learning_rate": 2e-4,
            "encoder_hidden_dim": 256,
            "encoder_layers": 2,
            "decoder_hidden_dim": 128,
            "decoder_layers": 2,
            "latent_size": latent_size,
            "embryo_latent_size": 2048,
            "empty_latent_size": 2048,
            "seq_len": 50,
            "image_size": 128,
        }

        # Required files for ConvLSTM model with latent split
        required_files = [
            "train.py",
            "raffael_model.py",
            "raffael_losses.py",
            "raffael_conv_lstm.py",
            "dataset_ivf.py",
            "train_model.sh",
            "training_config.txt",
            "training_config_latent_split.txt",
        ]

        # Generate unique repo name
        repo_name = generate_repo_name("convlstm-ls", config_for_hash, required_files, date_label)

        # Create comprehensive config for HuggingFace
        hf_config = {
            "model_type": "ConvLSTMAutoencoder",
            "architecture": "ConvLSTM Autoencoder with Latent Split",
            # Model architecture parameters
            "seq_len": 50,
            "input_channels": 1,
            "encoder_hidden_dim": 256,
            "encoder_layers": 2,
            "decoder_hidden_dim": 128,
            "decoder_layers": 2,
            "latent_size": latent_size,
            "use_classifier": False,
            "num_classes": 2,
            "use_latent_split": True,
            "embryo_latent_size": 2048,
            "empty_latent_size": 2048,
            "image_size": 128,
            # Ablation parameters
            "dropout_rate": dropout_rate,
            "use_convlstm": use_convlstm,
            "use_residual": use_residual,
            "use_batchnorm": use_batchnorm,
            # Loss configuration
            "loss_type": loss_type,
            "ms_ssim_weight": ms_ssim_weight,
            "rec_weight": rec_weight,
            "temporal_weight": temporal_weight,
            "loss_description": loss_description,
            # Training configuration
            "learning_rate": 2e-4,
            "weight_decay": 1e-5,
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "batch_size": 1,
            "epochs": 10,
            "gradient_clip": 5.0,
            # Dataset
            "dataset": "https://zenodo.org/records/7912264",
            "resize": 128,
            "normalization": "minmax01",
            # Reproducibility
            "repo_name": repo_name,
            "date": date_label,
            "hash": repo_name.split("-")[-2] if "-" in repo_name else "",
        }

        save_and_push_model(model, model_name + "-" + date_label, required_files, model_config=hf_config)
        val_metrics = {
            'mse': RunningStats(),
            'l1': RunningStats(),
            'ssim': RunningStats(),
            'temp': RunningStats()
        }
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for embryo_vol, _, _ in val_loader:
                embryo_vol = embryo_vol.to(DEVICE)  # (1, T, 1, H, W)
                val_recon, val_lat = model(embryo_vol, empty_well=False)
                _, empty_val_lat = model(embryo_vol, empty_well=True)
                val_lat = torch.cat([val_lat, empty_val_lat], dim= 2)
                B, T, C, H, W = embryo_vol.shape

                # MSE
                val_metrics['mse'].push(F.mse_loss(val_recon, embryo_vol).item())

                # L1
                val_metrics['l1'].push(F.l1_loss(val_recon, embryo_vol).item())

                # MS-SSIM
                val_recon_flat = val_recon.view(B * T, C, H, W)
                embryo_vol_flat = embryo_vol.view(B * T, C, H, W)
                ms_ssim_val = ms_ssim(val_recon_flat, embryo_vol_flat)
                val_metrics['ssim'].push((1 - ms_ssim_val).item())

                # Temporal smoothness of latents
                # val_lat is (B, T, latent_size)
                if T > 1:
                    lat_diff = torch.diff(val_lat, dim=1)  # (B, T-1, latent_size)
                    temporal_smooth = lat_diff.norm(dim=-1).mean()  # Average L2 norm of differences
        # Log to wandb with val_ prefix
        val_log_dict = {
            f"val_{key}": value.mean for key, value in val_metrics.items()
        }
        val_log_std_dict = {
            f"val_{key}_std": value.std_dev for key, value in val_metrics.items()
        }

        run.log(val_log_dict)
        run.log(val_log_std_dict)
        
    run.finish()
    gc.collect()
    torch.cuda.empty_cache()

def train_mse_distributed():
    print("hi")
def train_mse_single():
    print("hi")
def train():
    print("hi")

if __name__ == "__main__":
    import sys
    import argparse

    # Check if using old command line interface
    if len(sys.argv) > 1 and sys.argv[1] in ["mse_distributed", "mse_single", "convlstm", "convlstm_latent_split"]:
        mode = sys.argv[1]
        if mode == "mse_distributed":
            train_mse_distributed()
        elif mode == "mse_single":
            train_mse_single()
        elif mode == "convlstm":
            # Parse additional convlstm arguments with ablation support
            parser = argparse.ArgumentParser(description="Train ConvLSTM Autoencoder with Ablation Studies")
            parser.add_argument("mode", type=str, help="Training mode")

            # Loss ablation arguments
            parser.add_argument("--loss-type", type=str, default="l1", choices=["l1", "mse"],
                              help="Reconstruction loss type: l1 or mse (default: l1)")
            parser.add_argument("--ms-ssim-weight", type=float, default=0.5,
                              help="Weight for MS-SSIM loss (default: 0.5, set to 0 to disable)")
            parser.add_argument("--rec-weight", type=float, default=0.5,
                              help="Weight for reconstruction loss (default: 0.5, set to 0 to disable)")
            parser.add_argument("--temporal-weight", type=float, default=0.1,
                              help="Weight for temporal smoothness loss (default: 0.1, set to 0 to disable)")

            # Model ablation arguments
            parser.add_argument("--dropout-rate", type=float, default=0.1,
                              help="Dropout rate (default: 0.1, set to 0 to disable)")
            parser.add_argument("--no-convlstm", action="store_true",
                              help="Disable ConvLSTM (no temporal modeling)")
            parser.add_argument("--no-residual", action="store_true",
                              help="Disable residual connections")
            parser.add_argument("--no-batchnorm", action="store_true",
                              help="Disable batch normalization")
            parser.add_argument("--name", type=str, default="", help="model name duhh")
            parser.add_argument("--size", type=int, default=4096, help="lat size bruh")
            args = parser.parse_args()

            train_convlstm(
                loss_type=args.loss_type,
                ms_ssim_weight=args.ms_ssim_weight,
                rec_weight=args.rec_weight,
                temporal_weight=args.temporal_weight,
                dropout_rate=args.dropout_rate,
                use_convlstm=not args.no_convlstm,
                use_residual=not args.no_residual,
                use_batchnorm=not args.no_batchnorm,
                model_name = args.name,
                latent_size = args.size

            )
        elif mode == "convlstm_latent_split":
            # Parse additional convlstm_latent_split arguments with ablation support
            parser = argparse.ArgumentParser(description="Train ConvLSTM Autoencoder with Latent Split and Ablation Studies")
            parser.add_argument("mode", type=str, help="Training mode")

            # Loss ablation arguments
            parser.add_argument("--loss-type", type=str, default="l1", choices=["l1", "mse"],
                              help="Reconstruction loss type: l1 or mse (default: l1)")
            parser.add_argument("--ms-ssim-weight", type=float, default=0.5,
                              help="Weight for MS-SSIM loss (default: 0.5, set to 0 to disable)")
            parser.add_argument("--rec-weight", type=float, default=0.5,
                              help="Weight for reconstruction loss (default: 0.5, set to 0 to disable)")
            parser.add_argument("--temporal-weight", type=float, default=0.1,
                              help="Weight for temporal smoothness loss (default: 0.1, set to 0 to disable)")

            # Model ablation arguments
            parser.add_argument("--dropout-rate", type=float, default=0.1,
                              help="Dropout rate (default: 0.1, set to 0 to disable)")
            parser.add_argument("--no-convlstm", action="store_true",
                              help="Disable ConvLSTM (no temporal modeling)")
            parser.add_argument("--no-residual", action="store_true",
                              help="Disable residual connections")
            parser.add_argument("--no-batchnorm", action="store_true",
                              help="Disable batch normalization")

            parser.add_argument("--name", type=str, default="", help="model name duhh")
            
            parser.add_argument("--size", type=int, default=4096, help="lat size bruh")
            args = parser.parse_args()

            train_convlstm_latent_split(
                loss_type=args.loss_type,
                ms_ssim_weight=args.ms_ssim_weight,
                rec_weight=args.rec_weight,
                temporal_weight=args.temporal_weight,
                dropout_rate=args.dropout_rate,
                use_convlstm=not args.no_convlstm,
                use_residual=not args.no_residual,
                use_batchnorm=not args.no_batchnorm,
                model_name = args.name,
                latent_size = args.size
            )
    else:
        train()

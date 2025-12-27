import numpy as np
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
from raffael_losses import reconstruction_loss as convlstm_reconstruction_loss, temporal_smoothness_loss
import sys

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
from huggingface_hub import login
import shutil
import hashlib
import json
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
                repo_id=f"jenslundsgaard7-uw-madison/{repo_name}",
                repo_type="model"
            )
            print(f"Uploaded config.json to HuggingFace Hub")
        except Exception as e:
            print(f"Warning: Failed to upload config.json: {e}")

    # Upload model weights if they exist
    model_file = os.path.join(repo_name, "pytorch_model.bin")
    if os.path.exists(model_file):
        try:
            api.upload_file(
                path_or_fileobj=model_file,
                path_in_repo="pytorch_model.bin",
                repo_id=f"jenslundsgaard7-uw-madison/{repo_name}",
                repo_type="model"
            )
            print(f"Uploaded pytorch_model.bin to HuggingFace Hub")
        except Exception as e:
            print(f"Warning: Failed to upload pytorch_model.bin: {e}")

    # Upload additional required files
    for file_path in required_files:
        local_file = os.path.join(repo_name, os.path.basename(file_path))
        if os.path.exists(local_file):
            try:
                api.upload_file(
                    path_or_fileobj=local_file,
                    path_in_repo=os.path.basename(file_path),
                    repo_id=f"jenslundsgaard7-uw-madison/{repo_name}",
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


def train():
    """Original training with MS-SSIM loss and distributed training"""
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    rank, world_size, local_rank = setup_distributed()
    DEVICE = torch.device(f"cuda:{local_rank}")

    # Only print on main process
    is_main = rank == 0
    run = None
    # Initialize wandb only on main process
    if is_main:
        wandb.login(key=os.getenv("WANDB_KEY"))
        run = wandb.init(
            entity="jenslundsgaard7-uw-madison",
            project="IVF-Training",
            config={
                "learning_rate": 0.005,
                "architecture": "Conv LSTM Autoencoder",
                "dataset": "https://zenodo.org/records/7912264",
                "epochs": 10,
                "world_size": world_size,
                "loss": "MS-SSIM + L1",
            },
        )

        login(os.getenv("HF_KEY"))
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        print(DEVICE)

    model = Model()
    if os.path.exists("model_weights.pth"):
        try:
            checkpoint = torch.load("model_weights.pth", map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint)
            if is_main:
                print("Loaded model weights")
        except Exception as e:
            if is_main:
                print(f"Error loading weights: {e}")
                torch.save(model.state_dict(), "model_weights.pth")
    else:
        if is_main:
            torch.save(model.state_dict(), "model_weights.pth")
    model = model.to(DEVICE)

    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    learning_rate = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay = 1e-5 )

    ds = IVFSequenceDataset(os.path.abspath("index.csv"), resize=500, norm="minmax01")

    if world_size > 1:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    num_workers = max(4, 16 // world_size)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Important for DDP
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, steps_per_epoch=len(loader), epochs=10
    )
    for epoch in range(10):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        if sampler is not None:
            sampler.set_epoch(epoch)

        if is_main:
            pbar = tqdm(loader, desc=f"epoch {epoch}")
        else:
            pbar = loader
        total = 0.0
        count = 0
        for index, (embryo_vol, empty_well_vol, sample_vol) in enumerate(pbar):

            optimizer.zero_grad()

            embryo_vol = embryo_vol.view(-1,1,500,500)
            empty_well_vol = empty_well_vol.view(-1,1,500,500)
            sample_vol = sample_vol.view(-1,1,500,500)

            embryo_size = embryo_vol.shape[0]
            embryo_size = embryo_vol.shape[0]
            sample_size = sample_vol.shape[0]

            vol = torch.cat((embryo_vol, sample_vol), 0).to(DEVICE)
            empty_well_vol = empty_well_vol.to(DEVICE)


            recon, _ = model(vol, empty_well = False)
            empty_well_recon, _ = model(empty_well_vol, empty_well=True)

            # Use SSIM-based reconstruction loss
            # Reshape for reconstruction_loss function: (B, T, C, H, W)
            rec_loss_vol, rec_metrics_vol = reconstruction_loss(
                recon.unsqueeze(1), vol.unsqueeze(1), l1_weight=0.5, ms_ssim_weight=0.5
            )
            rec_loss_empty, rec_metrics_empty = reconstruction_loss(
                empty_well_recon.unsqueeze(1), empty_well_vol.unsqueeze(1), l1_weight=0.5, ms_ssim_weight=0.5
            )
            rec_loss = rec_loss_vol + rec_loss_empty


            _, lat = model(vol, empty_well=False)

            embryo_lat = lat[:embryo_size].clone()
            sample_lat = lat[embryo_size:].clone()

            embryo_lat1 = torch.cat((embryo_lat[1:], embryo_lat[5:], embryo_lat[10:], embryo_lat[20:]), 0).to(DEVICE)
            embryo_lat2 = torch.cat((embryo_lat[:-1], embryo_lat[:-5], embryo_lat[:-10], embryo_lat[:-20]), 0).to(DEVICE)
            temp_adj_sim = F.cosine_similarity(embryo_lat1, embryo_lat2).mean()
            rand_sample_sim = F.cosine_similarity(embryo_lat, sample_lat).mean()
            tcl = rand_sample_sim - temp_adj_sim
            loss = rec_loss + (0.03 * tcl)

            if torch.isnan(loss) or torch.isinf(loss):
                if is_main:
                    print(f"NaN/Inf detected, skipping batch")
                continue

            loss.backward()
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            if total_norm > 100:  # Warning threshold
                print(f"Warning: Large gradient norm: {total_norm:.2f}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scheduler.step()
            optimizer.step()
            total += loss.item()
            count += 1
            if is_main and (index % 50 == 0) and run is not None:
                run.log({
                    "step": epoch * len(loader) + index,
                    "loss": loss.item(),
                    "rec_loss": rec_loss.item(),
                    "tcl": tcl.item(),
                    "temp_adj_sim": temp_adj_sim.item(),
                    "rand_sample_sim": rand_sample_sim.item(),
                    "ms_ssim_vol": rec_metrics_vol["ms_ssim_value"],
                    "ms_ssim_empty": rec_metrics_empty["ms_ssim_value"],
                    "l1_loss_vol": rec_metrics_vol["l1_loss"],
                    "l1_loss_empty": rec_metrics_empty["l1_loss"],
                    "lr": scheduler.get_last_lr()[0]
                })

                if isinstance(pbar, tqdm):
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        rec=f"{rec_loss.item():.4f}",
                        tcl=f"{tcl.item():.4f}"
                    )
            del embryo_vol
            del empty_well_vol
            del sample_vol
            del vol
            del recon
            del lat
            del empty_well_recon
            del embryo_lat
            del sample_lat
            del embryo_lat1
            del embryo_lat2
            del rec_metrics_vol
            del rec_metrics_empty
            torch.cuda.empty_cache()

        avg_loss = total/max(1,count)
        if world_size > 1:
            avg_loss_tensor = torch.tensor([avg_loss], device=DEVICE)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = avg_loss_tensor.item()
        if is_main:
            run.log({"avg_loss": avg_loss})
            print(f"epoch {epoch} avg loss={avg_loss}:.4f")
            model_to_save = model.module if hasattr(model, 'module') else model
            # Save the state dict without moving the model to CPU
            torch.save(model_to_save.state_dict(), "model_weights.pth")

            # Create a temporary CPU copy for HuggingFace upload
            model_cpu = type(model_to_save)()  # Create new instance
            model_cpu.load_state_dict(model_to_save.state_dict())

            # Save with descriptive name
            date_label = datetime.now().strftime("%Y-%m-%d")
            repo_name = f"embryo-vision-msssim-{date_label}"

            # Required files for this model
            required_files = [
                "train.py",
                "model.py",
                "dataset_ivf.py",
            ]

            # Create config for HuggingFace
            hf_config = {
                "model_type": "Model",
                "architecture": "Conv LSTM Autoencoder",
                "loss": "MS-SSIM + L1",
                "learning_rate": 0.005,
                "optimizer": "Adam",
                "epochs": 10,
                "dataset": "https://zenodo.org/records/7912264",
                "l1_weight": 0.5,
                "ms_ssim_weight": 0.5,
                "tcl_weight": 0.03,
                "repo_name": repo_name,
                "date": date_label,
            }

            save_and_push_model(model_cpu, repo_name, required_files, model_config=hf_config)
            del model_cpu  # Clean up the CPU copy

    if is_main:
        run.finish()
    cleanup_distributed()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def train_mse_distributed():
    """Training with MSE loss and distributed training"""
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    rank, world_size, local_rank = setup_distributed()
    DEVICE = torch.device(f"cuda:{local_rank}")

    is_main = rank == 0
    run = None
    if is_main:
        wandb.login(key=os.getenv("WANDB_KEY"))
        run = wandb.init(
            entity="jenslundsgaard7-uw-madison",
            project="IVF-Training",
            config={
                "learning_rate": 0.005,
                "architecture": "Conv LSTM Autoencoder",
                "dataset": "https://zenodo.org/records/7912264",
                "epochs": 10,
                "world_size": world_size,
                "loss": "MSE",
            },
        )

        login(os.getenv("HF_KEY"))
        print(torch.cuda.memory_summary(device=None, abbreviated=False))
        print(DEVICE)

    model = Model()
    if os.path.exists("model_weights.pth"):
        try:
            checkpoint = torch.load("model_weights.pth", map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint)
            if is_main:
                print("Loaded model weights")
        except Exception as e:
            if is_main:
                print(f"Error loading weights: {e}")
    model = model.to(DEVICE)

    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    learning_rate = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    ds = IVFSequenceDataset(os.path.abspath("index.csv"), resize=500, norm="minmax01")

    if world_size > 1:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    num_workers = max(4, 16 // world_size)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, steps_per_epoch=len(loader), epochs=10
    )
    for epoch in range(10):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)

        if is_main:
            pbar = tqdm(loader, desc=f"epoch {epoch}")
        else:
            pbar = loader
        total = 0.0
        count = 0
        for index, (embryo_vol, empty_well_vol, sample_vol) in enumerate(pbar):

            optimizer.zero_grad()

            embryo_vol = embryo_vol.view(-1,1,500,500)
            empty_well_vol = empty_well_vol.view(-1,1,500,500)
            sample_vol = sample_vol.view(-1,1,500,500)

            embryo_size = embryo_vol.shape[0]
            sample_size = sample_vol.shape[0]

            vol = torch.cat((embryo_vol, sample_vol), 0).to(DEVICE)
            empty_well_vol = empty_well_vol.to(DEVICE)

            recon, _ = model(vol, empty_well=False)
            empty_well_recon, _ = model(empty_well_vol, empty_well=True)

            # MSE-based reconstruction loss
            rec_loss_vol = F.mse_loss(recon, vol)
            rec_loss_empty = F.mse_loss(empty_well_recon, empty_well_vol)
            rec_loss = rec_loss_vol + rec_loss_empty

            _, lat = model(vol, empty_well=False)

            embryo_lat = lat[:embryo_size].clone()
            sample_lat = lat[embryo_size:].clone()

            embryo_lat1 = torch.cat((embryo_lat[1:], embryo_lat[5:], embryo_lat[10:], embryo_lat[20:]), 0).to(DEVICE)
            embryo_lat2 = torch.cat((embryo_lat[:-1], embryo_lat[:-5], embryo_lat[:-10], embryo_lat[:-20]), 0).to(DEVICE)
            temp_adj_sim = F.cosine_similarity(embryo_lat1, embryo_lat2).mean()
            rand_sample_sim = F.cosine_similarity(embryo_lat, sample_lat).mean()
            tcl = rand_sample_sim - temp_adj_sim
            loss = rec_loss + (0.03 * tcl)

            if torch.isnan(loss) or torch.isinf(loss):
                if is_main:
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
            if is_main and (index % 50 == 0) and run is not None:
                run.log({
                    "step": epoch * len(loader) + index,
                    "loss": loss.item(),
                    "rec_loss": rec_loss.item(),
                    "tcl": tcl.item(),
                    "temp_adj_sim": temp_adj_sim.item(),
                    "rand_sample_sim": rand_sample_sim.item(),
                    "lr": scheduler.get_last_lr()[0]
                })

                if isinstance(pbar, tqdm):
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        rec=f"{rec_loss.item():.4f}",
                        tcl=f"{tcl.item():.4f}"
                    )
            del embryo_vol
            del empty_well_vol
            del sample_vol
            del vol
            del recon
            del lat
            del empty_well_recon
            del embryo_lat
            del sample_lat
            del embryo_lat1
            del embryo_lat2
            torch.cuda.empty_cache()

        avg_loss = total/max(1,count)
        if world_size > 1:
            avg_loss_tensor = torch.tensor([avg_loss], device=DEVICE)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = avg_loss_tensor.item()
        if is_main:
            run.log({"avg_loss": avg_loss})
            print(f"epoch {epoch} avg loss={avg_loss}:.4f")
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), "model_weights_mse.pth")

            # Save to HuggingFace with descriptive name
            model_cpu = type(model_to_save)()
            model_cpu.load_state_dict(model_to_save.state_dict())
            date_label = datetime.now().strftime("%Y-%m-%d")
            repo_name = f"embryo-vision-mse-distributed-{date_label}"

            # Required files for this model
            required_files = [
                "train.py",
                "model.py",
                "dataset_ivf.py",
            ]

            # Create config for HuggingFace
            hf_config = {
                "model_type": "Model",
                "architecture": "Conv LSTM Autoencoder",
                "loss": "MSE",
                "learning_rate": 0.005,
                "optimizer": "Adam",
                "epochs": 10,
                "world_size": world_size,
                "dataset": "https://zenodo.org/records/7912264",
                "tcl_weight": 0.03,
                "distributed": True,
                "repo_name": repo_name,
                "date": date_label,
            }

            save_and_push_model(model_cpu, repo_name, required_files, model_config=hf_config)
            del model_cpu

    if is_main:
        run.finish()
    cleanup_distributed()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def train_mse_single():
    """Training with MSE loss without distributed training"""
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
        config={
            "learning_rate": 0.005,
            "architecture": "Conv LSTM Autoencoder",
            "dataset": "https://zenodo.org/records/7912264",
            "epochs": 10,
            "world_size": 1,
            "loss": "MSE",
            "distributed": False,
        },
    )

    login(os.getenv("HF_KEY"))
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    print(DEVICE)

    model = Model()
    if os.path.exists("model_weights.pth"):
        try:
            checkpoint = torch.load("model_weights.pth", map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint)
            print("Loaded model weights")
        except Exception as e:
            print(f"Error loading weights: {e}")
    model = model.to(DEVICE)

    learning_rate = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    ds = IVFSequenceDataset(os.path.abspath("index.csv"), resize=500, norm="minmax01")

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, steps_per_epoch=len(loader), epochs=10
    )
    for epoch in range(10):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        total = 0.0
        count = 0
        for index, (embryo_vol, empty_well_vol, sample_vol) in enumerate(pbar):

            optimizer.zero_grad()

            embryo_vol = embryo_vol.view(-1,1,500,500)
            empty_well_vol = empty_well_vol.view(-1,1,500,500)
            sample_vol = sample_vol.view(-1,1,500,500)

            embryo_size = embryo_vol.shape[0]
            sample_size = sample_vol.shape[0]

            vol = torch.cat((embryo_vol, sample_vol), 0).to(DEVICE)
            empty_well_vol = empty_well_vol.to(DEVICE)

            recon, _ = model(vol, empty_well=False)
            empty_well_recon, _ = model(empty_well_vol, empty_well=True)

            # MSE-based reconstruction loss
            rec_loss_vol = F.mse_loss(recon, vol)
            rec_loss_empty = F.mse_loss(empty_well_recon, empty_well_vol)
            rec_loss = rec_loss_vol + rec_loss_empty

            _, lat = model(vol, empty_well=False)

            embryo_lat = lat[:embryo_size].clone()
            sample_lat = lat[embryo_size:].clone()

            embryo_lat1 = torch.cat((embryo_lat[1:], embryo_lat[5:], embryo_lat[10:], embryo_lat[20:]), 0).to(DEVICE)
            embryo_lat2 = torch.cat((embryo_lat[:-1], embryo_lat[:-5], embryo_lat[:-10], embryo_lat[:-20]), 0).to(DEVICE)
            temp_adj_sim = F.cosine_similarity(embryo_lat1, embryo_lat2).mean()
            rand_sample_sim = F.cosine_similarity(embryo_lat, sample_lat).mean()
            tcl = rand_sample_sim - temp_adj_sim
            loss = rec_loss + (0.05 * tcl)

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
                run.log({
                    "step": epoch * len(loader) + index,
                    "loss": loss.item(),
                    "rec_loss": rec_loss.item(),
                    "tcl": tcl.item(),
                    "temp_adj_sim": temp_adj_sim.item(),
                    "rand_sample_sim": rand_sample_sim.item(),
                    "lr": scheduler.get_last_lr()[0]
                })

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    rec=f"{rec_loss.item():.4f}",
                    tcl=f"{tcl.item():.4f}"
                )

        avg_loss = total/max(1,count)
        run.log({"avg_loss": avg_loss})
        print(f"epoch {epoch} avg loss={avg_loss}:.4f")
        torch.save(model.state_dict(), "model_weights_mse_single.pth")

        # Save to HuggingFace with descriptive name
        date_label = datetime.now().strftime("%Y-%m-%d")
        repo_name = f"embryo-vision-mse-{date_label}"

        # Required files for this model
        required_files = [
            "train.py",
            "model.py",
            "dataset_ivf.py",
        ]

        # Create config for HuggingFace
        hf_config = {
            "model_type": "Model",
            "architecture": "Conv LSTM Autoencoder",
            "loss": "MSE",
            "learning_rate": 0.005,
            "optimizer": "Adam",
            "epochs": 10,
            "dataset": "https://zenodo.org/records/7912264",
            "tcl_weight": 0.05,
            "distributed": False,
            "repo_name": repo_name,
            "date": date_label,
        }

        save_and_push_model(model, repo_name, required_files, model_config=hf_config)

    run.finish()
    gc.collect()
    torch.cuda.empty_cache()


def train_convlstm(
    loss_type="l1",
    ms_ssim_weight=0.5,
    rec_weight=0.5,
    temporal_weight=0.1,
    dropout_rate=0.1,
    use_convlstm=True,
    use_residual=True,
    use_batchnorm=True
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

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
        config={
            "learning_rate": 0.02,
            "architecture": "ConvLSTM Autoencoder",
            "model_features": model_description,
            "dataset": "https://zenodo.org/records/7912264",
            "epochs": 10,
            "loss": loss_description,
            "loss_type": loss_type,
            "ms_ssim_weight": ms_ssim_weight,
            "rec_weight": rec_weight,
            "temporal_weight": temporal_weight,
            "dropout_rate": dropout_rate,
            "use_convlstm": use_convlstm,
            "use_residual": use_residual,
            "use_batchnorm": use_batchnorm,
            "latent_size": 4096,
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

Loss Configuration:
  - Base Loss Type: {loss_type.upper()}
  - MS-SSIM Weight: {ms_ssim_weight} {'(DISABLED)' if ms_ssim_weight == 0 else ''}
  - Reconstruction Weight: {rec_weight} {'(DISABLED)' if rec_weight == 0 else ''}
  - Temporal Smoothness Weight: {temporal_weight} {'(DISABLED)' if temporal_weight == 0 else ''}
  - Combined Loss Function: {loss_description}

Model Architecture Ablations:
  - ConvLSTM: {'ENABLED' if use_convlstm else 'DISABLED'}
  - Residual Connections: {'ENABLED' if use_residual else 'DISABLED'}
  - Batch Normalization: {'ENABLED' if use_batchnorm else 'DISABLED'}
  - Dropout Rate: {dropout_rate} {'(DISABLED)' if dropout_rate == 0 else ''}
  - Model Features: {model_description}

Model Architecture:
  - Architecture: ConvLSTM Autoencoder
  - Sequence Length: 50
  - Input Channels: 1
  - Encoder Hidden Dim: 256
  - Encoder Layers: 2
  - Decoder Hidden Dim: 128
  - Decoder Layers: 2
  - Latent Size: 4096
  - Use Classifier: False
  - Image Size: 128x128

Training Configuration:
  - Learning Rate: 2e-4 (CosineAnnealingLR)
  - Weight Decay: 1e-5
  - Optimizer: Adam
  - Batch Size: 1
  - Epochs: 10
  - Gradient Clipping: 5.0
  - Device: {DEVICE}

Dataset:
  - Dataset: https://zenodo.org/records/7912264
  - Resize: 128x128
  - Normalization: minmax01

Model Files:
  - raffael_model.py (with ablation support)
  - raffael_conv_lstm.py (ConvLSTM implementation)
  - raffael_losses.py (Loss functions)

Reproducibility:
  - All ablation settings are logged in wandb config
  - Model stores ablation parameters for inference
  - Configuration saved to training_config_detailed.txt
"""

    with open("training_config_detailed.txt", "w") as f:
        f.write(config_content)

    print("Configuration saved to training_config_detailed.txt")

    model = ConvLSTMAutoencoder(
        seq_len=50,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        latent_size=4096,
        use_classifier=False,
        num_classes=2,
        use_latent_split=False,
        # Ablation parameters
        dropout_rate=dropout_rate,
        use_convlstm=use_convlstm,
        use_residual=use_residual,
        use_batchnorm=use_batchnorm
    )

    if os.path.exists("convlstm_model_weights.pth"):
        try:
            checkpoint = torch.load("convlstm_model_weights.pth", map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint)
            print("Loaded ConvLSTM model weights")
        except Exception as e:
            print(f"Error loading weights: {e}")

    model = model.to(DEVICE)

    learning_rate = 2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)


    ds = IVFSequenceDataset(os.path.abspath("index.csv"), resize=128, norm="minmax01")
    total_size = len(ds)

    train_size = int(0.95 * total_size)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(ds, [train_size, val_size], generator=generator)

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
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(loader) * 10)

    for epoch in range(10):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        total = 0.0
        count = 0

        for index, (embryo_vol, _, _) in enumerate(pbar):
            optimizer.zero_grad()

            embryo_vol = embryo_vol.to(DEVICE)  # (1, T, 1, 500, 500)

            # Forward pass - returns (reconstruction, latent_seq)
            embryo_recon, embryo_lat = model(embryo_vol)

            # Reconstruction loss using MS-SSIM + L1 or MSE (with configurable weights)
            if loss_type == "l1":
                rec_loss, rec_metrics = convlstm_reconstruction_loss(
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

            # Temporal smoothness loss (with configurable weight)
            # embryo_lat is (1, T, 4096) - encourages smooth transitions between frames
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
            "latent_size": 4096,
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
        print(f"Repository name: {repo_name} (length: {len(repo_name)})")

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
            "latent_size": 4096,
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

        save_and_push_model(model, repo_name, required_files, model_config=hf_config)
        val_loss = 0
        val_count = 0
        model.eval() # Set model to evaluation mode
        with torch.no_grad():
            for embryo_vol, _, _ in val_loader:
                embryo_vol = embryo_vol.to(DEVICE)
                val_recon, _ = model(embryo_vol)
                val_recon = val_recon.to(DEVICE)
                val_loss += torch.nn.functional.mse_loss(embryo_vol, val_recon).item()
                val_count += 1
        run.log({"val_mse": val_loss/val_count})

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
    use_batchnorm=True
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

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
        config={
            "learning_rate": 0.02,
            "architecture": "ConvLSTM Autoencoder with Latent Split",
            "model_features": model_description,
            "dataset": "https://zenodo.org/records/7912264",
            "epochs": 10,
            "loss": loss_description,
            "loss_type": loss_type,
            "ms_ssim_weight": ms_ssim_weight,
            "rec_weight": rec_weight,
            "temporal_weight": temporal_weight,
            "dropout_rate": dropout_rate,
            "use_convlstm": use_convlstm,
            "use_residual": use_residual,
            "use_batchnorm": use_batchnorm,
            "latent_size": 4096,
            "latent_split": True,
            "embryo_latent_size": 2048,
            "empty_latent_size": 2048,
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

Loss Configuration:
  - Base Loss Type: {loss_type.upper()}
  - MS-SSIM Weight: {ms_ssim_weight} {'(DISABLED)' if ms_ssim_weight == 0 else ''}
  - Reconstruction Weight: {rec_weight} {'(DISABLED)' if rec_weight == 0 else ''}
  - Temporal Smoothness Weight: {temporal_weight} {'(DISABLED)' if temporal_weight == 0 else ''}
  - Combined Loss Function: {loss_description}

Model Architecture Ablations:
  - ConvLSTM: {'ENABLED' if use_convlstm else 'DISABLED'}
  - Residual Connections: {'ENABLED' if use_residual else 'DISABLED'}
  - Batch Normalization: {'ENABLED' if use_batchnorm else 'DISABLED'}
  - Dropout Rate: {dropout_rate} {'(DISABLED)' if dropout_rate == 0 else ''}
  - Model Features: {model_description}

Model Architecture:
  - Architecture: ConvLSTM Autoencoder with LATENT SPLIT
  - Sequence Length: 50
  - Input Channels: 1
  - Encoder Hidden Dim: 256
  - Encoder Layers: 2
  - Decoder Hidden Dim: 128
  - Decoder Layers: 2
  - Total Latent Size: 4096
  - LATENT SPLIT ENABLED:
    * Empty Well Latent: 2048 (first half)
    * Embryo Latent: 2048 (second half)
  - Use Classifier: False
  - Image Size: 128x128

Training Configuration:
  - Learning Rate: 2e-4 (CosineAnnealingLR)
  - Weight Decay: 1e-5
  - Optimizer: Adam
  - Batch Size: 1
  - Epochs: 10
  - Gradient Clipping: 5.0
  - Device: {DEVICE}

Dataset:
  - Dataset: https://zenodo.org/records/7912264
  - Resize: 128x128
  - Normalization: minmax01

Model Files:
  - raffael_model.py (with ablation support)
  - raffael_conv_lstm.py (ConvLSTM implementation)
  - raffael_losses.py (Loss functions)

Reproducibility:
  - All ablation settings are logged in wandb config
  - Model stores ablation parameters for inference
  - Configuration saved to training_config_latent_split.txt
"""

    with open("training_config_latent_split.txt", "w") as f:
        f.write(config_content)

    print("Configuration saved to training_config_latent_split.txt")

    # Create model with LATENT SPLIT and ABLATION parameters
    model = ConvLSTMAutoencoder(
        seq_len=50,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        latent_size=4096,
        use_classifier=False,
        num_classes=2,
        use_latent_split=True,  # ENABLE LATENT SPLIT
        # Ablation parameters
        dropout_rate=dropout_rate,
        use_convlstm=use_convlstm,
        use_residual=use_residual,
        use_batchnorm=use_batchnorm
    )

    if os.path.exists("convlstm_latent_split_weights.pth"):
        try:
            checkpoint = torch.load("convlstm_latent_split_weights.pth", map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint)
            print("Loaded ConvLSTM model weights with latent split")
        except Exception as e:
            print(f"Error loading weights: {e}")

    model = model.to(DEVICE)

    learning_rate = 2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    ds = IVFSequenceDataset(os.path.abspath("index.csv"), resize=128, norm="minmax01")
    total_size = len(ds)

    train_size = int(0.95 * total_size)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(ds, [train_size, val_size], generator=generator)

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
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
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
            "latent_size": 4096,
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
        print(f"Repository name: {repo_name} (length: {len(repo_name)})")

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
            "latent_size": 4096,
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

        save_and_push_model(model, repo_name, required_files, model_config=hf_config)
        val_loss = 0
        val_count = 0
        model.eval() # Set model to evaluation mode
        with torch.no_grad():
            for embryo_vol, _, _ in val_loader:
                embryo_vol = embryo_vol.to(DEVICE)
                val_recon, _ = model(embryo_vol)
                val_recon = val_recon.to(DEVICE)
                val_loss += torch.nn.functional.mse_loss(embryo_vol, val_recon).item()
                val_count += 1
        run.log({"val_mse": val_loss/val_count})
            

    run.finish()
    gc.collect()
    torch.cuda.empty_cache()


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

            args = parser.parse_args()

            train_convlstm(
                loss_type=args.loss_type,
                ms_ssim_weight=args.ms_ssim_weight,
                rec_weight=args.rec_weight,
                temporal_weight=args.temporal_weight,
                dropout_rate=args.dropout_rate,
                use_convlstm=not args.no_convlstm,
                use_residual=not args.no_residual,
                use_batchnorm=not args.no_batchnorm
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

            args = parser.parse_args()

            train_convlstm_latent_split(
                loss_type=args.loss_type,
                ms_ssim_weight=args.ms_ssim_weight,
                rec_weight=args.rec_weight,
                temporal_weight=args.temporal_weight,
                dropout_rate=args.dropout_rate,
                use_convlstm=not args.no_convlstm,
                use_residual=not args.no_residual,
                use_batchnorm=not args.no_batchnorm
            )
    else:
        train()

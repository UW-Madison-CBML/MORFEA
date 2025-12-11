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
from raffael_losses import reconstruction_loss as convlstm_reconstruction_loss
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
                model_cpu = model.cpu()
                model_cpu.save_pretrained("IVF-Model")
                model_cpu.push_to_hub("IVF-Model")
        except Exception as e:
            if is_main:
                print(f"Error loading weights: {e}")
                torch.save(model.state_dict(), "model_weights.pth")
                model_cpu = model.cpu()
                model_cpu.save_pretrained("IVF-Model")
                model_cpu.push_to_hub("IVF-Model")
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

            # Save with date label
            date_label = datetime.now().strftime("%Y-%m-%d")
            repo_name = f"IVF-Model-{date_label}"
            model_cpu.save_pretrained(repo_name)
            model_cpu.push_to_hub(repo_name)
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

            # Save to HuggingFace with date label
            model_cpu = type(model_to_save)()
            model_cpu.load_state_dict(model_to_save.state_dict())
            date_label = datetime.now().strftime("%Y-%m-%d")
            repo_name = f"IVF-Model-MSE-{date_label}"
            model_cpu.save_pretrained(repo_name)
            model_cpu.push_to_hub(repo_name)
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

        # Save to HuggingFace with date label
        date_label = datetime.now().strftime("%Y-%m-%d")
        repo_name = f"IVF-Model-MSE-Single-{date_label}"
        model.save_pretrained(repo_name)
        model.push_to_hub(repo_name)

    run.finish()
    gc.collect()
    torch.cuda.empty_cache()


def train_convlstm():
    """Training ConvLSTM Autoencoder with MS-SSIM + L1 Loss (single GPU)"""
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
        config={
            "learning_rate": 0.005,
            "architecture": "ConvLSTM Autoencoder",
            "dataset": "https://zenodo.org/records/7912264",
            "epochs": 10,
            "loss": "MS-SSIM + L1 + TCL",
            "latent_size": 4000,
            "seq_len": 50,
            "distributed": False,
        },
    )

    login(os.getenv("HF_KEY"))
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    print(DEVICE)

    model = ConvLSTMAutoencoder(
        seq_len=50,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        latent_size=4000,
        use_classifier=False,
        num_classes=2
    )

    if os.path.exists("convlstm_model_weights.pth"):
        try:
            checkpoint = torch.load("convlstm_model_weights.pth", map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint)
            print("Loaded ConvLSTM model weights")
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

        for index, (embryo_vol, _, _) in enumerate(pbar):
            optimizer.zero_grad()

            # embryo_vol is (T, 1, 500, 500), we need (1, T, 1, 500, 500)
            embryo_vol = embryo_vol.to(DEVICE)  # (1, T, 1, 500, 500)

            # Forward pass - returns (reconstruction, latent_seq)
            embryo_recon, embryo_lat = model(embryo_vol)

            # Reconstruction loss using MS-SSIM + L1
            rec_loss_embryo, rec_metrics_embryo = convlstm_reconstruction_loss(
                embryo_recon, embryo_vol, l1_weight=0.5, ms_ssim_weight=0.5
            )
            rec_loss = rec_loss_embryo

            # Temporal contrastive loss on latents
            # embryo_lat and sample_lat are (1, T, 4000)
            embryo_lat = embryo_lat.squeeze(0)  # (T, 4000)


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
                run.log({
                    "step": epoch * len(loader) + index,
                    "loss": loss.item(),
                    "rec_loss": rec_loss.item(),
                    "tcl": tcl.item(),
                    "temp_adj_sim": temp_adj_sim.item(),
                    "rand_sample_sim": rand_sample_sim.item(),
                    "ms_ssim_embryo": rec_metrics_embryo["ms_ssim_value"],
                    "ms_ssim_sample": rec_metrics_sample["ms_ssim_value"],
                    "l1_loss_embryo": rec_metrics_embryo["l1_loss"],
                    "l1_loss_sample": rec_metrics_sample["l1_loss"],
                    "lr": scheduler.get_last_lr()[0]
                })

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    rec=f"{rec_loss.item():.4f}",
                    tcl=f"{tcl.item():.4f}"
                )


        avg_loss = total/max(1, count)
        run.log({"avg_loss": avg_loss})
        print(f"epoch {epoch} avg loss={avg_loss:.4f}")

        # Save the state dict
        torch.save(model.state_dict(), "convlstm_model_weights.pth")

        # Save to HuggingFace with date label
        date_label = datetime.now().strftime("%Y-%m-%d")
        repo_name = f"IVF-ConvLSTM-Model-{date_label}"
        model.save_pretrained(repo_name)
        model.push_to_hub(repo_name)

    run.finish()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "mse_distributed":
            train_mse_distributed()
        elif mode == "mse_single":
            train_mse_single()
        elif mode == "convlstm":
            train_convlstm()
        else:
            train()
    else:
        train()

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
import sys
from torch.utils.data import DataLoader
from dataset_ivf import IVFSequenceDataset
from tqdm import tqdm
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

def train():
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
            },
        )
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

    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay = 1e-5 )
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

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
            rec_loss = loss_fn(recon, vol) + loss_fn(empty_well_recon, empty_well_vol)
            
            with torch.no_grad():
                model.eval()  

            _, lat = model(vol, empty_well=False)

            with torch.no_grad():
                model.train()  

            embryo_lat = lat[:embryo_size].clone()
            sample_lat = lat[embryo_size:].clone()

            embryo_lat1 = torch.cat((embryo_lat[1:], embryo_lat[5:], embryo_lat[10:], embryo_lat[20:]), 0).to(DEVICE)
            embryo_lat2 = torch.cat((embryo_lat[:-1], embryo_lat[:-5], embryo_lat[:-10], embryo_lat[:-20]), 0).to(DEVICE)
            temp_adj_sim = F.cosine_similarity(embryo_lat1, embryo_lat2).mean()
            rand_sample_sim = F.cosine_similarity(embryo_lat, sample_lat).mean() 
            tcl = rand_sample_sim - temp_adj_sim
            loss = rec_loss + (0.05 * tcl)

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
            optimizer.step()
            total += loss.item()
            count += 1
            
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
            if is_main and (index % 50 == 0) and run is not None:
                run.log({
                    "step": epoch * len(loader) + index,
                    "loss": loss.item(),
                    "rec_loss": rec_loss.item(),
                    "tcl": tcl.item(),
                    "temp_adj_sim": temp_adj_sim.item(),
                    "rand_sample_sim": rand_sample_sim.item()
                })

                if isinstance(pbar, tqdm):
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        rec=f"{rec_loss.item():.4f}",
                        tcl=f"{tcl.item():.4f}"
                    )
        avg_loss = total/max(1,count)
        if world_size > 1:
            avg_loss_tensor = torch.tensor([avg_loss], device=DEVICE)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = avg_loss_tensor.item()
        if is_main:
            run.log({"lr": scheduler.get_last_lr()[0], "avg_loss": avg_loss})
            print(f"epoch {epoch} avg loss={avg_loss}:.4f")
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), "model_weights.pth")
        scheduler.step()
    if is_main:
        run.finish()
    cleanup_distributed()
    del model
    gc.collect()
    torch.cuda.empty_cache()
if __name__ == "__main__":
    train()

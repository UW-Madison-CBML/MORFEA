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
from ae_model import ConvLSTMAutoencoder
import sys
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from torch.utils.data import DataLoader
from dataset_ivf import IVFSequenceDataset
from dataset_ivf_embryo import IVFEmbryoDataset
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
import cebra
from cebra import CEBRA
gc.collect()
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import time
import umap
import sklearn
from huggingface_hub import login
import shutil
import hashlib
import json
from torchsummary import summary
from train_lstm_classifier import train_on as train_lstm_classifier_on
from export_kanakasabapathy_latents import export_kanakasabapathy

from dataset_ivf_embryo import read_gray, normalize_video
class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def push(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self):
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std_dev(self):
        return math.sqrt(self.variance)

VAL_EMBRYOS = []#"RS363-7", "CZ594-5","CJ261-10","RL747-8","TM272-9","LFA766-1","GT353-3","LGA881-2-5","LBE649-3","TH481-5","LTA908-2","BS648-7","GS955-7","HA1040-4","CM892-5","FC048-6","GC702-6","DI358-3","MM912-4","RK787-3","GSS052-2","OJ319-5","DML373-2","PS292-4","TM294-2","KT573-4","DJC641-4","FE14-020","LD400-1","MV930-2","MDCH869-4","AS662-2","LH1169-8","GA664-1","PMDPI029-1-3","DV116-3","FV709-11","GM456-3","RA361-4","LM844-1","DL020-3","VM570-4","MC833-6","LV613-2","ZS435-5","RM126-7","BK428-2","LS93-8","GS490-7","GF976-4","PMDPI029-1-11","DRL1048-1","BS294-7","CA658-12","RO793-2","GJ191-1","CC007-2","SL313-11","RC545-2-8","OJ319-9","PA289-8","TK319-10","SM686-7","KJ1077-3","BE645-10","BC167-4","VC581-1","FM162-6","PC758-2","HC459-6","DE069-10","GC340-3","BS596-5","PE256-2","LBE857-1","PH783-3","LS1045-4","CC455-3","DL617-6","BS1086-1","CK601-4","DA309-5","LTE064-1","KF460-4","LP181-1","GS349-4","LC47-8","GS205-6","EH309-8","BS1033-2","LL854-1","DHDPI042-6","BN356-6","PA145-2","GC340-1","MM334-5","AG274-2","BA518-7","BC973-4","BA1195-9","AM33-2","AB91-1","AB028-6","BC167-4","AL884-2","AM685-3"]
# TODO: add tcl or stricter smoothing loss
from info_nce import InfoNCE, info_nce
def temporal_smoothness_loss(z_seq, weight=0.1):
    if z_seq.size(1) < 2:
        return torch.tensor(0.0, device=z_seq.device)
    B,T,L = z_seq.shape
    """ 
    # Compute difference between adjacent timesteps
    diff1 = z_seq[:, 1:] - z_seq[:, :-1]  # (B, T-1, L)
    diff2 = z_seq[:, 2:] - z_seq[:, :-2]  # (B, T-2, L)
    diff4 = z_seq[:, 4:] - z_seq[:, :-4]  # (B, T-4, L)
    diff8 = z_seq[:, 8:] - z_seq[:, :-8]  # (B, T-8, L)
    diff = torch.cat((diff1, diff2, diff4, diff8), axis=1) # (B, 4T-15, L)
    smooth_loss = (diff ** 2).mean()"""
    loss = InfoNCE(negative_mode='paired') 
    query = z_seq[:, :-1, :].reshape(-1,L)
    positive_key = z_seq[:, 1:, :].reshape(-1,L)

    negative_keys = z_seq.roll(shifts=1, dims=0)[:, :-1, :].reshape(-1,1,L) # roll along batch dim
    # for unpaired we add the buffer dim 1 cause there's only a single negative key to compare with here

    output = loss(query, positive_key, negative_keys) 
    return weight * output


def save_and_push_model(model, repo_name, required_files, model_config=None):
    os.makedirs(repo_name, exist_ok=True)
    device = next(model.parameters()).device
    clean_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    # make a copy of the model on the cpu()
    model.load_state_dict(clean_state_dict)
    try:
        model.save_pretrained(repo_name)
        print(f"Saved model using save_pretrained")
    except Exception as e:
        print(f"save_pretrained failed ({e}), saving state dict only")
        torch.save(model.state_dict(), os.path.join(repo_name, "pytorch_model.bin"))
    if model_config is not None:
        config_path = os.path.join(repo_name, "config.json")
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        print(f"Saved config.json with ablation parameters")
    for file_path in required_files:
        if os.path.exists(file_path):
            shutil.copy2(file_path, repo_name)
            print(f"Added {file_path} to repository")
        else:
            print(f"Warning: {file_path} not found, skipping")
    try:
        model.push_to_hub(repo_name)
        print(f"Pushed model weights to {repo_name}")
    except Exception as e:
        print(f"Warning: push_to_hub failed ({e}), will upload manually")

    model.to(device)
    if model.decoder.lstm_dec is not None and model.encoder.lstm_enc is not None:
        model.encoder.lstm_enc.flatten_parameters()
        model.decoder.lstm_dec.flatten_parameters()
    api = HfApi()

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
    coords = torch.arange(size, dtype=torch.float32)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0) * g.unsqueeze(1)

# TODO just find a library with someone else's implementation
def ssim(img1, img2, kernel_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
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
    
    ms_ssim_val = torch.prod(torch.stack([mcs ** w for mcs, w in zip(mcs_list, weights[:-1])]))
    ms_ssim_val = (ssim_val ** weights[-1]) * ms_ssim_val
    
    return ms_ssim_val

def reconstruction_loss(x_rec, x_true, l1_weight=0.5, ms_ssim_weight=0.5):
    B, T, C, H, W = x_rec.shape
    
    x_rec_flat = x_rec.view(B * T, C, H, W)  # (B*T, 1, 128, 128)
    x_true_flat = x_true.view(B * T, C, H, W)  # (B*T, 1, 128, 128)
    
    l1_loss = F.l1_loss(x_rec, x_true)
    
    ms_ssim_val = ms_ssim(x_rec_flat, x_true_flat)
    ms_ssim_loss = 1 - ms_ssim_val
    
    total_loss = l1_weight * l1_loss + ms_ssim_weight * ms_ssim_loss
    
    return total_loss, {
        "l1_loss": l1_loss.item(),
        "ms_ssim_loss": ms_ssim_loss.item(),
        "ms_ssim_value": ms_ssim_val.item()
    }


def train_lstm(
    loss_type="l1",
    ms_ssim_weight=0.5,
    rec_weight=0.5,
    temporal_weight=0.1,
    dropout_rate=0.1,
    use_lstm=True,
    use_residual=True,
    use_batchnorm=True,
    model_name="", 
    latent_size = 4096,
    lr=2e-4,
    epochs=25,
    warm_restarts=False
    ):
    #hyperparameters:

    
    #epochs = 30
    #lr = 2e-4
    batch_size = 64
    #warm_restarts = False
    # ------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()
    
    #torch.autograd.detect_anomaly(True)
    # Build loss description for logging
   
    date_label = datetime.now().strftime("%Y-%m-%d")

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
        name=model_name +"-" + date_label,
        config={
            "lr": lr,
            "batch_size":batch_size,
            "architecture": "ConvLSTM Autoencoder",
            "dataset": "https://zenodo.org/records/7912264",
            "epochs": epochs,
            "train_split": "not ICM graded",
            "val_split": "ICM graded",
            "ms_ssim_weight": ms_ssim_weight,
            "rec_weight": rec_weight,
            "temporal_weight": temporal_weight,
            "dropout_rate": dropout_rate,
            "use_lstm": use_lstm,
            "use_residual": use_residual,
            "use_batchnorm": use_batchnorm,
            "latent_size": latent_size,
            "image_size": 128,
            "distributed": False,
            "warm_restarts":True,
        },
    )

    login(os.getenv("HF_KEY"))

    torch.cuda.init()
    model = ConvLSTMAutoencoder(
        None,
        input_channels=1,
        encoder_layers=2,
        decoder_layers=2,
        latent_size=latent_size,
        use_classifier=False,
        num_classes=2,
        use_latent_split=False,
        # Ablation parameters
        dropout_rate=dropout_rate,
        use_lstm=use_lstm,
        use_residual=use_residual,
        use_batchnorm=use_batchnorm
    )

    VAL_EMBRYOS = pd.read_csv("embryo_dataset_grades.csv").rename(columns={"video_name":"embryo_id"}).dropna(subset=["ICM"])["embryo_id"].astype(str).tolist()
    torch.cuda.init()
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
    run.log({"train_params":trainable_params,"params":all_params})
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    df = pd.read_csv(os.path.abspath("index.csv"))
    mask = df["cell_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    val_df = df[mask]
    train_df = df[~mask]
    train_dataset = IVFSequenceDataset(train_df, resize=128, norm="minmax01")
    val_dataset = IVFSequenceDataset(val_df, resize=128, norm="minmax01")
    print("val size: ", str(len(val_df) / len(df)))
    full_seq_df = pd.read_csv(os.path.abspath("index_embryo.csv")).rename(columns={"cell_id":"embryo_id"})
    full_seq_val_mask = full_seq_df["embryo_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    full_seq_df_val = full_seq_df[full_seq_val_mask] # just look at validation ICM embryos
    
    full_seq_df_train = full_seq_df[~full_seq_val_mask] # just look at validation ICM embryos
    full_seq_dataset_val = IVFEmbryoDataset(full_seq_df_val, resize=128, norm="minmax01")
    full_seq_dataset_train = IVFEmbryoDataset(full_seq_df_train, resize=128, norm="minmax01")

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False 
    )
    full_seq_loader_val = DataLoader(
        full_seq_dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False 
    )
    full_seq_loader_train = DataLoader(
        full_seq_dataset_train,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False 
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(loader) * epochs) if warm_restarts else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(loader) * epochs)

    for epoch in range(epochs):
        print(f"epoch {epoch}")
        print(torch.cuda.memory_summary(device=DEVICE, abbreviated=False))
        gc.collect()
        torch.cuda.empty_cache()

        model.train()
        if model.decoder.lstm_dec is not None and model.encoder.lstm_enc is not None:
            model.encoder.lstm_enc.flatten_parameters()
            model.decoder.lstm_dec.flatten_parameters()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        total = 0.0
        count = 0
        start_time = time.perf_counter()
        end_time = time.perf_counter()
        for index, (embryo_vol, _, _) in enumerate(pbar):
            optimizer.zero_grad()
            t0 = time.perf_counter()
            embryo_vol = embryo_vol.to(DEVICE)  # (1, T, 1, 500, 500)
            t1 = time.perf_counter()
            embryo_recon, embryo_lat = model(embryo_vol)
            t2 = time.perf_counter()
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

            if loss_type == "l1":
                rec_loss, rec_metrics = reconstruction_loss(
                    embryo_recon, embryo_vol, l1_weight=rec_weight, ms_ssim_weight=ms_ssim_weight
                )
            elif loss_type == "mse":
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
            t3 = time.perf_counter()
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
            if (index % 5 == 0):
                run.log({
                    "data_to_gpu_time": t1 - t0,
                    "model_forward_time": t2 - t1,
                    "grad_calc_time": t3 - t2,
                    "optimizer_step_time": end_time - t3,
                        })

            loss = loss.detach().cpu() 
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


        date_label = datetime.now().strftime("%Y-%m-%d")
        
        required_files = [
            "train_ae.py",
            "ae_model.py",
        ]


        hf_config = {
            "model_type": "ConvLSTMAutoencoder",
            "architecture": "ConvLSTM Autoencoder",
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
            "dropout_rate": dropout_rate,
            "use_lstm": use_lstm,
            "use_residual": use_residual,
            "use_batchnorm": use_batchnorm,
            "loss_type": loss_type,
            "ms_ssim_weight": ms_ssim_weight,
            "rec_weight": rec_weight,
            "temporal_weight": temporal_weight,
            "lr": lr,
            "weight_decay": 1e-5,
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "batch_size": 1,
            "epochs": 10,
            "gradient_clip": 5.0,
            "dataset": "https://zenodo.org/records/7912264",
            "resize": 128,
            "normalization": "minmax01",
            "date": date_label,
        }
        model_clone = ConvLSTMAutoencoder(None,
            input_channels=1,
            encoder_layers=2,
            decoder_layers=2,
            latent_size=latent_size,
            use_classifier=False,
            num_classes=2,
            use_latent_split=False,
            dropout_rate=dropout_rate,
            use_lstm=use_lstm,use_residual=use_residual,use_batchnorm=use_batchnorm)

        model_clone.load_state_dict(model.state_dict())
        save_and_push_model(model_clone, model_name +"-"+ date_label, required_files, model_config=hf_config)
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

                if T > 1:
                    val_metrics['temp'].push(temporal_smoothness_loss(val_lat).item())
        # Log to wandb with val_ prefix
        val_log_dict = {
            f"val_{key}": value.mean for key, value in val_metrics.items()
        }
        val_log_std_dict = {
            f"val_{key}_std": value.std_dev for key, value in val_metrics.items()
        }
        run.log(val_log_dict | val_log_std_dict)


        cebra_time_model = CEBRA(model_architecture="offset10-model-mse",
            batch_size=1024,
            learning_rate=5e-5,
            temperature=13,
            output_dimension=3,
            num_hidden_units=128,
            max_iterations=3000,
            distance="euclidean",
            conditional="time",
            device="cuda_if_available",
            verbose=True,
            time_offsets=10)
        cebra_latents = []
        cebra_labels = []
        model.eval()
        offset = 0
        with torch.no_grad():
            for embryo_vol in full_seq_loader_train:
                embryo_vol = embryo_vol.to(DEVICE)
                _, z_seq = model(embryo_vol)
                traj = z_seq.cpu().numpy()[0] # batch size one just use that batch
                cebra_latents.append(traj)
                cebra_labels.append((np.arange(len(traj)) + offset ).reshape(-1, 1).astype(np.float32))
                offset += len(traj) + 10000
        cebra_time_model.fit(np.concatenate(cebra_latents, axis=0), np.concatenate(cebra_labels, axis=0))
        
        #cebra_time_model.save("cebra_time_model.pt")


        rand_img = np.random.randint(40, 64) 
        with torch.no_grad():
            for i, embryo_vol in enumerate(full_seq_loader_val):
                if i % rand_img != 0:
                    continue
                embryo_vol = embryo_vol.to(DEVICE) 
                embryo_recon, z_seq = model(embryo_vol)
                traj = z_seq.cpu().numpy()[0] # batch size one just use that batch
                distance_mat = distance_matrix(traj,traj)
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(distance_mat, cmap='viridis')

                ax.set_xlabel("Time Index")
                ax.set_ylabel("Time Index")
                plt.colorbar(im, ax=ax)
                wandb.log({"temp_smoothness_val": wandb.Image(fig)})

                plt.close(fig)                
                
                embedding = sklearn.decomposition.PCA(n_components=2).fit_transform(traj)
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.scatter(embedding[:,0], embedding[:,1],c=np.linspace(0,1,embedding.shape[0]), cmap='viridis')

                ax.set_xlabel("PCA 1")
                ax.set_ylabel("PCA 2")
                plt.colorbar(im, ax=ax)
                wandb.log({"pca_val": wandb.Image(fig)})

                plt.close(fig)                
                
                cebra_embedding = cebra_time_model.transform(traj, session_id=0) # i guess dont batch it?
                fig, ax = plt.subplots(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                im = ax.scatter(cebra_embedding[:,0], cebra_embedding[:,1], cebra_embedding[:,2], c=np.linspace(0,1,cebra_embedding.shape[0]), cmap='viridis')

                ax.set_xlabel("Cebra 1")
                ax.set_ylabel("Cebra 2")
                ax.set_zlabel("Cebra 3")
                plt.colorbar(im, ax=ax)
                wandb.log({"cebra_val": wandb.Image(fig)})

                plt.close(fig) 
                # look at some validation recons
                rand_idx = np.random.randint(embryo_vol.shape[1]) 
                vol_img = embryo_vol[0, rand_idx, 0].cpu().detach().numpy()
                recon_img = embryo_recon[0, rand_idx, 0].cpu().detach().numpy()

                vol_img = (vol_img * 255).astype(np.uint8)
                recon_img = (recon_img * 255).astype(np.uint8)
                comparison = np.concatenate((vol_img, recon_img), axis=1)
     
                images = wandb.Image(comparison, caption="embryo_recon_val")
                run.log({"reconstruction_val": images})
        
        rand_img = np.random.randint(40, 64) 
        del cebra_time_model
        with torch.no_grad():
            for i, embryo_vol in enumerate(full_seq_loader_train):
                if i % rand_img != 0:
                    continue
                embryo_vol = embryo_vol.to(DEVICE) 
                _, z_seq = model(embryo_vol)
                traj = z_seq.cpu().detach().numpy()[0] # batch size one just use that batch
                distance_mat = distance_matrix(traj,traj)
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(distance_mat, cmap='viridis')

                ax.set_xlabel("Time Index")
                ax.set_ylabel("Time Index")
                plt.colorbar(im, ax=ax)
                wandb.log({"temp_smoothness_train": wandb.Image(fig)})

                plt.close(fig)   
        # export the kanakasabapathy latents
        
        metadata_df, kanakasabapathy_lats,imgs = export_kanakasabapathy(model)
        
        model.train()
        # spoof the ICM grades
        metadata_df['ICM'] = metadata_df["TE"]
        # not necessary for kanakasabapathy
        #metadata_df = metadata_df.rename(columns={"Image":"embryo_id"})
        idx = np.random.randint(len(metadata_df))
        vol_img = normalize_video([read_gray(metadata_df.iloc[idx]["path"], 128)], "minmax01")[0]
        recon_img = imgs[idx]

        vol_img = (vol_img * 255).astype(np.uint8)
        recon_img = (recon_img * 255).astype(np.uint8)
        comparison = np.concatenate((vol_img, recon_img), axis=1)

        images = wandb.Image(comparison, caption="kanakasabapathy_recon_val")
        run.log({"kanakasabapathy_recon_val": images,"kanakasabapathy_grade_sizes": wandb.Table(dataframe=metadata_df.groupby("TE").size())})

        # train the lstm model on the kanakasabapathy latents and log the loss to wandb
        kanakasabapathy_lats_df = pd.DataFrame(kanakasabapathy_lats, index=metadata_df.index, columns=[f"z_{i}" for i in range(kanakasabapathy_lats.shape[1])])
        
            
        kanakasabapathy_df = pd.concat([metadata_df, kanakasabapathy_lats_df], axis=1)
        
        embryo_ids = kanakasabapathy_df["embryo_id"].unique()
        np.random.shuffle(embryo_ids)
        # 30% seems about right?
        VAL_EMBRYOS = embryo_ids[:int(0.3 * len(embryo_ids))]
        kanakasabapathy_mask = kanakasabapathy_df["embryo_id"].isin(VAL_EMBRYOS)
        kanakasabapathy_val_df = kanakasabapathy_df[kanakasabapathy_mask]
        kanakasabapathy_df = kanakasabapathy_df[~kanakasabapathy_mask]
        train_lstm_classifier_on(kanakasabapathy_df, kanakasabapathy_val_df, {"latents":True, "te_lr":0.003, "icm_lr":0.003}, False, "kanakasabapathy", run, batch_size=128, epochs=30)


            

    run.finish()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys
    import argparse

    if len(sys.argv) > 1 and sys.argv[1] == "lstm":
        mode = sys.argv[1]
        parser = argparse.ArgumentParser(description="Train ConvLSTM Autoencoder with Ablation Studies")
        parser.add_argument("mode", type=str, help="Training mode")

        # loss ablation arguments
        parser.add_argument("--loss-type", type=str, default="l1", choices=["l1", "mse"],
                          help="Reconstruction loss type: l1 or mse (default: l1)")
        parser.add_argument("--ms-ssim-weight", type=float, default=0.5,
                          help="Weight for MS-SSIM loss (default: 0.5, set to 0 to disable)")
        parser.add_argument("--rec-weight", type=float, default=0.5,
                          help="Weight for reconstruction loss (default: 0.5, set to 0 to disable)")
        parser.add_argument("--temporal-weight", type=float, default=0.1,
                          help="Weight for temporal smoothness loss (default: 0.1, set to 0 to disable)")
        #model layer ablations
        parser.add_argument("--dropout-rate", type=float, default=0.1,
                          help="Dropout rate (default: 0.1, set to 0 to disable)")
        parser.add_argument("--no-lstm", action="store_true",
                          help="Disable ConvLSTM (no temporal modeling)")
        parser.add_argument("--no-residual", action="store_true",
                          help="Disable residual connections")
        parser.add_argument("--no-batchnorm", action="store_true",
                          help="Disable batch normalization")
        parser.add_argument("--name", type=str, default="", help="model name")
        parser.add_argument("--size", type=int, default=4096, help="lat dimensions")

        parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
        parser.add_argument("--epochs", type=int, default=25, help="epochs")
        parser.add_argument("--warm-restarts", action="store_true", help="turn on warm restarts lr scheduling, default is cosine annealing decreasing over the the whole run")
        args = parser.parse_args()

        train_lstm(
            loss_type=args.loss_type,
            ms_ssim_weight=args.ms_ssim_weight,
            rec_weight=args.rec_weight,
            temporal_weight=args.temporal_weight,
            dropout_rate=args.dropout_rate,
            use_lstm=not args.no_lstm,
            use_residual=not args.no_residual,
            use_batchnorm=not args.no_batchnorm,
            model_name = args.name,
            latent_size = args.size,
            lr=args.lr,
            epochs=args.epochs,
            warm_restarts=args.warm_restarts

        )
    elif len(sys.argv) > 1 and sys.argv[1] == "vit":
        print("not yet implemented")
        

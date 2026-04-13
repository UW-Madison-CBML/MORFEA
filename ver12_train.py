"""
Training Script — Ver10 ConvLSTM Autoencoder
=============================================
Changes from Ver09:
    - encoder_hidden_dim: 256 -> 512
    - decoder_hidden_dim: 256 -> 512
    - FrameDecoder: 8x8 -> 16x16 starting spatial
    - batch_size: 32 -> 16 (larger model needs more memory)

Usage:
    python ver12_train.py convlstm --name my_run
    python ver12_train.py convlstm --infonce-weight 0  # reconstruction only
"""

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ConvLSTMAutoencoder
from losses import compute_loss
from dataset_ivf import IVFSequenceDataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class RunningStats:
    """Online mean and std (Welford's algorithm)."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def push(self, x: float):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self._m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self._m2 / (self.n - 1)) if self.n > 1 else 0.0


def push_checkpoint(model: torch.nn.Module, repo_name: str, cfg: argparse.Namespace):
    try:
        from huggingface_hub import HfApi, login
        login(os.getenv("HF_KEY"))
    except Exception as e:
        print(f"  [hub] login failed: {e}")
        return
    os.makedirs(repo_name, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(repo_name, "pytorch_model.bin"))
    with open(os.path.join(repo_name, "config.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2)
    for fp in ["ver12_train.py", "ver12_model.py", "ver12_losses.py",
               "dataset_ivf.py", "ver12_build_index.py"]:
        if os.path.exists(fp):
            shutil.copy2(fp, repo_name)
    api = HfApi()
    for fname in os.listdir(repo_name):
        try:
            api.upload_file(
                path_or_fileobj=os.path.join(repo_name, fname),
                path_in_repo=fname,
                repo_id=f"JensLundsgaard/{repo_name}",
                repo_type="model",
            )
        except Exception as e:
            print(f"  [hub] upload failed {fname}: {e}")
    print(f"  [hub] pushed to JensLundsgaard/{repo_name}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    model: ConvLSTMAutoencoder,
    val_loader: DataLoader,
    device: torch.device,
    cfg: argparse.Namespace,
) -> dict:
    from pytorch_msssim import ms_ssim as _ms_ssim
    from losses import multiscale_temporal_loss

    stats = {k: RunningStats() for k in [
        "mse", "l1", "ms_ssim_loss", "temporal_loss"
    ]}
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            embryo_vol = batch[0].to(device)
            out = model(embryo_vol)
            B, T, C, H, W = embryo_vol.shape

            stats["mse"].push(F.mse_loss(out["reconstruction"], embryo_vol).item())
            stats["l1"].push(F.l1_loss(out["reconstruction"], embryo_vol).item())

            flat_r = out["reconstruction"].reshape(B * T, C, H, W)
            flat_t = embryo_vol.reshape(B * T, C, H, W)
            ms = _ms_ssim(flat_r, flat_t, data_range=1.0, size_average=True)
            stats["ms_ssim_loss"].push((1 - ms).item())

            if out["z_seq"].shape[1] > 25:
                _, tmp_m = multiscale_temporal_loss(out["z_seq"])
                stats["temporal_loss"].push(tmp_m["temporal_loss"])

    return stats


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def log_visuals(run, out: dict, embryo_vol: torch.Tensor, step: int):
    try:
        import wandb
        from scipy.spatial import distance_matrix as sp_dm
    except ImportError:
        return

    with torch.no_grad():
        orig  = (embryo_vol[0, -1, 0].cpu().numpy() * 255).astype(np.uint8)
        recon = (out["reconstruction"][0, -1, 0].cpu().numpy() * 255).astype(np.uint8)
        run.log({"vis/reconstruction": wandb.Image(
            np.concatenate([orig, recon], axis=1),
            caption="original | reconstruction",
        )}, step=step)

        traj = out["z_seq"][0].cpu().numpy()
        dm   = sp_dm(traj, traj)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(dm, cmap="viridis")
        ax.set_xlabel("Frame"); ax.set_ylabel("Frame")
        ax.set_title("Pairwise latent distance")
        plt.colorbar(im, ax=ax)
        run.log({"vis/latent_distance_matrix": wandb.Image(fig)}, step=step)
        plt.close(fig)

        norms = np.linalg.norm(traj, axis=-1)
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(norms)
        ax.set_xlabel("Frame"); ax.set_ylabel("||z_t||")
        run.log({"vis/latent_norm": wandb.Image(fig)}, step=step)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: argparse.Namespace):
    gc.collect()
    torch.cuda.empty_cache()

    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    date_label = datetime.now().strftime("%Y-%m-%d")
    print(f"Device: {DEVICE}")

    run = None
    if not cfg.no_wandb:
        try:
            import wandb
            wandb.login(key=os.getenv("WANDB_KEY"))
            run = wandb.init(
                entity="jenslundsgaard7-uw-madison",
                project="IVF-Training",
                name=f"{cfg.name}-{date_label}",
                config=vars(cfg),
            )
        except Exception as e:
            print(f"[wandb] init failed: {e}")

    model = ConvLSTMAutoencoder(
        seq_len=cfg.seq_len,
        input_channels=1,
        encoder_hidden_dim=cfg.encoder_dim,
        encoder_layers=cfg.encoder_layers,
        decoder_hidden_dim=cfg.decoder_dim,
        decoder_layers=cfg.decoder_layers,
        dropout_rate=cfg.dropout,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    if run:
        run.log({"trainable_params": n_params})

    dataset_kwargs = dict(resize=128, norm="minmax01")
    if cfg.extracted_dir:
        dataset_kwargs["extracted_dir"] = cfg.extracted_dir
    elif cfg.tar_file:
        dataset_kwargs["tar_file"] = cfg.tar_file

    train_ds = IVFSequenceDataset(cfg.index_train, **dataset_kwargs)
    val_ds   = IVFSequenceDataset(cfg.index_val,   **dataset_kwargs)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * cfg.epochs
    )

    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_stats = RunningStats()
        t0 = time.perf_counter()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            embryo_vol = batch[0].to(DEVICE)
            optimizer.zero_grad()

            out = model(embryo_vol)
            loss, metrics = compute_loss(
                out, embryo_vol,
                pixel_weight=cfg.pixel_weight,
                ms_ssim_weight=cfg.ms_ssim_weight,
                temporal_weight=cfg.temporal_weight,
                temporal_pairs=cfg.temporal_pairs,
                loss_type=cfg.loss_type,
            )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [warn] NaN/Inf at step {global_step}, skipping")
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            epoch_stats.push(loss.item())
            global_step += 1

            if global_step % 50 == 0:
                print(f"  step={global_step} " +
                      " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                if run:
                    run.log({
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": float(grad_norm),
                        **{f"train/{k}": v for k, v in metrics.items()},
                    }, step=global_step)

            if run and global_step % 200 == 0:
                log_visuals(run, out, embryo_vol, global_step)

        epoch_time = time.perf_counter() - t0
        print(f"Epoch {epoch}  avg={epoch_stats.mean:.4f}  time={epoch_time:.0f}s")
        if run:
            run.log({
                "epoch": epoch,
                "epoch_time": epoch_time,
                "train/avg_loss": epoch_stats.mean,
            }, step=global_step)

        val_stats = validate(model, val_loader, DEVICE, cfg)
        print("  val:", "  ".join(f"{k}={v.mean:.4f}" for k, v in val_stats.items()))
        if run:
            run.log(
                {**{f"val/{k}": v.mean for k, v in val_stats.items()},
                 **{f"val/{k}_std": v.std for k, v in val_stats.items()}},
                step=global_step,
            )

        torch.save(model.state_dict(), "model_weights.pth")
        print("  Saved model_weights.pth")

        if not cfg.no_hub:
            push_checkpoint(model, f"{cfg.name}-{date_label}", cfg)

    if run:
        run.finish()
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ver10 ConvLSTM — Reconstruction + Rolled InfoNCE"
    )
    p.add_argument("mode", choices=["convlstm"])
    p.add_argument("--name", default="convlstm_ver12")

    g = p.add_argument_group("Data")
    g.add_argument("--index-train",   default="index_train.csv")
    g.add_argument("--index-val",     default="index_val.csv")
    g.add_argument("--extracted-dir", default=None)
    g.add_argument("--tar-file",      default=None)
    g.add_argument("--num-workers",   type=int, default=4)

    g = p.add_argument_group("Loss")
    g.add_argument("--loss-type",      default="l1", choices=["l1", "mse"])
    g.add_argument("--pixel-weight",   type=float, default=0.5)
    g.add_argument("--ms-ssim-weight", type=float, default=0.5)
    g.add_argument("--temporal-weight", type=float, default=0.1,
                   help="Multi-scale temporal loss weight (0 to ablate)")
    g.add_argument("--temporal-pairs", type=int, default=4,
                   help="Number of random offset pairs per step")

    g = p.add_argument_group("Architecture")
    g.add_argument("--encoder-dim",    type=int,   default=512)
    g.add_argument("--encoder-layers", type=int,   default=2)
    g.add_argument("--decoder-dim",    type=int,   default=512)
    g.add_argument("--decoder-layers", type=int,   default=2)
    g.add_argument("--dropout",        type=float, default=0.0)
    g.add_argument("--seq-len",        type=int,   default=50)

    g = p.add_argument_group("Training")
    g.add_argument("--epochs",     type=int,   default=50)
    g.add_argument("--batch-size", type=int,   default=16,
                   help="Reduced from 32 due to larger model (512 hidden dim)")
    g.add_argument("--lr",         type=float, default=2e-4)

    g = p.add_argument_group("Integrations")
    g.add_argument("--no-wandb", action="store_true")
    g.add_argument("--no-hub",   action="store_true")

    return p


if __name__ == "__main__":
    cfg = build_parser().parse_args()
    train(cfg)

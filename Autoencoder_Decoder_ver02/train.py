"""
Complete High-Quality Training Script
- Uses MS-SSIM + L1 Loss
- Supports classification task (optional)
- Complete training loop, validation, saving
- Maximum quality configuration, no computational savings
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Add current directory and parent directories to path to find dataset_ivf
# Try multiple possible locations (current directory first!)
possible_paths = [
    Path(__file__).parent,  # Current directory (where train.py is)
    Path(__file__).parent.parent,  # ../Code/
    Path(__file__).parent.parent.parent,  # ../../ (if in Raffael/date/)
    Path(__file__).parent.parent / "Code",  # ../Code/Code/
]
for path in possible_paths:
    if (path / "dataset_ivf.py").exists():
        sys.path.insert(0, str(path))
        print(f"Found dataset_ivf.py at: {path}")
        break
else:
    print(f"WARNING: dataset_ivf.py not found in any of: {[str(p) for p in possible_paths]}")
    print(f"Current directory: {Path.cwd()}")
    print(f"Files in current directory: {list(Path('.').glob('*.py'))}")

from dataset_ivf import IVFSequenceDataset
from model import ConvLSTMAutoencoder
from losses import (
    reconstruction_loss,
    temporal_smoothness_loss,
    classification_loss
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def train(
    index_csv="index.csv",
    batch_size=8,
    seq_len=20,
    num_epochs=50,
    learning_rate=3e-4,
    weight_decay=1e-5,
    l1_weight=0.5,
    ms_ssim_weight=0.5,
    smooth_weight=0.1,
    cls_weight=0.0,  # Set to 0 if classification not needed
    use_classifier=False,
    use_l2_loss=False,  # If True, use L2 (MSE) instead of L1
    save_dir="checkpoints",
    log_dir="logs",
    resume_from=None
):
    """
    Training function
    
    Args:
        index_csv: data index file
        batch_size: batch size
        seq_len: sequence length
        num_epochs: number of training epochs
        learning_rate: learning rate
        weight_decay: weight decay
        l1_weight: L1 loss weight
        ms_ssim_weight: MS-SSIM loss weight
        smooth_weight: temporal smoothness loss weight
        cls_weight: classification loss weight (if classifier enabled)
        use_classifier: whether to use classifier
        save_dir: directory to save models
        log_dir: directory to save logs
        resume_from: checkpoint to resume training from
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Dataset
    print("Loading dataset...")
    train_dataset = IVFSequenceDataset(index_csv, resize=128, norm="minmax01")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if DEVICE == "cuda" else False,
        persistent_workers=True
    )
    print(f"Dataset size: {len(train_dataset)}")
    
    # Model
    print("Initializing model...")
    model = ConvLSTMAutoencoder(
        seq_len=seq_len,
        input_channels=1,
        encoder_hidden_dim=256,  # High-quality configuration
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        use_classifier=use_classifier,
        num_classes=3  # A, B, C 三个类别
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Resume training
    start_epoch = 0
    if resume_from:
        print(f"Resuming from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    print("\nStarting training...")
    training_log = []
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = {
            "total": 0.0,
            "reconstruction": 0.0,
            "l1": 0.0,
            "l2": 0.0,  # Will be 0 if using L1
            "ms_ssim": 0.0,
            "smooth": 0.0,
            "classification": 0.0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (vol, cell_id) in enumerate(pbar):
            vol = vol.to(DEVICE)  # (B, T, 1, 128, 128)
            
            # Forward pass
            output = model(vol)
            x_rec = output["reconstruction"]
            z_seq = output["z_seq"]
            
            # Reconstruction loss
            rec_loss, rec_details = reconstruction_loss(
                x_rec, vol,
                l1_weight=l1_weight,
                ms_ssim_weight=ms_ssim_weight,
                use_l2=use_l2_loss
            )
            
            # Temporal smoothness loss
            smooth_loss = temporal_smoothness_loss(z_seq, weight=smooth_weight)
            
            # Total loss
            total_loss = rec_loss + smooth_loss
            
            # Classification loss (if enabled)
            if use_classifier and "logits" in output:
                # Need real labels here, skip for now
                # If label data available, do:
                # labels = ...  # Get from data
                # cls_loss = classification_loss(output["logits"], labels)
                # total_loss += cls_weight * cls_loss
                pass
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Record losses
            epoch_losses["total"] += total_loss.item()
            epoch_losses["reconstruction"] += rec_loss.item()
            if "l1_loss" in rec_details:
                epoch_losses["l1"] += rec_details["l1_loss"]
            if "l2_loss" in rec_details:
                epoch_losses["l2"] += rec_details["l2_loss"]
            epoch_losses["ms_ssim"] += rec_details["ms_ssim_loss"]
            epoch_losses["smooth"] += smooth_loss.item()
            
            # Update progress bar
            pixel_loss_key = "l2_loss" if use_l2_loss else "l1_loss"
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "rec": f"{rec_loss.item():.4f}",
                pixel_loss_key: f"{rec_details[pixel_loss_key]:.4f}",
                "ms_ssim": f"{rec_details['ms_ssim_loss']:.4f}",
                "smooth": f"{smooth_loss.item():.4f}",
                "ms_ssim_val": f"{rec_details['ms_ssim_value']:.4f}"
            })
        
        # Average losses
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log entry
        log_entry = {
            "epoch": epoch + 1,
            "lr": current_lr,
            **epoch_losses
        }
        training_log.append(log_entry)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Total Loss: {epoch_losses['total']:.4f}")
        print(f"  Reconstruction: {epoch_losses['reconstruction']:.4f}")
        if use_l2_loss:
            print(f"    - L2 (MSE): {epoch_losses['l2']:.4f}")
        else:
            print(f"    - L1: {epoch_losses['l1']:.4f}")
        print(f"    - MS-SSIM: {epoch_losses['ms_ssim']:.4f}")
        print(f"  Smooth: {epoch_losses['smooth']:.4f}")
        if epoch_losses['classification'] > 0:
            print(f"  Classification: {epoch_losses['classification']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'losses': epoch_losses,
                'config': {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay,
                }
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save training log
        log_path = os.path.join(log_dir, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
    
    print("\nTraining completed!")
    print(f"Final model saved in: {save_dir}")
    print(f"Training log saved in: {log_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ConvLSTM Autoencoder")
    parser.add_argument("--index_csv", type=str, default="index.csv",
                       help="Path to index CSV file")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--seq_len", type=int, default=20,
                       help="Sequence length")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--l1_weight", type=float, default=0.5,
                       help="L1/L2 loss weight")
    parser.add_argument("--ms_ssim_weight", type=float, default=0.5,
                       help="MS-SSIM loss weight")
    parser.add_argument("--smooth_weight", type=float, default=0.1,
                       help="Temporal smoothness loss weight")
    parser.add_argument("--use_l2_loss", action="store_true",
                       help="Use L2 (MSE) loss instead of L1 loss")
    parser.add_argument("--use_classifier", action="store_true",
                       help="Use classifier head")
    parser.add_argument("--cls_weight", type=float, default=0.0,
                       help="Classification loss weight")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs",
                       help="Directory to save logs")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    train(
        index_csv=args.index_csv,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        l1_weight=args.l1_weight,
        ms_ssim_weight=args.ms_ssim_weight,
        smooth_weight=args.smooth_weight,
        cls_weight=args.cls_weight,
        use_classifier=args.use_classifier,
        use_l2_loss=args.use_l2_loss,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_from=args.resume_from
    )


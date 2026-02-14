"""
Extract Latent Vectors from Epoch 50 Checkpoint
- Uses CHTC model architecture (encoder.spatial_cnn + decoder.spatial_decoder)
- Extracts z_seq (full sequence) for t-PHATE analysis
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import argparse
import sys
import os

# Import model (CHTC uses Encoder/Decoder, local uses ConvLSTMAutoencoder)
try:
    from model import Encoder, Decoder
    # CHTC model structure - need to create Autoencoder class
    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()
        
        def forward(self, x):
            # x: [B, T, C, H, W]
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            z_seq = self.encoder(x_flat)  # [B*T, hidden_dim]
            z_seq = z_seq.view(B, T, -1)
            recon = self.decoder(z_seq)  # [B*T, C, H, W]
            recon = recon.view(B, T, C, H, W)
            return {
                'reconstruction': recon,
                'z_seq': z_seq,
                'z_last': z_seq[:, -1, :]
            }
    
    CHTC_MODEL_AVAILABLE = True
    print("Using CHTC model structure (Encoder + Decoder)")
except (ImportError, AttributeError):
    print("Using local model structure (ConvLSTMAutoencoder)")
    CHTC_MODEL_AVAILABLE = False
    from model import ConvLSTMAutoencoder

from dataset_ivf import IVFSequenceDataset


def extract_latents_epoch50(
    checkpoint_path="checkpoints/checkpoint_epoch_50.pt",
    index_csv="index.csv",
    output_dir="latents_epoch50",
    batch_size=8,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_z_seq=True  # Extract full sequence for t-PHATE
):
    """
    Extract latent vectors from epoch 50 checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        index_csv: Path to index CSV file
        output_dir: Directory to save latent vectors
        batch_size: Batch size for extraction
        device: Device to use
        use_z_seq: If True, save full z_seq (B, T, hidden_dim); 
                  if False, save only z_last (B, hidden_dim)
    """
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    config = checkpoint.get('config', {})
    seq_len = config.get('seq_len', 20)
    
    # Load model
    print("Loading model...")
    # Use local variable to avoid UnboundLocalError
    chtc_available = CHTC_MODEL_AVAILABLE
    use_chtc_model = False
    
    if chtc_available:
        try:
            model = Autoencoder()
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("✓ Model loaded (CHTC structure, strict=True)")
            use_chtc_model = True
        except Exception as e:
            print(f"Error loading CHTC model: {e}")
            print("Falling back to local model structure...")
            use_chtc_model = False
    
    if not use_chtc_model:
        from model import ConvLSTMAutoencoder
        model = ConvLSTMAutoencoder(
            seq_len=seq_len,
            input_channels=1,
            encoder_hidden_dim=256,
            encoder_layers=2,
            decoder_hidden_dim=128,
            decoder_layers=2,
            use_classifier=False
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✓ Model loaded (local structure, strict=False)")
    
    model.to(device)
    model.eval()
    
    # Load dataset
    print(f"Loading dataset from: {index_csv}")
    dataset = IVFSequenceDataset(index_csv, resize=128, norm="minmax01")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Load dataframe for metadata (start_idx)
    df = pd.read_csv(index_csv)
    print(f"Index CSV loaded: {len(df)} rows")
    
    # Extract latent vectors
    print("\nExtracting latent vectors...")
    all_z_seq = []
    all_z_last = []
    all_cell_ids = []
    all_start_indices = []
    
    with torch.no_grad():
        global_idx = 0
        batch_idx = 0
        for batch in tqdm(dataloader, desc="Extracting"):
            # dataset_ivf returns tuple: (sequence, cell_id)
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                sequences, cell_ids = batch
                # Get start_indices from dataframe
                start_indices = []
                for i in range(len(cell_ids)):
                    try:
                        if global_idx < len(df):
                            start_idx = int(df.iloc[global_idx].get('start_idx', 0))
                        else:
                            start_idx = 0
                    except:
                        start_idx = 0
                    start_indices.append(start_idx)
                    global_idx += 1
            else:
                # Fallback for dict format
                sequences = batch['sequence']
                cell_ids = batch['cell_id']
                start_indices = batch.get('start_idx', [0] * len(cell_ids))
            
            sequences = sequences.to(device)  # [B, T, C, H, W]
            
            # Forward pass to get latent vectors
            output = model(sequences)
            z_seq = output['z_seq']  # Could be [B, T, hidden_dim] or [B, T, C, H, W]
            z_last = output['z_last']  # Could be [B, hidden_dim] or [B, C, H, W]
            
            # Check if z_seq is spatial (has spatial dimensions) and pool if needed
            if len(z_seq.shape) == 5:
                # z_seq is spatial: [B, T, C, H, W] -> pool to [B, T, C]
                B, T, C, H, W = z_seq.shape
                # Global Average Pooling
                z_seq = z_seq.view(B, T, C, -1).mean(dim=-1)  # [B, T, C]
                if batch_idx == 0:
                    print(f"  [Info] z_seq was spatial [B, T, C, H, W], pooled to [B, T, C] = {z_seq.shape}")
            
            if len(z_last.shape) == 4:
                # z_last is spatial: [B, C, H, W] -> pool to [B, C]
                B, C, H, W = z_last.shape
                # Global Average Pooling
                z_last = z_last.view(B, C, -1).mean(dim=-1)  # [B, C]
                if batch_idx == 0:
                    print(f"  [Info] z_last was spatial [B, C, H, W], pooled to [B, C] = {z_last.shape}")
            
            # Move to CPU and convert to numpy
            z_seq_np = z_seq.cpu().numpy()  # [B, T, hidden_dim]
            z_last_np = z_last.cpu().numpy()  # [B, hidden_dim]
            
            all_z_seq.append(z_seq_np)
            all_z_last.append(z_last_np)
            all_cell_ids.extend(cell_ids)
            all_start_indices.extend(start_indices if isinstance(start_indices, list) else start_indices.tolist())
            
            batch_idx += 1
    
    # Concatenate all batches
    all_z_seq = np.concatenate(all_z_seq, axis=0)  # [N, T, hidden_dim]
    all_z_last = np.concatenate(all_z_last, axis=0)  # [N, hidden_dim]
    
    print(f"\nExtracted latent vectors:")
    print(f"  z_seq shape: {all_z_seq.shape}")
    print(f"  z_last shape: {all_z_last.shape}")
    print(f"  Total sequences: {len(all_cell_ids)}")
    
    # Save latent vectors
    if use_z_seq:
        # Save z_seq (full sequence) - needed for t-PHATE
        output_file = output_path / "latents_z_seq_epoch50.npy"
        np.save(output_file, all_z_seq)
        print(f"\n✓ Saved z_seq to: {output_file}")
    else:
        # Save z_last
        output_file = output_path / "latents_z_last_epoch50.npy"
        np.save(output_file, all_z_last)
        print(f"\n✓ Saved z_last to: {output_file}")
    
    # Also save z_last for reference
    np.save(output_path / "latents_z_last_epoch50.npy", all_z_last)
    
    # Save metadata
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get('epoch', 50),
        "index_csv": index_csv,
        "total_sequences": len(all_cell_ids),
        "z_seq_shape": list(all_z_seq.shape),
        "z_last_shape": list(all_z_last.shape),
        "cell_ids": all_cell_ids,
        "start_indices": all_start_indices,
        "model_config": config,
        "losses": checkpoint.get('losses', {})
    }
    
    metadata_file = output_path / "latents_metadata_epoch50.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_file}")
    
    # Save as CSV for easy viewing
    df_metadata = pd.DataFrame({
        "cell_id": all_cell_ids,
        "start_idx": all_start_indices,
        "latent_index": range(len(all_cell_ids))
    })
    csv_file = output_path / "latents_info_epoch50.csv"
    df_metadata.to_csv(csv_file, index=False)
    print(f"✓ Saved info CSV to: {csv_file}")
    
    print(f"\n✅ Extraction complete! Files saved in: {output_dir}/")
    return all_z_seq, all_z_last, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract latent vectors from epoch 50 checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch_50.pt",
                       help="Path to checkpoint file")
    parser.add_argument("--index_csv", type=str, default="index.csv",
                       help="Path to index CSV file")
    parser.add_argument("--output_dir", type=str, default="latents_epoch50",
                       help="Output directory for latent vectors")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--use_z_seq", action="store_true", default=True,
                       help="Save full z_seq (for t-PHATE)")
    
    args = parser.parse_args()
    
    extract_latents_epoch50(
        checkpoint_path=args.checkpoint,
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_z_seq=args.use_z_seq
    )


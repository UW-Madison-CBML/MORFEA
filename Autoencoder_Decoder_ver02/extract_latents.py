"""
Extract Latent Vectors from Trained Model
- Loads trained checkpoint
- Extracts latent vectors (z_seq and z_last) for all sequences
- Saves to numpy arrays for downstream analysis
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

from model import ConvLSTMAutoencoder
from dataset_ivf import IVFSequenceDataset


def extract_latents(
    checkpoint_path,
    index_csv="index.csv",
    output_dir="latents",
    batch_size=8,
    device="cuda" if torch.cuda.is_available() else "cpu",
    use_z_last=True,  # If True, save z_last; if False, save full z_seq
    seq_len=20
):
    """
    Extract latent vectors from trained model
    
    Args:
        checkpoint_path: Path to checkpoint file
        index_csv: Path to index CSV file
        output_dir: Directory to save latent vectors
        batch_size: Batch size for extraction
        device: Device to use
        use_z_last: If True, save only last latent (B, hidden_dim); 
                   if False, save full sequence (B, T, hidden_dim)
        seq_len: Sequence length
    """
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load model
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    config = checkpoint.get('config', {})
    model_seq_len = config.get('seq_len', seq_len)
    
    # Initialize model
    model = ConvLSTMAutoencoder(
        seq_len=model_seq_len,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        use_classifier=False
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
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
    
    # Extract latent vectors
    print("\nExtracting latent vectors...")
    all_z_last = []
    all_z_seq = []
    all_cell_ids = []
    all_start_indices = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting")):
            sequences = batch['sequence'].to(device)  # [B, T, C, H, W]
            cell_ids = batch['cell_id']
            start_indices = batch['start_idx']
            
            # Encode to get latent vectors
            z_seq, z_last = model.encode(sequences)  # z_seq: [B, T, hidden_dim], z_last: [B, hidden_dim]
            
            # Move to CPU and convert to numpy
            z_last_np = z_last.cpu().numpy()  # [B, hidden_dim]
            z_seq_np = z_seq.cpu().numpy()  # [B, T, hidden_dim]
            
            all_z_last.append(z_last_np)
            all_z_seq.append(z_seq_np)
            all_cell_ids.extend(cell_ids)
            all_start_indices.extend(start_indices.tolist())
    
    # Concatenate all batches
    all_z_last = np.concatenate(all_z_last, axis=0)  # [N, hidden_dim]
    all_z_seq = np.concatenate(all_z_seq, axis=0)  # [N, T, hidden_dim]
    
    print(f"\nExtracted latent vectors:")
    print(f"  z_last shape: {all_z_last.shape}")
    print(f"  z_seq shape: {all_z_seq.shape}")
    print(f"  Total sequences: {len(all_cell_ids)}")
    
    # Save latent vectors
    if use_z_last:
        # Save z_last (most common use case)
        output_file = output_path / "latents_z_last.npy"
        np.save(output_file, all_z_last)
        print(f"\n✓ Saved z_last to: {output_file}")
    else:
        # Save full z_seq
        output_file = output_path / "latents_z_seq.npy"
        np.save(output_file, all_z_seq)
        print(f"\n✓ Saved z_seq to: {output_file}")
    
    # Save metadata
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "index_csv": index_csv,
        "total_sequences": len(all_cell_ids),
        "z_last_shape": list(all_z_last.shape),
        "z_seq_shape": list(all_z_seq.shape),
        "cell_ids": all_cell_ids,
        "start_indices": all_start_indices,
        "model_config": config
    }
    
    metadata_file = output_path / "latents_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_file}")
    
    # Also save as CSV for easy viewing
    df_metadata = pd.DataFrame({
        "cell_id": all_cell_ids,
        "start_idx": all_start_indices,
        "latent_index": range(len(all_cell_ids))
    })
    csv_file = output_path / "latents_info.csv"
    df_metadata.to_csv(csv_file, index=False)
    print(f"✓ Saved info CSV to: {csv_file}")
    
    print(f"\n✅ Extraction complete! Files saved in: {output_dir}/")
    return all_z_last, all_z_seq, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract latent vectors from trained model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch_50.pt",
                       help="Path to checkpoint file")
    parser.add_argument("--index_csv", type=str, default="index.csv",
                       help="Path to index CSV file")
    parser.add_argument("--output_dir", type=str, default="latents",
                       help="Output directory for latent vectors")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--use_z_last", action="store_true", default=True,
                       help="Save z_last (final latent) instead of full z_seq")
    parser.add_argument("--use_z_seq", action="store_true", default=False,
                       help="Save full z_seq (all time steps) instead of z_last")
    
    args = parser.parse_args()
    
    # Determine which to save
    use_z_last = not args.use_z_seq if args.use_z_seq else args.use_z_last
    
    extract_latents(
        checkpoint_path=args.checkpoint,
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_z_last=use_z_last
    )


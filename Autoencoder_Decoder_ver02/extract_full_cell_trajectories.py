"""
Extract Full Cell Trajectories (One per Cell)
- Extracts complete latent trajectories for each cell
- Combines overlapping sequences to get full development timeline
- For t-PHATE analysis: one trajectory per cell
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
from collections import defaultdict

# Import model (CHTC uses Encoder/Decoder, local uses ConvLSTMAutoencoder)
try:
    from model import Encoder, Decoder
    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()
        
        def forward(self, x):
            # x: [B, T, C, H, W]
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            z_seq = self.encoder(x_flat)  # [B*T, hidden_dim or spatial]
            z_seq = z_seq.view(B, T, -1) if len(z_seq.shape) == 2 else z_seq.view(B, T, *z_seq.shape[1:])
            recon = self.decoder(z_seq)  # [B*T, C, H, W]
            recon = recon.view(B, T, C, H, W)
            return {
                'reconstruction': recon,
                'z_seq': z_seq,
                'z_last': z_seq[:, -1, :] if len(z_seq.shape) == 3 else z_seq[:, -1]
            }
    
    CHTC_MODEL_AVAILABLE = True
    print("Using CHTC model structure (Encoder + Decoder)")
except (ImportError, AttributeError):
    print("Using local model structure (ConvLSTMAutoencoder)")
    CHTC_MODEL_AVAILABLE = False
    from model import ConvLSTMAutoencoder

from dataset_ivf import IVFSequenceDataset


def extract_full_cell_trajectories(
    checkpoint_path="checkpoints/checkpoint_epoch_50.pt",
    index_csv="index.csv",
    output_dir="latents_full_cells",
    batch_size=1,  # Use 1 to process sequences individually
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_cells=None  # Limit number of cells to process
):
    """
    Extract full latent trajectories for each cell (one trajectory per cell)
    
    Args:
        checkpoint_path: Path to checkpoint file
        index_csv: Path to index CSV file
        output_dir: Directory to save latent vectors
        batch_size: Batch size (use 1 for individual sequences)
        device: Device to use
        max_cells: Maximum number of cells to process (None = all)
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
    
    # Load dataset and dataframe
    print(f"Loading dataset from: {index_csv}")
    dataset = IVFSequenceDataset(index_csv, resize=128, norm="minmax01")
    df = pd.read_csv(index_csv)
    print(f"Dataset size: {len(dataset)} sequences")
    print(f"Unique cells: {len(df['cell_id'].unique())}")
    
    # Group sequences by cell_id
    cell_sequences = defaultdict(list)
    for idx, row in df.iterrows():
        cell_id = row['cell_id']
        start_idx = row['start_idx']
        cell_sequences[cell_id].append((idx, start_idx))
    
    # Sort sequences by start_idx for each cell
    for cell_id in cell_sequences:
        cell_sequences[cell_id].sort(key=lambda x: x[1])
    
    unique_cells = sorted(cell_sequences.keys())
    if max_cells:
        unique_cells = unique_cells[:max_cells]
    
    print(f"\nProcessing {len(unique_cells)} cells...")
    
    # Extract trajectories for each cell
    all_cell_trajectories = {}
    all_cell_metadata = {}
    
    with torch.no_grad():
        for cell_idx, cell_id in enumerate(tqdm(unique_cells, desc="Processing cells")):
            seq_indices = [idx for idx, _ in cell_sequences[cell_id]]
            
            # Extract latent vectors for all sequences of this cell
            cell_latents = []
            cell_start_indices = []
            
            for seq_idx in seq_indices:
                # Get sequence
                sample = dataset[seq_idx]
                if isinstance(sample, tuple):
                    sequence, _ = sample
                else:
                    sequence = sample['sequence']
                
                sequence = sequence.unsqueeze(0).to(device)  # [1, T, C, H, W]
                
                # Forward pass
                output = model(sequence)
                z_seq = output['z_seq']  # [1, T, ...]
                
                # Handle spatial latents
                if len(z_seq.shape) == 5:
                    # [1, T, C, H, W] -> [1, T, C]
                    B, T, C, H, W = z_seq.shape
                    z_seq = z_seq.view(B, T, C, -1).mean(dim=-1)  # Global Average Pooling
                elif len(z_seq.shape) == 4:
                    # [1, T, C, H, W] but missing batch? Handle it
                    if z_seq.shape[0] != 1:
                        z_seq = z_seq.unsqueeze(0)
                
                z_seq = z_seq.squeeze(0).cpu().detach().numpy()  # [T, hidden_dim]
                
                # Get start_idx
                start_idx = df.iloc[seq_idx]['start_idx']
                
                cell_latents.append(z_seq)
                cell_start_indices.append(start_idx)
            
            # Combine sequences (handle overlap)
            # For now, just concatenate (you can add overlap handling later)
            if len(cell_latents) > 0:
                # Simple concatenation (may have overlap, but OK for now)
                full_trajectory = np.concatenate(cell_latents, axis=0)  # [Total_T, hidden_dim]
                
                all_cell_trajectories[cell_id] = full_trajectory
                all_cell_metadata[cell_id] = {
                    'num_sequences': len(cell_latents),
                    'start_indices': cell_start_indices,
                    'trajectory_length': len(full_trajectory),
                    'latent_dim': full_trajectory.shape[1] if len(full_trajectory.shape) > 1 else full_trajectory.shape[0]
                }
    
    print(f"\nExtracted trajectories for {len(all_cell_trajectories)} cells")
    
    # Save trajectories
    trajectories_dict = {cell_id: traj.tolist() for cell_id, traj in all_cell_trajectories.items()}
    
    # Save as numpy arrays (one file per cell, or combined)
    trajectories_np = np.array([all_cell_trajectories[cell_id] for cell_id in unique_cells], dtype=object)
    
    # Save metadata
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get('epoch', 50),
        "index_csv": index_csv,
        "total_cells": len(all_cell_trajectories),
        "cell_ids": unique_cells,
        "cell_metadata": all_cell_metadata,
        "model_config": config,
        "losses": checkpoint.get('losses', {})
    }
    
    metadata_file = output_path / "full_trajectories_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_file}")
    
    # Save each cell's trajectory separately
    for cell_id, trajectory in all_cell_trajectories.items():
        traj_file = output_path / f"trajectory_{cell_id}.npy"
        np.save(traj_file, trajectory)
    
    print(f"✓ Saved {len(all_cell_trajectories)} individual trajectory files")
    
    # Also save as combined file (for easier loading)
    combined_file = output_path / "all_trajectories.npz"
    np.savez(combined_file, **{cell_id: traj for cell_id, traj in all_cell_trajectories.items()})
    print(f"✓ Saved combined file: {combined_file}")
    
    # Save summary CSV
    summary_data = []
    for cell_id in unique_cells:
        meta = all_cell_metadata[cell_id]
        summary_data.append({
            'cell_id': cell_id,
            'num_sequences': meta['num_sequences'],
            'trajectory_length': meta['trajectory_length'],
            'latent_dim': meta['latent_dim']
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_file = output_path / "trajectories_summary.csv"
    df_summary.to_csv(summary_file, index=False)
    print(f"✓ Saved summary CSV: {summary_file}")
    
    print(f"\n✅ Extraction complete! Files saved in: {output_dir}/")
    print(f"\nSummary:")
    print(f"  Total cells: {len(all_cell_trajectories)}")
    print(f"  Average trajectory length: {np.mean([len(t) for t in all_cell_trajectories.values()]):.1f} time steps")
    print(f"  Latent dimension: {all_cell_metadata[unique_cells[0]]['latent_dim']}")
    
    return all_cell_trajectories, all_cell_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract full cell trajectories (one per cell)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch_50.pt",
                       help="Path to checkpoint file")
    parser.add_argument("--index_csv", type=str, default="index.csv",
                       help="Path to index CSV file")
    parser.add_argument("--output_dir", type=str, default="latents_full_cells",
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (use 1 for individual sequences)")
    parser.add_argument("--max_cells", type=int, default=None,
                       help="Maximum number of cells to process")
    
    args = parser.parse_args()
    
    extract_full_cell_trajectories(
        checkpoint_path=args.checkpoint,
        index_csv=args.index_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_cells=args.max_cells
    )


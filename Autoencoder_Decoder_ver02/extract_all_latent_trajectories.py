#!/usr/bin/env python3
"""
Extract and save all latent trajectories for all embryos

Directory structure:
    model_version_name/
        checkpoint.pt              # Model checkpoint (copied)
        latents/
            embryo_ZS435-5.npy     # Latent trajectory [T, latent_dim]
            embryo_RS363-7.npy
            ...
        metadata.json              # Information about extraction
"""
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_ivf import IVFSequenceDataset
from model import ConvLSTMAutoencoder


def load_model(checkpoint_path: str, device: str = "cpu") -> ConvLSTMAutoencoder:
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Try to determine model architecture from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get state dict
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Default parameters (adjust based on your model)
    encoder_hidden_dim = 256
    decoder_hidden_dim = 128
    
    # Try to infer from layer names
    for key in state_dict.keys():
        if 'encoder_lstm' in key:
            if 'weight_ih_l0' in key:
                # LSTM input size is first dimension of weight_ih
                # Output size is weight_ih.shape[0] // 4 (4 gates)
                encoder_hidden_dim = state_dict[key].shape[0] // 4
        elif 'decoder_lstm' in key:
            if 'weight_ih_l0' in key:
                decoder_hidden_dim = state_dict[key].shape[0] // 4
        elif 'frame_encoder.proj' in key and 'weight' in key:
            # frame_encoder.proj maps to encoder_hidden_dim
            encoder_hidden_dim = state_dict[key].shape[0]
        elif 'frame_decoder.fc' in key and 'weight' in key:
            # frame_decoder.fc maps from decoder_hidden_dim
            # We can infer from output size: 8*8*128 = 8192
            # But this is less reliable, so we'll use default
            pass
    
    # Create model with inferred or default parameters
    model = ConvLSTMAutoencoder(
        seq_len=20,  # Default, adjust if needed
        input_channels=1,
        encoder_hidden_dim=encoder_hidden_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        encoder_layers=2,
        decoder_layers=2
    )
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning: Error loading state dict: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Encoder hidden dim: {encoder_hidden_dim}")
    print(f"  Decoder hidden dim: {decoder_hidden_dim}")
    
    return model


def extract_latent_trajectory(model, sequence, device: str = "cpu") -> np.ndarray:
    """
    Extract latent trajectory from a single sequence
    
    Args:
        model: Trained model
        sequence: Input sequence tensor [1, T, C, H, W]
        device: Device to run on
    
    Returns:
        Latent trajectory [T, latent_dim]
    """
    model.eval()
    with torch.no_grad():
        # Add batch dimension if needed
        if sequence.dim() == 4:
            sequence = sequence.unsqueeze(0)
        
        sequence = sequence.to(device)
        
        # Forward pass
        output = model(sequence)
        
        # Get latent sequence
        if isinstance(output, dict):
            z_seq = output['z_seq']  # [B, T, latent_dim]
        else:
            # Fallback: try encode method
            z_seq, _ = model.encode(sequence)
        
        # Remove batch dimension
        z_seq = z_seq.squeeze(0).cpu().numpy()  # [T, latent_dim]
        
        return z_seq


def get_unique_embryos(dataset: IVFSequenceDataset, index_csv: str = None) -> List[str]:
    """
    Get list of unique embryo IDs from dataset
    Optimized: reads directly from CSV instead of iterating dataset
    
    Args:
        dataset: IVFSequenceDataset instance
        index_csv: Path to index CSV (optional, for faster reading)
    
    Returns:
        List of unique embryo IDs
    """
    print(f"[get_unique_embryos] Starting, index_csv={index_csv}")
    
    # Fast path: read directly from CSV if available
    if index_csv:
        csv_path = Path(index_csv)
        print(f"[get_unique_embryos] Checking CSV path: {csv_path} (exists: {csv_path.exists()}, absolute: {csv_path.is_absolute()})")
        
        if csv_path.exists():
            print("Scanning CSV for unique embryos (fast method)...")
            try:
                import pandas as pd
                print(f"[get_unique_embryos] Reading CSV: {csv_path}")
                df = pd.read_csv(csv_path)
                print(f"[get_unique_embryos] CSV loaded: {len(df)} rows, columns: {df.columns.tolist()}")
                unique_embryos = sorted(df['cell_id'].unique().tolist())
                print(f"✓ Found {len(unique_embryos)} unique embryos from CSV")
                return unique_embryos
            except Exception as e:
                print(f"⚠️  CSV reading failed: {e}, falling back to slow method")
        else:
            print(f"⚠️  CSV file not found at {csv_path}, falling back to slow method")
    
    # Fallback: iterate dataset (slow but works)
    print("Scanning dataset for unique embryos (slow method)...")
    print(f"[get_unique_embryos] Dataset length: {len(dataset)}")
    unique_embryos = set()
    for i in range(len(dataset)):
        try:
            _, cell_id = dataset[i]
            unique_embryos.add(cell_id)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} sequences...", end='\r', flush=True)
        except Exception as e:
            print(f"\n⚠️  Error processing sequence {i}: {e}")
            continue
    
    print(f"\n✓ Found {len(unique_embryos)} unique embryos")
    return sorted(list(unique_embryos))


def get_embryo_sequences(embryo_id: str, index_csv: str = None) -> List[int]:
    """
    Get all sequence indices for a specific embryo
    Optimized: reads directly from CSV instead of iterating dataset
    
    Args:
        embryo_id: Embryo ID to find sequences for
        index_csv: Path to index CSV
    
    Returns:
        List of sequence indices
    """
    if index_csv:
        csv_path = Path(index_csv)
        if csv_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                # Find all rows where cell_id matches
                indices = df[df['cell_id'] == embryo_id].index.tolist()
                return indices
            except Exception as e:
                print(f"⚠️  CSV reading failed: {e}, falling back to slow method")
    
    # Fallback: return empty list (should not happen if CSV is available)
    return []


def extract_all_trajectories(
    checkpoint_path: str,
    model_version_name: str,
    index_csv: str = "index.csv",
    data_root: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = 1,
    max_embryos: Optional[int] = None,
    output_base_dir: str = "model_latents"
):
    """
    Extract and save all latent trajectories for all embryos
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_version_name: Name for this model version (e.g., "v1_baseline", "v2_no_smooth")
        index_csv: Path to index CSV file
        data_root: Root directory for data (if None, uses default)
        device: Device to run on ("cpu" or "cuda")
        batch_size: Batch size for processing
        max_embryos: Maximum number of embryos to process (None = all)
        output_base_dir: Base directory for output
    """
    print("=" * 60)
    print("Extract All Latent Trajectories")
    print("=" * 60)
    print(f"Model version: {model_version_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Create output directory structure
    output_dir = Path(output_base_dir) / model_version_name
    latents_dir = output_dir / "latents"
    output_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Copy model checkpoint to output directory
    checkpoint_dest = output_dir / "checkpoint.pt"
    if Path(checkpoint_path).exists():
        shutil.copy2(checkpoint_path, checkpoint_dest)
        print(f"✓ Copied checkpoint to {checkpoint_dest}")
    else:
        print(f"⚠️  Warning: Checkpoint not found at {checkpoint_path}")
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Load dataset
    print(f"\nLoading dataset from {index_csv}...")
    if data_root:
        # If data_root is provided, we might need to adjust dataset loading
        # For now, assume index.csv has correct paths
        dataset = IVFSequenceDataset(index_csv=index_csv, resize=128, norm="minmax01")
    else:
        dataset = IVFSequenceDataset(index_csv=index_csv, resize=128, norm="minmax01")
    
    print(f"✓ Dataset loaded: {len(dataset)} sequences")
    
    # Get unique embryos (optimized: read from CSV directly)
    unique_embryos = get_unique_embryos(dataset, index_csv=index_csv)
    
    if max_embryos:
        unique_embryos = unique_embryos[:max_embryos]
        print(f"Limited to first {max_embryos} embryos")
    
    print(f"\nProcessing {len(unique_embryos)} embryos...")
    
    # Extract trajectories for each embryo
    results = {
        "model_version": model_version_name,
        "checkpoint_path": str(checkpoint_path),
        "extraction_date": datetime.now().isoformat(),
        "device": device,
        "total_embryos": len(unique_embryos),
        "embryos": {}
    }
    
    successful = 0
    failed = 0
    
    for idx, embryo_id in enumerate(unique_embryos):
        print(f"\n[{idx + 1}/{len(unique_embryos)}] Processing embryo: {embryo_id}")
        
        try:
            # Find all sequences for this embryo (optimized: read from CSV)
            print(f"  Finding sequences for {embryo_id}...", flush=True)
            embryo_sequences = get_embryo_sequences(embryo_id, index_csv=index_csv)
            
            if len(embryo_sequences) == 0:
                print(f"  ⚠️  No sequences found for {embryo_id}")
                results["embryos"][embryo_id] = {
                    "status": "no_sequences",
                    "num_sequences": 0
                }
                failed += 1
                continue
            
            print(f"  Found {len(embryo_sequences)} sequences")
            
            # Extract latent trajectories for all sequences
            all_latents = []
            for seq_num, seq_idx in enumerate(embryo_sequences, 1):
                print(f"    Processing sequence {seq_num}/{len(embryo_sequences)} (index {seq_idx})...", flush=True)
                sequence, _ = dataset[seq_idx]
                print(f"    Sequence loaded, shape: {sequence.shape}", flush=True)
                z_seq = extract_latent_trajectory(model, sequence, device)
                print(f"    Latent extracted, shape: {z_seq.shape}", flush=True)
                all_latents.append(z_seq)
            
            # Concatenate all sequences (or handle overlap if needed)
            # For now, we'll concatenate them
            if len(all_latents) > 1:
                # Simple concatenation (you might want to handle overlap differently)
                full_trajectory = np.concatenate(all_latents, axis=0)
            else:
                full_trajectory = all_latents[0]
            
            # Save to numpy file
            output_file = latents_dir / f"embryo_{embryo_id}.npy"
            np.save(output_file, full_trajectory)
            
            print(f"  ✓ Saved: {output_file}")
            print(f"    Shape: {full_trajectory.shape}")
            print(f"    Dtype: {full_trajectory.dtype}")
            
            results["embryos"][embryo_id] = {
                "status": "success",
                "num_sequences": len(embryo_sequences),
                "trajectory_shape": list(full_trajectory.shape),
                "trajectory_dtype": str(full_trajectory.dtype),
                "file": f"latents/embryo_{embryo_id}.npy"
            }
            
            successful += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {embryo_id}: {e}")
            results["embryos"][embryo_id] = {
                "status": "error",
                "error": str(e)
            }
            failed += 1
    
    # Save metadata
    results["successful"] = successful
    results["failed"] = failed
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print("=" * 60)
    print(f"Total embryos: {len(unique_embryos)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Latent trajectories: {latents_dir}")
    print(f"  - Model checkpoint: {output_dir / 'checkpoint.pt'}")
    print(f"  - Metadata: {metadata_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save all latent trajectories for all embryos"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--model_version",
        type=str,
        required=True,
        help="Model version name (e.g., 'v1_baseline', 'v2_no_smooth')"
    )
    parser.add_argument(
        "--index_csv",
        type=str,
        default="index.csv",
        help="Path to index CSV file (default: index.csv)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root directory for data (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,  # 改為 None，讓程序自動檢測（套用成功方法的邏輯）
        help="Device to run on (default: auto-detect: cuda if available, else cpu). Valid values: 'cpu' or 'cuda'"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (default: 1)"
    )
    parser.add_argument(
        "--max_embryos",
        type=int,
        default=None,
        help="Maximum number of embryos to process (default: all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_latents",
        help="Base output directory (default: model_latents)"
    )
    
    args = parser.parse_args()
    
    # Auto-detect device (套用上次成功方法的邏輯)
    # 就像 export_all_frame_latents_direct.py 一樣，直接在 Python 中處理 device
    if args.device and args.device in ["cpu", "cuda"]:
        device = args.device
    else:
        # 如果 device 參數無效或未指定，自動檢測（就像成功的方法一樣）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚠️  Device parameter invalid or missing, auto-detected: {device}")
    
    # 如果指定了 cuda 但實際不可用，降級到 cpu
    if device == "cuda" and not torch.cuda.is_available():
        print(f"⚠️  CUDA requested but not available, falling back to CPU")
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Auto-detect paths for CHTC (staging directory)
    GROUP_BASE = Path('/staging/groups/bhaskar_group/rho9')
    
    # Auto-detect checkpoint if not absolute path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        # Try different locations (staging first)
        possible_paths = [
            GROUP_BASE / 'checkpoints' / checkpoint_path.name,  # Staging first
            checkpoint_path,  # Current directory
            Path('checkpoints') / checkpoint_path.name,
            Path.home() / 'ivf_repo' / 'checkpoints' / checkpoint_path.name,
        ]
        
        for path in possible_paths:
            if path.exists():
                checkpoint_path = path
                print(f"Found checkpoint at: {checkpoint_path}")
                break
        else:
            raise FileNotFoundError(f"Could not find checkpoint: {args.checkpoint}")
    else:
        checkpoint_path = Path(args.checkpoint)
    
    extract_all_trajectories(
        checkpoint_path=str(checkpoint_path),
        model_version_name=args.model_version,
        index_csv=args.index_csv,
        data_root=args.data_root,
        device=device,  # 使用自動檢測的 device
        batch_size=args.batch_size,
        max_embryos=args.max_embryos,
        output_base_dir=args.output_dir
    )


if __name__ == "__main__":
    main()


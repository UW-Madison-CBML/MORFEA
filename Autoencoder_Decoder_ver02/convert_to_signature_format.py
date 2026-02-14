#!/usr/bin/env python3
"""
Convert latent data to format expected by export_signatures.py

The export_signatures.py script expects:
- latents/{model_name}.npy: numpy array of shape [N, latent_dim] where N is total number of frames
- latents/{model_name}.csv: CSV with 'cell_id' column, one row per frame

This script can convert from:
1. Individual .npy files (from extract_all_latent_trajectories.py)
   - Input: model_latents/{model_version}/latents/embryo_{cell_id}.npy
2. .npz file (from export_all_frame_latents_direct.py)
   - Input: *.npz with keys: Z, cell_id, frame_in_cell, etc.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
import re


def convert_from_individual_npy_files(
    latents_dir: str,
    model_name: str,
    output_dir: str = "latents"
):
    """
    Convert individual .npy files (one per embryo) to combined format
    
    Args:
        latents_dir: Directory containing embryo_*.npy files
        model_name: Model name for output files
        output_dir: Output directory (default: "latents")
    """
    print(f"=== Converting from individual .npy files ===")
    print(f"Input directory: {latents_dir}")
    print(f"Model name: {model_name}")
    print(f"Output directory: {output_dir}")
    
    # Find all embryo .npy files
    latent_files = glob(str(Path(latents_dir) / "embryo_*.npy"))
    if len(latent_files) == 0:
        # Try alternative pattern
        latent_files = glob(str(Path(latents_dir) / "*_z.npy"))
    
    if len(latent_files) == 0:
        raise FileNotFoundError(f"No .npy files found in {latents_dir}")
    
    print(f"\nFound {len(latent_files)} latent files")
    
    # Extract cell_id from filename
    def extract_cell_id(filename):
        # Pattern 1: embryo_{cell_id}.npy
        match = re.search(r'embryo_([^_]+)\.npy', filename)
        if match:
            return match.group(1)
        # Pattern 2: {cell_id}_z.npy
        match = re.search(r'([^_]+)_z\.npy', filename)
        if match:
            return match.group(1)
        # Fallback: use filename stem
        return Path(filename).stem.replace('embryo_', '').replace('_z', '')
    
    # Load all latents and collect cell_ids
    all_latents = []
    all_cell_ids = []
    
    for latent_file in sorted(latent_files):
        cell_id = extract_cell_id(latent_file)
        print(f"  Loading {cell_id} from {Path(latent_file).name}...")
        
        z = np.load(latent_file)
        print(f"    Shape: {z.shape}")
        
        # z should be [T, latent_dim] or [T, ...]
        if len(z.shape) == 1:
            # If 1D, assume it's a single frame
            z = z.reshape(1, -1)
        elif len(z.shape) > 2:
            # Flatten extra dimensions
            z = z.reshape(z.shape[0], -1)
        
        # Add all frames from this embryo
        num_frames = z.shape[0]
        all_latents.append(z)
        all_cell_ids.extend([cell_id] * num_frames)
    
    # Concatenate all latents
    Z = np.vstack(all_latents)
    print(f"\n✓ Combined shape: {Z.shape}")
    print(f"  Total frames: {Z.shape[0]}")
    print(f"  Latent dimension: {Z.shape[1]}")
    print(f"  Unique embryos: {len(set(all_cell_ids))}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save numpy array
    npy_file = output_path / f"{model_name}.npy"
    np.save(npy_file, Z)
    print(f"\n✓ Saved numpy array to: {npy_file}")
    
    # Save CSV with cell_id
    csv_file = output_path / f"{model_name}.csv"
    df = pd.DataFrame({'cell_id': all_cell_ids})
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved CSV to: {csv_file}")
    print(f"  CSV shape: {df.shape}")
    
    # Verify consistency
    assert len(all_cell_ids) == Z.shape[0], \
        f"Mismatch: {len(all_cell_ids)} cell_ids but {Z.shape[0]} latent vectors"
    
    print(f"\n✅ Conversion complete!")
    print(f"  Files ready for export_signatures.py:")
    print(f"    - {npy_file}")
    print(f"    - {csv_file}")
    
    return npy_file, csv_file


def convert_from_npz_file(
    npz_file: str,
    model_name: str,
    output_dir: str = "latents"
):
    """
    Convert .npz file to combined format
    
    Args:
        npz_file: Path to .npz file (from export_all_frame_latents_direct.py)
        model_name: Model name for output files
        output_dir: Output directory (default: "latents")
    """
    print(f"=== Converting from .npz file ===")
    print(f"Input file: {npz_file}")
    print(f"Model name: {model_name}")
    print(f"Output directory: {output_dir}")
    
    # Load .npz file
    data = np.load(npz_file, allow_pickle=True)
    print(f"\nKeys in .npz: {list(data.keys())}")
    
    # Extract data
    if 'Z' in data:
        Z = data['Z']
    elif 'latents' in data:
        Z = data['latents']
    else:
        raise KeyError(f"Could not find 'Z' or 'latents' in .npz file. Available keys: {list(data.keys())}")
    
    if 'cell_id' in data:
        cell_ids = data['cell_id']
        # Handle both string arrays and object arrays
        if cell_ids.dtype == object:
            cell_ids = [str(cid) for cid in cell_ids]
        else:
            cell_ids = cell_ids.astype(str)
    else:
        raise KeyError(f"Could not find 'cell_id' in .npz file. Available keys: {list(data.keys())}")
    
    print(f"\nLoaded data:")
    print(f"  Z shape: {Z.shape}")
    print(f"  cell_id length: {len(cell_ids)}")
    print(f"  Unique embryos: {len(set(cell_ids))}")
    
    # Verify consistency
    if len(cell_ids) != Z.shape[0]:
        print(f"⚠️  Warning: Mismatch between cell_ids ({len(cell_ids)}) and Z ({Z.shape[0]})")
        min_len = min(len(cell_ids), Z.shape[0])
        Z = Z[:min_len]
        cell_ids = cell_ids[:min_len]
        print(f"  Truncated to {min_len} frames")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save numpy array
    npy_file = output_path / f"{model_name}.npy"
    np.save(npy_file, Z)
    print(f"\n✓ Saved numpy array to: {npy_file}")
    
    # Save CSV with cell_id
    csv_file = output_path / f"{model_name}.csv"
    df = pd.DataFrame({'cell_id': cell_ids})
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved CSV to: {csv_file}")
    print(f"  CSV shape: {df.shape}")
    
    print(f"\n✅ Conversion complete!")
    print(f"  Files ready for export_signatures.py:")
    print(f"    - {npy_file}")
    print(f"    - {csv_file}")
    
    return npy_file, csv_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert latent data to format expected by export_signatures.py"
    )
    
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["individual_npy", "npz"],
        required=True,
        help="Input format: 'individual_npy' (from extract_all_latent_trajectories.py) or 'npz' (from export_all_frame_latents_direct.py)"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input path: directory for 'individual_npy' or file path for 'npz'"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name for output files (e.g., 'v1_baseline', 'epoch50')"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="latents",
        help="Output directory (default: 'latents')"
    )
    
    args = parser.parse_args()
    
    if args.input_type == "individual_npy":
        convert_from_individual_npy_files(
            latents_dir=args.input,
            model_name=args.model_name,
            output_dir=args.output_dir
        )
    elif args.input_type == "npz":
        convert_from_npz_file(
            npz_file=args.input,
            model_name=args.model_name,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()







#!/usr/bin/env python3
"""
验证 Aadhitya 的 latent 文件格式
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys


def verify_latents(npy_path, csv_path):
    """
    验证 latent 文件格式
    
    Args:
        npy_path: Path to latents.npy
        csv_path: Path to latents.csv
    """
    print(f"NPY file: {npy_path}")
    print(f"CSV file: {csv_path}")
    print()
    
    npy_file = Path(npy_path)
    csv_file = Path(csv_path)
    
    if not npy_file.exists():
        return False
    
    if not csv_file.exists():
        return False
    
    
    try:
        Z = np.load(npy_path)
        print(f"  Shape: {Z.shape}")
        print(f"  Dtype: {Z.dtype}")
        print(f"  Size: {Z.nbytes / 1024 / 1024:.2f} MB")
    except Exception as e:
        return False
    
    try:
        df = pd.read_csv(csv_path)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
    except Exception as e:
        return False
    
    if 'cell_id' not in df.columns:
        return False
    
    if 'time_step' in df.columns:
    
    if len(df) != Z.shape[0]:
        min_len = min(len(df), Z.shape[0])
    else:
    
    
    cell_counts = df['cell_id'].value_counts().head(10)
    for cell_id, count in cell_counts.items():
        print(f"  {cell_id}: {count} frames")
    
    return True


def check_export_signatures_compatibility(npy_path, csv_path):
    """
    检查是否可以直接用于 export_signatures.py
    """
    
    df = pd.read_csv(csv_path)
    
    # - latents/{model_name}.npy: [N, latent_dim]
    
    required_columns = ['cell_id']
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        return False
    
    print(f"     cp {npy_path} latents/aadhitya_v1.npy")
    print(f"     cp {csv_path} latents/aadhitya_v1.csv")
    print("     python export_signatures.py --name aadhitya_v1")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("  python verify_aadhitya_latents.py /staging/groups/bhaskar_group/ivf/latents.npy /staging/groups/bhaskar_group/ivf/latents.csv")
        sys.exit(1)
    
    npy_path = sys.argv[1]
    csv_path = sys.argv[2]
    
    if verify_latents(npy_path, csv_path):
        check_export_signatures_compatibility(npy_path, csv_path)







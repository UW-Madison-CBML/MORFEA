#!/usr/bin/env python3
"""
验证 Aadhitya 的 latent 文件格式
在 CHTC 上运行: python3 verify_aadhitya.py
"""
import numpy as np
import pandas as pd

npy_path = "/staging/groups/bhaskar_group/ivf/latents/latents.npy"
csv_path = "/staging/groups/bhaskar_group/ivf/latents/latents.csv"

print(f"NPY: {npy_path}")
print(f"CSV: {csv_path}")
print()

try:
    Z = np.load(npy_path)
    df = pd.read_csv(csv_path)
    
    print(f"✓ NPY shape: {Z.shape}")
    print(f"✓ CSV shape: {df.shape}")
    print(f"✓ CSV columns: {df.columns.tolist()}")
    
    if 'cell_id' in df.columns:
        print(f"✓ Unique embryos: {df['cell_id'].nunique()}")
        print(f"✓ Consistent: {len(df) == Z.shape[0]}")
        
        cell_counts = df['cell_id'].value_counts().head(10)
        for cell_id, count in cell_counts.items():
            print(f"  {cell_id}: {count} frames")
        
        print("  1. mkdir -p ~/latents")
        print("  2. cp /staging/groups/bhaskar_group/ivf/latents/latents.npy ~/latents/aadhitya_v1.npy")
        print("  3. cp /staging/groups/bhaskar_group/ivf/latents/latents.csv ~/latents/aadhitya_v1.csv")
        print("  4. python export_signatures.py --name aadhitya_v1")
    else:
        
except Exception as e:
    import traceback
    traceback.print_exc()







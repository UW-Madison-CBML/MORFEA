#!/usr/bin/env python3
"""
檢查本地已下載的 latent extraction 結果
"""
import json
import numpy as np
from pathlib import Path

def check_local_results():
    print("=" * 70)
    print("=" * 70)
    
    base_dir = Path("model_latents/v1_baseline")
    
    if not base_dir.exists():
        print("  scp -r rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/model_latents/v1_baseline ./model_latents/")
        return
    
    
    metadata_file = base_dir / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    if isinstance(value, list) and len(value) > 0:
                else:
                    print(f"   - {key}: {value}")
        except Exception as e:
    else:
    
    latents_dir = base_dir / "latents"
    if not latents_dir.exists():
        return
    
    npy_files = sorted(list(latents_dir.glob("*.npy")))
    
    if not npy_files:
        return
    
    
    total_size = sum(f.stat().st_size for f in npy_files) / (1024 * 1024)  # MB
    
    
    for i, file in enumerate(npy_files[:10], 1):
        try:
            data = np.load(file)
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   {i:2d}. {file.name}")
        except Exception as e:
    
    if len(npy_files) > 10:
    
    checkpoint_file = base_dir / "checkpoint.pt"
    if checkpoint_file.exists():
        size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
    else:
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_local_results()


#!/usr/bin/env python3
"""
檢查本地已下載的 latent extraction 結果
"""
import json
import numpy as np
from pathlib import Path

def check_local_results():
    print("=" * 70)
    print("本地 Latent Vector Extraction 結果檢查")
    print("=" * 70)
    
    base_dir = Path("model_latents/v1_baseline")
    
    if not base_dir.exists():
        print(f"\n⚠️  結果目錄不存在: {base_dir}")
        print("\n請先從 CHTC 下載結果：")
        print("  scp -r rho9@ap2001.chtc.wisc.edu:/staging/groups/bhaskar_group/rho9/model_latents/v1_baseline ./model_latents/")
        return
    
    print(f"\n📁 結果目錄: {base_dir}")
    
    # 檢查 metadata
    metadata_file = base_dir / "metadata.json"
    if metadata_file.exists():
        print(f"\n✅ 找到 metadata.json")
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"\n   Metadata 內容:")
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    print(f"   - {key}: {type(value).__name__} (長度: {len(value)})")
                    if isinstance(value, list) and len(value) > 0:
                        print(f"     前3項: {value[:3]}")
                else:
                    print(f"   - {key}: {value}")
        except Exception as e:
            print(f"   ⚠️  讀取 metadata 時發生錯誤: {e}")
    else:
        print(f"\n⚠️  metadata.json 不存在")
    
    # 檢查 latents 目錄
    latents_dir = base_dir / "latents"
    if not latents_dir.exists():
        print(f"\n⚠️  latents 目錄不存在: {latents_dir}")
        return
    
    npy_files = sorted(list(latents_dir.glob("*.npy")))
    
    if not npy_files:
        print(f"\n⚠️  latents 目錄為空，沒有找到 .npy 檔案")
        return
    
    print(f"\n✅ 找到 {len(npy_files)} 個 latent vector 檔案")
    
    # 統計資訊
    total_size = sum(f.stat().st_size for f in npy_files) / (1024 * 1024)  # MB
    
    print(f"\n   統計:")
    print(f"   - 總檔案數: {len(npy_files)}")
    print(f"   - 總大小: {total_size:.2f} MB")
    
    # 檢查前幾個檔案
    print(f"\n   檔案詳細資訊（前10個）:")
    for i, file in enumerate(npy_files[:10], 1):
        try:
            data = np.load(file)
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   {i:2d}. {file.name}")
            print(f"       大小: {size_mb:.3f} MB")
            print(f"       形狀: {data.shape}")
            print(f"       資料類型: {data.dtype}")
            print(f"       數值範圍: [{data.min():.4f}, {data.max():.4f}]")
        except Exception as e:
            print(f"   {i:2d}. {file.name} (讀取錯誤: {e})")
    
    if len(npy_files) > 10:
        print(f"\n   ... 還有 {len(npy_files) - 10} 個檔案")
    
    # 檢查 checkpoint
    checkpoint_file = base_dir / "checkpoint.pt"
    if checkpoint_file.exists():
        size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
        print(f"\n✅ 找到 checkpoint.pt ({size_mb:.2f} MB)")
    else:
        print(f"\n⚠️  checkpoint.pt 不存在")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_local_results()


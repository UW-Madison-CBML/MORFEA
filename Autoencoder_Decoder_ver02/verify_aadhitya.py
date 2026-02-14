#!/usr/bin/env python3
"""
验证 Aadhitya 的 latent 文件格式
在 CHTC 上运行: python3 verify_aadhitya.py
"""
import numpy as np
import pandas as pd

npy_path = "/staging/groups/bhaskar_group/ivf/latents/latents.npy"
csv_path = "/staging/groups/bhaskar_group/ivf/latents/latents.csv"

print("=== 验证 Aadhitya 的 Latent 文件 ===")
print(f"NPY: {npy_path}")
print(f"CSV: {csv_path}")
print()

try:
    # 加载文件
    print("加载文件...")
    Z = np.load(npy_path)
    df = pd.read_csv(csv_path)
    
    print(f"✓ NPY shape: {Z.shape}")
    print(f"✓ CSV shape: {df.shape}")
    print(f"✓ CSV columns: {df.columns.tolist()}")
    
    # 检查 cell_id 列
    if 'cell_id' in df.columns:
        print(f"✓ Unique embryos: {df['cell_id'].nunique()}")
        print(f"✓ Consistent: {len(df) == Z.shape[0]}")
        
        # 显示前几个胚胎的统计
        print("\n前10个胚胎的帧数:")
        cell_counts = df['cell_id'].value_counts().head(10)
        for cell_id, count in cell_counts.items():
            print(f"  {cell_id}: {count} frames")
        
        print("\n✅ 格式正确！可以直接使用 export_signatures.py")
        print("\n下一步操作:")
        print("  1. mkdir -p ~/latents")
        print("  2. cp /staging/groups/bhaskar_group/ivf/latents/latents.npy ~/latents/aadhitya_v1.npy")
        print("  3. cp /staging/groups/bhaskar_group/ivf/latents/latents.csv ~/latents/aadhitya_v1.csv")
        print("  4. python export_signatures.py --name aadhitya_v1")
    else:
        print("\n⚠️  警告: CSV 缺少 'cell_id' 列")
        print(f"  可用列: {df.columns.tolist()}")
        print("\n需要转换格式，可以使用 convert_to_signature_format.py")
        
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()







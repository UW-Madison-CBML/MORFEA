#!/bin/bash

NPY_PATH="/staging/groups/bhaskar_group/ivf/latents/latents.npy"
CSV_PATH="/staging/groups/bhaskar_group/ivf/latents/latents.csv"

echo "NPY: $NPY_PATH"
echo "CSV: $CSV_PATH"
echo ""

if [ ! -f "$NPY_PATH" ]; then
    exit 1
fi

if [ ! -f "$CSV_PATH" ]; then
    exit 1
fi

echo ""

ls -lh "$NPY_PATH"
ls -lh "$CSV_PATH"
echo ""

python3 << EOF
import numpy as np
import pandas as pd
import sys

npy_path = "$NPY_PATH"
csv_path = "$CSV_PATH"

print("=== 加载并验证文件 ===")
try:
    print("加载 NPY 文件...")
    Z = np.load(npy_path)
    print(f"✓ NPY 加载成功")
    print(f"  Shape: {Z.shape}")
    print(f"  Dtype: {Z.dtype}")
    print(f"  Size: {Z.nbytes / 1024 / 1024:.2f} MB")
    
    print("\n加载 CSV 文件...")
    df = pd.read_csv(csv_path)
    print(f"✓ CSV 加载成功")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    
    print("\n=== 检查格式 ===")
    if 'cell_id' not in df.columns:
        print("❌ 错误: CSV 缺少 'cell_id' 列")
        print(f"  可用列: {df.columns.tolist()}")
        sys.exit(1)
    print("✓ 找到 'cell_id' 列")
    
    if 'time_step' in df.columns:
        print("✓ 找到 'time_step' 列（可选）")
    
    print("\n=== 验证数据一致性 ===")
    if len(df) != Z.shape[0]:
        print(f"⚠️  警告: CSV 行数 ({len(df)}) 与 NPY 行数 ({Z.shape[0]}) 不匹配")
        min_len = min(len(df), Z.shape[0])
        print(f"  建议使用前 {min_len} 行")
    else:
        print(f"✓ CSV 和 NPY 行数一致: {len(df)}")
    
    print("\n=== 数据统计 ===")
    print(f"  总帧数: {len(df)}")
    print(f"  唯一胚胎数: {df['cell_id'].nunique()}")
    print(f"  Latent 维度: {Z.shape[1]}")
    
    print("\n每个胚胎的帧数（前10个）:")
    cell_counts = df['cell_id'].value_counts().head(10)
    for cell_id, count in cell_counts.items():
        print(f"  {cell_id}: {count} frames")
    
    print("\n✅ 验证完成！文件格式正确，可以直接使用 export_signatures.py")
    print("\n使用方法:")
    print("  1. 复制文件到 latents/ 目录:")
    print("     mkdir -p ~/latents")
    print("     cp $NPY_PATH ~/latents/aadhitya_v1.npy")
    print("     cp $CSV_PATH ~/latents/aadhitya_v1.csv")
    print("  2. 运行 Path Signature 计算:")
    print("     python export_signatures.py --name aadhitya_v1")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF







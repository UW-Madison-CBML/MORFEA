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
    print("=== 验证 Aadhitya 的 Latent 文件 ===")
    print(f"NPY file: {npy_path}")
    print(f"CSV file: {csv_path}")
    print()
    
    # 检查文件是否存在
    npy_file = Path(npy_path)
    csv_file = Path(csv_path)
    
    if not npy_file.exists():
        print(f"❌ 错误: {npy_path} 不存在")
        return False
    
    if not csv_file.exists():
        print(f"❌ 错误: {csv_path} 不存在")
        return False
    
    print("✓ 文件存在")
    
    # 加载文件
    print("\n加载文件...")
    try:
        Z = np.load(npy_path)
        print(f"✓ NPY 加载成功")
        print(f"  Shape: {Z.shape}")
        print(f"  Dtype: {Z.dtype}")
        print(f"  Size: {Z.nbytes / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"❌ 加载 NPY 失败: {e}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ CSV 加载成功")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"❌ 加载 CSV 失败: {e}")
        return False
    
    # 检查必需的列
    print("\n检查 CSV 格式...")
    if 'cell_id' not in df.columns:
        print(f"❌ 错误: CSV 缺少 'cell_id' 列")
        print(f"  可用列: {df.columns.tolist()}")
        return False
    print("✓ 找到 'cell_id' 列")
    
    if 'time_step' in df.columns:
        print("✓ 找到 'time_step' 列（可选）")
    
    # 验证一致性
    print("\n验证数据一致性...")
    if len(df) != Z.shape[0]:
        print(f"⚠️  警告: CSV 行数 ({len(df)}) 与 NPY 行数 ({Z.shape[0]}) 不匹配")
        min_len = min(len(df), Z.shape[0])
        print(f"  建议使用前 {min_len} 行")
    else:
        print("✓ CSV 和 NPY 行数一致")
    
    # 统计信息
    print("\n数据统计:")
    print(f"  总帧数: {len(df)}")
    print(f"  唯一胚胎数: {df['cell_id'].nunique()}")
    print(f"  Latent 维度: {Z.shape[1]}")
    
    # 显示每个胚胎的帧数（前10个）
    print("\n每个胚胎的帧数（前10个）:")
    cell_counts = df['cell_id'].value_counts().head(10)
    for cell_id, count in cell_counts.items():
        print(f"  {cell_id}: {count} frames")
    
    print("\n✅ 验证完成！")
    return True


def check_export_signatures_compatibility(npy_path, csv_path):
    """
    检查是否可以直接用于 export_signatures.py
    """
    print("\n=== 检查 export_signatures.py 兼容性 ===")
    
    df = pd.read_csv(csv_path)
    
    # export_signatures.py 期望的格式：
    # - latents/{model_name}.npy: [N, latent_dim]
    # - latents/{model_name}.csv: 只需要 'cell_id' 列
    
    required_columns = ['cell_id']
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        print(f"❌ 缺少必需的列: {missing}")
        return False
    
    print("✓ 格式兼容 export_signatures.py")
    print("\n使用方法:")
    print("  1. 复制文件到 latents/ 目录:")
    print(f"     cp {npy_path} latents/aadhitya_v1.npy")
    print(f"     cp {csv_path} latents/aadhitya_v1.csv")
    print("  2. 运行:")
    print("     python export_signatures.py --name aadhitya_v1")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python verify_aadhitya_latents.py <npy_path> <csv_path>")
        print("\n例子:")
        print("  python verify_aadhitya_latents.py /staging/groups/bhaskar_group/ivf/latents.npy /staging/groups/bhaskar_group/ivf/latents.csv")
        sys.exit(1)
    
    npy_path = sys.argv[1]
    csv_path = sys.argv[2]
    
    if verify_latents(npy_path, csv_path):
        check_export_signatures_compatibility(npy_path, csv_path)







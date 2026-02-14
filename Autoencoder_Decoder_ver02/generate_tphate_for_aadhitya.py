#!/usr/bin/env python3
"""
为 Aadhitya 的 latents 生成 T-PHATE 和 Curvature plot

注意：此脚本直接读取已存储的 latent vectors，不进行 inference。

输入格式:
- latents.npy: [N, latent_dim] - 所有胚胎的所有帧（已存储的 latent vectors）
- latents.csv: 包含 cell_id 和 time_step 列

输出:
- tphate_plots/ 文件夹: 包含每个胚胎的 3D T-PHATE plot（按时间着色）
- curvature_plots/ 文件夹: 包含每个胚胎的 3D T-PHATE plot（按曲率着色）
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Try to import tphate
try:
    import tphate
    TPHATE_AVAILABLE = True
except ImportError:
    TPHATE_AVAILABLE = False
    print("Warning: tphate not available. Install with: pip install tphate")


def apply_tphate_3d(latents, n_components=3, knn=5):
    """
    应用 3D T-PHATE
    
    Args:
        latents: numpy array [T, latent_dim]
        n_components: 输出维度 (应该是 3)
        knn: k-nearest neighbors
    
    Returns:
        embedding: numpy array [T, 3]
    """
    if not TPHATE_AVAILABLE:
        raise RuntimeError("TPHATE is required but not available. Install with: pip install tphate")
    
    print(f"  应用 3D T-PHATE (knn={knn}, n_components={n_components})...")
    tph = tphate.TPHATE(n_components=n_components, knn=knn, verbose=0)
    embedding = tph.fit_transform(latents)
    print(f"  ✓ 3D T-PHATE embedding shape: {embedding.shape}")
    return embedding


def compute_curvature(trajectory_3d):
    """
    计算 3D 轨迹的曲率
    
    Args:
        trajectory_3d: numpy array [T, 3]
    
    Returns:
        curvatures: numpy array [T]
    """
    T = trajectory_3d.shape[0]
    curvatures = np.zeros(T)
    
    for i in range(1, T-1):
        p_prev = trajectory_3d[i-1]
        p = trajectory_3d[i]
        p_next = trajectory_3d[i+1]
        
        # 计算三角形边长
        a = np.linalg.norm(p - p_prev)
        b = np.linalg.norm(p_next - p)
        c = np.linalg.norm(p_next - p_prev)
        
        # Heron's formula 计算三角形面积
        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)
        area = np.sqrt(np.maximum(area_sq, 0))
        
        # 计算曲率: kappa = 4 * area / (a * b * c)
        if area > 0 and a > 0 and b > 0 and c > 0:
            kappa = 4 * area / (a * b * c)
            curvatures[i] = kappa
        else:
            curvatures[i] = 0
    
    # 首尾点: 设置为邻居的值
    curvatures[0] = curvatures[1] if T > 1 else 0
    curvatures[-1] = curvatures[-2] if T > 1 else 0
    
    return curvatures


def plot_tphate_3d_trajectory(tphate_3d, cell_id, save_path):
    """
    绘制 3D T-PHATE 轨迹图（按时间着色）
    
    Args:
        tphate_3d: numpy array [T, 3]
        cell_id: 胚胎 ID
        save_path: 保存路径
    """
    print(f"  绘制 3D T-PHATE plot...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 按时间着色绘制散点
    scatter = ax.scatter(tphate_3d[:, 0], tphate_3d[:, 1], tphate_3d[:, 2],
                        c=range(len(tphate_3d)), cmap='viridis', s=50, alpha=0.7)
    
    # 添加轨迹线
    ax.plot(tphate_3d[:, 0], tphate_3d[:, 1], tphate_3d[:, 2],
           'k-', alpha=0.3, linewidth=1)
    
    # 标记起点和终点
    ax.scatter(tphate_3d[0, 0], tphate_3d[0, 1], tphate_3d[0, 2],
              c='green', s=200, marker='o', label='Start', edgecolors='black')
    ax.scatter(tphate_3d[-1, 0], tphate_3d[-1, 1], tphate_3d[-1, 2],
              c='red', s=200, marker='s', label='End', edgecolors='black')
    
    ax.set_xlabel('T-PHATE Component 1', fontsize=12)
    ax.set_ylabel('T-PHATE Component 2', fontsize=12)
    ax.set_zlabel('T-PHATE Component 3', fontsize=12)
    ax.set_title(f'T-PHATE Trajectory (3D): {cell_id}\n({len(tphate_3d)} time points)', 
                 fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Time Step', rotation=270, labelpad=20)
    
    ax.legend()
    
    # 添加统计信息
    trajectory_length = np.sum(np.sqrt(np.sum(np.diff(tphate_3d, axis=0)**2, axis=1)))
    textstr = f'Time Points: {len(tphate_3d)}\nTrajectory Length: {trajectory_length:.3f}'
    ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=75, bbox_inches='tight')  # 降低 DPI 到 75 以减少文件大小
    plt.close()
    
    print(f"  ✓ Saved to {save_path}")


def plot_curvature_tphate(tphate_3d, curvatures, cell_id, save_path):
    """
    绘制按曲率着色的 3D T-PHATE 轨迹图
    
    Args:
        tphate_3d: numpy array [T, 3]
        curvatures: numpy array [T]
        cell_id: 胚胎 ID
        save_path: 保存路径
    """
    print(f"  绘制 Curvature T-PHATE plot...")
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 按曲率着色绘制散点
    scatter = ax.scatter(tphate_3d[:, 0], tphate_3d[:, 1], tphate_3d[:, 2],
                        c=curvatures, cmap='viridis', s=50, alpha=0.7)
    
    # 添加轨迹线
    ax.plot(tphate_3d[:, 0], tphate_3d[:, 1], tphate_3d[:, 2],
           'k-', alpha=0.3, linewidth=1)
    
    # 标记起点和终点
    ax.scatter(tphate_3d[0, 0], tphate_3d[0, 1], tphate_3d[0, 2],
              c='green', s=200, marker='o', label='Start', edgecolors='black')
    ax.scatter(tphate_3d[-1, 0], tphate_3d[-1, 1], tphate_3d[-1, 2],
              c='red', s=200, marker='s', label='End', edgecolors='black')
    
    ax.set_xlabel('T-PHATE Component 1', fontsize=12)
    ax.set_ylabel('T-PHATE Component 2', fontsize=12)
    ax.set_zlabel('T-PHATE Component 3', fontsize=12)
    ax.set_title(f'T-PHATE Trajectory (3D) colored by Curvature: {cell_id}\n({len(tphate_3d)} time points)', 
                 fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Curvature', rotation=270, labelpad=20)
    
    ax.legend()
    
    # 添加统计信息
    max_curvature = np.max(curvatures)
    mean_curvature = np.mean(curvatures)
    textstr = f'Max Curvature: {max_curvature:.6f}\nMean Curvature: {mean_curvature:.6f}'
    ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=75, bbox_inches='tight')  # 降低 DPI 到 75 以减少文件大小
    plt.close()
    
    print(f"  ✓ Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='为 Aadhitya 的 latents 生成 T-PHATE 和 Curvature plot')
    parser.add_argument('--npy_file', type=str, 
                       default='/staging/groups/bhaskar_group/ivf/latents/latents.npy',
                       help='Path to latents.npy file')
    parser.add_argument('--csv_file', type=str,
                       default='/staging/groups/bhaskar_group/ivf/latents/latents.csv',
                       help='Path to latents.csv file')
    parser.add_argument('--output_base', type=str, default='aadhitya_v1',
                       help='Base directory name for output (default: aadhitya_v1)')
    parser.add_argument('--knn', type=int, default=5,
                       help='k-nearest neighbors for T-PHATE (default: 5)')
    parser.add_argument('--max_embryos', type=int, default=None,
                       help='Maximum number of embryos to process (None = all)')
    parser.add_argument('--val_set_file', type=str, default=None,
                       help='Path to file containing validation set cell_id list (one per line)')
    parser.add_argument('--val_set_csv', type=str, default=None,
                       help='Path to CSV file with cell_id column for validation set')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip embryos that already have output files')
    parser.add_argument('--start_from', type=int, default=0,
                       help='Start processing from this index (0-based, skip first N embryos)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("为 Aadhitya 的 Latents 生成 T-PHATE 和 Curvature Plot")
    print("=" * 60)
    print(f"NPY file: {args.npy_file}")
    print(f"CSV file: {args.csv_file}")
    print(f"Output base: {args.output_base}")
    print(f"KNN: {args.knn}")
    print("=" * 60)
    
    # 检查 TPHATE 是否可用
    if not TPHATE_AVAILABLE:
        raise RuntimeError("TPHATE is required but not available. Install with: pip install tphate")
    
    # 加载数据
    print("\n加载数据...")
    Z = np.load(args.npy_file)
    df = pd.read_csv(args.csv_file)
    
    print(f"✓ NPY shape: {Z.shape}")
    print(f"✓ CSV shape: {df.shape}")
    print(f"✓ CSV columns: {df.columns.tolist()}")
    
    # 验证数据一致性
    if len(df) != Z.shape[0]:
        print(f"⚠️  警告: CSV 行数 ({len(df)}) 与 NPY 行数 ({Z.shape[0]}) 不匹配")
        min_len = min(len(df), Z.shape[0])
        df = df.iloc[:min_len]
        Z = Z[:min_len]
        print(f"  截断到 {min_len} 行")
    
    # 按 cell_id 分组
    print(f"\n按 cell_id 分组...")
    grouped = df.groupby('cell_id')
    all_unique_embryos = df['cell_id'].unique()
    print(f"✓ 找到 {len(all_unique_embryos)} 个唯一胚胎")
    
    # 筛选 validation set（如果指定）
    unique_embryos = all_unique_embryos
    if args.val_set_file or args.val_set_csv:
        val_cell_ids = set()
        
        if args.val_set_file:
            print(f"\n从文件读取 validation set: {args.val_set_file}")
            val_file = Path(args.val_set_file)
            if val_file.exists():
                with open(val_file, 'r') as f:
                    val_cell_ids = set(line.strip() for line in f if line.strip())
                print(f"  ✓ 读取了 {len(val_cell_ids)} 个 validation cell_id")
            else:
                print(f"  ⚠️  文件不存在: {args.val_set_file}")
        
        if args.val_set_csv:
            print(f"\n从 CSV 读取 validation set: {args.val_set_csv}")
            val_csv = Path(args.val_set_csv)
            if val_csv.exists():
                val_df = pd.read_csv(val_csv)
                if 'cell_id' in val_df.columns:
                    val_cell_ids.update(val_df['cell_id'].astype(str).tolist())
                    print(f"  ✓ 从 CSV 读取了 {len(val_df)} 个 validation cell_id")
                else:
                    print(f"  ⚠️  CSV 文件缺少 'cell_id' 列")
            else:
                print(f"  ⚠️  文件不存在: {args.val_set_csv}")
        
        if val_cell_ids:
            unique_embryos = [eid for eid in all_unique_embryos if str(eid) in val_cell_ids]
            print(f"  ✓ Validation set 包含 {len(unique_embryos)} 个胚胎")
        else:
            print(f"  ⚠️  没有找到 validation set，使用所有胚胎")
    
    # 跳过前N个胚胎（如果指定）
    if args.start_from > 0:
        if args.start_from < len(unique_embryos):
            unique_embryos = unique_embryos[args.start_from:]
            print(f"  跳过前 {args.start_from} 个胚胎，从第 {args.start_from + 1} 个开始")
        else:
            print(f"  ⚠️  start_from ({args.start_from}) >= 总胚胎数 ({len(unique_embryos)})")
            return
    
    if args.max_embryos:
        unique_embryos = unique_embryos[:args.max_embryos]
        print(f"  限制处理前 {args.max_embryos} 个胚胎")
    
    # 创建输出目录
    output_dir = Path(args.output_base)
    tphate_dir = output_dir / 'tphate_plots'
    curvature_dir = output_dir / 'curvature_plots'
    
    tphate_dir.mkdir(parents=True, exist_ok=True)
    curvature_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n输出目录:")
    print(f"  - T-PHATE plots: {tphate_dir}")
    print(f"  - Curvature plots: {curvature_dir}")
    
    # 处理每个胚胎
    print(f"\n处理 {len(unique_embryos)} 个胚胎...")
    
    for idx, cell_id in enumerate(unique_embryos, 1):
        print(f"\n[{idx}/{len(unique_embryos)}] 处理胚胎: {cell_id}")
        
        # 如果启用 skip_existing，检查文件是否已存在
        if args.skip_existing:
            tphate_file = tphate_dir / f'{cell_id}_tphate.png'
            curvature_file = curvature_dir / f'{cell_id}_curvature.png'
            if tphate_file.exists() and curvature_file.exists():
                print(f"  ⏭️  跳过（文件已存在）")
                continue
        
        # 获取该胚胎的所有行
        embryo_rows = grouped.get_group(cell_id)
        embryo_indices = embryo_rows.index.values
        
        # 如果有 time_step 列，按时间排序
        if 'time_step' in df.columns:
            embryo_rows = embryo_rows.sort_values('time_step')
            embryo_indices = embryo_rows.index.values
        
        # 提取该胚胎的 latent vectors
        embryo_latents = Z[embryo_indices]
        print(f"  Latent shape: {embryo_latents.shape}")
        
        if len(embryo_latents) < 3:
            print(f"  ⚠️  胚胎帧数太少 ({len(embryo_latents)} < 3), 跳过")
            continue
        
        try:
            # 1. 生成 3D T-PHATE（只计算一次，同时用于两种plot）
            tphate_3d = apply_tphate_3d(embryo_latents, n_components=3, knn=args.knn)
            
            # 2. 生成 3D T-PHATE plot（按时间着色）
            tphate_plot_path = tphate_dir / f'{cell_id}_tphate.png'
            plot_tphate_3d_trajectory(tphate_3d, cell_id, tphate_plot_path)
            
            # 3. 计算曲率
            curvatures = compute_curvature(tphate_3d)
            print(f"  ✓ 曲率计算完成 (min={curvatures.min():.6f}, max={curvatures.max():.6f}, mean={curvatures.mean():.6f})")
            
            # 4. 生成 curvature plot（按曲率着色）
            curvature_plot_path = curvature_dir / f'{cell_id}_curvature.png'
            plot_curvature_tphate(tphate_3d, curvatures, cell_id, curvature_plot_path)
            
            print(f"  ✅ 完成: {cell_id}")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"T-PHATE plots: {tphate_dir}")
    print(f"  - 生成了 {len(list(tphate_dir.glob('*.png')))} 个文件")
    print(f"Curvature plots: {curvature_dir}")
    print(f"  - 生成了 {len(list(curvature_dir.glob('*.png')))} 个文件")


if __name__ == '__main__':
    main()


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
        
        a = np.linalg.norm(p - p_prev)
        b = np.linalg.norm(p_next - p)
        c = np.linalg.norm(p_next - p_prev)
        
        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)
        area = np.sqrt(np.maximum(area_sq, 0))
        
        if area > 0 and a > 0 and b > 0 and c > 0:
            kappa = 4 * area / (a * b * c)
            curvatures[i] = kappa
        else:
            curvatures[i] = 0
    
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
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(tphate_3d[:, 0], tphate_3d[:, 1], tphate_3d[:, 2],
                        c=range(len(tphate_3d)), cmap='viridis', s=50, alpha=0.7)
    
    ax.plot(tphate_3d[:, 0], tphate_3d[:, 1], tphate_3d[:, 2],
           'k-', alpha=0.3, linewidth=1)
    
    ax.scatter(tphate_3d[0, 0], tphate_3d[0, 1], tphate_3d[0, 2],
              c='green', s=200, marker='o', label='Start', edgecolors='black')
    ax.scatter(tphate_3d[-1, 0], tphate_3d[-1, 1], tphate_3d[-1, 2],
              c='red', s=200, marker='s', label='End', edgecolors='black')
    
    ax.set_xlabel('T-PHATE Component 1', fontsize=12)
    ax.set_ylabel('T-PHATE Component 2', fontsize=12)
    ax.set_zlabel('T-PHATE Component 3', fontsize=12)
    ax.set_title(f'T-PHATE Trajectory (3D): {cell_id}\n({len(tphate_3d)} time points)', 
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Time Step', rotation=270, labelpad=20)
    
    ax.legend()
    
    trajectory_length = np.sum(np.sqrt(np.sum(np.diff(tphate_3d, axis=0)**2, axis=1)))
    textstr = f'Time Points: {len(tphate_3d)}\nTrajectory Length: {trajectory_length:.3f}'
    ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=75, bbox_inches='tight')
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
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(tphate_3d[:, 0], tphate_3d[:, 1], tphate_3d[:, 2],
                        c=curvatures, cmap='viridis', s=50, alpha=0.7)
    
    ax.plot(tphate_3d[:, 0], tphate_3d[:, 1], tphate_3d[:, 2],
           'k-', alpha=0.3, linewidth=1)
    
    ax.scatter(tphate_3d[0, 0], tphate_3d[0, 1], tphate_3d[0, 2],
              c='green', s=200, marker='o', label='Start', edgecolors='black')
    ax.scatter(tphate_3d[-1, 0], tphate_3d[-1, 1], tphate_3d[-1, 2],
              c='red', s=200, marker='s', label='End', edgecolors='black')
    
    ax.set_xlabel('T-PHATE Component 1', fontsize=12)
    ax.set_ylabel('T-PHATE Component 2', fontsize=12)
    ax.set_zlabel('T-PHATE Component 3', fontsize=12)
    ax.set_title(f'T-PHATE Trajectory (3D) colored by Curvature: {cell_id}\n({len(tphate_3d)} time points)', 
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Curvature', rotation=270, labelpad=20)
    
    ax.legend()
    
    max_curvature = np.max(curvatures)
    mean_curvature = np.mean(curvatures)
    textstr = f'Max Curvature: {max_curvature:.6f}\nMean Curvature: {mean_curvature:.6f}'
    ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=75, bbox_inches='tight')
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
    print("=" * 60)
    print(f"NPY file: {args.npy_file}")
    print(f"CSV file: {args.csv_file}")
    print(f"Output base: {args.output_base}")
    print(f"KNN: {args.knn}")
    print("=" * 60)
    
    if not TPHATE_AVAILABLE:
        raise RuntimeError("TPHATE is required but not available. Install with: pip install tphate")
    
    Z = np.load(args.npy_file)
    df = pd.read_csv(args.csv_file)
    
    print(f"✓ NPY shape: {Z.shape}")
    print(f"✓ CSV shape: {df.shape}")
    print(f"✓ CSV columns: {df.columns.tolist()}")
    
    if len(df) != Z.shape[0]:
        min_len = min(len(df), Z.shape[0])
        df = df.iloc[:min_len]
        Z = Z[:min_len]
    
    grouped = df.groupby('cell_id')
    all_unique_embryos = df['cell_id'].unique()
    
    unique_embryos = all_unique_embryos
    if args.val_set_file or args.val_set_csv:
        val_cell_ids = set()
        
        if args.val_set_file:
            val_file = Path(args.val_set_file)
            if val_file.exists():
                with open(val_file, 'r') as f:
                    val_cell_ids = set(line.strip() for line in f if line.strip())
            else:
        
        if args.val_set_csv:
            val_csv = Path(args.val_set_csv)
            if val_csv.exists():
                val_df = pd.read_csv(val_csv)
                if 'cell_id' in val_df.columns:
                    val_cell_ids.update(val_df['cell_id'].astype(str).tolist())
                else:
            else:
        
        if val_cell_ids:
            unique_embryos = [eid for eid in all_unique_embryos if str(eid) in val_cell_ids]
        else:
    
    if args.start_from > 0:
        if args.start_from < len(unique_embryos):
            unique_embryos = unique_embryos[args.start_from:]
        else:
            return
    
    if args.max_embryos:
        unique_embryos = unique_embryos[:args.max_embryos]
    
    output_dir = Path(args.output_base)
    tphate_dir = output_dir / 'tphate_plots'
    curvature_dir = output_dir / 'curvature_plots'
    
    tphate_dir.mkdir(parents=True, exist_ok=True)
    curvature_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  - T-PHATE plots: {tphate_dir}")
    print(f"  - Curvature plots: {curvature_dir}")
    
    
    for idx, cell_id in enumerate(unique_embryos, 1):
        
        if args.skip_existing:
            tphate_file = tphate_dir / f'{cell_id}_tphate.png'
            curvature_file = curvature_dir / f'{cell_id}_curvature.png'
            if tphate_file.exists() and curvature_file.exists():
                continue
        
        embryo_rows = grouped.get_group(cell_id)
        embryo_indices = embryo_rows.index.values
        
        if 'time_step' in df.columns:
            embryo_rows = embryo_rows.sort_values('time_step')
            embryo_indices = embryo_rows.index.values
        
        embryo_latents = Z[embryo_indices]
        print(f"  Latent shape: {embryo_latents.shape}")
        
        if len(embryo_latents) < 3:
            continue
        
        try:
            tphate_3d = apply_tphate_3d(embryo_latents, n_components=3, knn=args.knn)
            
            tphate_plot_path = tphate_dir / f'{cell_id}_tphate.png'
            plot_tphate_3d_trajectory(tphate_3d, cell_id, tphate_plot_path)
            
            curvatures = compute_curvature(tphate_3d)
            
            curvature_plot_path = curvature_dir / f'{cell_id}_curvature.png'
            plot_curvature_tphate(tphate_3d, curvatures, cell_id, curvature_plot_path)
            
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("=" * 60)
    print(f"T-PHATE plots: {tphate_dir}")
    print(f"Curvature plots: {curvature_dir}")


if __name__ == '__main__':
    main()


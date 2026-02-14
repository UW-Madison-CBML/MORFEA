"""
Step 2: Preprocess Latents (嚴謹版)
- 標準化 latent vectors (z-score)
- (可選) PCA 降維去噪
- 輸出: Z_norm, Z_pca (可選)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def preprocess_latents(
    input_file="latents_all_frames.npz",
    output_file="latents_preprocessed.npz",
    do_pca=True,
    pca_components=32,
    remove_outliers=True,
    outlier_std_threshold=5.0
):
    """
    預處理 latent vectors
    
    Args:
        input_file: 輸入 .npz 文件（來自 Step 1）
        output_file: 輸出 .npz 文件
        do_pca: 是否做 PCA 降維
        pca_components: PCA 降到的維度
        remove_outliers: 是否移除異常值
        outlier_std_threshold: 異常值標準差閾值
    """
    print(f"=== Step 2: Preprocess Latents ===")
    
    # Load data
    print(f"\nLoading data from: {input_file}")
    data = np.load(input_file, allow_pickle=True)
    Z = data['Z']  # [N_frames, D]
    cell_id = data['cell_id']
    frame_in_cell = data['frame_in_cell']
    abs_time = data['abs_time'] if 'abs_time' in data else None
    sequence_idx = data['sequence_idx'] if 'sequence_idx' in data else None
    paths = data['paths'] if 'paths' in data else None
    
    print(f"  Original shape: {Z.shape}")
    print(f"  Total frames: {len(Z)}")
    
    # Step 2.1: Remove outliers (optional)
    if remove_outliers:
        print("\n[2.1] Removing outliers...")
        # 計算每個維度的 mean 和 std
        Z_mean = Z.mean(axis=0, keepdims=True)
        Z_std = Z.std(axis=0, keepdims=True) + 1e-6
        
        # 找出異常值（任何維度超過 threshold * std）
        z_scores = np.abs((Z - Z_mean) / Z_std)
        max_z_score = z_scores.max(axis=1)
        outlier_mask = max_z_score < outlier_std_threshold
        
        n_outliers = (~outlier_mask).sum()
        print(f"  Removed {n_outliers} outliers ({100*n_outliers/len(Z):.2f}%)")
        
        Z = Z[outlier_mask]
        cell_id = cell_id[outlier_mask]
        frame_in_cell = frame_in_cell[outlier_mask]
        if abs_time is not None:
            abs_time = abs_time[outlier_mask]
        if sequence_idx is not None:
            sequence_idx = sequence_idx[outlier_mask]
        if paths is not None:
            paths = paths[outlier_mask]
    
    # Step 2.2: Standardize (z-score)
    print("\n[2.2] Standardizing (z-score)...")
    Z_mean = Z.mean(axis=0, keepdims=True)
    Z_std = Z.std(axis=0, keepdims=True) + 1e-6
    Z_norm = (Z - Z_mean) / Z_std
    
    print(f"  Mean: {Z_norm.mean(axis=0)[:5]}... (should be ~0)")
    print(f"  Std: {Z_norm.std(axis=0)[:5]}... (should be ~1)")
    
    # Step 2.3: PCA (optional)
    Z_pca = None
    pca_model = None
    pca_explained_variance = None
    
    if do_pca:
        print(f"\n[2.3] Applying PCA (n_components={pca_components})...")
        pca = PCA(n_components=pca_components, random_state=42)
        Z_pca = pca.fit_transform(Z_norm)
        
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()
        
        print(f"  PCA shape: {Z_pca.shape}")
        print(f"  Explained variance: {explained_variance[:5]}")
        print(f"  Cumulative variance: {cumulative_variance[-1]:.4f} ({100*cumulative_variance[-1]:.2f}%)")
        
        pca_model = {
            'components': pca.components_,
            'mean': pca.mean_,
            'explained_variance_ratio': explained_variance,
            'n_components': pca_components
        }
        pca_explained_variance = explained_variance
    
    # Save preprocessed data
    print(f"\nSaving preprocessed data to: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'Z_norm': Z_norm,
        'cell_id': cell_id,
        'frame_in_cell': frame_in_cell,
    }
    
    if abs_time is not None:
        save_dict['abs_time'] = abs_time
    if sequence_idx is not None:
        save_dict['sequence_idx'] = sequence_idx
    
    if Z_pca is not None:
        save_dict['Z_pca'] = Z_pca
    
    if paths is not None:
        save_dict['paths'] = paths
    
    np.savez(output_path, **save_dict)
    print(f"✓ Saved to: {output_file}")
    
    # Save preprocessing metadata
    # Convert numpy types to Python native types for JSON serialization
    metadata = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "original_shape": [int(x) for x in Z.shape],
        "normalized_shape": [int(x) for x in Z_norm.shape],
        "do_pca": bool(do_pca),
        "pca_components": int(pca_components) if do_pca else None,
        "pca_explained_variance": float(cumulative_variance[-1]) if do_pca else None,
        "remove_outliers": bool(remove_outliers),
        "outlier_std_threshold": float(outlier_std_threshold),
        "Z_mean": [float(x) for x in Z_mean.flatten()],
        "Z_std": [float(x) for x in Z_std.flatten()],
    }
    
    if pca_model is not None:
        metadata['pca_components'] = [[float(x) for x in row] for row in pca_model['components']]
        metadata['pca_mean'] = [float(x) for x in pca_model['mean']]
        metadata['pca_explained_variance_ratio'] = [float(x) for x in pca_model['explained_variance_ratio']]
    
    metadata_file = output_path.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to: {metadata_file}")
    
    return Z_norm, Z_pca, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess latents (Step 2)")
    parser.add_argument("--input", type=str, default="latents_all_frames.npz",
                       help="Input .npz file from Step 1")
    parser.add_argument("--output", type=str, default="latents_preprocessed.npz",
                       help="Output .npz file")
    parser.add_argument("--no_pca", action="store_true",
                       help="Skip PCA dimensionality reduction")
    parser.add_argument("--pca_components", type=int, default=32,
                       help="Number of PCA components")
    parser.add_argument("--no_outlier_removal", action="store_true",
                       help="Skip outlier removal")
    parser.add_argument("--outlier_threshold", type=float, default=5.0,
                       help="Outlier standard deviation threshold")
    
    args = parser.parse_args()
    
    preprocess_latents(
        input_file=args.input,
        output_file=args.output,
        do_pca=not args.no_pca,
        pca_components=args.pca_components,
        remove_outliers=not args.no_outlier_removal,
        outlier_std_threshold=args.outlier_threshold
    )


#!/usr/bin/env python3
"""
比較 PCA 和 TPHATE 的 curvature 結果
"""

import numpy as np
from pathlib import Path
import argparse

def compare_curvature_results(video_name, data_dir):
    """
    比較 PCA 和 TPHATE 的 curvature 結果
    """
    data_path = Path(data_dir)
    
    # 載入 PCA 結果
    pca_file = data_path / f'curvature_data_pca_{video_name}.npz'
    tphate_file = data_path / f'curvature_data_tphate_{video_name}.npz'
    
    print("=" * 60)
    print(f"Comparing PCA vs TPHATE Curvature for {video_name}")
    print("=" * 60)
    
    # 載入 PCA
    if pca_file.exists():
        pca_data = np.load(pca_file, allow_pickle=True)
        pca_curvatures = pca_data['curvatures']
        pca_max = np.max(pca_curvatures)
        pca_max_idx = np.argmax(pca_curvatures)
        pca_mean = np.mean(pca_curvatures)
        print(f"\n📊 PCA Results:")
        print(f"  Max curvature: {pca_max:.6f} at frame {pca_max_idx}")
        print(f"  Mean curvature: {pca_mean:.6f}")
        print(f"  Total frames: {len(pca_curvatures)}")
    else:
        print(f"\n⚠️  PCA data not found: {pca_file}")
        pca_curvatures = None
        pca_max = None
    
    # 載入 TPHATE
    if tphate_file.exists():
        tphate_data = np.load(tphate_file, allow_pickle=True)
        tphate_curvatures = tphate_data['curvatures']
        tphate_max = np.max(tphate_curvatures)
        tphate_max_idx = np.argmax(tphate_curvatures)
        tphate_mean = np.mean(tphate_curvatures)
        print(f"\n📊 TPHATE Results:")
        print(f"  Max curvature: {tphate_max:.6f} at frame {tphate_max_idx}")
        print(f"  Mean curvature: {tphate_mean:.6f}")
        print(f"  Total frames: {len(tphate_curvatures)}")
    else:
        print(f"\n⚠️  TPHATE data not found: {tphate_file}")
        tphate_curvatures = None
        tphate_max = None
    
    # 比較
    if pca_max is not None and tphate_max is not None:
        print(f"\n🔍 Comparison:")
        
        # 檢查最大 curvature 值
        if pca_max > tphate_max:
            diff = pca_max - tphate_max
            pct_diff = (diff / tphate_max) * 100
            print(f"  Max Curvature Value:")
            print(f"    ✓ PCA has HIGHER max curvature")
            print(f"    Difference: {diff:.6f} ({pct_diff:.1f}% higher)")
        elif tphate_max > pca_max:
            diff = tphate_max - pca_max
            pct_diff = (diff / pca_max) * 100
            print(f"  Max Curvature Value:")
            print(f"    ✓ TPHATE has HIGHER max curvature")
            print(f"    Difference: {diff:.6f} ({pct_diff:.1f}% higher)")
        else:
            print(f"  Max Curvature Value:")
            print(f"    = Both have the same max curvature: {pca_max:.6f}")
        
        print(f"\n  Frame with Max Curvature:")
        print(f"    PCA:   frame {pca_max_idx} (curvature = {pca_max:.6f})")
        print(f"    TPHATE: frame {tphate_max_idx} (curvature = {tphate_max:.6f})")
        
        # 檢查是否同一個 frame
        if pca_max_idx == tphate_max_idx:
            print(f"\n  ✅ SAME FRAME: Both methods identify frame {pca_max_idx} as having max curvature")
            print(f"     This suggests the morphological change at this frame is significant")
            print(f"     regardless of the dimensionality reduction method used.")
        else:
            frame_diff = abs(pca_max_idx - tphate_max_idx)
            print(f"\n  ⚠️  DIFFERENT FRAMES: Max curvature occurs at different frames")
            print(f"     Frame difference: {frame_diff} frames")
            print(f"     This suggests:")
            print(f"     - PCA and TPHATE capture different aspects of the trajectory")
            print(f"     - The 'most curved' moment depends on the embedding method")
            print(f"     - TPHATE considers temporal structure, PCA is purely geometric")
        
        # 檢查 top 20 最大 curvature frames 的重疊
        pca_top20 = np.argsort(pca_curvatures)[-20:][::-1]
        tphate_top20 = np.argsort(tphate_curvatures)[-20:][::-1]
        overlap = len(set(pca_top20) & set(tphate_top20))
        
        print(f"\n  Top 20 High-Curvature Frames Overlap:")
        print(f"    PCA top 20:   {sorted(pca_top20)}")
        print(f"    TPHATE top 20: {sorted(tphate_top20)}")
        print(f"    Overlap: {overlap}/20 frames ({overlap*5}%)")
        if overlap >= 15:
            print(f"    ✅ Excellent agreement: Both methods identify very similar high-curvature regions")
        elif overlap >= 10:
            print(f"    ✅ Good agreement: Both methods identify similar high-curvature regions")
        elif overlap >= 5:
            print(f"    ⚠️  Moderate agreement: Some overlap but methods differ")
        elif overlap >= 1:
            print(f"    ⚠️  Partial agreement: Limited overlap between methods")
        else:
            print(f"    ❌ Low agreement: Methods identify different high-curvature regions")
        
        # 顯示重疊的 frames
        overlapping_frames = sorted(set(pca_top20) & set(tphate_top20))
        if len(overlapping_frames) > 0:
            print(f"\n  Overlapping frames (in both top 20): {overlapping_frames}")
        else:
            print(f"\n  No overlapping frames in top 20")
    
    print("\n" + "=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Compare PCA vs TPHATE curvature results')
    parser.add_argument('--video_name', type=str, required=True,
                       help='Video name (e.g., ZS435-5)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory containing curvature data files (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Auto-detect data directory
    if args.data_dir is None:
        # Try group directory first
        group_dir = Path('/staging/groups/bhaskar_group/rho9/curvature_analysis')
        local_dir = Path('curvature_analysis')
        
        if group_dir.exists():
            data_dir = group_dir
        elif local_dir.exists():
            data_dir = local_dir
        else:
            raise FileNotFoundError("Could not find data directory. Specify with --data_dir")
    else:
        data_dir = Path(args.data_dir)
    
    compare_curvature_results(args.video_name, data_dir)

if __name__ == '__main__':
    main()


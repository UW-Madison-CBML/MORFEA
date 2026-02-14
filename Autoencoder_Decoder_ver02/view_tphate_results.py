#!/usr/bin/env python3
"""
查看 TPHATE 结果的脚本
"""
import numpy as np
import json
from pathlib import Path

def view_tphate_results(results_dir="tphate_results"):
    
    results_path = Path(results_dir)
    
    print("=" * 60)
    print("TPHATE Results Summary")
    print("=" * 60)
    print()
    
    metadata_file = results_path / "tphate_segments_direct" / "segments_metadata.json"
    if metadata_file.exists():
        print("📊 Segment Information:")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        for seg_name, seg_info in metadata.items():
            n_frames = seg_info.get('n_frames', 0)
            frame_range = seg_info.get('frame_range', [])
            print(f"  {seg_name}: {n_frames} frames")
            if frame_range:
                print(f"    Frame range: {frame_range[0]} - {frame_range[1]}")
        print()
    
    tphate_file = results_path / "tphate_3d_results_direct.npz"
    if tphate_file.exists():
        print("📈 TPHATE 3D Embedding:")
        data = np.load(tphate_file, allow_pickle=True)
        
        if 'Z_tphate' in data:
            Z_tphate = data['Z_tphate']
            print(f"  Shape: {Z_tphate.shape}")
            print(f"  Total points: {len(Z_tphate)}")
            print(f"  Embedding dimensions: {Z_tphate.shape[1]}")
            print(f"  Value range:")
            print(f"    X: [{Z_tphate[:, 0].min():.3f}, {Z_tphate[:, 0].max():.3f}]")
            print(f"    Y: [{Z_tphate[:, 1].min():.3f}, {Z_tphate[:, 1].max():.3f}]")
            print(f"    Z: [{Z_tphate[:, 2].min():.3f}, {Z_tphate[:, 2].max():.3f}]")
        
        if 'cell_id' in data:
            cell_ids = data['cell_id']
            unique_cells = np.unique(cell_ids)
            print(f"  Unique cells: {len(unique_cells)}")
            print(f"    {', '.join(unique_cells[:5])}{'...' if len(unique_cells) > 5 else ''}")
        
        if 'frame_in_cell' in data:
            frames = data['frame_in_cell']
            print(f"  Frame range: {frames.min()} - {frames.max()}")
        print()
    
    latents_file = results_path / "latents_all_frames_direct.npz"
    if latents_file.exists():
        print("🧬 Original Latents:")
        data = np.load(latents_file, allow_pickle=True)
        
        if 'Z' in data:
            Z = data['Z']
            print(f"  Shape: {Z.shape}")
            print(f"  Total frames: {len(Z)}")
            print(f"  Latent dimension: {Z.shape[1]}")
        
        if 'cell_id' in data:
            cell_ids = data['cell_id']
            unique_cells = np.unique(cell_ids)
            print(f"  Unique cells: {len(unique_cells)}")
        print()
    
    preprocessed_file = results_path / "latents_preprocessed_direct.npz"
    if preprocessed_file.exists():
        print("🔧 Preprocessed Latents:")
        data = np.load(preprocessed_file, allow_pickle=True)
        
        if 'Z_norm' in data:
            Z_norm = data['Z_norm']
            print(f"  Shape: {Z_norm.shape}")
            print(f"  Normalized: ✓")
        
        if 'Z_pca' in data:
            Z_pca = data['Z_pca']
            print(f"  PCA shape: {Z_pca.shape}")
            print(f"  PCA components: {Z_pca.shape[1]}")
        print()
    
    vis_dir = results_path / "tphate_segments_direct"
    if vis_dir.exists():
        print("🖼️  Visualization Files:")
        vis_files = list(vis_dir.glob("*.png"))
        for f in sorted(vis_files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name} ({size_mb:.2f} MB)")
        print()
    
    print("=" * 60)
    print("✅ All results loaded successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Open visualization files in tphate_segments_direct/")
    print("  2. Analyze segment transitions (morphological stages)")
    print("  3. Perform TDA (persistent homology) analysis")
    print("  4. Clustering to find developmental subtypes")
    print("  5. Extract features for downstream classifier")

if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "tphate_results"
    view_tphate_results(results_dir)


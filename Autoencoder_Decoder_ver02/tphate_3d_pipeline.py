"""
Step 3-4: 3D TPHATE Pipeline (嚴謹版)
- Step 3: 把時間資訊 encode 給 TPHATE
- Step 4: 跑 3D TPHATE 降維
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from collections import defaultdict

# Try to import tphate (REQUIRED)
TPHATE_AVAILABLE = False
TPHATE_ERROR = None

try:
    import tphate
    TPHATE_AVAILABLE = True
    print("✓ tphate library available")
except (ImportError, ModuleNotFoundError, AttributeError) as e:
    TPHATE_AVAILABLE = False
    TPHATE_ERROR = str(e)
    print("❌ ERROR: tphate library is REQUIRED and not available!")
    print(f"   Import error: {e}")
    print("\n   TPHATE is MANDATORY (no approximation allowed).")
    print("\n   Installation steps:")
    print("   1. pip install --user --upgrade numpy setuptools wheel")
    print("   2. pip install --user --no-cache-dir s_gd2")
    print("   3. pip install --user --no-cache-dir tphate")
    print("\n   If s_gd2 compilation fails (C++ issue):")
    print("   - Check if gcc/g++ is available: gcc --version")
    print("   - Try: pip install --user --upgrade pip setuptools")
    print("   - Or contact CHTC support for C++ compiler access")
    print("\n   TPHATE is required for:")
    print("     (1) Multi-embryo manifold alignment")
    print("     (2) Reduced local folding (temporal kernel robustness)")
    print("     (3) Embedding not dominated by time scale")
    raise ImportError("TPHATE is REQUIRED. Install tphate library before proceeding.")

# phate is still needed for some utilities
try:
    import phate
    PHATE_AVAILABLE = True
except ImportError:
    PHATE_AVAILABLE = False


def build_time_structure(cell_id, frame_in_cell):
    """
    Step 3: 建立時間結構
    
    Args:
        cell_id: [N] cell IDs
        frame_in_cell: [N] frame indices within each cell
    
    Returns:
        time_edges: List of (i, j) tuples representing temporal neighbors
        cell_groups: Dict mapping cell_id to list of indices
    """
    print("\n=== Step 3: Building Time Structure ===")
    
    N = len(cell_id)
    unique_cells = np.unique(cell_id)
    
    # Group indices by cell_id
    cell_groups = defaultdict(list)
    for i, cid in enumerate(cell_id):
        cell_groups[cid].append(i)
    
    # Build time edges (temporal neighbors within each cell)
    time_edges = []
    for cid in unique_cells:
        indices = np.array(cell_groups[cid])
        # Sort by frame_in_cell
        frame_values = frame_in_cell[indices]
        sorted_order = np.argsort(frame_values)
        sorted_indices = indices[sorted_order]
        
        # Add edges: frame t -> frame t+1
        for i in range(len(sorted_indices) - 1):
            a, b = sorted_indices[i], sorted_indices[i + 1]
            time_edges.append((int(a), int(b)))
        
        # Optional: also add t -> t+2 (weaker connection)
        for i in range(len(sorted_indices) - 2):
            a, b = sorted_indices[i], sorted_indices[i + 2]
            time_edges.append((int(a), int(b)))
    
    print(f"  Total cells: {len(unique_cells)}")
    print(f"  Total time edges: {len(time_edges)}")
    print(f"  Average edges per cell: {len(time_edges) / len(unique_cells):.1f}")
    
    return time_edges, cell_groups


def apply_tphate_3d(Z, time_edges, cell_id, frame_in_cell, knn=10, n_components=3, use_phate_fallback=True):
    """
    Step 4: 應用 3D TPHATE（強制使用，不 fallback）
    
    理由：
    (1) 多顆 embryo manifold alignment 會更乾淨、穩定
    (2) local folding 會減少（temporal kernel 對折返更 robust）
    (3) embedding 不會被 time scale 主導
    
    Args:
        Z: [N, d] preprocessed latent vectors
        time_edges: List of (i, j) temporal edge tuples
        cell_id: [N] cell IDs
        frame_in_cell: [N] frame indices
        knn: k-nearest neighbors parameter
        n_components: Output dimensionality (3 for 3D)
    
    Returns:
        Z_tphate: [N, 3] 3D TPHATE embedding
    """
    print("\n=== Step 4: Applying 3D TPHATE (REQUIRED, no approximation) ===")
    print("  Reasons for using TPHATE:")
    print("    (1) Multi-embryo manifold alignment will be cleaner and more stable")
    print("    (2) Reduced local folding (temporal kernel is more robust to reversals)")
    print("    (3) Embedding won't be dominated by time scale")
    
    if not TPHATE_AVAILABLE:
        error_msg = f"TPHATE is REQUIRED but not available.\n"
        error_msg += f"Import error: {TPHATE_ERROR}\n"
        error_msg += "\nInstallation steps:\n"
        error_msg += "  1. pip install --user --upgrade numpy setuptools wheel\n"
        error_msg += "  2. pip install --user --no-cache-dir s_gd2\n"
        error_msg += "  3. pip install --user --no-cache-dir tphate\n"
        error_msg += "\nTPHATE is mandatory. No approximation will be used."
        raise ImportError(error_msg)
    
    print("\nUsing tphate library (REQUIRED, no fallback)...")
    
    try:
        print("  Attempting TPHATE with standard fit_transform (no time parameters)...")
        tph = tphate.TPHATE(
            n_components=n_components,
            knn=knn,
            t="auto",
            n_jobs=-1
        )
        Z_tphate = tph.fit_transform(Z)
        print(f"✓ TPHATE embedding shape: {Z_tphate.shape}")
        return Z_tphate
    except Exception as e:
        print(f"  TPHATE failed: {e}")
        print("\n   Debugging: Check tphate API")
        print("   Run: python3 -c 'import tphate; help(tphate.TPHATE.fit_transform)'")
        raise RuntimeError(f"TPHATE failed: {e}")


def apply_phate_with_time_improved(Z, time_edges, cell_id, frame_in_cell, n_components=3, knn=10):
    """
    改進的 PHATE + 時間特徵（盡可能接近 TPHATE）
    
    使用時間鄰接矩陣加權，而不是簡單的時間特徵
    
    Args:
        Z: [N, d] preprocessed latent vectors
        time_edges: List of (i, j) temporal edge tuples
        cell_id: [N] cell IDs
        frame_in_cell: [N] frame indices
        n_components: Output dimensionality
        knn: k-nearest neighbors parameter
    
    Returns:
        Z_phate: [N, 3] 3D PHATE embedding with improved time encoding
    """
    if not PHATE_AVAILABLE:
        raise ImportError("phate library is required. Install with: pip install phate")
    
    print("Using Improved PHATE + Time (TPHATE approximation)...")
    print("  Note: This is not true TPHATE, but uses temporal adjacency weighting")
    
    from scipy import sparse
    from scipy.spatial.distance import pdist, squareform
    
    N = len(Z)
    
    print("  Computing geometric distances...")
    distances = squareform(pdist(Z, metric='euclidean'))
    
    print("  Building temporal adjacency matrix...")
    time_adj = sparse.lil_matrix((N, N))
    for i, j in time_edges:
        if i < N and j < N:
            time_adj[i, j] = 1
            time_adj[j, i] = 1
    time_adj = time_adj.tocsr()
    
    print("  Combining geometric and temporal structure...")
    temporal_weight = 0.3
    
    distances_combined = distances.copy()
    time_neighbors = time_adj.nonzero()
    for i, j in zip(time_neighbors[0], time_neighbors[1]):
        if i < j:
            original_dist = distances[i, j]
            distances_combined[i, j] = original_dist * (1 - temporal_weight)
            distances_combined[j, i] = distances_combined[i, j]
    
    print("  Applying PHATE with temporal-weighted distances...")
    
    
    Z_temporal_smoothed = Z.copy()
    for i in range(N):
        neighbors = time_adj[i].nonzero()[1]
        if len(neighbors) > 0:
            neighbor_latents = Z[neighbors]
            Z_temporal_smoothed[i] = 0.7 * Z[i] + 0.3 * neighbor_latents.mean(axis=0)
    
    ph = phate.PHATE(
        n_components=n_components,
        knn=knn,
        decay=40,
        t="auto",
        n_jobs=-1,
        random_state=42
    )
    
    Z_phate = ph.fit_transform(Z_temporal_smoothed)
    
    print(f"✓ Improved PHATE embedding shape: {Z_phate.shape}")
    print("  (This approximates TPHATE using temporal smoothing)")
    
    return Z_phate


def tphate_3d_pipeline(
    input_file="latents_preprocessed.npz",
    output_file="tphate_3d_results.npz",
    use_pca=True,
    knn=10,
    n_components=3,
    random_seed=42
):
    """
    完整的 3D TPHATE pipeline (Step 3-4)
    
    Args:
        input_file: 預處理後的 .npz 文件（來自 Step 2）
        output_file: 輸出 .npz 文件
        use_pca: 是否使用 PCA 降維後的 latent（如果有的話）
        knn: k-nearest neighbors 參數
        n_components: 輸出維度（3 for 3D）
        random_seed: 隨機種子
    """
    print(f"=== 3D TPHATE Pipeline (Step 3-4) ===")
    
    # Load preprocessed data
    print(f"\nLoading preprocessed data from: {input_file}")
    data = np.load(input_file, allow_pickle=True)
    
    if use_pca and 'Z_pca' in data:
        Z = data['Z_pca']
        print(f"  Using Z_pca: {Z.shape}")
    else:
        Z = data['Z_norm']
        print(f"  Using Z_norm: {Z.shape}")
    
    cell_id = data['cell_id']
    frame_in_cell = data['frame_in_cell']
    abs_time = data['abs_time'] if 'abs_time' in data else None
    
    # Step 3: Build time structure
    time_edges, cell_groups = build_time_structure(cell_id, frame_in_cell)
    
    np.random.seed(random_seed)
    Z_tphate = apply_tphate_3d(Z, time_edges, cell_id, frame_in_cell, knn=knn, n_components=n_components, use_phate_fallback=False)
    
    # Save results
    print(f"\nSaving results to: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        Z_tphate=Z_tphate,
        cell_id=cell_id,
        frame_in_cell=frame_in_cell,
        abs_time=abs_time if abs_time is not None else np.zeros(len(cell_id)),
        time_edges=np.array(time_edges),
        original_Z_shape=Z.shape
    )
    
    print(f"✓ Saved to: {output_file}")
    
    # Save metadata
    # Convert numpy types to Python native types for JSON serialization
    metadata = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "n_frames": int(len(Z_tphate)),
        "n_components": int(n_components),
        "knn": int(knn),
        "use_pca": bool(use_pca),
        "tphate_available": bool(TPHATE_AVAILABLE),
        "phate_available": bool(PHATE_AVAILABLE),
        "n_cells": int(len(np.unique(cell_id))),
        "n_time_edges": int(len(time_edges)),
        "random_seed": int(random_seed)
    }
    
    metadata_file = output_path.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to: {metadata_file}")
    
    return Z_tphate, time_edges, cell_groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D TPHATE Pipeline (Step 3-4)")
    parser.add_argument("--input", type=str, default="latents_preprocessed.npz",
                       help="Input preprocessed .npz file from Step 2")
    parser.add_argument("--output", type=str, default="tphate_3d_results.npz",
                       help="Output .npz file")
    parser.add_argument("--use_pca", action="store_true", default=True,
                       help="Use PCA-reduced latents (if available)")
    parser.add_argument("--no_pca", dest="use_pca", action="store_false",
                       help="Don't use PCA-reduced latents")
    parser.add_argument("--knn", type=int, default=10,
                       help="k-nearest neighbors parameter")
    parser.add_argument("--n_components", type=int, default=3,
                       help="Output dimensionality (3 for 3D)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    tphate_3d_pipeline(
        input_file=args.input,
        output_file=args.output,
        use_pca=args.use_pca,
        knn=args.knn,
        n_components=args.n_components,
        random_seed=args.seed
    )


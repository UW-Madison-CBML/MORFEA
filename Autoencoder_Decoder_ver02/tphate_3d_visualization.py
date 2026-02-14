"""
3D T-PHATE Visualization using Standard Library
- Uses phate library for T-PHATE embedding
- Creates 3D visualizations of embryo development trajectories
- Based on latent vectors from epoch 50 checkpoint
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import json

try:
    import phate
    PHATE_AVAILABLE = True
except ImportError:
    print("Warning: phate library not available. Install with: pip install phate")
    PHATE_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: sklearn not available")
    SKLEARN_AVAILABLE = False


def load_latents(latents_file, metadata_file):
    """Load latent vectors and metadata"""
    print(f"Loading latents from: {latents_file}")
    z_seq = np.load(latents_file)  # [N, T, hidden_dim]
    
    print(f"Loading metadata from: {metadata_file}")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return z_seq, metadata


def apply_tphate_3d(z_seq, n_components=3, **kwargs):
    """
    Apply T-PHATE to latent sequences for 3D visualization
    
    Args:
        z_seq: [N, T, hidden_dim] - latent sequences
        n_components: Number of components (3 for 3D)
        **kwargs: Additional arguments for PHATE
    """
    if not PHATE_AVAILABLE:
        raise ImportError("phate library is required. Install with: pip install phate")
    
    N, T, hidden_dim = z_seq.shape
    
    # Flatten sequences for PHATE: [N*T, hidden_dim]
    z_flat = z_seq.reshape(-1, hidden_dim)
    print(f"Flattened shape: {z_flat.shape}")
    
    # Apply PHATE
    print("Applying PHATE embedding...")
    phate_op = phate.PHATE(
        n_components=n_components,
        knn=kwargs.get('knn', 5),
        decay=kwargs.get('decay', 40),
        t=kwargs.get('t', 'auto'),
        verbose=kwargs.get('verbose', 1)
    )
    
    z_phate_flat = phate_op.fit_transform(z_flat)  # [N*T, n_components]
    
    # Reshape back to sequences: [N, T, n_components]
    z_phate = z_phate_flat.reshape(N, T, n_components)
    
    return z_phate, phate_op


def plot_3d_tphate_trajectories(
    z_phate,  # [N, T, 3]
    cell_ids,
    output_dir="tphate_3d_results",
    n_embryos_to_plot=10,
    save_individual=True
):
    """
    Plot 3D T-PHATE trajectories
    
    Args:
        z_phate: [N, T, 3] - T-PHATE embeddings
        cell_ids: List of cell IDs
        output_dir: Output directory
        n_embryos_to_plot: Number of embryos to plot
        save_individual: Save individual embryo plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Get unique embryos
    unique_embryos = list(set(cell_ids))
    n_embryos_to_plot = min(n_embryos_to_plot, len(unique_embryos))
    
    print(f"\nPlotting {n_embryos_to_plot} embryos...")
    
    # Create main comparison plot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_embryos_to_plot))
    
    for i, cell_id in enumerate(unique_embryos[:n_embryos_to_plot]):
        # Find all sequences for this embryo
        embryo_indices = [j for j, cid in enumerate(cell_ids) if cid == cell_id]
        
        if len(embryo_indices) == 0:
            continue
        
        # Plot each sequence for this embryo
        for seq_idx in embryo_indices:
            z_traj = z_phate[seq_idx]  # [T, 3]
            T = len(z_traj)
            
            # Plot trajectory
            ax.plot(z_traj[:, 0], z_traj[:, 1], z_traj[:, 2],
                   color=colors[i], alpha=0.6, linewidth=2, label=cell_id if seq_idx == embryo_indices[0] else "")
            
            # Plot start and end points
            ax.scatter(z_traj[0, 0], z_traj[0, 1], z_traj[0, 2],
                      color=colors[i], s=100, marker='o', alpha=0.8)
            ax.scatter(z_traj[-1, 0], z_traj[-1, 1], z_traj[-1, 2],
                      color=colors[i], s=100, marker='s', alpha=0.8)
    
    ax.set_xlabel('T-PHATE Component 1', fontsize=12)
    ax.set_ylabel('T-PHATE Component 2', fontsize=12)
    ax.set_zlabel('T-PHATE Component 3', fontsize=12)
    ax.set_title('3D T-PHATE Trajectories: Embryo Development', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    comparison_file = output_path / "tphate_3d_comparison.png"
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comparison plot: {comparison_file}")
    
    # Save individual embryo plots
    if save_individual:
        for i, cell_id in enumerate(unique_embryos[:n_embryos_to_plot]):
            embryo_indices = [j for j, cid in enumerate(cell_ids) if cid == cell_id]
            
            if len(embryo_indices) == 0:
                continue
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            for seq_idx in embryo_indices:
                z_traj = z_phate[seq_idx]  # [T, 3]
                T = len(z_traj)
                
                # Color by time
                scatter = ax.scatter(z_traj[:, 0], z_traj[:, 1], z_traj[:, 2],
                                   c=range(T), cmap='viridis', s=50, alpha=0.7)
                
                # Plot trajectory line
                ax.plot(z_traj[:, 0], z_traj[:, 1], z_traj[:, 2],
                       'r-', alpha=0.3, linewidth=1)
            
            ax.set_xlabel('T-PHATE Component 1', fontsize=12)
            ax.set_ylabel('T-PHATE Component 2', fontsize=12)
            ax.set_zlabel('T-PHATE Component 3', fontsize=12)
            ax.set_title(f'3D T-PHATE: Embryo {cell_id}', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax, label='Time Step')
            
            plt.tight_layout()
            individual_file = output_path / f"tphate_3d_embryo_{cell_id}.png"
            plt.savefig(individual_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            if (i + 1) % 5 == 0:
                print(f"  Saved {i + 1}/{n_embryos_to_plot} individual plots...")
    
    print(f"\n✅ 3D T-PHATE visualization complete!")
    print(f"📁 Files saved in: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="3D T-PHATE visualization from epoch 50 latents")
    parser.add_argument("--latents_file", type=str, default="latents_epoch50/latents_z_seq_epoch50.npy",
                       help="Path to latent vectors file")
    parser.add_argument("--metadata_file", type=str, default="latents_epoch50/latents_metadata_epoch50.json",
                       help="Path to metadata file")
    parser.add_argument("--output_dir", type=str, default="tphate_3d_results",
                       help="Output directory for visualizations")
    parser.add_argument("--n_embryos", type=int, default=10,
                       help="Number of embryos to plot")
    parser.add_argument("--knn", type=int, default=5,
                       help="k-NN parameter for PHATE")
    parser.add_argument("--decay", type=int, default=40,
                       help="Decay parameter for PHATE")
    
    args = parser.parse_args()
    
    if not PHATE_AVAILABLE:
        print("❌ Error: phate library is required")
        print("Install with: pip install phate")
        return
    
    # Load latents
    z_seq, metadata = load_latents(args.latents_file, args.metadata_file)
    cell_ids = metadata['cell_ids']
    
    print(f"\nLoaded {len(z_seq)} sequences from {len(set(cell_ids))} unique embryos")
    
    # Apply T-PHATE
    z_phate, phate_op = apply_tphate_3d(z_seq, n_components=3, knn=args.knn, decay=args.decay)
    
    print(f"T-PHATE embedding shape: {z_phate.shape}")
    
    # Create visualizations
    plot_3d_tphate_trajectories(
        z_phate,
        cell_ids,
        output_dir=args.output_dir,
        n_embryos_to_plot=args.n_embryos,
        save_individual=True
    )


if __name__ == "__main__":
    main()


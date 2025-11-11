"""
TPHATE Visualization Script

Loads latent embeddings from a CSV file and creates TPHATE visualizations.

Usage:
    python visualize.py <csv_file>
    python visualize.py latents.csv
"""

import sys
import os
import tphate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def load_latents(csv_file):
    """Load latent embeddings from CSV file.

    Expected CSV format:
    - cell_id: Cell identifier
    - time_step: Time step in the sequence
    - z_0, z_1, ..., z_199: Latent dimensions

    Args:
        csv_file: Path to CSV file

    Returns:
        data: (num_samples, num_features) array of latent embeddings
        df: Original dataframe with metadata
    """
    print(f"Loading latent embeddings from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} samples with {len(df.columns)} columns")

    # Extract latent dimensions (all columns except cell_id and time_step)
    latent_cols = [col for col in df.columns if col.startswith('z_')]
    data = df[latent_cols].values
    print(f"  Using {len(latent_cols)} latent dimensions")

    return data, df


def apply_tphate(data, n_jobs=-1):
    """Apply TPHATE dimensionality reduction.

    Args:
        data: (num_samples, num_features) array
        n_jobs: Number of parallel jobs for computation

    Returns:
        tphate_data: (num_samples, 2) array of 2D projections
    """
    print(f"\nApplying TPHATE to {data.shape[0]} samples...")
    tphate_op = tphate.TPHATE(n_jobs=n_jobs)
    tphate_data = tphate_op.fit_transform(data)
    print(f"  TPHATE output shape: {tphate_data.shape}")

    return tphate_data


def plot_tphate_scatter(tphate_data, df, output_file="tphate_scatter.png"):
    """Create a simple scatter plot of TPHATE projection.

    Args:
        tphate_data: (num_samples, 2) TPHATE projection
        df: Original dataframe for metadata
        output_file: Output filename for the plot
    """
    print(f"Creating scatter plot...")
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(tphate_data[:, 0], tphate_data[:, 1], alpha=0.6, s=30)
    ax.set_xlabel("TPHATE Dimension 1", fontsize=12)
    ax.set_ylabel("TPHATE Dimension 2", fontsize=12)
    ax.set_title("TPHATE Projection of Latent Embeddings", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {output_file}")
    plt.close()


def plot_tphate_by_cell_id(tphate_data, df, output_file="tphate_by_cell_id.png"):
    """Create scatter plot colored by cell_id.

    Args:
        tphate_data: (num_samples, 2) TPHATE projection
        df: Original dataframe with cell_id column
        output_file: Output filename for the plot
    """
    print(f"Creating scatter plot colored by cell_id...")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get unique cell IDs and create colormap
    cell_ids = df['cell_id'].values
    unique_cells = np.unique(cell_ids)
    cmap = get_cmap('tab20', len(unique_cells))

    # Create color mapping
    color_map = {cell_id: cmap(i) for i, cell_id in enumerate(unique_cells)}
    colors = [color_map[cell_id] for cell_id in cell_ids]

    scatter = ax.scatter(tphate_data[:, 0], tphate_data[:, 1],
                         c=colors, alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
    ax.set_xlabel("TPHATE Dimension 1", fontsize=12)
    ax.set_ylabel("TPHATE Dimension 2", fontsize=12)
    ax.set_title("TPHATE Projection Colored by Cell ID", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add legend for cell IDs
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color_map[cell_id],
                                  markersize=8, label=cell_id)
                       for cell_id in unique_cells[:10]]  # Show first 10
    if len(unique_cells) > 10:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor='gray', markersize=8,
                                         label=f'... +{len(unique_cells)-10} more'))
    ax.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {output_file}")
    plt.close()


def plot_tphate_by_time_step(tphate_data, df, output_file="tphate_by_time_step.png"):
    """Create scatter plot colored by time_step.

    Args:
        tphate_data: (num_samples, 2) TPHATE projection
        df: Original dataframe with time_step column
        output_file: Output filename for the plot
    """
    print(f"Creating scatter plot colored by time_step...")
    fig, ax = plt.subplots(figsize=(12, 8))

    time_steps = df['time_step'].values

    scatter = ax.scatter(tphate_data[:, 0], tphate_data[:, 1],
                         c=time_steps, cmap='viridis', alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
    ax.set_xlabel("TPHATE Dimension 1", fontsize=12)
    ax.set_ylabel("TPHATE Dimension 2", fontsize=12)
    ax.set_title("TPHATE Projection Colored by Time Step", fontsize=14)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Time Step", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved to: {output_file}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <csv_file>")
        print("Example: python visualize.py latents.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    # Create output directory
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load data
    data, df = load_latents(csv_file)

    # Apply TPHATE
    tphate_data = apply_tphate(data)

    # Create visualizations
    plot_tphate_scatter(tphate_data, df, os.path.join(output_dir, "tphate_scatter.png"))
    #plot_tphate_by_cell_id(tphate_data, df, os.path.join(output_dir, "tphate_by_cell_id.png"))
    #plot_tphate_by_time_step(tphate_data, df, os.path.join(output_dir, "tphate_by_time_step.png"))

    print(f"\nVisualization complete! Check the '{output_dir}' directory for output files.")


if __name__ == "__main__":
    main()

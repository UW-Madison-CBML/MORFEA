#"""
#TPHATE Visualization Script
#
#Processes a batch of cell_ids and creates TPHATE scatter plots and
#trajectory path plots for each cell_id in the batch.
#
#Usage:
    #python visualize.py <latents_csv> <cell_line>
    #python visualize.py latents.csv "cell_id_1,cell_id_2,cell_id_3"
#
#Where cell_line is a comma-separated list of cell_ids.
#"""

import sys
import os
import tphate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import argparse


def load_latents(csv_file):
    """Load latent embeddings from CSV file.

    Expected CSV format:
    - cell_id: Cell identifier
    - time_step: Time step in the sequence
    - z_0, z_1, ..., z_N: Latent dimensions

    Args:
        csv_file: Path to CSV file

    Returns:
        df: Dataframe with latent embeddings and metadata
        latent_cols: List of latent dimension column names
    """
    print(f"Loading latent embeddings from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} samples with {len(df.columns)} columns")

    # Extract latent dimensions (all columns except cell_id and time_step)
    latent_cols = [col for col in df.columns if col.startswith('z_')]
    print(f"  Using {len(latent_cols)} latent dimensions")

    return df, latent_cols


def apply_tphate(data, n_jobs=-1):
    """Apply TPHATE dimensionality reduction.

    Args:
        data: (num_samples, num_features) array
        n_jobs: Number of parallel jobs for computation

    Returns:
        tphate_data: (num_samples, 2) array of 2D projections
    """
    print(f"  Applying TPHATE to {data.shape[0]} samples...")
    tphate_op = tphate.TPHATE(n_jobs=n_jobs)
    tphate_data = tphate_op.fit_transform(data)
    print(f"  TPHATE output shape: {tphate_data.shape}")

    return tphate_data


def plot_cell_scatter(cell_id, tphate_data, time_steps, output_dir="plots"):
    """Create scatter plot for a single cell_id.

    Args:
        cell_id: Cell ID string
        tphate_data: (num_samples, 2) TPHATE projection
        time_steps: Array of time steps for coloring
        output_dir: Output directory for plots
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(tphate_data[:, 0], tphate_data[:, 1],
                         c=time_steps, cmap='viridis', alpha=0.7, s=50,
                         edgecolors='black', linewidth=0.5)
    ax.set_xlabel("TPHATE Dimension 1", fontsize=12)
    ax.set_ylabel("TPHATE Dimension 2", fontsize=12)
    ax.set_title(f"TPHATE Scatter: {cell_id}", fontsize=14)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Time Step", fontsize=11)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"scatter_{cell_id}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved scatter plot: {output_file}")


def plot_cell_trajectory(cell_id, tphate_data, time_steps, output_dir="plots"):
    """Create trajectory path plot for a single cell_id.

    Args:
        cell_id: Cell ID string
        tphate_data: (num_samples, 2) TPHATE projection
        time_steps: Array of time steps for ordering path
        output_dir: Output directory for plots
    """
    # Sort by time step to draw trajectory in order
    sorted_indices = np.argsort(time_steps)
    sorted_tphate = tphate_data[sorted_indices]
    sorted_times = time_steps[sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw trajectory line
    ax.plot(sorted_tphate[:, 0], sorted_tphate[:, 1], 'k-', alpha=0.3, linewidth=1.5)

    # Draw points colored by time step
    scatter = ax.scatter(sorted_tphate[:, 0], sorted_tphate[:, 1],
                         c=sorted_times, cmap='viridis', alpha=0.8, s=50,
                         edgecolors='black', linewidth=0.5, zorder=5)

    # Mark start and end points
    ax.plot(sorted_tphate[0, 0], sorted_tphate[0, 1], 'go', markersize=12, label='Start', zorder=6)
    ax.plot(sorted_tphate[-1, 0], sorted_tphate[-1, 1], 'r*', markersize=20, label='End', zorder=6)

    ax.set_xlabel("TPHATE Dimension 1", fontsize=12)
    ax.set_ylabel("TPHATE Dimension 2", fontsize=12)
    ax.set_title(f"TPHATE Trajectory: {cell_id}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Time Step", fontsize=11)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"trajectory_{cell_id}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved trajectory plot: {output_file}")


def process_cell_id_batch(cell_ids, df, latent_cols, output_dir="plots"):
    """Process a batch of cell_ids: run TPHATE and create plots.

    Args:
        cell_ids: List of cell_id strings in batch
        df: Full dataframe with all data
        latent_cols: List of latent dimension column names
        output_dir: Output directory for plots
    """
    print(f"\n  Processing batch with {len(cell_ids)} cell_ids")

    # Process each cell_id individually with its own TPHATE run
    for cell_id in cell_ids:
        cell_df = df[df['cell_id'] == cell_id]
        num_cell_samples = len(cell_df)

        print(f"    Processing {cell_id} ({num_cell_samples} samples)")

        # Get latents for this cell_id only
        cell_latents = cell_df[latent_cols].values

        # Apply TPHATE to this cell_id's data individually
        cell_tphate = apply_tphate(cell_latents)
        cell_time_steps = cell_df['time_step'].values

        # Create scatter and trajectory plots
        plot_cell_scatter(cell_id, cell_tphate, cell_time_steps, output_dir)
        plot_cell_trajectory(cell_id, cell_tphate, cell_time_steps, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Visualize cell_id batch with TPHATE")
    parser.add_argument("latents_csv", help="Path to latents CSV file")
    parser.add_argument("cell_line", help="Cell group line (comma-separated cell_ids)")
    parser.add_argument("--output", type=str, default="plots", help="Output directory for plots")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.latents_csv):
        print(f"Error: Latents CSV not found: {args.latents_csv}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"Output directory: {args.output}\n")

    # Load latents
    df, latent_cols = load_latents(args.latents_csv)

    # Parse cell_ids from the line argument
    cell_ids = args.cell_line.strip().split(',')
    cell_ids = [cid.strip() for cid in cell_ids if cid.strip()]  # Clean up and filter empty

    if not cell_ids:
        print("Error: No cell_ids found in the provided line")
        sys.exit(1)

    print(f"Processing batch with {len(cell_ids)} cell_ids:")
    print(f"  Cell IDs: {', '.join(cell_ids)}\n")
    process_cell_id_batch(cell_ids, df, latent_cols, args.output)

    print(f"\n\nVisualization complete! Plots saved to: {args.output}")


if __name__ == "__main__":
    main()

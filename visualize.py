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
    print(f"Loading latent embeddings from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} samples with {len(df.columns)} columns")

    latent_cols = [col for col in df.columns if col.startswith('z_')]
    print(f"  Using {len(latent_cols)} latent dimensions")

    return df, latent_cols


def apply_tphate(data, n_jobs=-1):

    print(f"  Applying TPHATE to {data.shape[0]} samples...")
    tphate_op = tphate.TPHATE(n_jobs=n_jobs)
    tphate_data = tphate_op.fit_transform(data)
    print(f"  TPHATE output shape: {tphate_data.shape}")

    return tphate_data




def plot_cell_trajectory_circle(cell_id, tphate_data, time_steps, output_dir="plots"):
    fig, ax = plt.subplots(figsize=(10, 8))
    print("making plots with ", str(tphate_data.shape), " shape")
    ax.plot(tphate_data[:, 0], tphate_data[:, 1], 'k-', alpha=0.3, linewidth=1.5)
    
    curvature = np.abs(np.gradient(np.gradient(tphate_data[:, 1], tphate_data[:, 0]),  tphate_data[:, 0]))

    norm_curvature = (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature))
    colors = plt.cm.jet(norm_curvature)
    scatter = ax.scatter(tphate_data[:, 0], tphate_data[:, 1],
                         c=colors, cmap='viridis', alpha=0.8, s=50,
                         edgecolors='black', linewidth=0.5, zorder=5)

    ax.plot(tphate_data[0, 0], tphate_data[0, 1], 'go', markersize=12, label='Start', zorder=6)
    ax.plot(tphate_data[-1, 0], tphate_data[-1, 1], 'r*', markersize=20, label='End', zorder=6)

    ax.set_xlabel("TPHATE Dimension 1", fontsize=12)
    ax.set_ylabel("TPHATE Dimension 2", fontsize=12)
    ax.set_title(f"TPHATE Trajectory: {cell_id}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Time Step", fontsize=11)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"circle_{cell_id}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved trajectory plot: {output_file}")

def plot_cell_trajectory_velocity(cell_id, tphate_data, time_steps, output_dir="plots"):
    fig, ax = plt.subplots(figsize=(10, 8))
    print("making plots with ", str(tphate_data.shape), " shape")
    
    # Plot the trajectory line
    ax.plot(tphate_data[:, 0], tphate_data[:, 1], 'k-', alpha=0.3, linewidth=1.5)
    
    # Calculate velocity between consecutive points
    dx = np.diff(tphate_data[:, 0])
    dy = np.diff(tphate_data[:, 1])
    distances = np.sqrt(dx**2 + dy**2)
    
    # Calculate time differences
    dt = np.diff(time_steps)
    dt[dt == 0] = 1e-10  # Avoid division by zero
    
    # Velocity is distance / time
    velocities = distances / dt
    
    # Assign velocity to each point (use velocity leading INTO that point)
    # First point gets the first velocity, last point gets the last velocity
    point_velocities = np.concatenate([[velocities[0]], velocities])
    
    # Normalize velocities for color mapping
    norm_velocity = (point_velocities - np.min(point_velocities)) / (np.max(point_velocities) - np.min(point_velocities) + 1e-10)
    
    # Create scatter plot colored by velocity
    scatter = ax.scatter(tphate_data[:, 0], tphate_data[:, 1],
                         c=point_velocities, cmap='jet', alpha=0.8, s=50,
                         edgecolors='black', linewidth=0.5, zorder=5)
    
    # Mark start and end points
    ax.plot(tphate_data[0, 0], tphate_data[0, 1], 'go', markersize=12, label='Start', zorder=6)
    ax.plot(tphate_data[-1, 0], tphate_data[-1, 1], 'r*', markersize=20, label='End', zorder=6)
    
    # Labels and formatting
    ax.set_xlabel("TPHATE Dimension 1", fontsize=12)
    ax.set_ylabel("TPHATE Dimension 2", fontsize=12)
    ax.set_title(f"TPHATE Trajectory (Colored by Velocity): {cell_id}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Velocity (TPHATE units/time step)", fontsize=11)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"velocity_{cell_id}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved trajectory plot: {output_file}")


def plot_cell_trajectory_timestamp(cell_id, tphate_data, time_steps, output_dir="plots"):
    import pandas as pd
    import matplotlib.patches as mpatches
    
    fig, ax = plt.subplots(figsize=(12, 8))
    print("making plots with ", str(tphate_data.shape), " shape")
    
    # Read phase annotations
    phase_file = f"embryo_dataset_annotations/{cell_id}_phases.csv"
    phases_df = pd.read_csv(phase_file, header=None, names=['phase', 'start', 'end'])
    
    # Create a mapping from time step to phase
    phase_colors = plt.cm.tab20(np.linspace(0, 1, len(phases_df)))
    time_to_phase = {}
    time_to_color = {}
    
    for idx, row in phases_df.iterrows():
        phase_name = row['phase']
        start_frame = row['start']
        end_frame = row['end']
        color = phase_colors[idx]
        
        for t in range(start_frame, end_frame + 1):
            time_to_phase[t] = phase_name
            time_to_color[t] = color
    
    # Assign phase colors to each point in trajectory
    point_colors = []
    point_phases = []
    for t in time_steps:
        if t in time_to_phase:
            point_colors.append(time_to_color[t])
            point_phases.append(time_to_phase[t])
        else:
            point_colors.append([0.5, 0.5, 0.5, 1.0])  # Gray for undefined
            point_phases.append('Unknown')
    
    # Plot the trajectory line
    ax.plot(tphate_data[:, 0], tphate_data[:, 1], 'k-', alpha=0.2, linewidth=1.5)
    
    # Create scatter plot colored by phase
    ax.scatter(tphate_data[:, 0], tphate_data[:, 1],
               c=point_colors, alpha=0.8, s=50,
               edgecolors='black', linewidth=0.5, zorder=5)
    
    # Mark start and end points
    ax.plot(tphate_data[0, 0], tphate_data[0, 1], 'go', markersize=12, label='Start', zorder=6)
    ax.plot(tphate_data[-1, 0], tphate_data[-1, 1], 'r*', markersize=20, label='End', zorder=6)
    
    # Labels and formatting
    ax.set_xlabel("TPHATE Dimension 1", fontsize=12)
    ax.set_ylabel("TPHATE Dimension 2", fontsize=12)
    ax.set_title(f"TPHATE Trajectory (Colored by Phase): {cell_id}", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Create legend for phases
    legend_patches = []
    for idx, row in phases_df.iterrows():
        patch = mpatches.Patch(color=phase_colors[idx], label=row['phase'])
        legend_patches.append(patch)
    
    # Add start/end to legend
    legend_patches.append(mpatches.Patch(color='green', label='Start'))
    legend_patches.append(mpatches.Patch(color='red', label='End'))
    
    # Position legend outside plot area
    ax.legend(handles=legend_patches, fontsize=9, loc='center left', 
              bbox_to_anchor=(1, 0.5), framealpha=0.9)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"timestamp_{cell_id}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved trajectory plot: {output_file}")


def process_cell_id_batch(cell_ids, df, latent_cols, output_dir="plots"):
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
        plot_cell_trajectory_circle(cell_id, cell_tphate, cell_time_steps, output_dir)
        plot_cell_trajectory_velocity(cell_id, cell_tphate, cell_time_steps, output_dir)
        plot_cell_trajectory_timestamp(cell_id, cell_tphate, cell_time_steps, output_dir)

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

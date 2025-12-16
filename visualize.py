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
import warnings
import tphate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.gridspec import GridSpec
import argparse

# Suppress statsmodels warnings from TPHATE's autocorrelation calculations
warnings.filterwarnings('ignore', message='invalid value encountered in divide', category=RuntimeWarning)


def load_latents(csv_file):
    print(f"Loading latent embeddings from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} samples with {len(df.columns)} columns")

    latent_cols = [col for col in df.columns if col.startswith('z_')]
    print(f"  Using {len(latent_cols)} latent dimensions")

    return df, latent_cols


def apply_tphate(data, n_jobs=-1):

    print(f"  Applying TPHATE to {data.shape[0]} samples...")
    tphate_op = tphate.TPHATE(n_jobs=n_jobs, n_components=3)
    tphate_data = tphate_op.fit_transform(data)
    print(f"  TPHATE output shape: {tphate_data.shape}")

    return tphate_data



def fit_circle_2d(x, y, w=[]):

    A = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2

    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W,A)
        b = np.dot(W,b)

    # Solve by method of least squares with error handling
    try:
        c = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        # If SVD fails, return default values
        return 0, 0, 1e10

    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    r_sq = c[2] + xc**2 + yc**2

    # Ensure radius is positive and reasonable
    if r_sq <= 0:
        return xc, yc, 1e10
    r = np.sqrt(r_sq)
    return xc, yc, r

def rodrigues_rot(P, n0, n1):

    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[np.newaxis, :]
    
    # Get vector of rotation k and angle theta
    n0 = n0/np.linalg.norm(n0)
    n1 = n1/np.linalg.norm(n1)
    k = np.cross(n0,n1)
    k = k/np.linalg.norm(k)
    theta = np.arccos(np.dot(n0,n1))
    
    # Compute rotated points
    P_rot = np.zeros((len(P),3))
    for i in range(len(P)):
        P_rot[i] = P[i]*np.cos(theta) + np.cross(k,P[i])*np.sin(theta) + k*np.dot(k,P[i])*(1-np.cos(theta))

    return P_rot

def angle_between(u, v, n=None):
    
    if n is None:
        return np.arctan2(np.linalg.norm(np.cross(u,v)), np.dot(u,v))
    else:
        return np.arctan2(np.dot(n,np.cross(u,v)), np.dot(u,v))
    
def compute_curvature(nbd, traj, num_pts):

    kappa = []

    for pt_idx in range(0, num_pts):

        P = traj[max(0, pt_idx-nbd):min(num_pts, pt_idx+nbd),:]
        P_mean = P.mean(axis=0)
        P_centered = P - P_mean

        try:
            U, s, V = np.linalg.svd(P_centered, full_matrices=True)
            # Handle low-rank case: if last singular value is very small, use it anyway
            normal = V[-1, :]
            P_xy = rodrigues_rot(P_centered, normal, [0, 0, 1])
            xc, yc, r = fit_circle_2d(P_xy[:, 0], P_xy[:, 1])
            # Clamp curvature to reasonable range
            kappa.append(min(1.0 / r, 1e10) if r > 0 else 0)
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
            # If curvature computation fails, use zero curvature
            kappa.append(0)

    return kappa
def plot_cell_trajectory_circle(cell_id, tphate_data, time_steps, output_dir="plots"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    print("making plots with ", str(tphate_data.shape), " shape")

    # Plot trajectory line
    ax.plot(tphate_data[:, 0], tphate_data[:, 1], tphate_data[:, 2],
            'k-', alpha=0.3, linewidth=1.5)

    # Calculate curvature using the compute_curvature function
    n_points = len(tphate_data)
    nbd = 2  # Neighborhood size for curvature calculation
    curvature = compute_curvature(nbd, tphate_data, n_points)

    # Normalize curvature for color mapping
    curvature_range = np.max(curvature) - np.min(curvature)
    if curvature_range > 1e-10:
        norm_curvature = (curvature - np.min(curvature)) / curvature_range
    else:
        norm_curvature = np.zeros_like(curvature)

    colors = plt.cm.jet(norm_curvature)
    scatter = ax.scatter(tphate_data[:, 0], tphate_data[:, 1], tphate_data[:, 2],
                         c=colors, alpha=0.8, s=50,
                         edgecolors='black', linewidth=0.5, zorder=5)

    # Mark start and end
    ax.scatter(tphate_data[0, 0], tphate_data[0, 1], tphate_data[0, 2],
               c='green', marker='o', s=120, label='Start', zorder=6, edgecolors='black', linewidth=1)
    ax.scatter(tphate_data[-1, 0], tphate_data[-1, 1], tphate_data[-1, 2],
               c='red', marker='*', s=300, label='End', zorder=6, edgecolors='black', linewidth=1)

    ax.set_xlabel("TPHATE Dimension 1", fontsize=12)
    ax.set_ylabel("TPHATE Dimension 2", fontsize=12)
    ax.set_zlabel("TPHATE Dimension 3", fontsize=12)
    ax.set_title(f"TPHATE Trajectory (Colored by Curvature): {cell_id}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("Curvature", fontsize=11)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"circle_{cell_id}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved 3D trajectory plot: {output_file}")

def plot_cell_trajectory_velocity(cell_id, tphate_data, time_steps, output_dir="plots"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    print("making plots with ", str(tphate_data.shape), " shape")

    # Plot the trajectory line
    ax.plot(tphate_data[:, 0], tphate_data[:, 1], tphate_data[:, 2],
            'k-', alpha=0.3, linewidth=1.5)

    # Calculate velocity between consecutive points
    dx = np.diff(tphate_data[:, 0])
    dy = np.diff(tphate_data[:, 1])
    dz = np.diff(tphate_data[:, 2])
    distances = np.sqrt(dx**2 + dy**2 + dz**2)

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
    scatter = ax.scatter(tphate_data[:, 0], tphate_data[:, 1], tphate_data[:, 2],
                         c=point_velocities, cmap='jet', alpha=0.8, s=50,
                         edgecolors='black', linewidth=0.5, zorder=5)

    # Mark start and end points
    ax.scatter(tphate_data[0, 0], tphate_data[0, 1], tphate_data[0, 2],
               c='green', marker='o', s=120, label='Start', zorder=6, edgecolors='black', linewidth=1)
    ax.scatter(tphate_data[-1, 0], tphate_data[-1, 1], tphate_data[-1, 2],
               c='red', marker='*', s=300, label='End', zorder=6, edgecolors='black', linewidth=1)

    # Labels and formatting
    ax.set_xlabel("TPHATE Dimension 1", fontsize=12)
    ax.set_ylabel("TPHATE Dimension 2", fontsize=12)
    ax.set_zlabel("TPHATE Dimension 3", fontsize=12)
    ax.set_title(f"TPHATE Trajectory (Colored by Velocity): {cell_id}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label("Velocity (TPHATE units/time step)", fontsize=11)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f"velocity_{cell_id}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved 3D trajectory plot: {output_file}")


def plot_cell_trajectory_timestamp(cell_id, tphate_data, time_steps, output_dir="plots"):
    import pandas as pd
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
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
    ax.plot(tphate_data[:, 0], tphate_data[:, 1], tphate_data[:, 2],
            'k-', alpha=0.2, linewidth=1.5)

    # Create scatter plot colored by phase
    ax.scatter(tphate_data[:, 0], tphate_data[:, 1], tphate_data[:, 2],
               c=point_colors, alpha=0.8, s=50,
               edgecolors='black', linewidth=0.5, zorder=5)

    # Mark start and end points
    ax.scatter(tphate_data[0, 0], tphate_data[0, 1], tphate_data[0, 2],
               c='green', marker='o', s=120, label='Start', zorder=6, edgecolors='black', linewidth=1)
    ax.scatter(tphate_data[-1, 0], tphate_data[-1, 1], tphate_data[-1, 2],
               c='red', marker='*', s=300, label='End', zorder=6, edgecolors='black', linewidth=1)

    # Labels and formatting
    ax.set_xlabel("TPHATE Dimension 1", fontsize=12)
    ax.set_ylabel("TPHATE Dimension 2", fontsize=12)
    ax.set_zlabel("TPHATE Dimension 3", fontsize=12)
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
    print(f"    Saved 3D trajectory plot: {output_file}")


def create_merged_comparison_plot(cell_ids, df, latent_cols, grades_df, output_dir="plots"):
    """
    Create a merged 3D plot showing all cell trajectories colored by their grade category.
    """
    print(f"\n  Creating merged 3D comparison plot for {len(cell_ids)} cells")

    # Group cells by grade
    grade_groups = {}
    for cell_id in cell_ids:
        if grades_df is not None:
            grade_row = grades_df[grades_df['cell_id'] == cell_id]
            if not grade_row.empty:
                g1 = grade_row.iloc[0]['grade1']
                g2 = grade_row.iloc[0]['grade2']
                g1_str = str(g1) if not pd.isna(g1) else 'NA'
                g2_str = str(g2) if not pd.isna(g2) else 'NA'
                grade_cat = f"{g1_str}-{g2_str}"
            else:
                grade_cat = "Unknown"
        else:
            grade_cat = "Unknown"

        if grade_cat not in grade_groups:
            grade_groups[grade_cat] = []
        grade_groups[grade_cat].append(cell_id)

    # Create 3D plot
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Color palette for different grades
    unique_grades = sorted(grade_groups.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_grades)))
    grade_to_color = {grade: colors[i] for i, grade in enumerate(unique_grades)}

    # Plot each cell
    for grade_cat, cells in grade_groups.items():
        color = grade_to_color[grade_cat]

        for cell_id in cells:
            cell_df = df[df['cell_id'] == cell_id]
            if len(cell_df) == 0:
                continue

            # Get latents and apply TPHATE
            cell_latents = cell_df[latent_cols].values
            cell_tphate = apply_tphate(cell_latents)

            # Plot trajectory
            ax.plot(cell_tphate[:, 0], cell_tphate[:, 1], cell_tphate[:, 2], '-',
                   color=color, alpha=0.4, linewidth=1.5)
            ax.scatter(cell_tphate[:, 0], cell_tphate[:, 1], cell_tphate[:, 2],
                      c=[color], alpha=0.6, s=30, edgecolors='black', linewidth=0.3)

            # Mark start/end
            ax.scatter(cell_tphate[0, 0], cell_tphate[0, 1], cell_tphate[0, 2],
                      c='green', marker='o', s=60, alpha=0.5, edgecolors='black', linewidth=0.5)
            ax.scatter(cell_tphate[-1, 0], cell_tphate[-1, 1], cell_tphate[-1, 2],
                      c='red', marker='*', s=120, alpha=0.5, edgecolors='black', linewidth=0.5)

    # Add legend
    import matplotlib.patches as mpatches
    legend_patches = []
    for grade_cat in unique_grades:
        patch = mpatches.Patch(color=grade_to_color[grade_cat], label=f"Grade {grade_cat}")
        legend_patches.append(patch)

    ax.legend(handles=legend_patches, loc='best', fontsize=10)
    ax.set_xlabel("TPHATE Dimension 1", fontsize=12)
    ax.set_ylabel("TPHATE Dimension 2", fontsize=12)
    ax.set_zlabel("TPHATE Dimension 3", fontsize=12)
    ax.set_title("Merged Trajectories by Grade (3D)", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "merged_all_grades.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    Saved merged 3D comparison plot: {output_file}")


def process_cell_id_batch(cell_ids, df, latent_cols, output_dir="plots", grades_df=None, create_merged=False):
    print(f"\n  Processing batch with {len(cell_ids)} cell_ids")

    # Create merged comparison plot if requested
    if create_merged and len(cell_ids) > 1:
        create_merged_comparison_plot(cell_ids, df, latent_cols, grades_df, output_dir)

    # Process each cell_id individually with its own TPHATE run
    for cell_id in cell_ids:
        cell_df = df[df['cell_id'] == cell_id]
        num_cell_samples = len(cell_df)

        print(f"    Processing {cell_id} ({num_cell_samples} samples)")

        # Determine output subdirectory based on grade
        cell_output_dir = output_dir
        if grades_df is not None:
            grade_row = grades_df[grades_df['cell_id'] == cell_id]
            if not grade_row.empty:
                g1 = grade_row.iloc[0]['grade1']
                g2 = grade_row.iloc[0]['grade2']
                g1_str = str(g1) if not pd.isna(g1) else 'NA'
                g2_str = str(g2) if not pd.isna(g2) else 'NA'
                grade_cat = f"{g1_str}-{g2_str}"
                cell_output_dir = os.path.join(output_dir, grade_cat)
                os.makedirs(cell_output_dir, exist_ok=True)
            else:
                cell_output_dir = os.path.join(output_dir, "Unknown")
                os.makedirs(cell_output_dir, exist_ok=True)

        # Get latents for this cell_id only
        cell_latents = cell_df[latent_cols].values

        # Apply TPHATE to this cell_id's data individually
        cell_tphate = apply_tphate(cell_latents)
        cell_time_steps = cell_df['time_step'].values

        # Create scatter and trajectory plots
        plot_cell_trajectory_circle(cell_id, cell_tphate, cell_time_steps, cell_output_dir)
        plot_cell_trajectory_velocity(cell_id, cell_tphate, cell_time_steps, cell_output_dir)
        plot_cell_trajectory_timestamp(cell_id, cell_tphate, cell_time_steps, cell_output_dir)

def filter_cells_by_grade(grades_df, grade_filter):
    """
    Filter cell_ids based on grade criteria.

    Args:
        grades_df: DataFrame with columns ['cell_id', 'grade1', 'grade2']
        grade_filter: Filter criteria, can be:
            - "all": all cells
            - Single letter: "A", "B", "C" (matches if either grade matches)
            - Exact category: "A-A", "B-NA", "NA-C", etc.
            - Special: "any_A", "any_B", "any_C" (at least one of the grades)

    Returns:
        List of cell_ids matching the filter
    """
    if grades_df is None or grade_filter == "all":
        return None  # Will process all cells

    matching_cells = []

    for _, row in grades_df.iterrows():
        cell_id = row['cell_id']
        g1 = row['grade1']
        g2 = row['grade2']

        # Convert to strings for comparison
        g1_str = str(g1) if not pd.isna(g1) else 'NA'
        g2_str = str(g2) if not pd.isna(g2) else 'NA'

        # Check filter criteria
        if grade_filter in ['A', 'B', 'C']:
            # Single letter: match if either grade matches
            if g1_str == grade_filter or g2_str == grade_filter:
                matching_cells.append(cell_id)

        elif grade_filter.startswith('any_'):
            # any_X: at least one grade is X
            target_grade = grade_filter.split('_')[1]
            if g1_str == target_grade or g2_str == target_grade:
                matching_cells.append(cell_id)

        elif '-' in grade_filter:
            # Exact category match
            g1_filter, g2_filter = grade_filter.split('-')
            if g1_str == g1_filter and g2_str == g2_filter:
                matching_cells.append(cell_id)

    return matching_cells


def process_all_cells_batched(df, latent_cols, grades_df, output_dir, cell_ids_filter=None,
                               create_merged=False, batch_size=50):
    """
    Process all cells in batches for memory efficiency.

    Args:
        df: DataFrame with latents
        latent_cols: List of latent column names
        grades_df: DataFrame with grades
        output_dir: Output directory
        cell_ids_filter: Optional list of cell_ids to process (None = all)
        create_merged: Whether to create merged comparison plots
        batch_size: Number of cells to process before clearing memory
    """
    # Get unique cell_ids
    if cell_ids_filter is not None:
        all_cell_ids = [cid for cid in cell_ids_filter if cid in df['cell_id'].values]
    else:
        all_cell_ids = df['cell_id'].unique().tolist()

    total_cells = len(all_cell_ids)
    print(f"\nProcessing {total_cells} cells in batches of {batch_size}")

    # Process in batches
    for batch_start in range(0, total_cells, batch_size):
        batch_end = min(batch_start + batch_size, total_cells)
        batch_cells = all_cell_ids[batch_start:batch_end]

        print(f"\n{'='*60}")
        print(f"Batch {batch_start//batch_size + 1}: Processing cells {batch_start+1} to {batch_end}")
        print(f"{'='*60}")

        # Process this batch
        process_cell_id_batch(batch_cells, df, latent_cols, output_dir, grades_df,
                            create_merged=(create_merged and batch_start == 0))

        # Clear memory between batches
        import gc
        gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize cell trajectories with TPHATE",
        epilog="""
Examples:
  # Process specific cells
  python visualize.py latents.csv --cells "cell1,cell2,cell3"

  # Process all cells
  python visualize.py latents.csv --all

  # Process cells with grade B
  python visualize.py latents.csv --grade-filter "any_B" --by-grade

  # Process all A-A grade cells with merged plot
  python visualize.py latents.csv --grade-filter "A-A" --by-grade --create-merged
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("latents_csv", help="Path to latents CSV file")
    parser.add_argument("--cells", type=str, help="Comma-separated list of cell_ids to process")
    parser.add_argument("--all", action="store_true", help="Process all cells in the CSV")
    parser.add_argument("--grade-filter", type=str,
                       help="Filter cells by grade (e.g., 'A', 'B', 'A-A', 'any_B')")
    parser.add_argument("--output", type=str, default="plots", help="Output directory for plots")
    parser.add_argument("--by-grade", action="store_true", help="Organize plots by embryo grades")
    parser.add_argument("--grades-file", type=str, default="embryo_dataset_grades.csv",
                       help="Path to grades CSV file")
    parser.add_argument("--create-merged", action="store_true", help="Create merged comparison plot")
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Number of cells to process per batch (for memory efficiency)")

    # Legacy support for old interface
    parser.add_argument("cell_line", nargs='?', help="(Legacy) Cell group line (comma-separated cell_ids)")

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

    # Load grades if requested
    grades_df = None
    if args.by_grade or args.create_merged or args.grade_filter:
        if os.path.exists(args.grades_file):
            print(f"Loading grades from: {args.grades_file}")
            grades_df = pd.read_csv(args.grades_file, header=None, names=['cell_id', 'grade1', 'grade2'])
            print(f"  Loaded grades for {len(grades_df)} cell_ids\n")
        else:
            print(f"Warning: Grades file not found: {args.grades_file}")
            if args.grade_filter:
                print("Cannot use --grade-filter without grades file")
                sys.exit(1)
            print("Proceeding without grade organization\n")

    # Determine which cells to process
    cell_ids = None

    if args.all:
        # Process all cells
        print("Processing all cells in the CSV")
        cell_ids = None  # Will be handled in process_all_cells_batched

    elif args.grade_filter:
        # Filter by grade
        print(f"Filtering cells by grade: {args.grade_filter}")
        cell_ids = filter_cells_by_grade(grades_df, args.grade_filter)
        if not cell_ids:
            print(f"No cells found matching grade filter: {args.grade_filter}")
            sys.exit(1)
        print(f"Found {len(cell_ids)} cells matching filter")

    elif args.cells:
        # Parse cell_ids from --cells argument
        cell_ids = args.cells.strip().split(',')
        cell_ids = [cid.strip() for cid in cell_ids if cid.strip()]

    elif args.cell_line:
        # Legacy support: positional argument
        cell_ids = args.cell_line.strip().split(',')
        cell_ids = [cid.strip() for cid in cell_ids if cid.strip()]

    else:
        print("Error: Must specify --all, --cells, or --grade-filter")
        print("Use --help for more information")
        sys.exit(1)

    # Process cells
    if cell_ids is None:
        # Process all cells in batches
        process_all_cells_batched(df, latent_cols, grades_df, args.output,
                                 cell_ids_filter=None,
                                 create_merged=args.create_merged,
                                 batch_size=args.batch_size)
    elif len(cell_ids) > args.batch_size:
        # Process filtered cells in batches
        process_all_cells_batched(df, latent_cols, grades_df, args.output,
                                 cell_ids_filter=cell_ids,
                                 create_merged=args.create_merged,
                                 batch_size=args.batch_size)
    else:
        # Process small batch directly
        print(f"Processing {len(cell_ids)} cell_ids:")
        if len(cell_ids) <= 10:
            print(f"  Cell IDs: {', '.join(cell_ids)}\n")
        else:
            print(f"  First 10: {', '.join(cell_ids[:10])}...\n")
        process_cell_id_batch(cell_ids, df, latent_cols, args.output, grades_df, args.create_merged)

    print(f"\n\nVisualization complete! Plots saved to: {args.output}")


if __name__ == "__main__":
    main()

"""
Visualization Utilities for Grade-based Analysis

Provides utility functions for creating merged plots and grade-based comparisons
of TPHATE trajectories.

Usage:
    python visualize_utils.py latents.csv --grade-filter "B" --plot-type circle
    python visualize_utils.py latents.csv --grade-filter "A-A" --plot-type velocity
    python visualize_utils.py latents.csv --compare-grades --output comparison_plots
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tphate

# Suppress warnings
warnings.filterwarnings('ignore', message='invalid value encountered in divide', category=RuntimeWarning)


def load_latents(csv_file):
    """Load latent embeddings from CSV"""
    print(f"Loading latent embeddings from: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} samples with {len(df.columns)} columns")

    latent_cols = [col for col in df.columns if col.startswith('z_')]
    print(f"  Using {len(latent_cols)} latent dimensions")

    return df, latent_cols


def load_grades(grades_file="embryo_dataset_grades.csv"):
    """Load embryo grades from CSV"""
    if not os.path.exists(grades_file):
        print(f"Warning: Grades file not found: {grades_file}")
        return None

    print(f"Loading grades from: {grades_file}")
    grades_df = pd.read_csv(grades_file, header=None, names=['cell_id', 'grade1', 'grade2'])
    print(f"  Loaded grades for {len(grades_df)} cell_ids")

    return grades_df


def get_grade_category(grade1, grade2):
    """Convert grade1, grade2 into a category string"""
    g1_str = str(grade1) if not pd.isna(grade1) else 'NA'
    g2_str = str(grade2) if not pd.isna(grade2) else 'NA'
    return f"{g1_str}-{g2_str}"


def filter_cells_by_grade(grades_df, grade_filter):
    """
    Filter cell_ids based on grade criteria.

    Args:
        grades_df: DataFrame with columns ['cell_id', 'grade1', 'grade2']
        grade_filter: Filter criteria, can be:
            - Single letter: "A", "B", "C" (matches if either grade matches)
            - Exact category: "A-A", "B-NA", "NA-C", etc.
            - Special: "any_A", "any_B", "any_C" (at least one of the grades)

    Returns:
        List of cell_ids matching the filter
    """
    if grades_df is None:
        return []

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
            category = get_grade_category(g1, g2)
            if category == grade_filter:
                matching_cells.append(cell_id)

        elif grade_filter == 'all':
            # Include all cells
            matching_cells.append(cell_id)

    return matching_cells


def apply_tphate(data, n_jobs=-1):
    """Apply TPHATE to data"""
    print(f"  Applying TPHATE to {data.shape[0]} samples...")
    tphate_op = tphate.TPHATE(n_jobs=n_jobs)
    tphate_data = tphate_op.fit_transform(data)
    print(f"  TPHATE output shape: {tphate_data.shape}")

    return tphate_data


def plot_single_trajectory(ax, tphate_data, time_steps, cell_id, color='blue', alpha=0.6):
    """Plot a single trajectory on given axes"""
    # Plot line
    ax.plot(tphate_data[:, 0], tphate_data[:, 1], '-', color=color, alpha=alpha*0.5, linewidth=1.5)

    # Plot points
    ax.scatter(tphate_data[:, 0], tphate_data[:, 1], c=color, alpha=alpha, s=30,
               edgecolors='black', linewidth=0.3, zorder=5)

    # Mark start and end
    ax.plot(tphate_data[0, 0], tphate_data[0, 1], 'go', markersize=8, zorder=6)
    ax.plot(tphate_data[-1, 0], tphate_data[-1, 1], 'r*', markersize=12, zorder=6)


def create_merged_plot(cell_ids, df, latent_cols, output_file, title="Merged Trajectories"):
    """
    Create a single plot with multiple cell trajectories overlaid.

    Args:
        cell_ids: List of cell_ids to include
        df: DataFrame with latents
        latent_cols: List of latent column names
        output_file: Path to save the plot
        title: Plot title
    """
    print(f"\nCreating merged plot with {len(cell_ids)} cells")

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(cell_ids)))

    for idx, cell_id in enumerate(cell_ids):
        cell_df = df[df['cell_id'] == cell_id]
        if len(cell_df) == 0:
            print(f"  Warning: No data for {cell_id}")
            continue

        print(f"  Processing {cell_id} ({len(cell_df)} samples)")

        # Get latents and apply TPHATE
        cell_latents = cell_df[latent_cols].values
        cell_tphate = apply_tphate(cell_latents)
        cell_time_steps = cell_df['time_step'].values

        # Plot trajectory
        plot_single_trajectory(ax, cell_tphate, cell_time_steps, cell_id,
                             color=colors[idx], alpha=0.7)

    ax.set_xlabel("TPHATE Dimension 1", fontsize=12)
    ax.set_ylabel("TPHATE Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Legend
    ax.plot([], [], 'go', markersize=8, label='Start')
    ax.plot([], [], 'r*', markersize=12, label='End')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved merged plot: {output_file}")


def create_grid_plot(cell_ids, df, latent_cols, output_file, title="Grid Comparison", max_cols=4):
    """
    Create a grid of individual trajectory plots.

    Args:
        cell_ids: List of cell_ids to include
        df: DataFrame with latents
        latent_cols: List of latent column names
        output_file: Path to save the plot
        title: Plot title
        max_cols: Maximum number of columns in grid
    """
    print(f"\nCreating grid plot with {len(cell_ids)} cells")

    n_cells = len(cell_ids)
    n_cols = min(max_cols, n_cells)
    n_rows = (n_cells + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3)

    for idx, cell_id in enumerate(cell_ids):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        cell_df = df[df['cell_id'] == cell_id]
        if len(cell_df) == 0:
            print(f"  Warning: No data for {cell_id}")
            continue

        print(f"  Processing {cell_id} ({len(cell_df)} samples)")

        # Get latents and apply TPHATE
        cell_latents = cell_df[latent_cols].values
        cell_tphate = apply_tphate(cell_latents)
        cell_time_steps = cell_df['time_step'].values

        # Plot trajectory
        plot_single_trajectory(ax, cell_tphate, cell_time_steps, cell_id,
                             color='blue', alpha=0.8)

        ax.set_title(f"{cell_id}", fontsize=10)
        ax.set_xlabel("TPHATE Dim 1", fontsize=9)
        ax.set_ylabel("TPHATE Dim 2", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, y=0.995)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved grid plot: {output_file}")


def is_grade_at_least(grade_str, min_grade='B'):
    """
    Check if a grade meets the minimum threshold.

    Args:
        grade_str: Grade string (e.g., 'A', 'B', 'C', 'NA')
        min_grade: Minimum grade threshold ('A' or 'B')

    Returns:
        True if grade >= min_grade
    """
    if grade_str == 'NA' or pd.isna(grade_str):
        return False

    grade_order = {'A': 0, 'B': 1, 'C': 2}
    if grade_str not in grade_order or min_grade not in grade_order:
        return False

    return grade_order[grade_str] <= grade_order[min_grade]


def filter_cells_by_grade_threshold(grades_df, min_grade='B'):
    """
    Filter cells where at least one grade meets the minimum threshold.

    Args:
        grades_df: DataFrame with columns ['cell_id', 'grade1', 'grade2']
        min_grade: Minimum grade threshold ('A' or 'B')

    Returns:
        List of cell_ids where grade1 OR grade2 >= min_grade
    """
    matching_cells = []

    for _, row in grades_df.iterrows():
        cell_id = row['cell_id']
        g1 = row['grade1']
        g2 = row['grade2']

        g1_str = str(g1) if not pd.isna(g1) else 'NA'
        g2_str = str(g2) if not pd.isna(g2) else 'NA'

        # Include if either grade meets threshold
        if is_grade_at_least(g1_str, min_grade) or is_grade_at_least(g2_str, min_grade):
            matching_cells.append(cell_id)

    return matching_cells


def create_grade_comparison_plots(df, latent_cols, grades_df, output_dir="comparison_plots"):
    """
    Create comparison plots for all grade categories.

    Args:
        df: DataFrame with latents
        latent_cols: List of latent column names
        grades_df: DataFrame with grades
        output_dir: Output directory for plots
    """
    print(f"\nCreating grade comparison plots")
    os.makedirs(output_dir, exist_ok=True)

    # Get unique grade categories
    grade_categories = {}
    for _, row in grades_df.iterrows():
        cell_id = row['cell_id']
        grade_cat = get_grade_category(row['grade1'], row['grade2'])

        if grade_cat not in grade_categories:
            grade_categories[grade_cat] = []
        grade_categories[grade_cat].append(cell_id)

    print(f"\nFound {len(grade_categories)} grade categories:")
    for grade_cat, cells in grade_categories.items():
        print(f"  {grade_cat}: {len(cells)} cells")

    # Create merged plot for each category
    for grade_cat, cells in grade_categories.items():
        if len(cells) == 0:
            continue

        # Merged overlay plot
        merged_file = os.path.join(output_dir, f"merged_{grade_cat}.png")
        create_merged_plot(cells, df, latent_cols, merged_file,
                          title=f"Merged Trajectories: Grade {grade_cat}")

        # Grid plot
        grid_file = os.path.join(output_dir, f"grid_{grade_cat}.png")
        create_grid_plot(cells, df, latent_cols, grid_file,
                        title=f"Individual Trajectories: Grade {grade_cat}")

    # Create plots for "any_X" categories
    for grade_letter in ['A', 'B', 'C']:
        cells = filter_cells_by_grade(grades_df, grade_letter)
        if len(cells) == 0:
            continue

        print(f"\nProcessing any_{grade_letter}: {len(cells)} cells")

        # Merged plot
        merged_file = os.path.join(output_dir, f"merged_any_{grade_letter}.png")
        create_merged_plot(cells, df, latent_cols, merged_file,
                          title=f"Merged Trajectories: Any Grade {grade_letter}")

        # Grid plot
        grid_file = os.path.join(output_dir, f"grid_any_{grade_letter}.png")
        create_grid_plot(cells, df, latent_cols, grid_file,
                        title=f"Individual Trajectories: Any Grade {grade_letter}")


def main():
    parser = argparse.ArgumentParser(description="Visualization utilities for grade-based analysis")
    parser.add_argument("latents_csv", help="Path to latents CSV file")
    parser.add_argument("--output", type=str, default="plots", help="Output directory")
    parser.add_argument("--grades-file", type=str, default="embryo_dataset_grades.csv",
                       help="Path to grades CSV file")
    parser.add_argument("--compare-grades", action="store_true",
                       help="Create comparison plots for all grades")
    parser.add_argument("--grade-filter", type=str,
                       help="Filter by grade (e.g., 'A', 'B', 'A-A', 'any_B')")
    parser.add_argument("--grade-threshold", type=str, choices=['A', 'B'],
                       help="Create aggregated plot for cells at least this grade (A or B)")
    parser.add_argument("--plot-type", type=str, choices=['merged', 'grid', 'both'],
                       default='both', help="Type of plot to create")
    parser.add_argument("--max-cells", type=int, help="Maximum number of cells to include")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.latents_csv):
        print(f"Error: Latents CSV not found: {args.latents_csv}")
        sys.exit(1)

    # Load data
    df, latent_cols = load_latents(args.latents_csv)
    grades_df = load_grades(args.grades_file)

    if grades_df is None:
        print("Error: Cannot proceed without grades file")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Execute based on arguments
    if args.compare_grades:
        # Create all grade comparison plots
        create_grade_comparison_plots(df, latent_cols, grades_df, args.output)

    elif args.grade_threshold:
        # Create aggregated plot for cells at grade threshold or better
        cells = filter_cells_by_grade_threshold(grades_df, args.grade_threshold)

        if len(cells) == 0:
            print(f"Error: No cells found at grade {args.grade_threshold} or better")
            sys.exit(1)

        print(f"\nFound {len(cells)} cells graded {args.grade_threshold} or better")

        # Limit cells if requested
        if args.max_cells and len(cells) > args.max_cells:
            print(f"Limiting to first {args.max_cells} cells")
            cells = cells[:args.max_cells]

        # Create plots
        if args.plot_type in ['merged', 'both']:
            output_file = os.path.join(args.output, f"merged_at_least_{args.grade_threshold}.png")
            create_merged_plot(cells, df, latent_cols, output_file,
                             title=f"Merged Trajectories: Grade {args.grade_threshold}+")

        if args.plot_type in ['grid', 'both']:
            output_file = os.path.join(args.output, f"grid_at_least_{args.grade_threshold}.png")
            create_grid_plot(cells, df, latent_cols, output_file,
                           title=f"Individual Trajectories: Grade {args.grade_threshold}+")

    elif args.grade_filter:
        # Filter cells by grade
        cells = filter_cells_by_grade(grades_df, args.grade_filter)

        if len(cells) == 0:
            print(f"Error: No cells found matching grade filter: {args.grade_filter}")
            sys.exit(1)

        print(f"\nFound {len(cells)} cells matching grade filter '{args.grade_filter}'")

        # Limit cells if requested
        if args.max_cells and len(cells) > args.max_cells:
            print(f"Limiting to first {args.max_cells} cells")
            cells = cells[:args.max_cells]

        # Create plots
        if args.plot_type in ['merged', 'both']:
            output_file = os.path.join(args.output, f"merged_{args.grade_filter}.png")
            create_merged_plot(cells, df, latent_cols, output_file,
                             title=f"Merged Trajectories: Grade {args.grade_filter}")

        if args.plot_type in ['grid', 'both']:
            output_file = os.path.join(args.output, f"grid_{args.grade_filter}.png")
            create_grid_plot(cells, df, latent_cols, output_file,
                           title=f"Individual Trajectories: Grade {args.grade_filter}")

    else:
        print("Error: Must specify --compare-grades, --grade-threshold, or --grade-filter")
        sys.exit(1)

    print(f"\n\nVisualization complete! Plots saved to: {args.output}")


if __name__ == "__main__":
    main()

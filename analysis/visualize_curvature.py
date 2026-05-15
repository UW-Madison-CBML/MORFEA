#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

from geometric_features import calculate_curvatures
def load_phase_annotations(embryo_id, phases_dir='embryo_dataset_annotations'):
    """Load phase annotations for a given embryo_id."""
    phase_file = Path(phases_dir) / f"{embryo_id}_phases.csv"
    
    if not phase_file.exists():
        return None
    
    phases = []
    with open(phase_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                phase_name = parts[0]
                start = int(parts[1])
                end = int(parts[2])
                phases.append((phase_name, start, end))
    
    return phases

def get_phase_color(timepoint, phases):
    """Get the phase color for a given timepoint."""
    if phases is None:
        return 'gray'
    
    # Define a color map for phases
    phase_colors = {
        'tPB2': '#1f77b4',
        'tPNa': '#ff7f0e',
        'tPNf': '#2ca02c',
        't2': '#d62728',
        't3': '#9467bd',
        't4': '#8c564b',
        't5': '#e377c2',
        't6': '#7f7f7f',
        't7': '#bcbd22',
        't8': '#17becf',
        't9+': '#aec7e8',
        'tM': '#ffbb78',
    }
    
    for phase_name, start, end in phases:
        if start <= timepoint <= end:
            return phase_colors.get(phase_name, 'gray')
    
    return 'gray'

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <model_name>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    
    # Load CSV and NPY files
    csv_path = Path('latents') / f"{model_name}.csv"
    npy_path = Path('latents') / f"{model_name}.npy"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist")
        sys.exit(1)
    
    if not npy_path.exists():
        print(f"Error: {npy_path} does not exist")
        sys.exit(1)
    
    # Load data
    df = pd.read_csv(csv_path)
    latents = np.load(npy_path)
    
    # Add latent columns to dataframe
    latent_cols = []
    n_dims = latents.shape[1]
    for i in range(n_dims):
        df[f'z_{i}'] = latents[:, i]
        latent_cols.append(f"z_{i}")
    df.rename(columns={"cell_id":"embryo_id", "video_name":"embryo_id"}) 
    print(df.head())
    # Group by embryo_id
    grouped = df.groupby('embryo_id')
    count = 0 
    # Create plots for each embryo
    for embryo_id, group in grouped:
        count += 1
        # Sort by timepoint if available
        if 'time_step' in group.columns:
            group = group.sort_values('time_step')
        
        trajectory = group[latent_cols].values.astype(np.float32)
        if(count % 50 == 0):
            print(trajectory)
        # Calculate curvatures
        curvatures = calculate_curvatures(trajectory, offset=20, how="triangle")
        
        # Load phase annotations
        phases = load_phase_annotations(embryo_id)
        
        # Create timepoints array
        if 'time_step' in group.columns:
            timepoints = group['time_step'].values
        else:
            timepoints = np.arange(len(trajectory))
        
        # Get colors for each timepoint
        colors = [get_phase_color(tp, phases) for tp in timepoints]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot curvature with phase colors
        for i in range(len(timepoints) - 1):
            ax.plot(timepoints[i:i+2], curvatures[i:i+2], 
                   color=colors[i], linewidth=2)
        
        ax.set_xlabel('Timepoint', fontsize=12)
        ax.set_ylabel('Curvature (1/radius)', fontsize=12)
        ax.set_title(f'Trajectory Curvature for Embryo {embryo_id}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add legend for phases if available
        if phases is not None:
            phase_colors = {
                'tPB2': '#1f77b4', 'tPNa': '#ff7f0e', 'tPNf': '#2ca02c',
                't2': '#d62728', 't3': '#9467bd', 't4': '#8c564b',
                't5': '#e377c2', 't6': '#7f7f7f', 't7': '#bcbd22',
                't8': '#17becf', 't9+': '#aec7e8', 'tM': '#ffbb78',
            }
            legend_elements = [Patch(facecolor=color, label=phase)
                             for phase, color in phase_colors.items()
                             if any(p[0] == phase for p in phases)]
            ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        # Save figure
        output_path = Path('curvature_plots') / f"{embryo_id}_curvature.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot for embryo {embryo_id}")
    
    print(f"\nAll plots saved to curvature_plots/")

if __name__ == "__main__":
    main()

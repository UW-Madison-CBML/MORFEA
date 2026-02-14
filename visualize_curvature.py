#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import least_squares
def fit_circle_curvature(points, how=""):
    """
    Fit a circle to 3 consecutive points and return curvature (1/radius).
    If points are collinear or too close, return 0.
    """
    if(how == "triangle"):
    
        # Get three points
        points = points[::max(1,len(points)//3)]
        p1, p2, p3 = points[0], points[1], points[2]

        # Calculate the radius using the circumradius formula
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        
        # Area using Heron's formula
        s = (a + b + c) / 2
        area_squared = s * (s - a) * (s - b) * (s - c)
        
        if area_squared <= 0:
            return 0  # Collinear points
        print(area_squared)
        
        area = np.sqrt(area_squared)
         
        print(area)
        if area == 0:
            return 0
        
        # Radius = (a*b*c) / (4*Area)
        radius = (a * b * c) / (4 * area)
        print(radius)
        if radius == 0:
            return 0
        print(1/radius) 
        return 1 / radius
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(points)
        pca = PCA(n_components=2)
        pca.fit(X_scaled)
        points_2d = pca.transform(X_scaled)       
        def circle_residuals(params, points):
            xc, yc, R = params
            # Calculate distance from each point to the center (xc, yc)
            distances = np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)
            # The residual is the difference between these distances and the radius R
            return distances - R


        x = points[:,0]
        y = points[:,1]
        points = np.column_stack((x, y))

        x0 = [np.mean(x), np.mean(y), np.std(np.sqrt(x**2 + y**2))]
        try:
            res = least_squares(circle_residuals, x0, args=(points,))
        except ValueError as e:
            print(e)
            return 0
        if not res.success:
            return 0
        _, _, radius = res.x

        if(radius == 0):
            return 0 
        return 1/radius
        

def calculate_curvatures(trajectory):
    """Calculate curvature for each point in trajectory using sliding window."""
    offset = 6
    curvatures = []
    
    for i in range(len(trajectory)):
        if i < offset:
            points = trajectory[i:i+(2*offset)]
            curvatures.append(fit_circle_curvature(points))

        elif i >= len(trajectory) - offset:
            points = trajectory[i-(2*offset):i]
            curvatures.append(fit_circle_curvature(points))

        else:
            points = trajectory[i-offset:i+offset]
            curvatures.append(fit_circle_curvature(points))
    
    return np.array(curvatures)

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
        curvatures = calculate_curvatures(trajectory)
        
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

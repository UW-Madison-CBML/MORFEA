import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import os
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
def get_mat(traj, embryo_id, model_name, annotations_dir):
    output_dir = f"{model_name}_distances"
    if not os.path.exists(output_dir):
        raise ValueError("output folder DNE")
    dist_matrix = distance_matrix(traj, traj)
    phase_matrix = np.full(dist_matrix.shape, np.nan)
    annotation_file = os.path.join(annotations_dir, f"{embryo_id}_phases.csv")
    df = pd.read_csv(annotation_file, names=['stage_id', 'stage_begin', 'stage_end'])
    for i, row in df.iterrows():
        start, end = int(row["stage_begin"]), int(row["stage_end"])
        phase_matrix[start:end+1, :] = i  
    mask = np.triu(np.ones_like(phase_matrix, dtype=bool), k=1)
    phase_matrix_lower = np.ma.masked_where(mask, phase_matrix)
    fig, ax = plt.subplots(figsize=(10, 8)) # Slightly wider for the legend
    im_dist = ax.imshow(dist_matrix, cmap='viridis', interpolation='none')
    colors = list(plt.cm.Set3(range(12))) + list(plt.cm.Set2(range(8)))
    custom_cmap = ListedColormap(colors)
    colors = colors[:len(df)] 

    custom_cmap = mcolors.ListedColormap(colors)

    bounds = np.arange(len(df) + 1)
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
    im_phase = ax.imshow(phase_matrix_lower, 
                     cmap=custom_cmap, 
                     interpolation='none', 
                     norm=norm)
    num_phases = len(df)
    patches = [
        mpatches.Patch(color=custom_cmap(i), label=row['stage_id'])
        for i, row in df.iterrows()
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.25, 1), loc='upper left', title="Phases")
    ax.set_title(f"{embryo_id} Distance Matrix", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Time Index")
    """ax.xaxis.set_major_locator(ticker.MultipleLocator(32))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(32))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(32, offset=-0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(32, offset=-0.5))
    ax.grid(True, which='minor', color='white', linestyle='-', linewidth=0.8, alpha=0.6, zorder=1)"""
    plt.colorbar(im_dist, ax=ax, label='Distance', shrink=0.8)
    plt.savefig(os.path.join(output_dir, f'{embryo_id}_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    """fig, ax = plt.subplots(figsize=(8, 6))
    im_dist = ax.imshow(dist_matrix, cmap='viridis', interpolation='none')
    im_phase = ax.imshow(phase_matrix_lower, cmap='Set3', interpolation='none', alpha=0.6)

    patches = [mpatches.Patch(color=plt.cm.Set3(i/len(phases)), label=p[0]) for i, p in enumerate(phases)]
    ax.legend(handles=patches, bbox_to_anchor=(1.15, 1), loc='upper left', title="Phases")
    ax.set_title(f"{embryo_id} Distance Matrix", fontsize=14, fontweight='bold')

    ax.set_xlabel("Time Index")
    ax.set_ylabel("Time Index")
    # Set major ticks (with labels) every 32 values
    ax.xaxis.set_major_locator(ticker.MultipleLocator(32, offset=-0.5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(32, offset=-0.5))
    ax.grid(True, which='major', color='white', linestyle='-', linewidth=0.8, alpha=0.6, zorder=1)
    plt.colorbar(im)
    plt.savefig(os.path.join(output_dir, f'{embryo_id}_matrix.png'), dpi=300, bbox_inches='tight')

    plt.close(fig) """

def main(model_name):
    file_name = "latents/"+ model_name
    #file_name =
    keys = pd.read_csv(file_name+".csv").rename(columns={"cell_id":"embryo_id", "video_name":"embryo_id"})
    values = np.load(file_name+'.npy')
    if(len(keys) != values.shape[0]):
        raise ValueError("keys and values sizes do not match")
    lat_columns = [f"z_{i}" for i in range(values.shape[1])]
    values_df = pd.DataFrame(values, columns=lat_columns)
    df = pd.concat([keys, values_df], axis = 1)
    df = df.dropna(subset=["ICM"])
    # This returns a DataFrame where each row is a cell_id with its signature
    sizes = df.groupby("embryo_id")["time_step"].size()
    print(sizes.idxmax())
    max_points = sizes.max()
    df.groupby("embryo_id").apply(lambda group: get_mat(group[lat_columns].to_numpy().astype(np.float32), group.name, model_name, "embryo_dataset_annotations"))
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()

    main(args.name)

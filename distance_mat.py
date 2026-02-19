import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import os
def get_mat(traj, embryo_id, model_name):
    output_dir = f"{model_name}_distances"
    if not os.path.exists(output_dir):
        raise ValueError("output folder DNE")
    dist_matrix = distance_matrix(traj, traj)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(dist_matrix, cmap='viridis')
    ax.set_title(f"{embryo_id} Distance Matrix", fontsize=14, fontweight='bold')

    ax.set_xlabel("Time Index")
    ax.set_ylabel("Time Index")
    plt.colorbar(im)

    plt.savefig(output_dir+f'{embryo_id}_distance_matrix.png', dpi=300, bbox_inches='tight')

    plt.close(fig) 

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
    # This returns a DataFrame where each row is a cell_id with its signature
    sizes = df.groupby("embryo_id")["time_step"].size()
    print(sizes.idxmax())
    max_points = sizes.max()
    df.groupby("embryo_id").apply(lambda group: get_mat(group[lat_columns].to_numpy().astype(np.float32), group.name, model_name))
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()

    main(args.name)

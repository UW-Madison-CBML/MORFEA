import wandb
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import umap
from sklearn_extra.cluster import KMedoids

VAL_EMBRYOS =[
    "RG434-11",
    "RC1103-1",
    "LV488-7",
    "QC211-6",
    "BM016-2",
    "LM184-3",
    "RMN410-3",
    "PA145-1",
    "RO793-2",
    "PV361-2",
    "RC755-7",
    "VC581-3",
    "VC581-11",
    "ADM715-1-2",
    "LS1045-4",
    "GA800-4",
    "GJ191-1",
    "JV227-2",
    "LA367-4",
    "BN356-6",
    "TN611-7",
    "AHS115-5",
    "LCF544-2",
    "JV227-5",
    "CAV074-8",
    "AL702-9",
    "VH99-3",
    "GE218-3",
    "CC455-3",
    "DA1054-5",
    "ME378-4",
    "BA560-1",
    "PA145-2",
    "DSM138-5",
    "FN852-1",
    "TJ297-4",
    "RC755-9",
    "PA289-8",
    "LS93-8",
    "GA817-1-8",
    "AM918-2-5",
    "LNA592-9",
    ]
phases = ['pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM','tSB','tB', 'tEB', 'tHB', 'post_phase']
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler
def addAnnotations(group_name, group, annotations_dir):
    annotation_file = os.path.join(annotations_dir, f"{group_name}_phases.csv")
    df = pd.read_csv(annotation_file, names=['stage_id', 'stage_begin', 'stage_end'])

    new_column = []
    lat_cols = [column for column in group.columns if column.startswith("z_")]
    
    new_column += ["pre_phase"] * (df.iloc[0]["stage_begin"] - 1)
    col_len_seq = []
    for index, row in df.iterrows():
        new_column += [row["stage_id"]] * (row["stage_end"] - row["stage_begin"]+1)
        col_len_seq.append(len(new_column))

    new_column += ["post_phase"] * (len(group) - len(new_column))
    new_column = new_column[:len(group)]
     
    group["phase"] = np.array([phases.index(i) for i in new_column]) / len(phases)
    trajectory = group[lat_cols].to_numpy().astype(np.float32)
     
    distance_mat = distance_matrix(trajectory, trajectory)
    scaled_data = StandardScaler().fit_transform(distance_mat)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(scaled_data)
    plt.scatter(embedding[:, 0], embedding[:,1], c=group["phase"].to_numpy(), cmap="viridis")

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("Distance Matric Feature UMAP Cluster")

    plt.savefig(os.path.join("stage_clusters", f"{embryo_id}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main(model_name):
     
    lat_df = pd.read_csv(os.path.abspath(f"latents/{model_name}.csv")).rename(columns={"cell_id":"embryo_id"})
    lat_np = np.load(os.path.abspath(f"latents/{model_name}.npy"))
    if(len(lat_df) != lat_np.shape[0]):
        raise ValueError("keys and values sizes do not match")
    lat_columns = [f"z_{i}" for i in range(lat_np.shape[1])]
    values_df = pd.DataFrame(lat_np, columns=lat_columns)
    df = pd.concat([lat_df, values_df], axis = 1)
    df.iloc[:10].groupby("embryo_id", group_keys = False).apply(lambda group:addAnnotations(group.name,group,self.annotations_dir))
         
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")
 
 
    parser.add_argument("--name", help="Model name. Must have already exported latents")
  
    args = parser.parse_args()
 
    main(args.name)

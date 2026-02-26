import wandb
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import umap
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
 
    def push(self, x):
        """Add a new value and update statistics."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
 
    @property
    def variance(self):
        """Returns sample variance (unbiased). Use self.m2 / self.n for population."""
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0
 
    @property
    def std_dev(self):
        """Returns sample standard deviation."""
        return math.sqrt(self.variance)

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
PHASES = ['pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM','tSB','tB', 'tEB', 'tHB', 'post_phase']
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib
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
     
    group["phase"] = np.array([PHASES.index(i) for i in new_column])
    trajectory = group[lat_cols].to_numpy().astype(np.float32)
     
    distance_mat = distance_matrix(trajectory, trajectory)
    scaled_mat = StandardScaler().fit_transform(distance_mat)


    reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, n_components=2, random_state=42)
    mat_embedding = scaled_mat # just use plain matrix for clustering, we get umap embedding for visuals 
    embedding = reducer.fit_transform(scaled_mat)
    cmap = matplotlib.colormaps["tab20"].resampled(18)
    plt.scatter(embedding[:, 0], embedding[:,1], c=group["phase"].to_numpy() / 18, cmap=cmap)
    legend_handles = []
    for i, phase_name in enumerate(PHASES):
        color = cmap(i)
        patch = patches.Patch(color=color, label=phase_name)
        legend_handles.append(patch)

    plt.legend(handles=legend_handles, title="Phases", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("Distance Matrix Feature UMAP Phases")
   
    plt.savefig(os.path.join("stage_clusters", f"{group_name}_umap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    kmedoids = KMedoids(n_clusters=18, random_state=0, method="pam")

    kmedoids.fit(mat_embedding)

    mat_labels = kmedoids.labels_ 

    plt.scatter(embedding[:, 0], embedding[:,1], c=mat_labels / 18 , cmap=cmap)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("Distance Matric Feature UMAP Cluster")
   
    plt.savefig(os.path.join("stage_clusters", f"{group_name}_clusters.png"), dpi=300, bbox_inches='tight')

    plt.close()
    print(group_name)
    phase = [PHASES[i] for i in group['phase'].to_numpy()]
    
    mat_ari = adjusted_rand_score(phase, mat_labels)
    mat_nmi = normalized_mutual_info_score(phase, mat_labels)
    #traj_ari = adjusted_rand_score(phase, traj_labels)
    #traj_nmi = normalized_mutual_info_score(phase, traj_labels)
    out_df = pd.DataFrame({"embryo_id":group_name, "nmi":mat_nmi, "ari":mat_ari}, index=[0])
    confusion_df = pd.crosstab(mat_labels, pd.Categorical(phase, categories=PHASES), dropna=False)
    print(confusion_df.columns)
    for col in confusion_df:  
        prominence = confusion_df.loc[confusion_df[col].idxmax()][col] /( 0.000001 + confusion_df[col].sum())
        out_df[col] = [prominence]
        

    return out_df #"traj_nmi":traj_nmi, "traj_ari":traj_ari

def main(model_name):
    annotations_dir = "embryo_dataset_annotations" 
    lat_df = pd.read_csv(os.path.abspath(f"latents/{model_name}.csv")).rename(columns={"cell_id":"embryo_id"})
    lat_np = np.load(os.path.abspath(f"latents/{model_name}.npy"))
    if(len(lat_df) != lat_np.shape[0]):
        raise ValueError("keys and values sizes do not match")
    lat_columns = [f"z_{i}" for i in range(lat_np.shape[1])]
    values_df = pd.DataFrame(lat_np, columns=lat_columns)
    df = pd.concat([lat_df, values_df], axis = 1)
    #df = df[df['embryo_id'].isin(df['embryo_id'].unique()[:10])]
    cluster_stats = df.groupby("embryo_id", group_keys = False).apply(lambda group:addAnnotations(group.name,group,annotations_dir)).reset_index()
    print(model_name)
    print("mat nmi: ", cluster_stats["nmi"].mean(), " +- ", cluster_stats["nmi"].std())
    print("mat ari: ", cluster_stats["ari"].mean(), " +- ", cluster_stats["ari"].std())
    print("best mat nmi:", cluster_stats.loc[cluster_stats["nmi"].idxmax()]["embryo_id"])
    print("best mat ari:", cluster_stats.loc[cluster_stats["ari"].idxmax()]["embryo_id"])
    for i in  PHASES:
        print("$\\num{", cluster_stats[i].mean() ,"} \\pm \\num{", cluster_stats[i].std()  , "}$&")
    print("\\\\")
    #print("traj nmi: ", cluster_stats["traj_nmi"].mean(), " +- ", cluster_stats["traj_nmi"].std())
    #print("traj ari: ", cluster_stats["traj_ari"].mean(), " +- ", cluster_stats["traj_ari"].std())
    #print("best traj nmi:", cluster_stats.loc[cluster_stats["traj_nmi"].idxmax()]["embryo_id"])
    #print("best traj ari:", cluster_stats.loc[cluster_stats["traj_ari"].idxmax()]["embryo_id"])    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")
 
 
    parser.add_argument("--name", help="Model name. Must have already exported latents")
  
    args = parser.parse_args()
 
    main(args.name)

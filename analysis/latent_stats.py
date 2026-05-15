import pandas as pd
import numpy as np
from geometric_features import get_acc, get_vel, calculate_curvatures, get_path_sigs,get_path_sig
from scipy.spatial import distance_matrix
import os
import matplotlib.pyplot as plt
import umap as UMAP
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA 
from sklearn.cluster import KMeans 
from visualize_cebra import plot_sequences
cebra_latent_cols =  ["cebra_0","cebra_1", "cebra_2"]
def add_geo_features(group, latent_cols):
    trajectory = group[latent_cols].to_numpy()
    cebra_traj = group[cebra_latent_cols].to_numpy()
    path_sigs, basis = get_path_sigs(cebra_traj, 3, return_feature_labels = True)
    
    curv_5 = calculate_curvatures(trajectory, offset=5, how="triangle")
    curv_10 = calculate_curvatures(trajectory, offset=10, how="triangle")
    curv_20 = calculate_curvatures(trajectory, offset=20, how="triangle")
    vel = get_vel(trajectory)
    acc = get_acc(trajectory)
    displacement = distance_matrix(trajectory, [trajectory[0]]).flatten() # distance mat's shape is (M,1)
    #-------------------------------------------------------
    cebra_curv_5 = calculate_curvatures(cebra_traj, offset=5, how="triangle")
    cebra_curv_10 = calculate_curvatures(cebra_traj, offset=10, how="triangle")
    cebra_curv_20 = calculate_curvatures(cebra_traj, offset=20, how="triangle")
    cebra_vel = get_vel(cebra_traj)
    cebra_acc = get_acc(cebra_traj)
    cebra_displacement = distance_matrix(cebra_traj, [cebra_traj[0]]).flatten() # distance mat's shape is (M,1)

    path_sigs_df = pd.DataFrame(path_sigs, columns=[f"ps_{basis[i]}" for i in range(path_sigs.shape[1])], index = group.index)
    cebra_features_df = pd.DataFrame({
        "cebra_curv_5":cebra_curv_5, 
        "cebra_curv_10":cebra_curv_10, 
        "cebra_curv_20":cebra_curv_20, 
        "cebra_vel":cebra_vel,
        "cebra_acc":cebra_acc,
        "cebra_displacement":cebra_displacement
                                }, index=group.index)

    features_df = pd.DataFrame({
        "curv_5":curv_5, 
        "curv_10":curv_10, 
        "curv_20":curv_20, 
        "vel":vel,
        "acc":acc,
        "displacement":displacement
                                }, index=group.index)
    return pd.concat([group, features_df, cebra_features_df, path_sigs_df], axis=1)

PHASES = ['pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM','tSB','tB', 'tEB', 'post_phase']

def print_next_to(strings): # strings must be of same # of lines
    line_rows = list(zip(*[s.split("\n") for s in strings]))
    all_lines = np.array(line_rows).flatten()
    padding_len = max([len(line) for line in all_lines])
    print("\n".join([" ".join([line.ljust(padding_len, " ") for line in line_row]) for line_row in line_rows]))

def main(model_name, grade):
    latent_cols = None
    df = None
    if(not os.path.exists("latent_stats.csv")):
        metadata_df = pd.read_csv(os.path.join("latents", f"{model_name}.csv"), keep_default_na=False)
        latents_np = np.load(os.path.join("latents", f"{model_name}.npy"))
        latent_cols = [f"z_{i}" for i in range(latents_np.shape[1])]
        latents_df = pd.DataFrame(latents_np, columns = latent_cols, index=metadata_df.index)
        cebra_np = np.load(os.path.join("cebra_latents", f"{model_name}.npy"))
        cebra_df = pd.DataFrame(cebra_np, columns=["cebra_0", "cebra_1", "cebra_2"], index=metadata_df.index)
        df = pd.concat([metadata_df, latents_df,cebra_df], axis=1)
        df = df.groupby("embryo_id").apply(lambda group:add_geo_features(group,latent_cols), include_groups=False).reset_index()
        df = df[~(df[grade] == "NA")]
        df.to_csv(os.path.join(os.getcwd(), "latent_stats.csv"))
    else:
        df = pd.read_csv(os.path.join(os.getcwd(), "latent_stats.csv"))
        latent_cols = [col for col in df.columns if col.startswith("z_")]

    # get curvature by grade
    #grade_curv_groups = df.groupby(grade)[["curv_5", "curv_10", "curv_20"]]
    #print(grade_curve_groups.mean(),"\n", grade_curv_groups.std())
    
    features = [col for col in df.columns if col.startswith("ps")]
    
    
    # turn each graded embryo into a single path signature feature, by picking along tSB to tEB (each path sig is from just those latents)
    def group_to_path_sig(group):
        ps_group = pd.DataFrame([get_path_sig(group[cebra_latent_cols],3)], columns=features)
        ps_group[grade] = group.iloc[0][grade]
        return ps_group
    blastocyst_cebra_df = df[(df['phase'].str.contains('tEB', regex=True))][cebra_latent_cols + [grade, "embryo_id"]]
    
    cebra_path_sigs_df = blastocyst_cebra_df.groupby("embryo_id").apply(group_to_path_sig, include_groups=False).reset_index()#
    

    colors = ["#FF0000", "#FFFF00", '#00FF00']
    grades = ["C", "B", "A"]
    
    kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(cebra_path_sigs_df[features].to_numpy())
    cebra_path_sigs_df["cluster"] = kmeans.labels_
    
    print(cebra_path_sigs_df)
    cebra_path_sigs_df.to_csv(os.path.join(os.getcwd(), "path_sigs_by_embryo.csv"))
    blastocyst_cebra_df = blastocyst_cebra_df.merge(cebra_path_sigs_df, how="left", left_on="embryo_id", right_on="embryo_id")
    
    print(cebra_path_sigs_df.groupby([grade, 'cluster']).size())
    
    plot_sequences([group[cebra_latent_cols].to_numpy() for _, group in blastocyst_cebra_df.groupby("embryo_id")], "path_sig_clusters", cmap="phase", c=[(group["cluster"].to_numpy() * 71) % 17 for _, group in blastocyst_cebra_df.groupby("embryo_id")])
    for cluster in range(8):
        cluster_df = blastocyst_cebra_df[blastocyst_cebra_df["cluster"] == cluster] 
        plot_sequences([group[cebra_latent_cols].to_numpy() for _, group in cluster_df.groupby("embryo_id")], f"path_sig_clusters_{cluster}", cmap="phase", c=[(group["cluster"].to_numpy() * 71) % 17 for _, group in cluster_df.groupby("embryo_id")])
    #umap = UMAP.UMAP(n_components=3) 
    #embedding = umap.fit_transform(cebra_latents_group_df[features].to_numpy())
    #pca = PCA(n_components=3)
    #embedding = pca.fit_transform(cebra_latents_group_df[features].to_numpy())
    #ica = FastICA(n_components=3)
    #embedding = ica.fit_transform(cebra_latents_group_df[features].to_numpy())
    #embedding = cebra_latents_group_df[["ps_1","ps_2","ps_[2,3]"]].to_numpy()
    #fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    #ax.scatter(embedding[:, 0], embedding[:,1], embedding[:,2], c=[colors[grades.index(g)] for g in cebra_latents_group_df[grade].to_list()])
    #fig.savefig(os.path.join(os.getcwd(), "path_sigs.jpg"))
    #plt.close(fig)
    
    
    # TODO: use ground truth time annotations to compare growth to literature






if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])

import pandas as pd
import numpy as np
from geometric_features import get_acc, get_vel, calculate_curvatures, get_path_sigs,get_path_sig
from scipy.spatial import distance_matrix
import os
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import umap as UMAP
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA 
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
    grade_phase_groups = df.groupby([grade, "phase"])
    # ----------------------------------------------------------------------------------
    # do some formatting for latex and run t-test
    
    features = [col for col in df.columns if col.startswith("ps")]
    grade_phase_feature_groups = grade_phase_groups[features]
    grade_phase_df = pd.concat([
            grade_phase_feature_groups.mean().reindex(PHASES, level="phase"), 
            grade_phase_feature_groups.std().reindex(PHASES, level="phase").rename(mapper=lambda n: n + "_std", axis = 1),
            pd.DataFrame(grade_phase_groups[["time_step"]].size().reindex(PHASES, level="phase"), columns=["count"])
        ],axis=1)    
    grade_phase_df = grade_phase_df.reindex(sorted(grade_phase_df.columns), axis=1)
    styled_df = grade_phase_df.style #.loc[pd.IndexSlice[:,"t9+":], :].style
    styled_df = styled_df.format_index(formatter= lambda s:s.replace("_",r"\_"),axis=1) # format column names
    styled_df = styled_df.format_index(formatter= lambda s:s.replace("_",r"\_"),axis=0) # format index names

    #print(styled_df.to_latex(hrules=True))
    print(grade_phase_df)
    grade_phase_df.to_csv(os.path.join(os.getcwd(), "path_sigs_by_phase.csv"))
    #print_next_to([g + " \n" + str(grade_phase_df.loc[g]) for g in ["A","B","C"]])
    t_test_series = [] 
    for column in features:
        t_s = []
        for phase in PHASES:
            a_df = grade_phase_feature_groups.get_group(("A", phase))
            a_np = a_df[column].to_numpy()
            bc_df = pd.concat([grade_phase_feature_groups.get_group(("B", phase)), grade_phase_feature_groups.get_group(("C", phase))], axis=0)
            bc_np = bc_df[column].to_numpy()
            t = ttest_ind(a_np, bc_np, equal_var=False)
            t_s.append(t.pvalue)
        t_test_series.append(pd.Series(t_s, index=PHASES))
    t_tests_df = pd.DataFrame({features[i]:t_test_series[i] for i in range(len(features))})
    t_tests_df.to_csv(os.path.join(os.getcwd(), "t_test.csv"))
    
    # turn each graded embryo into a single path signature feature, by picking along tSB to tEB (each path sig is from just those latents)
    def group_to_path_sig(group):
        ps_group = pd.DataFrame([get_path_sig(group[cebra_latent_cols],3)], columns=features)
        ps_group[grade] = group.iloc[0][grade]
        return ps_group
    cebra_latents_group_df = df[(df['phase'].str.contains('tSB|tB|tEB', regex=True))][cebra_latent_cols + [grade, "embryo_id"]].groupby("embryo_id").apply(group_to_path_sig, include_groups=False).reset_index()#

    print(cebra_latents_group_df)
    cebra_latents_group_df.to_csv(os.path.join(os.getcwd(), "path_sigs_by_embryo.csv"))
    colors = ["#FF0000", "#FFFF00", '#00FF00']
    grades = ["C", "B", "A"]
    umap = UMAP.UMAP(n_components=3) 
    embedding = umap.fit_transform(cebra_latents_group_df[features].to_numpy())
    #pca = PCA(n_components=3)
    #embedding = pca.fit_transform(cebra_latents_group_df[features].to_numpy())
    #ica = FastICA(n_components=3)
    #embedding = ica.fit_transform(cebra_latents_group_df[features].to_numpy())
    #embedding = cebra_latents_group_df[["ps_1","ps_2","ps_[2,3]"]].to_numpy()
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    ax.scatter(embedding[:, 0], embedding[:,1], embedding[:,2], c=[colors[grades.index(g)] for g in cebra_latents_group_df[grade].to_list()])
    fig.savefig(os.path.join(os.getcwd(), "path_sigs.jpg"))
    plt.close(fig)
    
    
    # TODO: use ground truth time annotations to compare growth to literature






if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])

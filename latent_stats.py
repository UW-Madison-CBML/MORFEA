import pandas as pd
import numpy as np
from geometric_features import get_acc, get_vel, calculate_curvatures, get_path_sigs
from scipy.spatial import distance_matrix
import os
from scipy.stats import ttest_ind
def add_geo_features(group, latent_cols):
    trajectory = group[latent_cols].to_numpy()
    cebra_traj = group[["cebra_0","cebra_1", "cebra_2"]].to_numpy()
    path_sigs = get_path_sigs(cebra_traj, 3)
    curv_5 = calculate_curvatures(trajectory, offset=5, how="triangle")
    curv_10 = calculate_curvatures(trajectory, offset=10, how="triangle")
    curv_20 = calculate_curvatures(trajectory, offset=20, how="triangle")
    vel = get_vel(trajectory)
    acc = get_acc(trajectory)
    displacement = distance_matrix(cebra_trajectory, [cebra_trajectory[0]]).flatten() # distance mat's shape is (M,1)
    cebra_curv_5 = calculate_curvatures(cebra_trajectory, offset=5, how="triangle")
    cebra_curv_10 = calculate_curvatures(cebra_trajectory, offset=10, how="triangle")
    cebra_curv_20 = calculate_curvatures(cebra_trajectory, offset=20, how="triangle")
    cebra_vel = get_vel(cebra_trajectory)
    cebra_acc = get_acc(cebra_trajectory)
    cebra_displacement = distance_matrix(cebra_trajectory, [cebra_trajectory[0]]).flatten() # distance mat's shape is (M,1)

    path_sigs_df = pd.DataFrame(path_sigs, columns=[f"path_sig_{i}" for i in range(path_sigs.shape[1])], index = group.index)
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

    # get curvature by grade
    #grade_curv_groups = df.groupby(grade)[["curv_5", "curv_10", "curv_20"]]
    #print(grade_curve_groups.mean(),"\n", grade_curv_groups.std())
    grade_phase_groups = df.groupby([grade, "phase"])
    # ----------------------------------------------------------------------------------
    # do some formatting for latex and run t-test
    
    column = "acc"
    features = ["vel", "acc"]
    grade_phase_feature_groups = grade_phase_groups[features]
    grade_phase_df = pd.concat([
            grade_phase_feature_groups.mean().reindex(PHASES, level="phase"), 
            grade_phase_feature_groups.std().reindex(PHASES, level="phase").rename(mapper=lambda n: n + "_std", axis = 1),
            pd.DataFrame(grade_phase_groups[["time_step"]].size().reindex(PHASES, level="phase"), columns=["count"])
        ],axis=1)    
    grade_phase_df = grade_phase_df.reindex(sorted(grade_phase_df.columns), axis=1)
    styled_df = grade_phase_df.loc[pd.IndexSlice[:,"t9+":], :].style
    styled_df = styled_df.format_index(formatter= lambda s:s.replace("_",r"\_"),axis=1) # format column names
    styled_df = styled_df.format_index(formatter= lambda s:s.replace("_",r"\_"),axis=0) # format index names

    #print(styled_df.to_latex(hrules=True))
    print_next_to([g + " \n" + str(grade_phase_df.loc[g]) for g in ["A","B","C"]])
    for phase in PHASES:
        print(phase)
        a_df = grade_phase_feature_groups.get_group(("A", phase))
        a_np = a_df[column].to_numpy()
        bc_df = pd.concat([grade_phase_feature_groups.get_group(("B", phase)), grade_phase_feature_groups.get_group(("C", phase))], axis=0)
        bc_np = bc_df[column].to_numpy()



        print("welch t-test: ", ttest_ind(a_np, bc_np, equal_var=False))
    # TODO: use ground truth time annotations to compare growth to literature





if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])

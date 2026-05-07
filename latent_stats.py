import pandas as pd
import numpy as np
from geometric_features import get_acc, get_vel, calculate_curvatures
from scipy.spatial import distance_matrix
import os
def add_geo_features(group, latent_cols):
    trajectory = group[latent_cols].to_numpy()
    curv_5 = calculate_curvatures(trajectory, offset=5, how="triangle")
    curv_10 = calculate_curvatures(trajectory, offset=10, how="triangle")
    curv_20 = calculate_curvatures(trajectory, offset=20, how="triangle")
    vel = get_vel(trajectory)
    acc = get_acc(trajectory)
    displacements = distance_matrix(trajectory, [trajectory[0]]).flatten() # distance mat's shape is (M,1)
    features_df = pd.DataFrame({
        "curv_5":curv_5, 
        "curv_10":curv_10, 
        "curv_20":curv_20, 
        "vel":vel,
        "acc":acc,
        "displacements":displacements
                                }, index=group.index)
    return pd.concat([group, features_df], axis=1)

PHASES = ['pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM','tSB','tB', 'tEB', 'tHB', 'post_phase']

def print_next_to(strings): # strings must be of same # of lines
    line_rows = list(zip(*[s.split("\n") for s in strings]))
    all_lines = np.array(line_rows).flatten()
    padding_len = max([len(line) for line in all_lines])
    print("\n".join([" ".join([line.ljust(padding_len, " ") for line in line_row]) for line_row in line_rows]))

def main(model_name, grade):

    metadata_df = pd.read_csv(os.path.join("latents", f"{model_name}.csv"), keep_default_na=False)
    latents_np = np.load(os.path.join("latents", f"{model_name}.npy"))
    latent_cols = [f"z_{i}" for i in range(latents_np.shape[1])]
    latents_df = pd.DataFrame(latents_np, columns = latent_cols, index=metadata_df.index)
    df = pd.concat([metadata_df, latents_df], axis=1)
    df = df.groupby("embryo_id").apply(lambda group:add_geo_features(group,latent_cols), include_groups=False).reset_index()
    df = df[~(df[grade] == "NA")]
    # get curvature by grade
    #grade_curv_groups = df.groupby(grade)[["curv_5", "curv_10", "curv_20"]]
    #print(grade_curve_groups.mean(),"\n", grade_curv_groups.std())
    grade_phase_groups = df.groupby([grade, "phase"])
    # ----------------------------------------------------------------------------------
    # get curvature during phase by grade
    
    grade_phase_curv_groups = grade_phase_groups[["curv_5", "curv_10", "curv_20"]]
    grade_phase_df = pd.concat([
            grade_phase_curv_groups.mean().reindex(PHASES, level="phase"), 
            grade_phase_curv_groups.std().reindex(PHASES, level="phase").rename(mapper=lambda n: n + "_std", axis = 1),
            pd.DataFrame(grade_phase_groups[["time_step"]].size().reindex(PHASES, level="phase"), columns=["count"])
        ],axis=1)    
    grade_phase_df = grade_phase_df.reindex(sorted(grade_phase_df.columns), axis=1)
    styled_df = grade_phase_df.loc[pd.IndexSlice[:,"t9+":], :].style
    styled_df = styled_df.format_index(formatter= lambda s:s.replace("_",r"\_"),axis=1) # format column names
    styled_df = styled_df.format_index(formatter= lambda s:s.replace("_",r"\_"),axis=0) # format index names

    print(styled_df.to_latex(hrules=True))
    print_next_to([g + " \n" + str(grade_phase_df.loc[g]) for g in ["A","B","C"]])
    # get displacement by phase by grade
    # get std dev of each coord by grade
    # TODO: use ground truth time annotations to compare growth to literature





if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])

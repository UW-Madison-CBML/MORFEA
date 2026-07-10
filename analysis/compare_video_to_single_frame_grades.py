import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
def get_tb(group):
    """ 
    group: group of latents by embryo_id with phase annotations
        returns:
        The single frame df from preferrably tB, if if not will try the first frame of tEB and then the last frame of tSB
    
    """
    tb_rows = group[group['phase'] == "tB"]
    if(len(tb_rows) > 0):
        out_df = tb_rows.iloc[int(0.5 * len(tb_rows)): int(0.5 * len(tb_rows)) + 1] # grab the single frame
        print("tB: ",len(out_df))
        return out_df
    teb_rows = group[group['phase'] == "tEB"]
    if(len(teb_rows) > 0):
        out_df = teb_rows.iloc[0:1]
        print("tEB: ",len(out_df))
        return out_df
    tsb_rows = group[group['phase'] == "tSB"]
    if(len(tsb_rows) > 0):
        out_df = tsb_rows.iloc[-1:]
        print("tSB: ", len(out_df))
        return out_df

    return pd.DataFrame(columns = group.columns)
def get_early(group):
    """ 
    group: group of latents by embryo_id with phase annotations
        returns:
        The single frame df from preferrably pre_phase, if if not will try the first frame of tPB2 and then the first frame of tPNa
    
    """
    pre_phase_rows = group[group['phase'] == "pre_phase"]
    if(len(pre_phase_rows) > 0):
        out_df = pre_phase_rows.iloc[0: 1] # grab the single frame
        print("pre_phase: ",len(out_df))
        return out_df
    tpb2_rows = group[group['phase'] == "tPB2"]
    if(len(tpb2_rows) > 0):
        out_df = tpb2_rows.iloc[0:1]
        print("tPB2: ",len(out_df))
        return out_df
    tpna_rows = group[group['phase'] == "tPNa"]
    if(len(tpna_rows) > 0):
        out_df = tpna_rows.iloc[0:1]
        print("tPNa: ", len(out_df))
        return out_df

    return pd.DataFrame(columns = group.columns)


GRADE_COLORS = {"A":"#00FF00","B":"#FFFF00", "C":"#FF0000"}
def main(model_name):
    video_latents = np.load(os.path.join("latents",f"{model_name}.npy")) 
    assert not np.isnan(video_latents).any(), "video dataset has nans"
    print(np.isnan(video_latents).any())
    video_metadata = pd.read_csv(os.path.join("latents",f"{model_name}.csv")) 
    lat_cols = [f"z_{i}" for i in range(video_latents.shape[1])]
    video_latents_df = pd.DataFrame(video_latents, columns = lat_cols, index=video_metadata.index)
    video_df = pd.concat([video_metadata, video_latents_df], axis=1)
    video_df = video_df[video_df["TE"].isin(["A","B","C"])]
    tb_frames = video_df.groupby("embryo_id").apply(get_early).reset_index(drop=True)
    video_latents = tb_frames[lat_cols].to_numpy()
    
    
  

    

    # just load this in
    single_frame_latents = np.load(os.path.join("kanakasabapathy_latents",f"{model_name}.npy")) 
    single_frame_metadata = pd.read_csv(os.path.join("kanakasabapathy_latents",f"{model_name}.csv")) 

    assert not np.isnan(single_frame_latents).any(), "single frame nans"
    assert not np.isnan(video_latents).any(), "single frame from video dataset nans"
    
    # ok now just plot them 
    video_colors = ["#FF0000"] * video_latents.shape[0]
    single_frame_colors = ["#00FF00"] * single_frame_latents.shape[0]
    latents = np.concatenate([video_latents, single_frame_latents], axis=0) 
    colors = video_colors + single_frame_colors
    scaler = StandardScaler()
    pca = PCA(n_components=3)
    embeddings= pca.fit_transform(scaler.fit_transform(latents))
 
    fig, ax = plt.subplots(figsize = (6,8), subplot_kw={"projection":"3d"})
    ax.scatter(embeddings[:, 0], embeddings[:,1], embeddings[:,2], c=colors)
    
    fig.savefig("comparison.png")
     
    plt.close(fig)
    
    video_grade_colors = [GRADE_COLORS[g] for g in tb_frames["TE"].to_list()]
    single_frame_grade_colors = [GRADE_COLORS[g] for g in single_frame_metadata["TE"].to_list()]
    grade_colors = video_grade_colors + single_frame_grade_colors
 
    fig, ax = plt.subplots(figsize = (6,8), subplot_kw={"projection":"3d"})
    ax.scatter(embeddings[:, 0], embeddings[:,1], embeddings[:,2], c=grade_colors)
    
    fig.savefig("grade_comparison.png")
     
    plt.close(fig)
    

    

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "")

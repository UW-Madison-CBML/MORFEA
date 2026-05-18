# grade_lstm_dataset.py - load signatures as well as their grades.
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from geometric_features import get_acc, get_vel, calculate_curvatures, get_path_sigs
ImageFile.LOAD_TRUNCATED_IMAGES = True

from scipy.spatial import distance_matrix
def add_annotations(group_name, group, features):
   
    lat_cols = [column for column in group.columns if column.startswith("z_")]
    trajectory = group[lat_cols].to_numpy()
    cebra_cols = ["cebra_0", "cebra_1", "cebra_2"]

    if (features['curvature']):
        
        curv12 = calculate_curvatures(trajectory, offset=12, retrospective=True, how='triangle')
        curv12 = np.nan_to_num(curv12, nan=0.0, posinf=0.0, neginf=0.0)
        curv12 *= (1 / (np.std(curv12) + 0.0001))
        curv20 = calculate_curvatures(trajectory, offset=20, retrospective=True, how='triangle')
        curv20 = np.nan_to_num(curv20, nan=0.0, posinf=0.0, neginf=0.0)
        curv20 *= (1 / (np.std(curv20) + 0.0001))
        curv4 = calculate_curvatures(trajectory, offset=4, retrospective=True, how='triangle')
        curv4 = np.nan_to_num(curv4, nan=0.0, posinf=0.0, neginf=0.0)
        curv4 *= (1 / (np.std(curv4) + 0.0001))
        group = pd.concat([group,pd.DataFrame({"z_curv12":curv12, "z_curv20":curv20, "curv4":curv4}, index = group.index)], axis=1)

    if (features['cebra_ps']):
        
        cebra_trajectory = group[cebra_cols]
        sigs = get_path_sigs(cebra_trajectory, 3)
        sigs_df = pd.DataFrame(sigs, columns = [f"z_cebra_sig_{feature}" for feature in range(sigs.shape[1])], index=group.index)
        group = pd.concat([group, sigs_df], axis=1)
    if (features['pca_ps']):
        
        pca_trajectory = group[[col for col in group.columns if col.startswith("pca")]]
        sigs = get_path_sigs(pca_trajectory, 3)
        sigs_df = pd.DataFrame(sigs, columns = [f"z_pca_sig_{feature}" for feature in range(sigs.shape[1])], index=group.index)
        group = pd.concat([group, sigs_df], axis=1)

    if (features['umap_ps']):
        
        umap_trajectory = group[[col for col in group.columns if col.startswith("umap")]]
        sigs = get_path_sigs(umap_trajectory, 3)
        sigs_df = pd.DataFrame(sigs, columns = [f"z_umap_sig_{feature}" for feature in range(sigs.shape[1])], index=group.index)
        group = pd.concat([group, sigs_df], axis=1)



    if(features['distance_mat']):
        mat = distance_matrix(np.array([trajectory[0]]), trajectory).flatten()
        
        group["z_dist"] = mat
    
    if(features['acceleration']):
        group['z_acc'] = get_acc(trajectory)
        
    if(features['velocity']):
        group['z_vel'] = get_vel(trajectory)
    if(features['cebra_latents']):
        cebra_df = pd.DataFrame(cebra_traj, columns=["z_cebra_0","z_cebra_1", "z_cebra_2"], index=group.index)
        group = pd.concat([group, cebra_df], axis=1)
    if (not features['latents']):
        group = group.drop(columns=lat_cols)
    
    return group


class GradeLSTMDataset(Dataset):
    """
    Dataset that loads complete embryo sequences.
    Each sample is one full embryo with all its frames.
    """
    def __init__(self, latents_df, grade, features, keep_na=False, return_whole_seqs=False):
        """
        Args: pd.read_csv(grades_csv, keep_default_na=False).
        """
        self.keep_na = keep_na
        self.return_whole_seqs = return_whole_seqs
        self.grade = grade
        self.lat_cols = [col for col in latents_df.columns if col.startswith("z_")] 
        self.df = latents_df
        if(not self.keep_na):
            self.df = self.df.dropna(subset=[self.grade])
        #self.df = self.df.sort_values(["embryo_id", "time_step"])
        
        self.df = self.df.groupby("embryo_id", group_keys = False).apply(lambda group:add_annotations(group.name,group, features)).reset_index()
        self.lat_cols = [col for col in self.df.columns if col.startswith("z_")] # recalculate lat cols after adding features
        self.groups = self.df.groupby("embryo_id") 
        self.grade_options = ["A", "B", "C", "NA"] if self.keep_na else ["A","B","C"]

    def __getitem__(self, idx):
        rows = None
        if self.return_whole_seqs:
            _, rows = list(self.groups)[idx]
        else:
            row = self.df.iloc[idx]
            embryo_id = row["embryo_id"]
            group = self.groups.get_group(embryo_id)
            iloc_idx = group.index.get_loc(row.name)
            rows = group.iloc[:max(8, iloc_idx)]

        lat_seq = rows[self.lat_cols].to_numpy(dtype=np.float32)
        grade_index = self.grade_options.index(rows.iloc[-1][self.grade])
        return torch.tensor(lat_seq.astype(np.float32)), torch.tensor(grade_index)

    def __len__(self):
        return len(self.df) if self.return_whole_seqs == False else len(self.groups)

# grade_lstm_dataset.py - load signatures as well as their grades.
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from geometric_features import get_acc, get_vel, calculate_curvatures
ImageFile.LOAD_TRUNCATED_IMAGES = True

from scipy.spatial import distance_matrix
def add_annotations(group_name, group, features):
   
    lat_cols = [column for column in group.columns if column.startswith("z_")]
    trajectory = group[lat_cols].to_numpy()

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


    if(features['distance_mat']):
        mat = distance_matrix(np.array([trajectory[0]]), trajectory).flatten()
        
        group["z_dist"] = mat
    
    if(features['acceleration']):
        group['z_acc'] = get_acc(trajectory)
        
    if(features['velocity']):
        group['z_vel'] = get_vel(trajectory)


    if (not features['latents']):
        group = group.drop(columns=lat_cols)
    
    return group


class GradeLSTMDataset(Dataset):
    """
    Dataset that loads complete embryo sequences.
    Each sample is one full embryo with all its frames.
    """
    def __init__(self, latents_df, grades_df, grade, features, keep_na=False, return_whole_seqs=False):
        """
        Args: pd.read_csv(grades_csv, keep_default_na=False).
        """
        self.keep_na = keep_na
        self.return_whole_seqs = return_whole_seqs
        self.grade = grade
        self.latents_df = latents_df.rename(columns={"cell_id":"embryo_id", "video_name":"embryo_id"})
        self.grades_df = grades_df.rename(columns={"cell_id":"embryo_id", "video_name":"embryo_id"})
        if(not("embryo_id" in self.latents_df.columns and "embryo_id" in self.grades_df.columns and "time_step" in self.latents_df.columns)):
            print(self.latents_df.head())
            print(self.grades_df.head())
            raise ValueError("no embryo_id column")
        self.lat_cols = [col for col in self.latents_df.columns if col.startswith("z_")] 
        self.df = self.latents_df.merge(self.grades_df, how="left", left_on="embryo_id", right_on="embryo_id")[["embryo_id", self.grade, "time_step"] + self.lat_cols] if self.keep_na else self.latents_df.merge(self.grades_df, how="left", left_on="embryo_id", right_on="embryo_id")[["embryo_id", self.grade, "time_step"] + self.lat_cols].dropna(subset=[self.grade])
        self.df = self.df.sort_values(["embryo_id", "time_step"])
        
        self.df = self.df.groupby("embryo_id", group_keys = False).apply(lambda group:add_annotations(group.name,group, features)).reset_index()
        self.lat_cols = [col for col in self.df.columns if col.startswith("z_")] # recalculate lat cols after adding features
        self.groups = self.df.groupby("embryo_id") 
        self.grade_options = ["A", "B", "C", "NA"] if self.keep_na else ["A","B","C"]

    def __getitem__(self, idx):
        rows = None
        if self.return_whole_seqs:
            _, rows = list(self.groups)[idx]
            print(f"embryo_id: {rows.iloc[-1]['embryo_id']}, grade: {rows.iloc[-1][self.grade]}")
        else:
            row = self.df.iloc[idx]
            embryo_id = row["embryo_id"]
            group = self.groups.get_group(embryo_id)
            rows = group.loc[:max(idx + 8, group.index[8])]

        lat_seq = rows[self.lat_cols].to_numpy(dtype=np.float32)
        grade_index = self.grade_options.index(rows.iloc[-1][self.grade])
        return torch.tensor(lat_seq.astype(np.float32)), torch.tensor(grade_index)

    def __len__(self):
        return len(self.df) if self.return_whole_seqs == False else len(self.groups)

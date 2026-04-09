import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis
import iisignature
import umap
from geometric_features import calculate_curvatures, get_path_sigs
from torch.nn.utils.rnn import pad_sequence 
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.spatial import distance_matrix
def get_annotations_col(embryo_id, group_len, annotations_dir):
    annotation_file = os.path.join(annotations_dir, f"{group_name}_phases.csv")
    df = pd.read_csv(annotation_file, names=['stage_id', 'stage_begin', 'stage_end'])

    new_column = []
    
    pca_cols = [column for column in group.columns if column.startswith("pca_")]
    new_column += ["pre_phase"] * (df.iloc[0]["stage_begin"] - 1)
    col_len_seq = []
    for index, row in df.iterrows():
        new_column += [row["stage_id"]] * (row["stage_end"] - row["stage_begin"]+1)
        col_len_seq.append(len(new_column))

    


    new_column += ["post_phase"] * (group_len - len(new_column))
    new_column = new_column[:group_len]
    return new_column
 
def addAnnotations(group_name, group, annotations_dir, curvature = True, velocity = True, latents = True, acceleration = True, path_signatures = True, distance_mat=True):
   
    lat_cols = [column for column in group.columns if column.startswith("z_")]
    new_column = get_annotations_col(group_name, len(group), annotations_dir)
    group["phase"] = new_column

    trajectory = group[lat_cols].to_numpy()
    pca_trajectory = group[pca_cols].to_numpy()

    if (curvature):
        
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


    if (path_signatures):
        sigs = get_path_sigs(pca_trajectory, 2)
        sigs_df = pd.DataFrame(sigs, columns = [f"z_sig_{feature}" for feature in range(sigs.shape[1])])
        sigs_df.index = group.index
        group = pd.concat([group, sigs_df], axis=1)
    if(distance_mat):
        mat = distance_matrix(np.array([trajectory[0]]), trajectory).flatten()
        
        group["z_dist"] = mat
    
    if(acceleration):
        print("")         


    if (not latents):
        group = group.drop(columns=lat_cols)
    
    print(f"Group {group_name} NaNs: {group.isna().sum().sum()}") 
    return group

class StageDataset(Dataset):
    """
    Dataset that loads complete embryo sequences.
    Each sample is one full embryo with all its frames.
    """
    def __init__(self, latents_df, annotations_dir, curvature=True, velocity=True, latents=True, acceleration=True, path_signatures=True, return_embryo_id=False, distance_mat=True): # preparing latents_df outside of the class i.e. from .csv .npy in latents/
        """
        Args: pd.read_csv(grades_csv, keep_default_na=False).
        """
        self.latents_df = latents_df
        
        print("nan cols just lats", self.latents_df.columns[self.latents_df.isna().any()])
        values = self.latents_df[[i for i in self.latents_df.columns if i.startswith("z_")]].to_numpy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(values)

        pca = umap.UMAP(n_neighbors=10, random_state=42, n_components=8) #PCA(n_components=20) 
        pca_results = pca.fit_transform(scaled_data)
        
        self.pca_latents_df = pd.DataFrame(pca_results, columns=[f"pca_{i}" for i in range(pca_results.shape[1])], index=self.latents_df.index)
        self.latents_df = pd.concat([self.latents_df, self.pca_latents_df], axis=1)
        self.annotations_dir = annotations_dir
        self.return_embryo_id = return_embryo_id
        sizes = self.latents_df.groupby("embryo_id")["time_step"].size()
        self.max_points = sizes.max()
        
        self.df = self.latents_df.groupby("embryo_id", group_keys = False).apply(lambda group:addAnnotations(group.name,group,self.annotations_dir, curvature = curvature, velocity = velocity, latents = latents, acceleration = acceleration, path_signatures = path_signatures, distance_mat=distance_mat)).reset_index()
        self.groups = self.df.groupby("embryo_id")
        self.seqlength = 64
        print("nan cols", self.latents_df.columns[self.latents_df.isna().any()])
        

        self.phases = ['pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM','tSB','tB', 'tEB', 'tHB', 'post_phase']
        self.lat_cols = [column for column in self.df.columns if column.startswith("z_")]

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        group = self.groups.get_group(row["embryo_id"])

        
        #seqindex = int(((row["time_step"] - 1) / len(group)) * (len(group) - self.seqlength - 1))

        seq_df = group.loc[:max(group.index[0] + 5, idx+1)]

        if (self.return_embryo_id):

            return torch.tensor(seq_df[self.lat_cols].to_numpy()), torch.tensor([self.phases.index(r["phase"]) for _,r in seq_df.iterrows()], dtype = torch.long), row["embryo_id"]

        return torch.tensor(seq_df[self.lat_cols].to_numpy()), torch.tensor([self.phases.index(r["phase"]) for _,r in seq_df.iterrows()], dtype = torch.long)
        

    def __len__(self):
        return len(self.df)

    def pad_collate(self, batch):
        if(self.return_embryo_id):
            (data, labels,embryo_id) = zip(*batch)
        else:  
            (data, labels) = zip(*batch)
        
        data_padded = pad_sequence(data, batch_first=True, padding_value=0)
        
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
        mask = torch.zeros(labels_padded.shape, dtype=torch.bool)
        for i, seq in enumerate(labels):
            mask[i, :len(seq)] = True
        
        if(self.return_embryo_id):
            return data_padded, labels_padded, mask, embryo_id
        return data_padded, labels_padded, mask

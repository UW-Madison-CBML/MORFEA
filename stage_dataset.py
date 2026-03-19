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
from geometric_features import calculate_curvatures, get_path_sigs
 
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.spatial import distance_matrix
def addAnnotations(group_name, group, annotations_dir, curvature = True, velocity = True, latents = True, acceleration = True, path_signatures = True, distance_mat=True):
    annotation_file = os.path.join(annotations_dir, f"{group_name}_phases.csv")
    df = pd.read_csv(annotation_file, names=['stage_id', 'stage_begin', 'stage_end'])

    new_column = []
    lat_cols = [column for column in group.columns if column.startswith("z_")]
    
    pca_cols = [column for column in group.columns if column.startswith("pca_")]
    new_column += ["pre_phase"] * (df.iloc[0]["stage_begin"] - 1)
    col_len_seq = []
    for index, row in df.iterrows():
        new_column += [row["stage_id"]] * (row["stage_end"] - row["stage_begin"]+1)
        col_len_seq.append(len(new_column))

    


    new_column += ["post_phase"] * (len(group) - len(new_column))
    new_column = new_column[:len(group)]
    
    group["phase"] = new_column

    trajectory = group[lat_cols].to_numpy()
    pca_trajectory = group[pca_cols].to_numpy()

    if (curvature):
        group["z_curvature_12"] = calculate_curvatures(group, offset=12, retrospective=True)
        group["z_curvature_12"] = group["z_curvature_12"] * (1 / (group["z_curvature_12"].std() + 0.0001))
        group["z_curvature_20"] = calculate_curvatures(group, offset=20, retrospective=True)
        group["z_curvature_20"] = group["z_curvature_20"] * (1 / (group["z_curvature_20"].std() + 0.0001))
        group["z_curvature_4"] = calculate_curvatures(group, offset=4, retrospective=True)
        group["z_curvature_4"] = group["z_curvature_4"] * (1 / (group["z_curvature_4"].std() + 0.0001))


    if (path_signatures):
        sigs = get_path_sigs(pca_traj)
        for feature in range(sigs.shape[1]):
            group[f"z_sig_{feature}"] = sigs[:, feature]
        
    if(distance_mat):
        mat = distance_matrix(np.array([trajectory[0]]), trajectory).flatten()
        
        group["z_dist"] = mat
    
    if(acceleration):
        print("")         


    if (not latents):
        group = group.drop(columns=lat_cols)
    
    
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
        values = self.latents_df[[i for i in self.latents_df.columns if i.startswith("z_")]].to_numpy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(values)

        pca = PCA(n_components=20) 
        pca_results = pca.fit_transform(scaled_data)
        
        self.pca_latents_df = pd.DataFrame(pca_results, columns=[f"pca_{i}" for i in range(pca_results.shape[1])])
        self.latents_df = pd.concat([self.latents_df, self.pca_latents_df], axis=1)
        self.annotations_dir = annotations_dir
        self.return_embryo_id = return_embryo_id
        sizes = self.latents_df.groupby("embryo_id")["time_step"].size()
        self.max_points = sizes.max()
        
        self.df = self.latents_df.groupby("embryo_id", group_keys = False).apply(lambda group:addAnnotations(group.name,group,self.annotations_dir, curvature = curvature, velocity = velocity, latents = latents, acceleration = acceleration, path_signatures = None, distance_mat=distance_mat)).reset_index()
        self.groups = self.df.groupby("embryo_id")
        self.seqlength = 64
        

        self.phases = ['pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM','tSB','tB', 'tEB', 'tHB', 'post_phase']
        self.lat_cols = [column for column in self.df.columns if column.startswith("z_")]

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        group = self.groups.get_group(row["embryo_id"])

        
        seqindex = int(((row["time_step"] - 1) / len(group)) * (len(group) - self.seqlength - 1))

        seq_df = group.iloc[seqindex : seqindex + self.seqlength]

        if (self.return_embryo_id):

            return seq_df[self.lat_cols].to_numpy(), torch.tensor([self.phases.index(r["phase"]) for _,r in seq_df.iterrows()], dtype = torch.long), row["embryo_id"]

        return seq_df[self.lat_cols].to_numpy(), torch.tensor([self.phases.index(r["phase"]) for _,r in seq_df.iterrows()], dtype = torch.long)
        

        # [EMBRYO1, 0, .... (latent vector),
        # EMBRYO1, 1, .... (latent vector)]
        # [EMBRYO2, 0, .... (latent vector),
        # EMBRYO2, 1, .... (latent vector)]
        # grab index with self.df.loc[idx]
        # that row['stage'] will be in some set of ["stage0","stage1","stage2"...],
        # CEL does not expect [0,0,1...] but rather 1 "stage1", 3 "stage3"
        # return velocity, curvature, stage (stage is an long 64 integer)


    def __len__(self):
        return len(self.df)

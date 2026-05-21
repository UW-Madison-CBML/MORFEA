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
from geometric_features import calculate_curvatures, get_path_sigs, get_acc, get_vel
from torch.nn.utils.rnn import pad_sequence 
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.spatial import distance_matrix
def get_annotations_col(group_name, group_len, annotations_dir):
    annotation_file = os.path.join(annotations_dir, f"{group_name}_phases.csv")
    df = pd.read_csv(annotation_file, names=['stage_id', 'stage_begin', 'stage_end'])

    new_column = []
    
    new_column += ["pre_phase"] * (df.iloc[0]["stage_begin"] - 1)
    col_len_seq = []
    for index, row in df.iterrows():
        new_column += [row["stage_id"]] * (row["stage_end"] - row["stage_begin"]+1)
        col_len_seq.append(len(new_column))


    new_column += ["post_phase"] * (group_len - len(new_column))
    new_column = new_column[:group_len]
    return new_column
 
def add_annotations(group_name, group, annotations_dir, features):
    cebra_cols = ["cebra_0", "cebra_1", "cebra_2"] 
    lat_cols = cebra_cols if features["cebra"] else [column for column in group.columns if column.startswith("z_")] 
    new_column = get_annotations_col(group_name, len(group), annotations_dir)
    group["phase"] = new_column

    trajectory = group[lat_cols].to_numpy()

    if ('curvature' in features.keys() and features['curvature']):
        
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


    if ('cebra_ps' in features.keys() and eatures['cebra_ps']):
        
        cebra_trajectory = group[cebra_cols]
        sigs = get_path_sigs(cebra_trajectory, 3)
        sigs_df = pd.DataFrame(sigs, columns = [f"z_cebra_sig_{feature}" for feature in range(sigs.shape[1])], index=group.index)
        group = pd.concat([group, sigs_df], axis=1)
    if ('pca_ps' in features.keys() and eatures['pca_ps']):
        
        pca_trajectory = group[[col for col in group.columns if col.startswith("pca")]]
        sigs = get_path_sigs(pca_trajectory, 3)
        sigs_df = pd.DataFrame(sigs, columns = [f"z_pca_sig_{feature}" for feature in range(sigs.shape[1])], index=group.index)
        group = pd.concat([group, sigs_df], axis=1)

    if ('umap_ps' in features.keys() and features['umap_ps']):
        
        umap_trajectory = group[[col for col in group.columns if col.startswith("umap")]]
        sigs = get_path_sigs(umap_trajectory, 3)
        sigs_df = pd.DataFrame(sigs, columns = [f"z_umap_sig_{feature}" for feature in range(sigs.shape[1])], index=group.index)
        group = pd.concat([group, sigs_df], axis=1)


    if('distance_mat' in features.keys() and features['distance_mat']):
        mat = distance_matrix(np.array([trajectory[0]]), trajectory).flatten()
        
        group["z_dist"] = mat
    
    if('acceleration' in features.keys() and features['acceleration']):
        group['z_acc'] = get_acc(trajectory)
        
    if('velocity' in features.keys() and features['velocity']):
        group['z_vel'] = get_vel(trajectory)


    if ('latents' in features.keys() and not features['latents']):
        group = group.drop(columns=lat_cols)
    
    return group

class StageDataset(Dataset):
    def __init__(self, latents_df, annotations_dir, features, return_embryo_id=False, return_whole_seqs=False): # preparing latents_df outside of the class i.e. from .csv .npy in latents/
        self.latents_df = latents_df
        self.return_whole_seqs = return_whole_seqs 
        
        
         
        self.annotations_dir = annotations_dir
        self.return_embryo_id = return_embryo_id
        sizes = self.latents_df.groupby("embryo_id")["time_step"].size()
        self.max_points = sizes.max()
        
        self.df = self.latents_df.groupby("embryo_id", group_keys = False).apply(lambda group:add_annotations(group.name,group,self.annotations_dir, features)).reset_index()
        self.groups = self.df.groupby("embryo_id")
        self.seqlength = 64
        

        self.phases = ['pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM','tSB','tB', 'tEB', 'tHB', 'post_phase']
        self.lat_cols = [column for column in self.df.columns if column.startswith("z_")] # need to recalculate lat cols as the actual features in the training sequences

    def __getitem__(self, idx):
        seq_df = None
        
        row = None
        if(self.return_whole_seqs):
            _, seq_df = list(self.groups)[idx]
            row = seq_df.iloc[0]
            
        else:
            row = self.df.iloc[idx]
            group = self.groups.get_group(row["embryo_id"])

            group_idx = group.index.get_loc(row.name)
    
            seq_df = group.iloc[:max(16, group_idx + 1)]

        if (self.return_embryo_id):
            return torch.tensor(seq_df[self.lat_cols].to_numpy()), torch.tensor([self.phases.index(r) for r in seq_df['phase'].to_list()], dtype = torch.long), row["embryo_id"]
        else:
            return torch.tensor(seq_df[self.lat_cols].to_numpy()), torch.tensor([self.phases.index(r) for r in seq_df['phase'].to_list()], dtype = torch.long)
        

    def __len__(self):
        return len(self.groups) if self.return_whole_seqs else len(self.df)

    def pad_collate(self, batch):
            
        if(self.return_embryo_id):
            (data, labels, embryo_ids) = zip(*batch)
        else:  
            (data, labels) = zip(*batch)
        print("DEBUG: Raw label lengths in batch:", [len(l) for l in labels]) 

        data_padded = pad_sequence(data, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)

        mask = torch.zeros(labels_padded.shape, dtype=torch.bool)
        for i, seq in enumerate(labels):
            mask[i, :seq.shape[0]] = True 
        
        print("masks :", mask.sum(dim=1))
        if(self.return_embryo_id):
            return data_padded, labels_padded, labels, mask, embryo_ids
        else:
            return data_padded, labels_padded, labels, mask

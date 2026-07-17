import torch
from torch.utils.data import Dataset
import iisignature
from geometric_features import get_path_sig 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm 
tqdm.pandas()
class PathSigGradeDataset(Dataset):
    PHASES = ['pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM','tSB','tB', 'tEB', 'tHB', 'post_phase']
    GRADES = ["A", "B", "C"]
    def __init__(self, df, time_offsets, pca_dim=8, depth=3, grade="TE"):
        """
        df: the latents df with embryo_id and stage annotations, sorted by embryo_id and time_step
        time_offsets: the offsets for the time correlation
        pca_dim: dimension of the pca cols, df should already have pca applied
        depth: the path sig depth
        """
        self.pca_dim = pca_dim
        self.pca_cols = [f"pca_{i}" for i in range(pca_dim)]
        self.time_offsets = time_offsets
        self.depth = depth
        self.num_features = len(iisignature.basis(iisignature.prepare(pca_dim+1, depth)))
        self.ps_cols = [f"path_sig_{i}" for i in range(self.num_features)]
        self.grade = grade

        def path_sig_agg(group):
            pca_traj = group[self.pca_cols].to_numpy()
            path_sig = get_path_sig(pca_traj, depth, time_offsets=self.time_offsets)
            out_df = pd.DataFrame(path_sig[None, :], columns = self.ps_cols)
            out_df["phase"] = group.name[1] # the second key is 'phase'
            out_df["grade"] = group.iloc[0][grade]
            return out_df 

        self.df = df.groupby(["embryo_id", "phase"]).progress_apply(path_sig_agg).reset_index(drop=True)
        self.df[self.ps_cols] = self.df[self.ps_cols].astype(np.float32)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return torch.from_numpy(self.df[self.ps_cols].iloc[idx].to_numpy()), torch.tensor(self.__class__.PHASES.index(row["phase"]), dtype=torch.long), torch.tensor(self.__class__.GRADES.index(row["grade"]), dtype=torch.long)

    def __len__(self):
        return len(self.df)

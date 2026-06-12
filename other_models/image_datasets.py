import torch
from torch.data.utils import Dataset
import pandas as pd
from dataset_ivf_embryo import read_gray, normalize_video

class ImageGradeDataset(Dataset):
    def __init__(self, index_df, grade_df):
        # index_df cols = embryo_id, embryo_paths
        # grade_df cols = embryo_id, TE, ICM, keep_default_na=False
        index_df = index_df.merge(grade_df, how="left", left_on="embryo_id", right_on="embryo_id")
        dfs = [] 
        for idx, row in index_df.iterrows():
            # embryo_id,num_frames,embryo_paths
            df = pd.DataFrame({"path":row["embryo_paths"].split("|"), "embryo_id": row["embryo_id"], "TE":row["TE"], "ICM":row["ICM"]})
            dfs.append(df)
        self.df = pd.concat(dfs, axis=0, ignore_index=True)
        
        
    def __getitem__(self, idx):
        return normalize_video([read_gray(self.df.iloc[idx]["path"], 128)], "minmax01")[0]
        
    def __len__(self):
        return len(self.df)



        

# signature_dataset.py - load signatures as well as their grades.
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SignatureDataset(Dataset):
    """
    Dataset that loads complete embryo sequences.
    Each sample is one full embryo with all its frames.
    """
    def __init__(self, signature_csv, grades_csv):
        """
        Args:
            index_csv: Path to index CSV with columns: embryo_id, embryo_paths
            resize: Target image size (square)
            norm: Normalization method ("zscore" or "minmax01")
            max_frames: Maximum number of frames to load per embryo (None = load all)
        """
        sig_df = pd.read_csv(signature_csv).rename(columns={"cell_id":"embryo_id", "video_name":"embryo_id"})
        grades_df = pd.read_csv(grades_csv, keep_default_na=False).rename(columns={"cell_id":"embryo_id", "video_name":"embryo_id"})
        if(not("embryo_id" in sig_df.columns and "embryo_id" in grades_df.columns)):
            raise ValueError("no embryo_id column")
            print(sig_df.head())
            print(grades_df.head())
        self.df = sig_df.merge(grades_df, how="left", left_on="embryo_id", right_on="embryo_id")

    def __getitem__(self, idx):
        grade_options = ["A","B","C","NA"]
        signature = self.df.iloc[idx][[i for i in self.df.columns.tolist() if i[:2] == "s_"]].to_numpy() 
        te,icm = self.df.iloc[idx][["TE","ICM"]].values
        te_grade = np.array([1 if j == te else 0 for j in grades_options]) 
        icm_grade = np.array([1 if j == icm else 0 for j in grades_options])


        return signature

    def __len__(self):
        return len(self.df)

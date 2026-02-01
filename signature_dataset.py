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
    def __init__(self, sig_df, grades_df, grade, keep_na=False):
        """
        Args: pd.read_csv(grades_csv, keep_default_na=False).
        """
        self.keep_na = keep_na
        self.grade = grade
        self.sig_df = sig_df.rename(columns={"cell_id":"embryo_id", "video_name":"embryo_id"})
        self.grades_df = grades_df.rename(columns={"cell_id":"embryo_id", "video_name":"embryo_id"})
        if(not("embryo_id" in self.sig_df.columns and "embryo_id" in self.grades_df.columns)):
            print(self.sig_df.head())
            print(self.grades_df.head())
            raise ValueError("no embryo_id column")
        self.sig_cols = [col for col in self.sig_df.columns if col.startswith("s_")]
        self.df =  self.sig_df.merge(self.grades_df, how="left", left_on="embryo_id", right_on="embryo_id")[["embryo_id", self.grade] + self.sig_cols] if keep_na else self.sig_df.merge(self.grades_df, how="left", left_on="embryo_id", right_on="embryo_id")[["embryo_id", self.grade] + self.sig_cols].dropna(subset=[self.grade])
        sig_data = self.df[self.sig_cols].values
        self.mean = sig_data.mean(axis=0)
        self.std = (sig_data.std(axis=0) + 1e-8).astype(np.float32)  # Avoid division by zero

    def __getitem__(self, idx):
        grade_options = ["A", "B", "C", "NA"] if self.keep_na else ["A","B","C"]
        signature = self.df.iloc[idx][[i for i in self.df.columns.tolist() if i[:2] == "s_"]].to_numpy(dtype=np.float32)
        signature = (signature - self.mean) / self.std
        g = self.df.iloc[idx][self.grade]
        g_grade = grade_options.index(g)
        return signature.astype(np.float32), g_grade

    def __len__(self):
        return len(self.df)

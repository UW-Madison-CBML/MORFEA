# grade_lstm_dataset.py - load signatures as well as their grades.
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class GradeLSTMDataset(Dataset):
    """
    Dataset that loads complete embryo sequences.
    Each sample is one full embryo with all its frames.
    """
    def __init__(self, latents_df, grades_df, grade, keep_na=False):
        """
        Args: pd.read_csv(grades_csv, keep_default_na=False).
        """
        self.keep_na = keep_na
        self.grade = grade
        self.latents_df = latents_df.rename(columns={"cell_id":"embryo_id", "video_name":"embryo_id"})
        self.grades_df = grades_df.rename(columns={"cell_id":"embryo_id", "video_name":"embryo_id"})
        if(not("embryo_id" in self.sig_df.columns and "embryo_id" in self.grades_df.columns)):
            print(self.sig_df.head())
            print(self.grades_df.head())
            raise ValueError("no embryo_id column")
        self.lat_cols = [col for col in self.sig_df.columns if col.startswith("z_")]
        self.df =  self.latents_df.merge(self.grades_df, how="left", left_on="embryo_id", right_on="embryo_id")[["embryo_id", self.grade] + self.lat_cols] if self.keep_na else self.latents_df.merge(self.grades_df, how="left", left_on="embryo_id", right_on="embryo_id")[["embryo_id", self.grade] + self.lat_cols].dropna(subset=[self.grade])
        self.df = self.df.sort_values(["embryo_id", "time_step"])
        sig_data = self.df[self.lat_cols].values
        self.mean = sig_data.mean(axis=0)
        self.std = (sig_data.std(axis=0) + 1e-8).astype(np.float32)  # Avoid division by zero

    def __getitem__(self, idx):
        grade_options = ["A", "B", "C", "NA"] if self.keep_na else ["A","B","C"]
        embryo_id = self.df.iloc[idx]["embryo_id"]
        rows = self.df.iloc[:idx][self.df["embryo_id"] == embryo_id]
        lat_seq = rows[[i for i in self.df.columns.tolist() if i[:2] == "z_"]].to_numpy(dtype=np.float32)
        #lat_seq = (signature - self.mean)
        grade_index = grade_options.index(rows[-1][self.grade])
        return lat_seq.astype(np.float32), grade_index

    def __len__(self):
        return len(self.df)

import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def addAnnotations(group_name, group, annotations_dir):
    annotation_file = os.path.join(annotations_dir, f"{group_name}_phases.csv")
    df = pd.read_csv(annotation_file, names=['stage_id', 'stage_begin', 'stage_end'])
    new_column = []

    new_column += ["pre_phase"] * (df.iloc[0]["stage_begin"] - 1)
    col_len_seq = []
    for index, row in df.iterrows():
        new_column += [row["stage_id"]] * (row["stage_end"] - row["stage_begin"]+1)
        col_len_seq.append(len(new_column))

    


    new_column += ["post_phase"] * (len(group) - len(new_column))
    new_column = new_column[:len(group)]
    
    group["phase"] = new_column

    
    return group

class StageDataset(Dataset):
    """
    Dataset that loads complete embryo sequences.
    Each sample is one full embryo with all its frames.
    """
    def __init__(self, latents_df, annotations_dir): # preparing latents_df outside of the class i.e. from .csv .npy in latents/
        """
        Args: pd.read_csv(grades_csv, keep_default_na=False).
        """
        self.latents_df = latents_df
        self.annotations_dir = annotations_dir


        self.df = self.latents_df.groupby("embryo_id", group_keys = False).apply(lambda group:addAnnotations(group.name,group,self.annotations_dir)).reset_index()
        self.groups = self.df.groupby("embryo_id")
        self.seqlength = 64


        self.phases = ['t2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tB', 'tEB', 'tHB', 'tM', 'tPB2', 'tPNa', 'tPNf', 'tSB', 'pre_phase', 'post_phase']
        self.lat_cols = [column for column in self.df.columns if column.startswith("z_")]
        # 1. dim reduce latents_df
        # dir/EMBRYO_ID_annotions.csv
        #   stage_id, stage_begin, stage_end
        #   stage1, 10, 50
        #   stage2, 51, 70
        # 2. pick features (velocity, curvature, etc.) and labels (stage)
        # 3. build self.df

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        group = self.groups.get_group(row["embryo_id"])

        seqindex = int(((row["time_step"] - 1) / len(group)) * (len(group) - self.seqlength - 1))

        seq_df = group.iloc[seqindex : seqindex + self.seqlength]
        return seq_df[self.lat_cols].to_numpy(), torch.tensor([self.phases.index(row["phase"]) for _,row in seq_df.iterrows()], dtype = torch.long)


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


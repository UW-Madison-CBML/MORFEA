# dataset_ivf_embryo.py - Load complete embryo sequences
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_gray(path, resize):
    img = Image.open(path).convert('L')

    if img is None:
        raise FileNotFoundError(path)

    if resize is not None:
        img = img.resize((resize, resize), Image.BILINEAR)

    return np.array(img, dtype="float32")

def normalize_video(vol, norm):
    if norm == "zscore":
        m, s = vol.mean(), vol.std() + 1e-6
        vol = (vol - m) / s
    elif norm == "minmax01":
        lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
        vol = (vol - lo) / (hi - lo + 1e-6)
        vol = np.clip(vol, 0, 1)
    return vol

class IVFEmbryoDataset(Dataset):
    def __init__(self, df, resize=128, norm="minmax01", max_frames=None):
        self.df = df
        self.resize = resize
        self.norm = norm
        self.max_frames = None

    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if pd.isna(row["embryo_paths"]):
            print(f"Row {idx} has missing embryo_paths: ", row.to_string(index=False))
            raise ValueError(f"Row {idx} has missing embryo_paths")

        embryo_paths = row["embryo_paths"].split("|")

        if self.max_frames is not None and len(embryo_paths) > self.max_frames:
            embryo_paths = embryo_paths[:self.max_frames]

        embryo_frames = [read_gray(p, self.resize) for p in embryo_paths]
        embryo_vol = np.stack(embryo_frames, axis=0)  # (T, H, W)

        embryo_vol = normalize_video(embryo_vol, self.norm)

        embryo_vol = embryo_vol[:, None, :, :]

        return torch.from_numpy(embryo_vol)

    def __len__(self):
        return len(self.df)

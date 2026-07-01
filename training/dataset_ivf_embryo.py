# dataset_ivf_embryo.py - Load complete embryo sequences
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision.transforms import v2
from torchvision import tv_tensors
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_gray(path, resize, crop):
    img = Image.open(path).convert("L")

    if img is None:
        raise FileNotFoundError(path)
    if(crop > 0):
        img = img.crop((crop,crop, 500-crop, 500-crop))
    if resize is not None and img.size != (resize, resize):
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
    def __init__(self, df, crop=45, resize=128, norm="minmax01", return_embryo_id=False):
        self.crop = crop
        self.df = df
        self.resize = resize
        self.norm = norm
        self.augment = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomHorizontalFlip(p=0.5),
        ])
        self.return_embryo_id = return_embryo_id

    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if pd.isna(row["embryo_paths"]):
            print(f"Row {idx} has missing embryo_paths: ", row.to_string(index=False))
            raise ValueError(f"Row {idx} has missing embryo_paths")

        embryo_paths = row["embryo_paths"].split("|")


        embryo_frames = [read_gray(p, self.resize, self.crop) for p in embryo_paths]
        embryo_vol = np.stack(embryo_frames, axis=0)  # (T, H, W)

        embryo_vol = normalize_video(embryo_vol, self.norm)

        embryo_vol = embryo_vol[:, None, :, :]
        if self.return_embryo_id:
            return torch.from_numpy(embryo_vol), row["embryo_id"]
            
        else:
            return torch.from_numpy(embryo_vol)

    def __len__(self):
        return len(self.df)

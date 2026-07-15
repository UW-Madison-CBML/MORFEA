# dataset_ivf_embryo.py - Load complete embryo sequences
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.tv_tensors import Video
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
    def __init__(self, df, crop=45, resize=128, norm="minmax01", return_embryo_id=False, return_augment=False):
        self.crop = crop
        self.df = df
        self.resize = resize
        self.norm = norm
        self.augment = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5), 
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomCrop(size=500-(2*self.crop)),
            v2.Resize(size=self.resize)
        ])

        self.return_embryo_id = return_embryo_id
        self.return_augment = return_augment

    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if pd.isna(row["embryo_paths"]):
            print(f"Row {idx} has missing embryo_paths: ", row.to_string(index=False))
            raise ValueError(f"Row {idx} has missing embryo_paths")

        embryo_paths = row["embryo_paths"].split("|")


        embryo_frames1 = [read_gray(p, self.resize, self.crop) for p in embryo_paths]
        embryo_frames2 = [read_gray(p, None, 0) for p in embryo_paths]
        embryo_vol1 = np.stack(embryo_frames1, axis=0)  
        embryo_vol2 = np.stack(embryo_frames2, axis=0)  
        embryo_vol1 = normalize_video(embryo_vol1, self.norm)
        embryo_vol2 = normalize_video(embryo_vol2, self.norm)
        embryo_vol1 = embryo_vol1[:,None, :, :] 
        embryo_vol2 = embryo_vol2[:,None, :, :] 
        torch_vol1 = torch.from_numpy(embryo_vol1) 
        torch_vol2 = torch.from_numpy(embryo_vol2) 
        if self.return_embryo_id and self.return_augment:
            return torch_vol1, self.augment(Video(torch_vol2)), row["embryo_id"]
        elif self.return_augment:
            return torch_vol1, self.augment(Video(torch_vol2))
        elif self.return_embryo_id:
            return torch_vol1, row["embryo_id"]
        else:
            return torch_vol1

    def __len__(self):
        return len(self.df)

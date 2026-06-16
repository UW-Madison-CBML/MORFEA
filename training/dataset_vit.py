# dataset for old ConvAE architecture
# dataset_ivf.py
import numpy as np, pandas as pd, torch 
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import os
from torchvision.transforms import v2
from torchvision.tv_tensors import Video
ImageFile.LOAD_TRUNCATED_IMAGES = True
class VITDataset(Dataset):
    def __init__(self, df, resize=224, norm="minmax01"):
        #self.df = pd.read_csv(index_csv)
        self.df = df
        self.resize = resize
        self.norm = norm
        self.transforms = v2.Compose([
            v2.CenterCrop(10),
            v2.Resize((resize,resize)),
            v2.Normalize(mean=[0.485], std=[0.229]),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def _read_gray(self, path):
        img = Image.open(path)
        if img is None:
            raise FileNotFoundError(path)
        img = img.resize((224,224), Image.BILINEAR)
        return np.array(img, dtype="float32")

    def _normalize_video(self, vol):
        if self.norm == "zscore":
            m, s = vol.mean(), vol.std() + 1e-6
            vol = (vol - m) / s
        elif self.norm == "minmax01":
            lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
            vol = (vol - lo) / (hi - lo + 1e-6)
            vol = np.clip(vol, 0, 1)
        return vol


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if pd.isna(row["embryo_paths"]) or pd.isna(row["empty_well_paths"]) or pd.isna(row["sample_paths"]):
            print(f"Row {idx} has missing path data: ", row.to_string(index = False))
            raise ValueError(f"Row {idx} has missing path data")

        embryo_paths = row["embryo_paths"].split("|")

        embryo_frames = [self._read_gray(p) for p in embryo_paths]
        embryo_vol = np.stack(embryo_frames, axis=0)  
        embryo_vol = self._normalize_video(embryo_vol)

        embryo_vol = embryo_vol[:,None, :, :] 

        #return self.transforms(torch.from_numpy(embryo_vol))
        return torch.from_numpy(embryo_vol)

    def __len__(self):
        return len(self.df)


   

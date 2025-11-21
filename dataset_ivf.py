# dataset_ivf.py
import numpy as np, pandas as pd, torch 
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
class IVFSequenceDataset(Dataset):
    def __init__(self, index_csv, resize=500, norm="minmax01"):
        self.df = pd.read_csv(index_csv)
        self.resize = resize
        self.norm = norm

    def _read_gray(self, path):
        img = np.array(Image.open(path), dtype = "float32")
        if img is None: 
            raise FileNotFoundError(path)
        return img.astype(np.float32)

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
            #raise ValueError(f"Row {idx} has missing path data")
            print(f"Row {idx} has missing path data: ", row.to_string(index = False))
            return torch.tensor([]), torch.tensor([]), torch.tensor([])           

        embryo_paths = row["embryo_paths"].split("|")
        empty_well_paths = row["empty_well_paths"].split("|")
        sample_paths = row["sample_paths"].split("|")

        embryo_frames = [self._read_gray(p) for p in embryo_paths]
        embryo_vol = np.stack(embryo_frames, axis=0)  
        embryo_vol = self._normalize_video(embryo_vol)
        embryo_vol = embryo_vol[:,None, :, :] 

        empty_well_frames = [self._read_gray(p) for p in empty_well_paths]
        empty_well_vol = np.stack(empty_well_frames, axis=0)  
        empty_well_vol = self._normalize_video(empty_well_vol)
        empty_well_vol = empty_well_vol[:,None, :, :]        

        sample_frames = [self._read_gray(p) for p in sample_paths]
        sample_vol = np.stack(sample_frames, axis=0)  
        sample_vol = self._normalize_video(sample_vol)
        sample_vol = sample_vol[:,None, :, :]
        return torch.from_numpy(embryo_vol), torch.from_numpy(empty_well_vol),  torch.from_numpy(sample_vol) 

    def __len__(self):
        return len(self.df)


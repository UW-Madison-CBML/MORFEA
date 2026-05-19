# dataset_ivf_embryo.py - Load complete embryo sequences
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class IVFEmbryoDataset(Dataset):
    def __init__(self, df, resize=128, norm="minmax01", max_frames=None):
        self.df = df
        self.resize = resize
        self.norm = norm
        self.max_frames = None

    def _read_gray(self, path):
        img = Image.open(path)
        if img is None:
            raise FileNotFoundError(path)

        if self.resize is not None:
            img = img.resize((self.resize, self.resize), Image.BILINEAR)

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

        if pd.isna(row["embryo_paths"]):
            print(f"Row {idx} has missing embryo_paths: ", row.to_string(index=False))
            raise ValueError(f"Row {idx} has missing embryo_paths")

        embryo_paths = row["embryo_paths"].split("|")

        if self.max_frames is not None and len(embryo_paths) > self.max_frames:
            embryo_paths = embryo_paths[:self.max_frames]

        embryo_frames = [self._read_gray(p) for p in embryo_paths]
        embryo_vol = np.stack(embryo_frames, axis=0)  # (T, H, W)

        embryo_vol = self._normalize_video(embryo_vol)

        embryo_vol = embryo_vol[:, None, :, :]

        return torch.from_numpy(embryo_vol)

    def __len__(self):
        return len(self.df)

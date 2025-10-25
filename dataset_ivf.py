# dataset_ivf.py
import cv2, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class IVFSequenceDataset(Dataset):
    def __init__(self, index_csv, resize=128, norm="minmax01"):
        self.df = pd.read_csv(index_csv)
        self.resize = resize
        self.norm = norm

    def _read_gray(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: 
            raise FileNotFoundError(path)
        img = cv2.resize(img, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
        # 輕量去雜訊（可選）
        img = cv2.GaussianBlur(img, (3,3), 1.0)
        return img.astype(np.float32)

    def _normalize_video(self, vol):  # vol: [T,H,W]
        # per-sequence normalization（嚴謹）
        if self.norm == "zscore":
            m, s = vol.mean(), vol.std() + 1e-6
            vol = (vol - m) / s
        elif self.norm == "minmax01":
            lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
            vol = (vol - lo) / (hi - lo + 1e-6)
            vol = np.clip(vol, 0, 1)
        return vol

    def __getitem__(self, idx):
        paths = self.df.iloc[idx]["paths"].split("|")
        frames = [self._read_gray(p) for p in paths]
        vol = np.stack(frames, axis=0)  # [T,H,W]
        vol = self._normalize_video(vol)
        vol = vol[:, None, :, :]        # [T,1,128,128]
        return torch.from_numpy(vol), self.df.iloc[idx]["cell_id"]

    def __len__(self):
        return len(self.df)


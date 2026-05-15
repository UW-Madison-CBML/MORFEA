# dataset_ivf_embryo.py - Load complete embryo sequences
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class IVFEmbryoDataset(Dataset):
    """
    Dataset that loads complete embryo sequences.
    Each sample is one full embryo with all its frames.
    """
    def __init__(self, df, resize=128, norm="minmax01", max_frames=None):
        """
        Args:
            index_csv: Path to index CSV with columns: embryo_id, embryo_paths
            resize: Target image size (square)
            norm: Normalization method ("zscore" or "minmax01")
            max_frames: Maximum number of frames to load per embryo (None = load all)
        """
        self.df = df
        self.resize = resize
        self.norm = norm
        self.max_frames = None

    def _read_gray(self, path):
        """Read and resize a grayscale image."""
        img = Image.open(path)
        if img is None:
            raise FileNotFoundError(path)

        # Resize if needed
        if self.resize is not None:
            img = img.resize((self.resize, self.resize), Image.BILINEAR)

        return np.array(img, dtype="float32")

    def _normalize_video(self, vol):
        """Normalize a video volume."""
        if self.norm == "zscore":
            m, s = vol.mean(), vol.std() + 1e-6
            vol = (vol - m) / s
        elif self.norm == "minmax01":
            lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
            vol = (vol - lo) / (hi - lo + 1e-6)
            vol = np.clip(vol, 0, 1)
        return vol

    def __getitem__(self, idx):
        """
        Returns:
            embryo_vol: Tensor of shape (T, 1, H, W) where T is the number of frames
        """
        row = self.df.iloc[idx]

        if pd.isna(row["embryo_paths"]):
            print(f"Row {idx} has missing embryo_paths: ", row.to_string(index=False))
            raise ValueError(f"Row {idx} has missing embryo_paths")

        embryo_paths = row["embryo_paths"].split("|")

        # Limit frames if max_frames is set
        if self.max_frames is not None and len(embryo_paths) > self.max_frames:
            embryo_paths = embryo_paths[:self.max_frames]

        # Load all frames
        embryo_frames = [self._read_gray(p) for p in embryo_paths]
        embryo_vol = np.stack(embryo_frames, axis=0)  # (T, H, W)

        # Normalize the entire sequence
        embryo_vol = self._normalize_video(embryo_vol)

        # Add channel dimension: (T, H, W) -> (T, 1, H, W)
        embryo_vol = embryo_vol[:, None, :, :]

        return torch.from_numpy(embryo_vol)

    def __len__(self):
        return len(self.df)

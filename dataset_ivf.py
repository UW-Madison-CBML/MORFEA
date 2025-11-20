# dataset_ivf.py
# PyTorch Dataset for IVF embryo timelapse sequences
import os
from pathlib import Path

from PIL import Image, ImageFile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Try to import build_index (will be available on GPU node)
try:
    import build_index
except ImportError:
    build_index = None


class IVFSequenceDataset(Dataset):
    """
    Dataset for loading IVF embryo timelapse sequences.
    
    Args:
        index_csv: Path to CSV file with columns: cell_id, start_idx, paths
        resize: Target image size (default: 128)
        norm: Normalization method - "minmax01" or "zscore" (default: "minmax01")
    """
    
    def __init__(self, index_csv="index.csv", resize=128, norm="minmax01"):
        self.index_csv = index_csv
        self.resize = resize
        self.norm = norm
        
        index_path = Path(index_csv)
        
        # ★★ Key: If index.csv doesn't exist, build it on GPU node ★★
        if not index_path.exists():
            print(f"[dataset_ivf] {index_csv} not found, building it with build_index.py...", flush=True)
            if build_index is None:
                raise ImportError(
                    "[dataset_ivf] build_index module not available. "
                    "Make sure build_index.py is present in the working directory "
                    "and included in transfer_input_files."
                )
            # build_index.main() will create 'index.csv' in CWD
            build_index.main()
            
            # Check again after building
            if not index_path.exists():
                raise FileNotFoundError(
                    f"[dataset_ivf] After running build_index.main(), still no {index_csv}. "
                    "Check that 'data' symlink -> /project/bhaskar_group/ivf exists "
                    "and contains valid cell image folders."
                )
            print(f"[dataset_ivf] ✓ Successfully created {index_csv}", flush=True)
        
        # At this point index.csv must exist, load it
        self.df = pd.read_csv(index_path)
        print(f"[dataset_ivf] Loaded index with {len(self.df)} rows", flush=True)

    def _read_gray(self, path):
        """Read and preprocess a single grayscale image using Pillow"""
        try:
            img = Image.open(path)
            img.load()  # Force loading to catch truncated images early
            img = img.convert("L")  # Convert to grayscale
            img = img.resize((self.resize, self.resize), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            # Light denoising using simple box filter (alternative to GaussianBlur)
            # For simplicity, we skip denoising here - can add if needed
            return arr
        except (OSError, IOError, Image.UnidentifiedImageError) as e:
            # Return a black image if file is corrupted/truncated
            print(f"Warning: Could not read image {path}, using black image. Error: {e}")
            return np.zeros((self.resize, self.resize), dtype=np.float32)
        except Exception as e:
            print(f"Warning: Unexpected error reading {path}, using black image. Error: {e}")
            return np.zeros((self.resize, self.resize), dtype=np.float32)

    def _normalize_video(self, vol):  # vol: [T, H, W]
        """Normalize video sequence per-sequence"""
        if self.norm == "zscore":
            m, s = vol.mean(), vol.std() + 1e-6
            vol = (vol - m) / s
        elif self.norm == "minmax01":
            lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
            vol = (vol - lo) / (hi - lo + 1e-6)
            vol = np.clip(vol, 0, 1)
        return vol

    def __getitem__(self, idx):
        """Get a single sequence"""
        paths = self.df.iloc[idx]["paths"].split("|")
        frames = [self._read_gray(p) for p in paths]
        vol = np.stack(frames, axis=0)  # [T, H, W]
        vol = self._normalize_video(vol)
        vol = vol[:, None, :, :]  # [T, 1, H, W] - add channel dimension
        return torch.from_numpy(vol), self.df.iloc[idx]["cell_id"]

    def __len__(self):
        return len(self.df)


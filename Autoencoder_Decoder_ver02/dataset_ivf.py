# dataset_ivf.py
# PyTorch Dataset for IVF embryo timelapse sequences
import os
import tarfile
import io
from pathlib import Path

from PIL import Image, ImageFile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Tar.gz file path (if data is in tar.gz)
TAR_FILE = "/staging/groups/bhaskar_group/ivf/embryo_dataset.tar.gz"

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
    
    def __init__(self, index_csv="index.csv", resize=128, norm="minmax01", tar_file=None):
        self.index_csv = index_csv
        self.resize = resize
        self.norm = norm
        self.tar_file = tar_file or TAR_FILE if os.path.exists(TAR_FILE) else None
        
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
            # Call build_index.main() directly (output will go to stdout/stderr)
            try:
                build_index.main()
            except SystemExit as e:
                # build_index uses sys.exit(1) on failure
                code = e.code if hasattr(e, 'code') and e.code is not None else 1
                if code != 0:
                    raise RuntimeError(
                        f"[dataset_ivf] build_index.main() failed with exit code {code}.\n"
                        "This usually means the 'data' symlink is broken or points to wrong path. "
                        "Check that data symlink -> /project/bhaskar_group/ivf exists and is accessible. "
                        "See build_index output above for details."
                    ) from e
                # If exit code is 0, that's fine (shouldn't happen but handle it)
            except Exception as e:
                raise RuntimeError(
                    f"[dataset_ivf] build_index.main() raised exception: {e}\n"
                    "Check that 'data' symlink -> /project/bhaskar_group/ivf exists "
                    "and contains valid cell image folders."
                ) from e
            
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
        """Read and preprocess a single grayscale image using Pillow
        Supports both regular file paths and tar.gz files
        """
        try:
            # Check if path is in tar.gz format (starts with "embryo_dataset/")
            # and tar.gz file exists
            if self.tar_file and os.path.exists(self.tar_file) and path.startswith("embryo_dataset/"):
                try:
                    with tarfile.open(self.tar_file, 'r:gz') as tar:
                        member = tar.getmember(path)
                        img_data = tar.extractfile(member)
                        if img_data:
                            img_bytes = img_data.read()
                            img = Image.open(io.BytesIO(img_bytes))
                            img.load()
                            img = img.convert("L")
                            img = img.resize((self.resize, self.resize), Image.BILINEAR)
                            arr = np.array(img, dtype=np.float32)
                            return arr
                except (KeyError, tarfile.TarError) as e:
                    # File not in tar.gz, fall through to try regular file path
                    pass
            
            # Try regular file path
            if os.path.exists(path):
                img = Image.open(path)
                img.load()
                img = img.convert("L")
                img = img.resize((self.resize, self.resize), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32)
                return arr
            else:
                # Path doesn't exist, return black image
                # Don't print warning for every missing file (too verbose)
                return np.zeros((self.resize, self.resize), dtype=np.float32)
                
        except (OSError, IOError, Image.UnidentifiedImageError) as e:
            # Don't print for every error (too verbose)
            return np.zeros((self.resize, self.resize), dtype=np.float32)
        except Exception as e:
            # Don't print for every error (too verbose)
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

    def _convert_path(self, path):
        """Convert local path to CHTC staging path if needed
        Returns path that can be used with tar.gz (format: embryo_dataset/...)
        """
        path_str = str(path)
        
        # If tar.gz exists, always convert to tar path format
        if self.tar_file and os.path.exists(self.tar_file):
            # Extract relative path from any format
            if "/Users/grnho/Desktop/Project IVF/embryo_dataset" in path_str:
                # From local path
                rel_path = path_str.split("embryo_dataset/", 1)[1] if "embryo_dataset/" in path_str else None
                if rel_path:
                    return f"embryo_dataset/{rel_path}"
            elif "/staging/groups/bhaskar_group/ivf/embryo_dataset" in path_str:
                # From staging path (already converted but wrong format)
                rel_path = path_str.split("embryo_dataset/", 1)[1] if "embryo_dataset/" in path_str else None
                if rel_path:
                    return f"embryo_dataset/{rel_path}"
            elif "embryo_dataset" in path_str:
                # Extract from any path containing embryo_dataset
                path_parts = Path(path_str).parts
                if "embryo_dataset" in path_parts:
                    idx = path_parts.index("embryo_dataset")
                    rel_parts = path_parts[idx+1:]
                    rel_path = "/".join(rel_parts)
                    return f"embryo_dataset/{rel_path}"
        
        # If no tar.gz, try regular file paths
        if "/Users/grnho/Desktop/Project IVF/embryo_dataset" in path_str:
            rel_path = path_str.split("embryo_dataset/", 1)[1] if "embryo_dataset/" in path_str else None
            if rel_path:
                possible_staging_paths = [
                    f"/staging/groups/bhaskar_group/ivf/embryo_dataset/{rel_path}",
                    f"/staging/groups/bhaskar_group/rho9/ivf_data/embryo_dataset/{rel_path}",
                ]
                for staging_path in possible_staging_paths:
                    if os.path.exists(staging_path):
                        return staging_path
        
        return path_str
    
    def __getitem__(self, idx):
        """Get a single sequence"""
        paths = self.df.iloc[idx]["paths"].split("|")
        # Convert paths to staging if needed
        paths = [self._convert_path(p) for p in paths]
        frames = [self._read_gray(p) for p in paths]
        vol = np.stack(frames, axis=0)  # [T, H, W]
        vol = self._normalize_video(vol)
        vol = vol[:, None, :, :]  # [T, 1, H, W] - add channel dimension
        return torch.from_numpy(vol), self.df.iloc[idx]["cell_id"]

    def __len__(self):
        return len(self.df)


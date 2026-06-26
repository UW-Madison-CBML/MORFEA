# dataset for old ConvAE architecture
# dataset_ivf.py
import numpy as np, pandas as pd, torch 
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import os
from torchvision.transforms import v2
from torchvision.tv_tensors import Video
ImageFile.LOAD_TRUNCATED_IMAGES = True
class IVFSequenceDataset(Dataset):
    def __init__(self, df, crop=45, resize=500, norm="minmax01", inference=False):
        self.crop=crop
        #self.df = pd.read_csv(index_csv)
        self.df = df
        self.resize = resize
        self.norm = norm
        self.inference = inference
        self.augment = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])


    def _read_gray(self, path):
        img = Image.open(path)
        if img is None:
            raise FileNotFoundError(path)
        # Resize if needed
        if self.resize is not None:
            # crop
            img = img.crop((self.crop,self.crop,500-self.crop,500-self.crop))
            #resize
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
        if pd.isna(row["embryo_paths"]) or pd.isna(row["empty_well_paths"]) or pd.isna(row["sample_paths"]):
            print(f"Row {idx} has missing path data: ", row.to_string(index = False))
            raise ValueError(f"Row {idx} has missing path data")

        embryo_paths = row["embryo_paths"].split("|")
        embryo_frames = [self._read_gray(p) for p in embryo_paths]
        embryo_vol = np.stack(embryo_frames, axis=0)  
        embryo_vol = self._normalize_video(embryo_vol)
        embryo_vol = embryo_vol[:,None, :, :] 
        torch_vol = torch.from_numpy(embryo_vol) 
        #if(self.inference):
        #    return torch_vol
        #else:
        #    re
        #return self.augment(Video(torch_vol))
        return torch_vol, self.augment(Video(torch_vol))

    def __len__(self):
        return len(self.df)

if __name__ == "__main__":

    ds = IVFSequenceDataset(pd.read_csv(os.path.abspath("index.csv")), resize=128, norm="minmax01")
    total_size = len(ds)

    train_size = int(0.85 * total_size)
    val_size = total_size - train_size

    generator = torch.Generator().manual_seed(42)
    _, val_set = torch.utils.data.random_split(ds, [train_size, val_size], generator=generator)
    val_df = ds.df.iloc[val_set.indices]
    pd.set_option('display.max_rows', None)
    print(val_df[['cell_id']])


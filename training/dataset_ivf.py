# dataset for old ConvAE architecture
# dataset_ivf.py
import numpy as np, pandas as pd, torch 
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import os
from torchvision.transforms import v2
from torchvision.tv_tensors import Video
from torchvision.transforms import InterpolationMode 
ImageFile.LOAD_TRUNCATED_IMAGES = True
def read_gray(path, resize, crop):
    img = Image.open(path).convert("L")
    if img is None:
        raise FileNotFoundError(path)
    #Resize if needed
    if crop > 0:
        # crop
        img = img.crop((crop,crop,500-crop,500-crop))

    if resize is not None:
        img = img.resize((resize, resize), Image.BILINEAR)
        
    return np.array(img, dtype="float32")


class IVFSequenceDataset(Dataset):
    def __init__(self, df, crop=45, resize=500, norm="minmax01", inference=False, sequences = True):
        self.crop=crop
        #self.df = pd.read_csv(index_csv)
        self.df = df
        self.resize = resize
        self.norm = norm
        self.inference = inference
        self.augment = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5), #v2.RandomRotation([-180,180], interpolation=InterpolationMode.BILINEAR),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomCrop(size=500-(2*self.crop)),
            v2.Resize(size=self.resize)
        ])
        self.sequences = sequences


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
        embryo_frames1 = [read_gray(p, self.resize, self.crop) for p in embryo_paths]
        embryo_frames2 = [read_gray(p, None, 0) for p in embryo_paths]
        embryo_vol1 = np.stack(embryo_frames1, axis=0)  
        embryo_vol2 = np.stack(embryo_frames2, axis=0)  
        embryo_vol1 = self._normalize_video(embryo_vol1)
        embryo_vol2 = self._normalize_video(embryo_vol2)
        embryo_vol1 = embryo_vol1[:,None, :, :] 
        embryo_vol2 = embryo_vol2[:,None, :, :] 
        torch_vol1 = torch.from_numpy(embryo_vol1) 
        torch_vol2 = torch.from_numpy(embryo_vol2) 
        #if(self.inference):
        #    return torch_vol
        #else:
        #    re
        #return self.augment(Video(torch_vol))
        return torch_vol1, self.augment(Video(torch_vol2))

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


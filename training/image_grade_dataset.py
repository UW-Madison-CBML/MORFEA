import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import v2
from torchvision.tv_tensors import Video
class ImageGradeDataset(Dataset):
    def __init__(self, df, grade, crop=50, resize=128, norm="minmax01", max_frames=None, return_whole_seqs=False):
        
        df = df.dropna(subset=[grade])
        self.df = pd.concat([pd.DataFrame({"embryo_id":row["embryo_id"], "image_path":row["embryo_paths"].split("|"), grade:row[grade]}) for _, row in df.iterrows()], axis=0, ignore_index=True)
        self.groups = self.df.groupby("embryo_id")
        self.crop = crop
        self.resize = resize
        self.norm = norm
        self.GRADES = ["A", "B", "C"]
        self.return_whole_seqs = return_whole_seqs
        self.augment = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])

       
        self.grade = grade
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
        rows=None
        if self.return_whole_seqs:
            _, group = list(self.groups)[idx]
            rows = group
        else:
            row = self.df.iloc[idx]

            group = self.groups.get_group(row["embryo_id"])

            group_idx = group.index.get_loc(row.name)
            rows = group.iloc[:max(5, group_idx)]

        embryo_paths = rows["image_path"].to_list()

        embryo_frames = [self._read_gray(p) for p in embryo_paths]
        embryo_vol = np.stack(embryo_frames, axis=0)  # (T, H, W)

        embryo_vol = self._normalize_video(embryo_vol)

        embryo_vol = embryo_vol[:, None, :, :]

        return torch.from_numpy(embryo_vol), torch.tensor(self.GRADES.index(rows.iloc[0][self.grade]))

    def __len__(self):
        return len(self.groups) if self.return_whole_seqs else len(self.df) 
    def pad_collate(self, batch):
        signals = [item[0] for item in batch]
        targets = [item[1] for item in batch]
    
        lengths = torch.tensor([len(s) for s in signals])
    
        signals_padded = torch.nn.utils.rnn.pad_sequence(
            signals, batch_first=True, padding_value=0.0
        )
    
        targets = torch.tensor(targets)
    
        return signals_padded, targets, lengths 

class SingleFrameDataset(Dataset):
    GRADES = ["A", "B", "C"]
    def __init__(self, df, image_size=224):
        self.df = df 
        self.resize = image_size
        self.crop = 0
    def _read_gray(self, path):
        img = Image.open(path).convert("L")
        if img is None:
            raise FileNotFoundError(path)
        # Resize if needed
        if self.resize is not None:
            # crop
            img = img.crop((self.crop,self.crop,500-self.crop,500-self.crop))
            #resize
            img = img.resize((self.resize, self.resize), Image.BILINEAR)
            
        return np.array(img, dtype="float32")


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return torch.tensor(self._read_gray(row["path"]))[None,:,:], torch.tensor(self.__class__.GRADES.index(row["TE"]), dtype=torch.long)

    def __len__(self):
        return len(self.df)


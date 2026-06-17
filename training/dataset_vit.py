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
            v2.ToDtype(torch.float32),
            v2.CenterCrop(400),
            v2.Resize((resize,resize)),
            v2.Normalize(mean=[0.485], std=[0.229]),
        ])

    def _read_gray(self, path):
        img = Image.open(path)
        if img is None:
            raise FileNotFoundError(path)
        return np.array(img, dtype="uint8")

    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        

        embryo_paths = row["embryo_paths"].split("|")

        embryo_frames = [self._read_gray(p) for p in embryo_paths]
        embryo_vol = np.stack(embryo_frames, axis=0)  

        embryo_vol = embryo_vol[:,None, :, :] 

        return self.transforms(torch.from_numpy(embryo_vol)),torch.from_numpy(embryo_vol)

    def __len__(self):
        return len(self.df)



if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    index_df = pd.read_csv(os.path.abspath("index.csv"))
    dataset = VITDataset(index_df)
    print(dataset.df.head())
    loader = DataLoader(dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False 
    )
    for embryo1,embryo2 in loader:
        plt.imshow(embryo1.squeeze()[0].numpy())
        plt.show()
        plt.imshow(embryo2.squeeze()[0].numpy())
        plt.show()
    
    

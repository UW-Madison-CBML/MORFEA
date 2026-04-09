import torch
import torch.nn.functional as F
from image_stage_model import ImageStageModel
from torch.utils.data import DataLoader
import wandb
import os
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.nn.utils.rnn import pad_sequence 
class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
 
    def push(self, x):
        """Add a new value and update statistics."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
 
    @property
    def variance(self):
        """Returns sample variance (unbiased). Use self.m2 / self.n for population."""
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0
 
    @property
    def std_dev(self):
        return math.sqrt(self.variance)
def get_annotations_col(embryo_id, group_len, annotations_dir):
    annotation_file = os.path.join(annotations_dir, f"{embryo_id}_phases.csv")
    df = pd.read_csv(annotation_file, names=['stage_id', 'stage_begin', 'stage_end'])

    new_column = []
    
    new_column += ["pre_phase"] * (df.iloc[0]["stage_begin"] - 1)
    col_len_seq = []
    for index, row in df.iterrows():
        new_column += [row["stage_id"]] * (row["stage_end"] - row["stage_begin"]+1)
        col_len_seq.append(len(new_column))

    


    new_column += ["post_phase"] * (group_len - len(new_column))
    new_column = new_column[:group_len]
    return new_column

class StageImageDataset(Dataset):
    def __init__(self, df, resize=128, norm="minmax01", return_embryo_id=False):
        self.resize = resize
        self.norm = norm
        self.return_embryo_id = return_embryo_id
        print("og df len: ", str(df["num_frames"].sum()))
        self.df = pd.concat([pd.DataFrame({"embryo_id":df.iloc[i]["embryo_id"],"phase":get_annotations_col(df.iloc[i]["embryo_id"],df.iloc[i]["num_frames"], "embryo_dataset_annotations"),"frame":df.iloc[i]["embryo_paths"].split("|")}) for i in range(len(df))], ignore_index=True).reset_index()
        print("neq df len: ", str(len(self.df)))
        print(self.df.head(100))
        self.groups = self.df.groupby("embryo_id")
        self.seqlength = 64
        self.phases = ['pre_phase', 'tPB2', 'tPNa', 'tPNf', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM','tSB','tB', 'tEB', 'tHB', 'post_phase']


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
        row = self.df.iloc[idx]
        

        group = self.groups.get_group(row["embryo_id"])

        
        #seqindex = int(((row["time_step"] - 1) / len(group)) * (len(group) - self.seqlength - 1))

        seq_df = group.loc[:max(group.index[0] + 5, idx+1)]

        embryo_paths = seq_df["frame"]
        embryo_frames = [self._read_gray(p) for p in embryo_paths]
        embryo_vol = np.stack(embryo_frames, axis=0)  # (T, H, W)

        embryo_vol = self._normalize_video(embryo_vol)

        embryo_vol = embryo_vol[:, None, :, :]


        if (self.return_embryo_id):

            return torch.tensor(embryo_vol), torch.tensor([self.phases.index(r["phase"]) for _,r in seq_df.iterrows()], dtype = torch.long), row["embryo_id"]

        return torch.tensor(embryo_vol), torch.tensor([self.phases.index(r["phase"]) for _,r in seq_df.iterrows()], dtype = torch.long)

    def __len__(self):
        return len(self.df)
    def pad_collate(self, batch):
        if(self.return_embryo_id):
            (data, labels,embryo_id) = zip(*batch)
        else:  
            (data, labels) = zip(*batch)
        
        data_padded = pad_sequence(data, batch_first=True, padding_value=0)
        
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
        mask = torch.zeros(labels_padded.shape, dtype=torch.bool)
        for i, seq in enumerate(labels):
            mask[i, :len(seq)] = True
        
        if(self.return_embryo_id):
            return data_padded, labels_padded, mask, embryo_id
        return data_padded, labels_padded, mask
 
VAL_EMBRYOS =[
    "RG434-11",
    "RC1103-1",
    "LV488-7",
    "QC211-6",
    "BM016-2",
    "LM184-3",
    "RMN410-3",
    "PA145-1",
    "RO793-2",
    "PV361-2",
    "RC755-7",
    "VC581-3",
    "VC581-11",
    "ADM715-1-2",
    "LS1045-4",
    "GA800-4",
    "GJ191-1",
    "JV227-2",
    "LA367-4",
    "BN356-6",
    "TN611-7",
    "AHS115-5",
    "LCF544-2",
    "JV227-5",
    "CAV074-8",
    "AL702-9",
    "VH99-3",
    "GE218-3",
    "CC455-3",
    "DA1054-5",
    "ME378-4",
    "BA560-1",
    "PA145-2",
    "DSM138-5",
    "FN852-1",
    "TJ297-4",
    "RC755-9",
    "PA289-8",
    "LS93-8",
    "GA817-1-8",
    "AM918-2-5",
    "LNA592-9",
    ]

 
    
    
    
    
def main():
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
        name=f"image_phase"
    )
 
    learning_rate = 0.001
    df = pd.read_csv(os.path.abspath("index_embryo.csv"))
    mask = df["embryo_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    val_df = df[mask]
    df = df[~mask]
 
    dataset = StageImageDataset(df)
    dataset_val = StageImageDataset(val_df)
    #weights = get_class_weights(os.path.abspath("embryo_dataset_annotations"), df.groupby("embryo_id").size().reset_index(name='counts'), dataset.phases).to(DEVICE)
    crit = torch.nn.CrossEntropyLoss()
    model = ImageStageModel()
    torch.backends.cudnn.enabled = False
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
 
 
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        collate_fn = lambda x: dataset.pad_collate(x)
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
        
        #collate_fn = lambda x: dataset.pad_collate(x) # i dont think this is necessary for batch of 1
    )
    print(len(loader))
    for epoch in range(8):
        print(epoch)
        model.train()
        for lats, labels,masks in loader:
            lats = lats.to(DEVICE).float()
            labels = labels.to(DEVICE).long()
            masks = masks.to(DEVICE)
            loss = model(lats, masks, tags=labels)
            #loss = crit(logits.view(-1, 18), labels.view(-1)) + 0.1 * monotonicity_loss(logits)
             
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run.log({"loss": loss.item()})
        model.eval()
        loss_stats = RunningStats()
        acc_stats = RunningStats()

        stats_dict = {} 

        with torch.no_grad():
            for lats, labels, embryo_id in loader_val:
                lats = lats.to(DEVICE).float()
                labels = labels.to(DEVICE).long()
                logits = model(lats, None) # no mask for single batch

                embryo_id = embryo_id[0]

                if (embryo_id in stats_dict.keys()):
                    stats_dict[embryo_id].push(loss.item())
                else:
                    stats_dict[embryo_id] = RunningStats()
 
                # Calculate accuracy
                preds = logits.view(-1,18).argmax(dim=1)  # Get predicted class (0, 1, or 2)
                acc_stats.push((preds == labels.view(-1)).sum().item()/labels.view(-1).shape[0])
        run.log({
            "val_acc":acc_stats.mean,
            "val_acc_std":acc_stats.std_dev})
        highest_std = sorted(list(stats_dict.items()), key = lambda x: (x[1].std_dev))[:5]
        return
        model.eval()
        for embryo, _ in highest_std:
            with open(os.path.join("embryo_dataset_annotations", f"{embryo}_phases.csv"), 'r') as file:
                print("Ground Truth:\n", file.read())
            with torch.no_grad():
                data = dataset_val.df[dataset_val.df["embryo_id"] == embryo][dataset_val.lat_cols].to_numpy()
                print(data.shape)
                whole_seq = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                logits = model(whole_seq, None)
                
                pred_indices = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
                print(pred_indices)

                print("Predicted Phases:")
                current = ""
                len_seq = 0
                current_idx = 0
                for i, idx in enumerate(pred_indices):
                    if(i == 0 or pred_indices[i-1] != idx):
                        print(f"{current} {current_idx}, {current_idx + len_seq} times")
                        current = dataset_val.phases[idx]
                        current_idx += len_seq
                        len_seq = 1
                    
                    else:
                        len_seq += 1
                
                 


 
 
if __name__ == "__main__":
    main() 
    

import torch
import torch.nn.functional as F
from stage_dataset import StageDataset
from stage_model import StageModel
from torch.utils.data import DataLoader
import wandb
import os
import pandas as pd
import numpy as np
import math
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
        """Returns sample standard deviation."""
        return math.sqrt(self.variance)
 
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
def get_class_weights(annotations_dir, lat_group_sizes, phases):
    stage_id_freq = {phase: 0 for phase in phases}
    for i, row in lat_group_sizes.iterrows():
        annotations_df = pd.read_csv(os.path.join(annotations_dir, f"{row['embryo_id']}_phases.csv"), header=None, names=["stage_id","stage_begin", "stage_end"])
        stage_id_freq["pre_phase"] += max(0,annotations_df.iloc[0]["stage_begin"] - 1)
        stage_id_freq["post_phase"] += max(0, row["counts"] - annotations_df.iloc[-1]["stage_end"])
        for _, phase_row in annotations_df.iterrows():
            stage_id_freq[phase_row["stage_id"]] += max(0,phase_row["stage_end"] - phase_row["stage_begin"])
    out_tensor = torch.tensor([stage_id_freq[phase] for phase in phases])
    return out_tensor / out_tensor.sum()

def monotonicity_loss(batched_logits_seq, temp=8.0):
    B, T, C = batched_logits_seq.shape
    device = batched_logits_seq.device
    probs = F.softmax(batched_logits_seq * temp, dim=-1) # (B, T, C)
    indices = torch.arange(C, device=device, dtype=torch.float32) # (C) 
    soft_indices = (probs * indices).sum(dim=-1) # (B, T) 
    diffs = soft_indices.diff(dim=-1) # (B, T-1) 
    monotone_violations = F.leaky_relu(-diffs, negative_slope=0.1) 
    return monotone_violations.mean()
 
    
    
    
    
def main(model_name, curvature = True, velocity = True, acceleration = True, path_signatures = True, latents = True, distance_mat=True):
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
        name=f"{model_name}-phase-{'c' if curvature else ''}{'v' if velocity else ''}{'p' if path_signatures else ''}{'l' if latents else ''}{'d' if distance_mat else ''}{'a' if acceleration else ''}",
    )
 
    learning_rate = 0.001
    lat_df = pd.read_csv(os.path.abspath(f"latents/{model_name}.csv")).rename(columns={"cell_id":"embryo_id"})
    lat_np = np.load(os.path.abspath(f"latents/{model_name}.npy"))
    if(len(lat_df) != lat_np.shape[0]):
        raise ValueError("keys and values sizes do not match")
    lat_columns = [f"z_{i}" for i in range(lat_np.shape[1])]
    values_df = pd.DataFrame(lat_np, columns=lat_columns)
    df = pd.concat([lat_df, values_df], axis = 1)
    mask = df["embryo_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    val_df = df[mask]
    df = df[~mask]
 
    dataset = StageDataset(df, "embryo_dataset_annotations", latents=latents, velocity=velocity, acceleration=acceleration, curvature=curvature, distance_mat=distance_mat, path_signatures=path_signatures)
    dataset_val = StageDataset(val_df, "embryo_dataset_annotations", latents=latents, velocity=velocity, acceleration=acceleration, curvature=curvature, distance_mat=distance_mat, path_signatures=path_signatures, return_embryo_id=True)
    weights = get_class_weights(os.path.abspath("embryo_dataset_annotations"), df.groupby("embryo_id").size().reset_index(name='counts'), dataset.phases).to(DEVICE)
    crit = torch.nn.CrossEntropyLoss(weight=weights)
    model = StageModel(input_size = len(dataset.lat_cols))
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
 
 
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    print(len(loader))
    for epoch in range(8):
        print(epoch)
        model.train()
        for lats, labels in loader:
            lats = lats.to(DEVICE).float()
            labels = labels.to(DEVICE).long()
            logits = model(lats)
            loss = crit(logits.view(-1, 18), labels.view(-1)) + 0.1 * monotonicity_loss(logits)
             
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
                logits = model(lats)
                loss = crit(logits.view(-1, 18), labels.view(-1))
                loss_stats.push(loss.item())

                embryo_id = embryo_id[0]

                if (embryo_id in stats_dict.keys()):
                    stats_dict[embryo_id].push(loss.item())
                else:
                    stats_dict[embryo_id] = RunningStats()
 
                # Calculate accuracy
                preds = logits.view(-1,18).argmax(dim=1)  # Get predicted class (0, 1, or 2)
                acc_stats.push((preds == labels.view(-1)).sum().item()/labels.view(-1).shape[0])
        run.log({"val_loss":loss_stats.mean,
            "val_loss_std":loss_stats.std_dev,
            "val_acc":acc_stats.mean,
            "val_acc_std":acc_stats.std_dev})
        highest_std = sorted(list(stats_dict.items()), key = lambda x: (x[1].std_dev))[:5]
        model.eval()
        for embryo, _ in highest_std:
            with open(os.path.join("embryo_dataset_annotations", f"{embryo}_phases.csv"), 'r') as file:
                print("Ground Truth:\n", file.read())
            with torch.no_grad():
                data = dataset_val.df[dataset_val.df["embryo_id"] == embryo][dataset_val.lat_cols].to_numpy()
                print(data.shape)
                whole_seq = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                logits = model(whole_seq)
                
                pred_indices = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

                print("Predicted Phases:")
                current = ""
                len_seq = 0
                for i, idx in enumerate(pred_indices):
                    if(i == 0 or pred_indices[i-1] != idx):
                        print(f"{current} {len_seq} times")
                        current = dataset_val.phases[idx]
                        len_seq = 1
                    
                    else:
                        len_seq += 1
                
                 


 
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")
 
 
    parser.add_argument("--name", help="Model name. Must have already exported latents")
    parser.add_argument("--curvature",action="store_true", help="Use to include curvature")
    parser.add_argument("--latents",action="store_true", help="Use to include latents")
    parser.add_argument("--path-signatures",action="store_true", help="Use to include path signatures")
    parser.add_argument("--velocity",action="store_true", help="Use to include velocity")
    parser.add_argument("--acceleration", action="store_true", help="Use to include acceleration")
    parser.add_argument("--distance-mat", action="store_true", help="Use to include distance to first frame")
  
    args = parser.parse_args()
 
    main(args.name,curvature=args.curvature,latents=args.latents,velocity=args.velocity, acceleration=args.acceleration, distance_mat=args.distance_mat, path_signatures=args.path_signatures)

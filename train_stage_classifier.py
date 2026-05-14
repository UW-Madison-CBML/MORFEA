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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
def recall_precision_f1(confusion_mat, i):
    recall = 0 if confusion_mat[:, i].sum() == 0 else confusion_mat[i,i]/confusion_mat[:, i].sum()
            
    precision = 0 if confusion_mat[i,:].sum() == 0 else confusion_mat[i,i]/ confusion_mat[i, :].sum()
    f1 = 0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    return recall, precision, f1

class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
 
    def push(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
 
    @property
    def variance(self):
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0
 
    @property
    def std_dev(self):
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

    
    
    
    
def main(model_name, features):
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
        name=f"{model_name}-phase-{'c' if features['curvature'] else ''}{'v' if features['velocity'] else ''}{'p' if features['path_signatures'] else ''}{'l' if features['latents'] else ''}{'d' if features['distance_mat'] else ''}{'a' if features['acceleration'] else ''}",
    )
 
    learning_rate = 0.001
    lat_df = pd.read_csv(os.path.join("latents", f"{model_name}.csv")).rename(columns={"cell_id":"embryo_id"}) # metadata
    lat_np = np.load(os.path.join("latents",f"{model_name}.npy"))
    cebra_np = np.load(os.path.join("cebra_latents",f"{model_name}.npy"))
    if(len(lat_df) != lat_np.shape[0]):
        raise ValueError("keys and values sizes do not match")
    lat_columns = [f"z_{i}" for i in range(lat_np.shape[1])]
    values_df = pd.DataFrame(lat_np, columns=lat_columns, index=lat_df.index)
    cebra_df = pd.DataFrame(cebra_np, columns=["cebra_0", "cebra_1", "cebra_2"], index=lat_df.index)
    df = pd.concat([lat_df, values_df, cebra_df], axis = 1)
    mask = df["embryo_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    val_df = df[mask]
    df = df[~mask]
 
    dataset = StageDataset(df, "embryo_dataset_annotations", features)
    dataset_val = StageDataset(val_df, "embryo_dataset_annotations", features, return_embryo_id=True, return_whole_seqs=True)
    #weights = get_class_weights(os.path.abspath("embryo_dataset_annotations"), df.groupby("embryo_id").size().reset_index(name='counts'), dataset.phases).to(DEVICE)
    crit = torch.nn.CrossEntropyLoss()
    model = StageModel(input_size = len(dataset.lat_cols))
    torch.backends.cudnn.enabled = False
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
 
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        collate_fn = lambda x: dataset.pad_collate(x)
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn = lambda x: dataset.pad_collate(x)
    )

    epochs = 8
    scheduler = CosineAnnealingLR(optimizer, len(loader) * epochs)
    print(len(loader))
    for epoch in range(epochs):
        print(epoch)
        model.train()
        for lats, labels,masks in loader:
            lats = lats.to(DEVICE).float()
            labels = labels.to(DEVICE).long()
            masks = masks.to(DEVICE)
            loss = model(lats, masks, tags=labels)
            #loss = crit(logits.view(-1, 18), labels.view(-1)) + 0.1 * monotonicity_loss(logits)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
             
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            run.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
        model.eval()
        acc_stats = RunningStats()
        area_between_curves_stats = RunningStats()

        stats_dict = {} 

        with torch.no_grad():
            for lats, labels, mask, embryo_id in loader_val:
                B, T, L = lats.shape
                # labels.shape = B, T 
                lats = lats.to(DEVICE).float()
                logits = model(lats, mask) # no mask for single batch

                embryo_id = embryo_id[0]
                preds = logits.view(B,T,18).detach().cpu().argmax(dim=1) # i think this is the right ordr
                labels = labels.view(B,T)
                acc = (preds == labels).sum().item()/labels.shape[0]
                area_between_curves = torch.abs(preds - labels).sum().item()
                acc_stats.push(acc)
                area_between_curves_stats.push(area_between_curves)

                if (embryo_id in stats_dict.keys()):
                    stats_dict[embryo_id].push(acc)
                else:
                    stats_dict[embryo_id] = RunningStats()
                    stats_dict[embryo_id].push(acc)
                    
 
        run.log({"val_acc":acc_stats.mean, "val_acc_std":acc_stats.std_dev,"area_between_curves_mean":area_between_curves_stats.mean,"area_between_curves_stddev":area_between_curves_stats.std_dev})
        highest_std = sorted(list(stats_dict.items()), key = lambda x: -1*(x[1].mean))[:10]
        f1_stats = {phase: RunningStats() for phase in dataset_val.phases}
        recall_stats = {phase: RunningStats() for phase in dataset_val.phases}
        precision_stats = {phase: RunningStats() for phase in dataset_val.phases}
        model.eval()
        for embryo, _ in highest_std:
            print(embryo)
            with open(os.path.join("embryo_dataset_annotations", f"{embryo}_phases.csv"), 'r') as file:
                print("Ground Truth:\n", file.read())
            
            ground_truth_indices = np.array([dataset_val.phases.index(phase) for phase in dataset_val.df[dataset_val.df["embryo_id"] == embryo]["phase"].tolist()])
            with torch.no_grad():
                data = dataset_val.df[dataset_val.df["embryo_id"] == embryo][dataset_val.lat_cols].to_numpy()
                whole_seq = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                logits = model(whole_seq, None)
                
                pred_indices = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

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
                confusion_mat = torch.einsum("ti,tj->ij", F.one_hot(torch.from_numpy(pred_indices), num_classes=18), F.one_hot(torch.from_numpy(ground_truth_indices), num_classes=18))
                for i in range(18):
                    recall, precision, f1 = recall_precision_f1(confusion_mat, i) 
                    f1_stats[dataset_val.phases[i]].push(f1)
                    recall_stats[dataset_val.phases[i]].push(recall)
                    precision_stats[dataset_val.phases[i]].push(precision)
                 
                if(len(pred_indices) != len(ground_truth_indices)):
                    print("bad index lists") 
                else:
                    fig, ax = plt.subplots()          
                    cmap = plt.get_cmap('tab20c')  
                    x = np.arange(len(pred_indices))
                    for i in range(len(x)-1):
                        ax.plot([x[i], x[i+1]], [pred_indices[i], pred_indices[i]], color="red", lw=2)
                        ax.plot([x[i+1], x[i+1]], [pred_indices[i], pred_indices[i+1]], color="red", lw=2)
                    for i in range(len(x)-1):
                        color = cmap(ground_truth_indices[i])
                        
                        ax.plot([x[i], x[i+1]], [ground_truth_indices[i], ground_truth_indices[i]], color=color, lw=3)
                        ax.plot([x[i+1], x[i+1]], [ground_truth_indices[i], ground_truth_indices[i+1]], color="black", lw=1)
                    


                    ax.set_ylabel('Phase')
                    ax.set_xlabel('Timestep')
                    ax.set_title(model_name + " " + embryo)
                    legend_elements = [Patch(facecolor=cmap(i), label=phase) for i, phase in enumerate(dataset_val.phases)]
                    ax.legend(handles=legend_elements, title="Phases") 
                    run.log({"pred_vs_truth": wandb.Image(fig)}) 
                    plt.close(fig)
        for phase in val_dataset.phases:
            run.log({f"{phase} recall": recall_stats.mean, f"{phase} recall std": recall_stats.std_dev, 
                f"{phase} f1": f1_stats.mean, f"{phase} f1 std": f1_stats.std_dev, 
                f"{phase} precision": precision_stats.mean, f"{phase} precision std": precision_stats.std_dev})



 
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a stage classifier using latents and their morphodynamic features.")
 
 
    parser.add_argument("--name", help="Model name. Must have already exported latents and cebra latents")
    parser.add_argument("--curvature",action="store_true", help="Use to include curvature")
    parser.add_argument("--latents",action="store_true", help="Use to include latents")
    parser.add_argument("--path-signatures",action="store_true", help="Use to include path signatures from cebra latents")
    parser.add_argument("--velocity",action="store_true", help="Use to include velocity")
    parser.add_argument("--acceleration", action="store_true", help="Use to include acceleration")
    parser.add_argument("--distance-mat", action="store_true", help="Use to include distance to first frame")
    parser.add_argument("--cebra", action="store_true", help="Use to derive features from cebra latents")
  
    args = parser.parse_args()
 
    main(args.name, vars(args))

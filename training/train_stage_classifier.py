import torch
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay
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
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

    
    
    
    
def main(model_name, features, lr=0.0001):
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
        name=f"{model_name}-phase-{features['run_name']}",
    )
 
    learning_rate = lr
    lat_df = pd.read_csv(os.path.join("latents", f"{model_name}.csv")).rename(columns={"cell_id":"embryo_id"}) # metadata
    lat_np = np.load(os.path.join("latents",f"{model_name}.npy"))
    cebra_np = np.load(os.path.join("cebra_latents",f"{model_name}.npy"))
    if(len(lat_df) != lat_np.shape[0]):
        raise ValueError("keys and values sizes do not match")
    lat_columns = [f"z_{i}" for i in range(lat_np.shape[1])]
    values_df = pd.DataFrame(lat_np, columns=lat_columns, index=lat_df.index)
    cebra_df = pd.DataFrame(cebra_np, columns=["cebra_0", "cebra_1", "cebra_2"], index=lat_df.index)
   
    # for validation splits reasons, need to prepare these outside of stage dataset 
    umap = UMAP(n_components=8)
    pca = PCA(n_components=8)
    std_scaler = StandardScaler()
    umap_df = pd.DataFrame(umap.fit_transform(lat_np), columns=[f"umap_{i}" for i in range(8)], index=lat_df.index)
    pca_df = pd.DataFrame(pca.fit_transform(std_scaler.fit_transform(lat_np)), columns=[f"pca_{i}" for i in range(8)], index=lat_df.index)
    
    df = pd.concat([lat_df, values_df, cebra_df, umap_df, pca_df], axis = 1)
    mask = df["embryo_id"].isin(VAL_EMBRYOS)
    val_df = df[mask]
    df = df[~mask]
 
    dataset = StageDataset(df, "embryo_dataset_annotations", features)
    dataset_val = StageDataset(val_df, "embryo_dataset_annotations", features, return_embryo_id=True, return_whole_seqs=True)
    #weights = get_class_weights(os.path.abspath("embryo_dataset_annotations"), df.groupby("embryo_id").size().reset_index(name='counts'), dataset.phases).to(DEVICE)
    crit = torch.nn.CrossEntropyLoss()
    model = StageModel(input_size = len(dataset.lat_cols), num_classes=len(dataset.phases))
    torch.backends.cudnn.enabled = False
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}"
    )
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
 
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
        collate_fn = lambda x: dataset.pad_collate(x)
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=16,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
        collate_fn = lambda x: dataset_val.pad_collate(x)
    )

    precision_recall_df = pd.DataFrame()
    epochs = 8
    scheduler = CosineAnnealingLR(optimizer, len(loader) * epochs)
    print(len(loader))
    for epoch in range(epochs):
        print(epoch)
        model.train()
        for lats, labels, _, masks in loader:
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
        acc_top_1_stats = RunningStats()

        acc_top_2_stats = RunningStats()
        acc_top_5_stats = RunningStats()

        f1_stats = {phase: RunningStats() for phase in dataset_val.phases}
        recall_stats = {phase: RunningStats() for phase in dataset_val.phases}
        precision_stats = {phase: RunningStats() for phase in dataset_val.phases}
        
        sum_confusion_mat = np.zeros((len(dataset_val.phases), len(dataset_val.phases))

        with torch.no_grad():
            for lats, _, labels, mask, embryo_ids in loader_val:
            
                model.eval()
                B, T, L = lats.shape
                # labels.shape = B, T 
                lats = lats.to(DEVICE).float()
                mask_gpu = mask.to(DEVICE)
                logits, emissions = model(lats, mask_gpu) # logits is a list of variable length tensor sequences of one hot's
                emissions = emissions.cpu()
                emissions = [emission_seq.masked_select(mask_seq).view((-1, len(dataset_val.phases))) for emission_seq, mask_seq in zip(emissions, mask_seq)]

                preds = [logit_seq.detach().cpu().argmax(dim=-1) for logit_seq in logits] # logits come out one-hot
                for pred, label, embryo_id, emissions_seq in zip(preds, labels, embryo_ids, emissions): # all outer most sizes are B
                    # top_1 is calculated from decoded seqs, top_k for k > 1 is based on emissions
                    acc_top_1= (pred == label).sum().item()/label.shape[0]
                    acc_top_1_stats.push(acc_top_1)
                    label_one_hots = F.one_hot(label, num_classes=len(dataset_val.phases))
                    _, top_2_indices = emissions_seq.topk(2, dim=-1)
                    _, top_5_indices = emissions_seq.topk(5, dim=-1)
                     
                    zeros = torch.zeros_like(label_one_hots)
                    
                    top_2_hots = zeros.scatter_(dim=-1, index=top_2_indices, value=1)
                    top_5_hots = zeros.scatter_(dim=-1, index=top_5_indices, value=1)
                    acc_top_2 = torch.einsum("ti,ti->",top_2_hots, label)/label_one_hots.shape[0]
                    acc_top_5 = torch.einsum("ti,ti->",top_5_hots, label)/label_one_hots.shape[0]
                    acc_top_2_stats.push(acc_top_2)
                    acc_top_5_stats.push(acc_top_5)
                    
                    # log for individual embryos
                    if(len(pred) != len(label)):
                        print("bad index lists") 
                    else:
                        # --------------------------------------------
                        # draw comparison step plots
                        fig, ax = plt.subplots()          
                        cmap = plt.get_cmap('tab20c')  
                        x = np.arange(len(pred))
                        for i in range(len(x)-1):
                            ax.plot([x[i], x[i+1]], [pred[i], pred[i]], color="red", lw=2)
                            ax.plot([x[i+1], x[i+1]], [pred[i], pred[i+1]], color="red", lw=2)
                        for i in range(len(x)-1):
                            color = cmap(label[i])
                            
                            ax.plot([x[i], x[i+1]], [label[i], label[i]], color=color, lw=3)
                            ax.plot([x[i+1], x[i+1]], [label[i], label[i+1]], color="black", lw=1)
                        


                        ax.set_ylabel('Phase')
                        ax.set_xlabel('Timestep')
                        ax.set_title(model_name + " " + embryo_id)
                        legend_elements = [Patch(facecolor=cmap(i), label=phase) for i, phase in enumerate(dataset_val.phases)]
                        ax.legend(handles=legend_elements, title="Phases") 
                        run.log({"pred_vs_truth": wandb.Image(fig)}) 
                        plt.close(fig)
                # -----------------------------------------------------
                # now look at the confusion matrix for precision recall calculations
                confusion_vol = torch.stack([torch.einsum("ti,tj->ij", F.one_hot(pred.to(dtype=torch.long), num_classes=model.num_classes), F.one_hot(label.to(dtype=torch.long), num_classes=model.num_classes)) for pred, label in zip(preds, labels)])
                confusion_mat = confusion_vol.sum(dim=0)
                sum_confusion_mat = sum_confusion_mat + confusion_mat.numpy() 
                for i in range(model.num_classes):
                    recall, precision, f1 = recall_precision_f1(confusion_mat, i) 
                    f1_stats[dataset_val.phases[i]].push(f1)
                    recall_stats[dataset_val.phases[i]].push(recall)
                    precision_stats[dataset_val.phases[i]].push(precision)
        for phase in dataset_val.phases:
            run.log({f"{phase} recall": recall_stats[phase].mean, f"{phase} recall std": recall_stats[phase].std_dev, 
                f"{phase} f1": f1_stats[phase].mean, f"{phase} f1 std": f1_stats[phase].std_dev, 
                f"{phase} precision": precision_stats[phase].mean, f"{phase} precision std": precision_stats[phase].std_dev})

        disp = ConfusionMatrixDisplay(confusion_matrix=sum_confusion_mat, display_labels=dataset_val.phases)
        disp.plot(cmap='Blues')
        fig = disp.figure_
        ax = disp.ax_
        ax.set_title("Confusion Matrix Example")
        run.log({"confusion_matrix": wandb.Image(fig)}) 
        plt.close(fig)

        
        f1_stats = {key: (value.mean, value.std_dev) for key,value in f1_stats.items()}
        precision_stats = {key: (value.mean, value.std_dev) for key,value in precision_stats.items()}
        recall_stats = {key: (value.mean, value.std_dev) for key,value in recall_stats.items()}

        precision_recall_df = pd.concat(
            [
            pd.DataFrame(f1_stats.values(), index=f1_stats.keys(), columns=["f1","f1_std"]),    
            pd.DataFrame(precision_stats.values(), index=precision_stats.keys(), columns=["precision","precision_std"]),    
            pd.DataFrame(recall_stats.values(), index=recall_stats.keys(), columns=["recall","recall_std"])
            ], axis=1)
        print(precision_recall_df)
        for stat in ["precision","recall", "f1"]:
            precision_recall_df[stat] = [f"$\\num{{{mean}}} \\pm \\num{{{std}}}$" for mean, std in zip(precision_recall_df[stat], precision_recall_df[f"{stat}_std"])]
        precision_recall_df = precision_recall_df.drop(columns = [col for col in precision_recall_df.columns if "std" in col])
        prf_style = precision_recall_df.style
        print(prf_style.to_latex())
            
        #print("transition matrix: ", model.crf.transitions.detach().cpu())
 
        run.log({"val_acc_top_1":acc_top_1_stats.mean, "val_acc_top_1_std":acc_top_1_stats.std_dev, "val_acc_top_5":acc_top_5_stats.mean, "val_acc_top_5_std":acc_top_5_stats.std_dev, "val_acc_top_5":acc_top_5_stats.mean, "val_acc_top_5_std":acc_top_5_stats.std_dev})

        # ------------------------------------------------
        #highest_std = sorted(list(stats_dict.items()), key = lambda x: -1*(x[1].mean))[:10]
        #for embryo, _ in highest_std:
                       
                        


 
 
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a stage classifier using latents and their morphodynamic features.")
 
 
    parser.add_argument("--model-name", help="Model name. Must have already exported latents and cebra latents")
    parser.add_argument("--run-name", help="WandB run name.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")

    parser.add_argument("--latents",action="store_true", help="Use to include latents")
    parser.add_argument("--cebra", action="store_true", help="Use to derive features from cebra latents")

    parser.add_argument("--cebra-ps",action="store_true", help="Use to include path signatures from cebra latents")
    parser.add_argument("--umap-ps",action="store_true", help="Use to include path signatures from UMAP latents")
    parser.add_argument("--pca-ps",action="store_true", help="Use to include path signatures from PCA latents")

    parser.add_argument("--curvature",action="store_true", help="Use to include curvature")
    parser.add_argument("--velocity",action="store_true", help="Use to include velocity")
    parser.add_argument("--acceleration", action="store_true", help="Use to include acceleration")
    parser.add_argument("--distance-mat", action="store_true", help="Use to include distance to first frame")

  
    args = parser.parse_args()
 
    main(args.model_name, vars(args), lr=args.lr)

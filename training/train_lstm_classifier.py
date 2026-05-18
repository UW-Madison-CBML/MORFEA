import torch
from grade_lstm_dataset import GradeLSTMDataset
from grade_lstm_model import GradeLSTMClassifier
from torch.utils.data import DataLoader
import wandb
import os
import pandas as pd
import numpy as np
import math
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

def collate_padd(batch):
    signals = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    lengths = torch.tensor([len(s) for s in signals])
    
    signals_padded = torch.nn.utils.rnn.pad_sequence(
        signals, batch_first=True, padding_value=0.0
    )
    
    targets = torch.tensor(targets)
    
    return signals_padded, targets, lengths # Return all three
def recall_precision_f1(confusion_mat, i):
    recall = 0 if confusion_mat[:, i].sum() == 0 else confusion_mat[i,i]/confusion_mat[:, i].sum()
            
    precision = 0 if confusion_mat[i,:].sum() == 0 else confusion_mat[i,i]/ confusion_mat[i, :].sum()
    f1 = 0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    return recall, precision, f1

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
    "RS363-7",
    "JE021-4",
    "ZS435-6",
    "QC211-2",
    "GE1055-6",
    "LBS371-1-8",
    "CAV074-1",
    "RA803-4",
    ]

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


def main(model_name, features):
    te_lr = features['te_lr']
    icm_lr = features['icm_lr']
    run_name = features['run_name']
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        name = run_name,
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
    )
    
    torch.backends.cudnn.enabled = False
    KEEP_NA = False
    
    grade_options = ["A", "B", "C", "NA"] if KEEP_NA else ["A","B","C"]
    learning_rate = 0.00001
    metadata_df = pd.read_csv(os.path.abspath(f"latents/{model_name}.csv"), keep_default_na=(not KEEP_NA))
    latents = np.load(os.path.join("latents",f"{model_name}.npy"))
    lat_cols = [f"z_{i}" for i in range(latents.shape[1])]
    cebra_latents = np.load(os.path.join("cebra_latents", f"{model_name}.npy"))
    #normalize latents here
    #lat_mean = latents.mean(axis=0)
    #lat_std_dev = np.std(latents, axis=0) + 1e-8
    #latents =  (latents - lat_mean) / lat_std_dev

    umap = UMAP(n_components=8)
    pca = PCA(n_components=8)
    std_scaler = StandardScaler()
    umap_df = pd.DataFrame(umap.fit_transform(latents), columns=[f"umap_{i}" for i in range(8)], index=metadata_df.index)
    pca_df = pd.DataFrame(pca.fit_transform(std_scaler.fit_transform(latents)), columns=[f"pca_{i}" for i in range(8)], index=metadata_df.index)

    latents_df = pd.concat([metadata_df, pd.DataFrame(latents, columns=lat_cols, index=metadata_df.index), pd.DataFrame(cebra_latents, columns=["cebra_0", "cebra_1", "cebra_2"], index=metadata_df.index), umap_df, pca_df], axis=1)
    #latents_df = latents_df[latents_df['phase'].str.contains("tM|tSB|tB|tEB")] # just classify around the blastocyst stage
    
    te_graded = latents_df.dropna(subset=["TE"])["embryo_id"].unique().tolist()
    np.random.shuffle(te_graded)
    val_ratio = 0.15
    VAL_EMBRYOS = te_graded[:int(val_ratio * len(te_graded))] 
    for embryo in ["RS363-7","JE021-4","ZS435-6","QC211-2","GE1055-6","LBS371-1-8","CAV074-1","RA803-4",]: # remove a bunch of C's
        if embryo in VAL_EMBRYOS:
            VAL_EMBRYOS.remove(embryo)
    for embryo in ["PH664-7","JV227-2","LLN873-1","LA367-4","RA580-2","MC373-5","DV210-4"]: # add a bunch of C's
        VAL_EMBRYOS.append(embryo)
    for embryo in VAL_EMBRYOS:
        print(f"{embryo}: {latents_df[latents_df['embryo_id'] == embryo].iloc[0]['TE']}")
    mask = latents_df["embryo_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    val_df = latents_df[mask]
    print(f"val_ratio={val_ratio}, actual val ratio: {len(val_df)/len(latents_df)}")
    latents_df = latents_df[~mask]
    # latents df will have the grade columns already
    dataset_te = GradeLSTMDataset(latents_df, "TE", features, keep_na=KEEP_NA) 
    dataset_icm = GradeLSTMDataset(latents_df, "ICM", features, keep_na=KEEP_NA)
    dataset_te_val = GradeLSTMDataset(val_df, "TE", features, keep_na=KEEP_NA, return_whole_seqs=True) 
    dataset_icm_val = GradeLSTMDataset(val_df, "ICM", features, keep_na=KEEP_NA, return_whole_seqs=True)
    crit_te = torch.nn.CrossEntropyLoss(weight= torch.tensor([0.4,0.4,0.2], device=DEVICE))
    crit_icm = torch.nn.CrossEntropyLoss(weight= torch.tensor([0.4,0.4,0.2], device=DEVICE))
    model_te = GradeLSTMClassifier(len(dataset_te.lat_cols), keep_na=KEEP_NA)
    model_te = model_te.to(DEVICE)
    model_icm = GradeLSTMClassifier(len(dataset_icm.lat_cols), keep_na=KEEP_NA)
    model_icm = model_icm.to(DEVICE)
    epochs = 8 
    optimizer_te = torch.optim.Adam(model_te.parameters(), lr=te_lr, weight_decay=1e-4)
    optimizer_icm = torch.optim.Adam(model_icm.parameters(), lr=icm_lr, weight_decay=1e-4)


    loader_te = DataLoader(
        dataset_te,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_padd)
    loader_icm = DataLoader(
        dataset_icm,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True, 
        collate_fn=collate_padd)
    loader_te_val = DataLoader(
        dataset_te_val,
        batch_size=len(VAL_EMBRYOS),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_padd)
    loader_icm_val = DataLoader(
        dataset_icm_val,
        batch_size=len(VAL_EMBRYOS),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_padd)
    scheduler_te = CosineAnnealingLR(optimizer_te, len(loader_te) * epochs)
    scheduler_icm = CosineAnnealingLR(optimizer_icm, len(loader_icm) * epochs)

    for epoch in range(epochs):
        model_te.train(); model_icm.train()
        # TE grades
        for features, targets, lengths in loader_te:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE).long()
            label = model_te(features, lengths)
            loss = crit_te(label, targets)

            optimizer_te.zero_grad() 
            loss.backward() 
            optimizer_te.step()

            scheduler_te.step()
            run.log({"te": loss.item()})
        # ICM grades
        for features, targets, lengths in loader_icm:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE).long()
            
            label = model_icm(features, lengths)
            loss = crit_icm(label, targets)

            optimizer_icm.zero_grad() 
            loss.backward() 
            optimizer_icm.step()
            scheduler_icm.step()
            run.log({"icm": loss.item()})
    
        te_acc_stats = RunningStats()
        icm_acc_stats = RunningStats()

        model_te.eval(); model_icm.eval()
        te_confusion_mat = torch.zeros((3,3))
        icm_confusion_mat = torch.zeros((3,3))
        with torch.no_grad():
            for features, targets, lengths in loader_te_val:
                
                print(f"targets shape: {targets.shape}")

                print(f"features shape: {features.shape}")

                features = features.to(DEVICE)
                logits = model_te(features, lengths)
                
                preds = logits.cpu().argmax(dim=-1)
                
                te_acc_stats.push((preds == targets).sum().item()/targets.shape[0]) 
                
                targets = F.one_hot(targets, num_classes=3)
                preds = F.one_hot(preds, num_classes=3)


                te_confusion_mat += torch.einsum('ti,tj->ij', preds, targets) # this einsum calculates confusion mats


            for features, targets, lengths in loader_icm_val:
                features = features.to(DEVICE)
                logits = model_icm(features, lengths)

                preds = logits.cpu().argmax(dim=-1) 

                icm_acc_stats.push((preds == targets).sum().item()/targets.shape[0])
                targets = F.one_hot(targets, num_classes=3)
                preds = F.one_hot(preds, num_classes=3)

                icm_confusion_mat += torch.einsum('ti,tj->ij', preds, targets)
                
        
        for i, g in enumerate(grade_options):
            recall, precision, f1 = recall_precision_f1(te_confusion_mat, i)
            run.log({f"te_{g}_recall": recall,f"te_{g}_precision": precision,f"te_{g}_f1":f1})
            #---------------------------------------------------------------------------------- 
            
            recall, precision, f1 = recall_precision_f1(icm_confusion_mat, i)
            run.log({f"icm_{g}_recall": recall,f"icm_{g}_precision": precision,f"icm_{g}_f1":f1})
        print(grade_options)
        print("te confusion:"); print(te_confusion_mat)
        print("icm confusion:"); print(icm_confusion_mat)
        run.log({"te_acc_mean": te_acc_stats.mean, "te_acc_std":te_acc_stats.std_dev, "icm_acc_mean":icm_acc_stats.mean, "icm_acc_std":icm_acc_stats.std_dev})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train seq-> grade lstm classifier.")
 
 
    parser.add_argument("--name", help="Model name. Must have already exported latents")
    parser.add_argument("--run-name", help="Wandb run name")

    parser.add_argument("--latents",action="store_true", help="Use to include latents")
    parser.add_argument("--cebra-latents", action="store_true", help="Use to include distance to first frame")

    parser.add_argument("--curvature",action="store_true", help="Use to include curvature")
    parser.add_argument("--velocity",action="store_true", help="Use to include velocity")
    parser.add_argument("--acceleration", action="store_true", help="Use to include acceleration")
    parser.add_argument("--distance-mat", action="store_true", help="Use to include distance to first frame")

    parser.add_argument("--cebra-ps", action="store_true", help="Use to include distance to first frame")
    parser.add_argument("--umap-ps", action="store_true", help="Use to include distance to first frame")
    parser.add_argument("--pca-ps", action="store_true", help="Use to include distance to first frame")

    parser.add_argument("--te-lr", type=float, default=0.0001, help="TE Learning rate")
    parser.add_argument("--icm-lr", type=float, default=0.0001,help="ICM Learning rate")
  
    args = parser.parse_args()
 
    main(args.name, vars(args))


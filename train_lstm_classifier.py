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

def collate_fn_padd(batch):
    signals = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    lengths = torch.tensor([len(s) for s in signals])
    
    signals_padded = torch.nn.utils.rnn.pad_sequence(
        signals, batch_first=True, padding_value=0.0
    )
    
    targets = torch.tensor(targets)
    
    return signals_padded, targets, lengths # Return all three
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


def main(model_name):
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
    )
    
    KEEP_NA = False
    
    grade_options = ["A", "B", "C", "NA"] if KEEP_NA else ["A","B","C"]
    learning_rate = 0.00001
    latents_df = pd.read_csv(os.path.abspath(f"latents/{model_name}.csv")).rename(columns={"embryo_id":"cell_id"}).drop(columns=['TE',"ICM","grade1","grade2","te","icm"], axis=1, errors='ignore')
    latents = np.load(os.path.abspath(f"latents/{model_name}.npy"))
    #normalize latents here
    lat_mean = latents.mean(axis=0)
    lat_std_dev = np.std(latents, axis=0) + 1e-8
    latents = (latents - lat_mean) / lat_std_dev

    lat_cols = [f"z_{i}" for i in range(latents.shape[1])]
    latents_df[lat_cols] = latents
    
    grades_df = pd.read_csv(os.path.abspath(f"embryo_dataset_grades.csv"), keep_default_na=(not KEEP_NA))
    mask = latents_df["cell_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    val_df = latents_df[mask]
    print(len(val_df)/len(latents_df))
    latents_df = latents_df[~mask]

    dataset_te = GradeLSTMDataset(latents_df, grades_df, "TE", keep_na=KEEP_NA) 
    dataset_icm = GradeLSTMDataset(latents_df, grades_df, "ICM", keep_na=KEEP_NA)
    dataset_te_val = GradeLSTMDataset(val_df, grades_df, "TE", keep_na=KEEP_NA, return_whole_seqs=True) 
    dataset_icm_val = GradeLSTMDataset(val_df, grades_df, "ICM", keep_na=KEEP_NA, return_whole_seqs=True)
    lat_size = len(lat_cols)
    crit_te = torch.nn.CrossEntropyLoss()
    crit_icm = torch.nn.CrossEntropyLoss()
    model_te = GradeLSTMClassifier(lat_size, keep_na=KEEP_NA)
    model_te = model_te.to(DEVICE)
    model_icm = GradeLSTMClassifier(lat_size, keep_na=KEEP_NA)
    model_icm = model_icm.to(DEVICE)
    optimizer_te = torch.optim.Adam(model_te.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_icm = torch.optim.Adam(model_icm.parameters(), lr=learning_rate, weight_decay=1e-5)


    loader_te = DataLoader(
        dataset_te,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_padd)
    loader_icm = DataLoader(
        dataset_icm,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True, collate_fn=collate_fn_padd)
    loader_te_val = DataLoader(
        dataset_te_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True) # no collate for 1 batch
    loader_icm_val = DataLoader(
        dataset_icm_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    for epoch in range(20):
        model_te.train(); model_icm.train()
        for sig, te, lengths in loader_te:
            sig = sig.to(DEVICE).view(32,-1, lat_size)
            te = te.to(DEVICE).long()
            if -1 in te:
                continue 
            label = model_te(sig, lengths)
            loss = crit_te(label, te)

            optimizer_te.zero_grad() 
            loss.backward() 
            optimizer_te.step()
            run.log({"te": loss.item()})

        for sig, icm, lengths in loader_icm:
            sig = sig.to(DEVICE).view(32,-1, lat_size)
            icm = icm.to(DEVICE).long()
            if -1 in icm:
                continue
            
            label = model_icm(sig, lengths)
            loss = crit_icm(label, icm)

            optimizer_icm.zero_grad() 
            loss.backward() 
            optimizer_icm.step()
            run.log({"icm": loss.item()})
    
    te_acc_stats = RunningStats()
    icm_acc_stats = RunningStats()

    model_te.eval(); model_icm.eval()
    te_confusion_mat = np.zeros((3,3))
    icm_confusion_mat = np.zeros((3,3))
    with torch.no_grad():
        for sig, te, lengths in loader_te_val:
            sig = sig.to(DEVICE)
            logits = model_te(sig, lengths)
            
            preds = logits.cpu()[0].argmax(dim=1).item() # batch size 1
            te = te[0]
            te_acc_stats.push((preds == te).sum()) 
            te_confusion_mat[preds, te] += 1 
        for sig, icm, lengths in loader_icm_val:
            sig = sig.to(DEVICE)

            logits = model_icm(sig, lengths)
            preds = logits.cpu()[0].argmax(dim=1).item()  # batch size 1
            icm = icm[0]

            icm_acc_stats.push((preds == icm).sum())
            icm_confusion_mat[preds, icm] += 1 
    
    for i, g in enumerate(grade_options):
        te_recall = 0 if te_confusion_mat[:, g].sum() == 0 else te_confusion_mat[g,g]/te_confusion_mat[:, g].sum()
        
        te_precision = 0 if te_confusion_mat[g,:].sum() == 0 else te_confusion_mat[g,g]/ te_confusion_mat[g, :].sum()
        te_f1 = 0
        if (precision + recall) > 0:
            te_f1 = 2 * (precision * recall) / (precision + recall)
        run.log({f"te_{g}_recall": te_recall,f"te_{g}_precision": te_precision,f"te_{g}_f1":te_f1})
        #---------------------------------------------------------------------------------- 
        icm_recall = 0 if icm_confusion_mat[:, g].sum() == 0 else icm_confusion_mat[g,g]/icm_confusion_mat[:, g].sum()
        
        icm_precision = 0 if icm_confusion_mat[g,:].sum() == 0 else icm_confusion_mat[g,g]/ icm_confusion_mat[g, :].sum()
        icm_f1 = 0
        if (precision + recall) > 0:
            icm_f1 = 2 * (precision * recall) / (precision + recall)

        run.log({f"icm_{g}_recall": icm_recall,f"icm_{g}_precision": icm_precision,f"icm_{g}_f1":icm_f1})

    run.log({"te_acc_mean": te_acc_stats.mean, "te_acc_std":te_acc_stats.std_dev, "icm_acc_mean":icm_acc_stats.mean, "icm_acc_std":icm_acc_stats.std_dev})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()

    main(args.name)

import torch
from grade_lstm_dataset import GradeLSTMDataset
from grade_lstm_model import GradeLSTMClassifier
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


def main(model_name):
    torch.cuda.empty_cache()
    torch.autograd.detect_anomaly(True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
    )
    
    learning_rate = 0.001
    latents_df = pd.read_csv(os.path.abspath(f"latents/{model_name}.csv")).rename(columns={"embryo_id":"cell_id"}).drop(columns=['TE',"ICM","grade1","grade2","te","icm"], axis=1, errors='ignore')
    latents = np.load(os.path.abspath(f"latents/{model_name}.npy"))
    lat_cols = [f"z_{i}" for i in range(latents.shape[1])]
    latents_df[lat_cols] = latents

    grades_df = pd.read_csv(os.path.abspath(f"embryo_dataset_grades.csv"), keep_default_na=False)
    mask = latents_df["cell_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    val_df = latents_df[mask]
    print(len(val_df)/len(latents_df))
    latents_df = latents_df[~mask]


    dataset_te = GradeLSTMDataset(sigs_df, grades_df, "TE", keep_na=True) 
    dataset_icm = GradeLSTMDataset(sigs_df, grades_df, "ICM", keep_na=True)
    dataset_te_val = GradeLSTMDataset(val_df, grades_df, "TE", keep_na=True) 
    dataset_icm_val = GradeLSTMDataset(val_df, grades_df, "ICM", keep_na=True)
    lat_size = len(lat_cols)
    crit_te = torch.nn.CrossEntropyLoss()
    crit_icm = torch.nn.CrossEntropyLoss()
    model_te = GradeLSTMClassifier(lat_size, keep_na=True)
    model_te = model_te.to(DEVICE)
    model_icm = GradeLSTMClassifier(lat_size, keep_na=True)
    model_icm = model_icm.to(DEVICE)
    optimizer_te = torch.optim.Adam(model_te.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_icm = torch.optim.Adam(model_icm.parameters(), lr=learning_rate, weight_decay=1e-5)


    loader_te = DataLoader(
        dataset_te,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False)
    loader_icm = DataLoader(
        dataset_icm,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False)
    loader_te_val = DataLoader(
        dataset_te_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False)
    loader_icm_val = DataLoader(
        dataset_icm_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False)

    for epoch in range(20):
        model_te.train(); model_icm.train()
        for sig, te in loader_te:
            sig = sig.squeeze().to(DEVICE)
            te = te.squeeze().to(DEVICE).long()
            label = model_te(sig)
            loss = crit_te(label, te)

            optimizer_te.zero_grad() 
            loss.backward() 
            optimizer_te.step()
            run.log({"te": loss.item()})

        for sig, icm in loader_icm:
            sig = sig.squeeze().to(DEVICE)
            icm = icm.squeeze().to(DEVICE).long()

            label = model_icm(sig)
            loss = crit_icm(label, icm)

            optimizer_icm.zero_grad() 
            loss.backward() 
            optimizer_icm.step()
            run.log({"icm": loss.item()})
    
    te_loss_stats = RunningStats()
    icm_loss_stats = RunningStats()
    te_acc_stats = RunningStats()
    icm_acc_stats = RunningStats()

    model_te.eval(); model_icm.eval()
    with torch.no_grad():
        for sig, te in loader_te_val:
            sig = sig.to(DEVICE)
            te = te.to(DEVICE).long()
            logits = model_te(sig)
            loss = crit_te(logits, te)
            te_loss_stats.push(loss.item())
            
            # Calculate accuracy
            preds = logits.argmax(dim=1)  # Get predicted class (0, 1, or 2)
            te_acc_stats.push((preds == te).sum().item()/te.shape[0])
        for sig, icm in loader_icm_val:
            sig = sig.to(DEVICE)
            icm = icm.to(DEVICE).long()

            logits = model_icm(sig)
            loss = crit_icm(logits, icm)
            icm_loss_stats.push(loss.item())
    print("TE: " + str(te_loss_stats.mean) + " +- " + str(te_loss_stats.std_dev))
    print("ICM: " + str(icm_loss_stats.mean) + " +- " + str(icm_loss_stats.std_dev))
    print("TE Acc: " + str(te_acc_stats.mean) + " +- " + str(te_acc_stats.std_dev))
    print("ICM Acc: " + str(icm_acc_stats.mean) + " +- " + str(icm_acc_stats.std_dev))



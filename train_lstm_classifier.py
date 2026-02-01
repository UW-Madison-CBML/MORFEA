import torch
from grade_lstm_dataset import GradeLSTMDataset
from grade_lstm_model import GradeLSTMClassifier
from torch.utils.data import DataLoader
import wandb
import os
import pandas as pd
import numpy as np
import math
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
    sigs_df = pd.read_csv(os.path.abspath(f"signatures/{model_name}_sigs.csv")).rename(columns={"embryo_id":"cell_id"})
    grades_df = pd.read_csv(os.path.abspath(f"embryo_dataset_grades.csv"))
    mask = sigs_df["cell_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    val_df = sigs_df[mask]
    print(len(val_df)/len(sigs_df))
    sigs_df = sigs_df[~mask]


    dataset_te = SignatureDataset(sigs_df, grades_df, "TE", keep_na=True) 
    dataset_icm = SignatureDataset(sigs_df, grades_df, "ICM", keep_na=True)
    dataset_te_val = SignatureDataset(val_df, grades_df, "TE", keep_na=True) 
    dataset_icm_val = SignatureDataset(val_df, grades_df, "ICM", keep_na=True)
    sig_size = len([i for i in sigs_df.columns if i[:2] == "s_"])
    crit_te = torch.nn.CrossEntropyLoss()
    crit_icm = torch.nn.CrossEntropyLoss()
    model_te = SignatureClassifier(sig_size)
    model_te = model_te.to(DEVICE)
    model_icm = SignatureClassifier(sig_size)
    model_icm = model_icm.to(DEVICE)
    optimizer_te = torch.optim.Adam(model_te.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_icm = torch.optim.Adam(model_icm.parameters(), lr=learning_rate, weight_decay=1e-5)


    loader_te = DataLoader(
        dataset_te,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,

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
    # 'batch' is a list of tuples (sequence, label)
    # Assume sequences are 1D tensors for simplicity, e.g., torch.tensor([1, 2, 3])

    # Separate sequences and labels
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)

    # Get original lengths
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)

    # Pad the sequences to the maximum length in the batch.
    # batch_first=True makes the output tensor shape (batch_size, max_length, feature_dim)
    # The default padding_value is 0.0
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

    # Note: It's important to sort the batch by length in descending order for
    # pack_padded_sequence to work correctly.
    lengths, sorted_idx = lengths.sort(descending=True)
    padded_sequences = padded_sequences[sorted_idx]
    labels = labels[sorted_idx]

    return padded_sequences, labels, lengths
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


    dataset_te = GradeLSTMDataset(latents_df, grades_df, "TE", keep_na=True) 
    dataset_icm = GradeLSTMDataset(latents_df, grades_df, "ICM", keep_na=True)
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
        batch_size=10,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_padd)
    loader_icm = DataLoader(
        dataset_icm,
        batch_size=10,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False, collate_fn=collate_fn_padd)
    loader_te_val = DataLoader(
        dataset_te_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False, collate_fn=collate_fn_padd)
    loader_icm_val = DataLoader(
        dataset_icm_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False, collate_fn=collate_fn_padd)

    for epoch in range(20):
        model_te.train(); model_icm.train()
        for sig, te in loader_te:
            sig = sig.to(DEVICE).view(10,-1, lat_size)
            te = te.to(DEVICE).long()
            if -1 in te:
                continue 
            print(sig.shape)
            label = model_te(sig)
            loss = crit_te(label, te)

            optimizer_te.zero_grad() 
            loss.backward() 
            optimizer_te.step()
            run.log({"te": loss.item()})

        for sig, icm in loader_icm:
            sig = sig.to(DEVICE).view(10,-1, lat_size)
            icm = icm.to(DEVICE).long()
            if -1 in icm:
                continue
            
            print(sig.shape)
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()

    main(args.name)

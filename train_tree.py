from signature_dataset import SignatureDataset
import os
import pandas as pd
import numpy as np
import math
from sklearn import tree
from torch.utils.data import DataLoader
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
def main(model_name):

    keep_na = True 
    learning_rate = 0.001
    sigs_df = pd.read_csv(os.path.abspath(f"signatures/{model_name}_sigs.csv")).rename(columns={"embryo_id":"cell_id"})
    grades_df = pd.read_csv(os.path.abspath(f"embryo_dataset_grades.csv"), keep_default_na=(not keep_na))
    mask = sigs_df["cell_id"].str.contains("|".join(VAL_EMBRYOS), regex=True)
    val_df = sigs_df[mask]
    print(len(val_df)/len(sigs_df))
    sigs_df = sigs_df[~mask]


    dataset_te = SignatureDataset(sigs_df, grades_df, "TE", keep_na=keep_na) 
    dataset_icm = SignatureDataset(sigs_df, grades_df, "ICM", keep_na=keep_na)
    dataset_te_val = SignatureDataset(val_df, grades_df, "TE", keep_na=keep_na) 
    dataset_icm_val = SignatureDataset(val_df, grades_df, "ICM", keep_na=keep_na)
    sig_size = len([i for i in sigs_df.columns if i[:2] == "s_"])


    loader_te = DataLoader(
        dataset_te,
        batch_size=len(dataset_te.df),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    loader_icm = DataLoader(
        dataset_icm,
        batch_size=len(dataset_icm.df),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    loader_te_val = DataLoader(
        dataset_te_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    loader_icm_val = DataLoader(
        dataset_icm_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    model_te = tree.DecisionTreeClassifier()
    model_icm = tree.DecisionTreeClassifier()

    def get_full_data(loader):
        sigs, labels = [], []
        for sig, label in loader:
            sigs.append(sig.numpy())
            labels.append(label.numpy())
        return np.concatenate(sigs), np.concatenate(labels)

    X_te, y_te = get_full_data(loader_te)
    X_icm, y_icm = get_full_data(loader_icm)

    model_te.fit(X_te, y_te)
    model_icm.fit(X_icm, y_icm)

    te_acc_stats = RunningStats()
    icm_acc_stats = RunningStats()

    for sig, te in loader_te_val:
        preds = model_te.predict(sig.numpy()) 
        
        acc = (preds == te.numpy()).mean()
        te_acc_stats.push(acc)

    for sig, icm in loader_icm_val:
        preds = model_icm.predict(sig.numpy())
        acc = (preds == icm.numpy()).mean()
        icm_acc_stats.push(acc)
    print("TE Acc: " + str(te_acc_stats.mean) + " +- " + str(te_acc_stats.std_dev))

    print("ICM Acc: " + str(icm_acc_stats.mean) + " +- " + str(icm_acc_stats.std_dev))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="A simple script using argparse to greet a user.")


    parser.add_argument("--name", help="Model name. Must have already exported latents")

    args = parser.parse_args()

    main(args.name)

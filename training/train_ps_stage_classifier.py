import torch
from torch.utils.data import Dataloader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from ps_stage_dataset import PathSigStageDataset
import torch.nn.functional as F
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class MLP(torch.nn.Module):
    def __init__(self, in_size):
        self.lin1 = torch.nn.Linear(in_size, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, 128)
        self.lin4 = torch.nn.Linear(128, len(PathSigStageDataset.PHASES))
        self.dropout = torch.nn.Dropout(0.3)
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.lin4(F.relu(self.lin3(x)))
        return x

def main(model_name):
    # hyperparameters 
    lr = 0.0004
    epochs = 8
    batch_size = 1024
    pca_dim = 8
    
    # load csv 
    metadata_df = pd.read_csv(os.path.join("latents", f"{model_name}.csv"))
    latents = np.load(os.path.join("latents", f"{model_name}.npy"))

    latents_df = pd.DataFrame(latents, columns = [f"z_{i}" for i in range(latents.shape[1])], index=metadata_df.index)
    pca_df = pd.DataFrame(PCA(n_components=pca_dim).fit_transform(StandardScaler().fit_transform(latents)), columns = [f"pca_{i}" for i in range(pca_dim)], index=metadata_df.index)
    df = pd.concat([metadata_df, latents_df, pca_df], axis=1)
    
    # shuffle df
    df = df.sample(frac=1, replace=False)
    
    train_df = df.iloc[:int(0.8 * len(df))]
    val_df = df.iloc[int(0.8 * len(df)):]

    loader = DataLoader(
        PathSigStageDataset(train_df, ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )    
    val_loader = DataLoader(
        PathSigStageDataset(val_df, ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )


    

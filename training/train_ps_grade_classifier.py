import torch
from torch.utils.data import DataLoader
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from ps_grade_dataset import PathSigGradeDataset
import torch.nn.functional as F
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from stats_utils import prfcm, disp_cm

class MLP(torch.nn.Module):
    def __init__(self, in_size):
        self.stage_embedding = torch.nn.Embedding(len(PathSigGradeDataset.PHASES), 3)
        self.lin1 = torch.nn.Linear(in_size + 3, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, 128)
        self.lin4 = torch.nn.Linear(128, len(PathSigGradeDataset.GRADES))
        self.dropout = torch.nn.Dropout(0.3)
    def forward(self, x, stage_classes):
        x = torch.cat([x, self.stage_embedding(stage_classes)], dim=1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.dropout(x)
        x = self.lin4(F.relu(self.lin3(x)))
        return x

# agg function for stage level confusion mats
def cm_agg_plot(group):
    gt = torch.from_numpy(group["gt"].to_numpy())
    pred = torch.from_numpy(group["pred"].to_numpy())
    _, cm = prfcm(gt, pred, 3)
    fig, ax = plt.subplots(figsize=(8,6))
    disp_cm(cm,PathSigGradeDataset.GRADES, fig, ax)
    img = wandb.Image(fig)
    plt.close(fig)
    return img

def main(model_name):
    # hyperparameters 
    lr = 0.0004
    epochs = 8
    batch_size = 1024
    pca_dim = 8
    depth = 3
    time_offsets = 5.0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
    # wandb 
    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="IVF-Training",
        name="ps_stage",
        config={
            "lr": lr,
            "batch_size":batch_size
        },
    )   
    # load csv 
    metadata_df = pd.read_csv(os.path.join("latents", f"{model_name}.csv"))
    latents = np.load(os.path.join("latents", f"{model_name}.npy"))

    pca_df = pd.DataFrame(PCA(n_components=pca_dim).fit_transform(StandardScaler().fit_transform(latents)), columns = [f"pca_{i}" for i in range(pca_dim)], index=metadata_df.index)
    df = pd.concat([metadata_df, pca_df], axis=1)
    # drop na's 
    df = df[df['TE'].isin(["A", "B", "C"])]
    
    # shuffle df
    df = df.sample(frac=1, replace=False)
    
    train_df = df.iloc[:int(0.8 * len(df))]
    val_df = df.iloc[int(0.8 * len(df)):]
    
    ds = PathSigGradeDataset(train_df, time_offsets, pca_dim=pca_dim, depth=depth)
    val_ds = PathSigGradeDataset(val_df, time_offsets, pca_dim=pca_dim, depth=depth)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )

    model = MLP(ds.num_features) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for features, stage_classes, targets in loader:
            optimizer.zero_grad()
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            stage_classes = stage_classes.to(DEVICE)
            logits = model(features, stage_classes)  
            loss = crit(logits, targets)
            loss.backward()
            optimizer.step()
            run.log({"loss":loss.item()})
        dfs = []    
        with torch.no_grad():
            for features, stage_classes, targets in val_loader:
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                stage_classes = stage_classes.to(DEVICE)
                logits = model(features, stage_classes)  
                loss = crit(logits, targets)
                run.log({"val_loss":loss.item()})
                pred = logits.cpu().argmax(dim=1).numpy()
                gt = targets.cpu().numpy()
                
                stage_classes = [PathSigGradeDataset.PHASES.index(s) for s in stage_classes.cpu().numpy()]
                dfs.append(pd.DataFrame({"pred": pred, "gt":gt, "stage_class":stage_classes}))
        image_dict = {}
        results_df = pd.concat(dfs, axis=0, ignore_index=True)
        image_dict["all_stage_cm"] = cm_agg_plot(results_df)
        images = results_df.groupby("stage_class").apply(cm_agg_plot).reset_index()
        for stage, img in images.items():
            image_dict[f"{stage}_cm"] = img
        run.log(image_dict)
            

            
        
if __name__ == "__main__":
    import sys
    main(sys.argv[1]) 

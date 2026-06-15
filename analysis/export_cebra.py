import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os
from torch.utils.data import DataLoader
from dataset_ivf_embryo import IVFEmbryoDataset
from datetime import datetime, timedelta
import cebra
from cebra import CEBRA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def export_latents_to_csv(model_name):
    cebra_time_model = CEBRA(model_architecture="offset10-model-mse",
                        batch_size=1024,
                        learning_rate=1e-5,
                        temperature=13,
                        output_dimension=3,
                        num_hidden_units=128,
                        max_iterations=5000,
                        distance="euclidean",
                        conditional="time",
                        device="cuda_if_available",
                        verbose=True,
                        time_offsets=10)

    lat_df = pd.read_csv(os.path.abspath(f"latents/{model_name}.csv"), keep_default_na=False).rename(columns={"cell_id":"embryo_id"})
    lat_np = np.load(os.path.abspath(f"latents/{model_name}.npy"))
    if(len(lat_df) != lat_np.shape[0]):
        raise ValueError("keys and values sizes do not match")
    lat_columns = [f"z_{i}" for i in range(lat_np.shape[1])]
    values_df = pd.DataFrame(lat_np, columns=lat_columns)
    df = pd.concat([lat_df, values_df], axis = 1)
        
    val_mask = df["embryo_id"].str.contains("|".join(df[~(df["ICM"] == "NA")]["embryo_id"].unique().tolist()), regex=True)
    train_df = df[~val_mask]    
    cebra_latents = []
    cebra_labels = []
    offset = 0
    for embryo_id, group in train_df.groupby("embryo_id"):
            traj = group[lat_columns].to_numpy()
            cebra_latents.append(traj)
            cebra_labels.append((np.arange(len(traj)) + offset).reshape(-1, 1).astype(np.float32))
            offset += len(traj) + 10000
    cebra_time_model.fit(np.concatenate(cebra_latents, axis=0), np.concatenate(cebra_labels, axis=0))

    cebra_out_lats = cebra_time_model.transform(df[lat_columns].to_numpy())

    print("cebra:", len(cebra_out_lats))
    print("latents:", len(df))
    np.save(f"{model_name}.npy",cebra_out_lats)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A simple script using argparse.")

    parser.add_argument("--name", type=str, help="Name of the model", default="JensLundsgaard/IVF-ConvLSTM-Model-2025-12-15")

    args = parser.parse_args()


    export_latents_to_csv(
        args.name,
    )

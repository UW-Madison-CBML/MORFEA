# export_latents_unique.py - Export latent embeddings to CSV
import numpy as np
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from dataset_ivf import IVFSequenceDataset
from model import Model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def export_latents_to_csv(checkpoint="model_weights.pth", output_csv="latents.csv"):
    """
    Export latent embeddings from the model to a CSV file.

    Each row contains: cell_id, time_step, and 2000 latent dimensions

    Args:
        checkpoint: Path to model weights file
        output_csv: Output CSV filename
    """
    # Load dataset
    if not os.path.exists("index.csv"):
        raise FileNotFoundError("index.csv not found. Make sure dataset index is available.")

    ds = IVFSequenceDataset("index.csv", resize=500, norm="minmax01")
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # Load model
    print(f"Loading model from: {checkpoint}")
    model = Model()
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")

    # Collect all latent vectors
    all_latents = []
    cell_ids = []
    time_steps = []

    print(f"\nExtracting latent embeddings from {len(ds)} sequences...")
    for idx, (embryo_vol, empty_well_vol, sample_vol) in enumerate(loader):
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(ds)} sequences")

        # Get metadata from dataframe
        row = ds.df.iloc[idx]
        cell_id = row.get("cell_id", f"cell_{idx}")
        start_idx = row.get("start_idx", 0)

        # Reshape embryo volume like in training
        embryo_vol = embryo_vol.view(-1, 1, 500, 500).to(DEVICE)

        model.eval()
        with torch.no_grad():
            _, z_seq = model(embryo_vol, empty_well=False)

        # z_seq shape: (T, 2000) where T is number of frames
        z = z_seq.cpu().numpy()  # Shape: (T, 2000)

        # Add one row per time step
        for t in range(z.shape[0]):
            all_latents.append(z[t])
            cell_ids.append(cell_id)
            time_steps.append(int(t + start_idx))

    # Create DataFrame
    print(f"\nCreating CSV with {len(all_latents)} samples...")
    latent_columns = [f"z_{i}" for i in range(2000)]

    latents_array = np.array(all_latents)  # Shape: (num_samples, 2000)
    df = pd.DataFrame(latents_array, columns=latent_columns)
    df.insert(0, "cell_id", cell_ids)
    df.insert(1, "time_step", time_steps)
    normed_df = pd.DataFrame(columns = df.columns)
    # for each key (cell_id, time_step), we should only have one row but there are two because of sequence overlap. Latent vector is no longer a function of sequence so theoretically it won't matter whether we average or take the first of the two
    normed_df = df.groupby(['cell_id', 'time_step']).agg({
      **{f'z_{i}': 'mean' for i in range(2000)}
    }).reset_index().sort_values(by=['cell_id','time_step'], ascending=[True, True])
    # Save to CSV
    normed_df.to_csv(output_csv, index=False)
    print(f"Latent embeddings saved to: {output_csv}")
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  File size: {os.path.getsize(output_csv) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    export_latents_to_csv(checkpoint="model_weights.pth", output_csv="latents.csv")


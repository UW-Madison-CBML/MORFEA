# export_latents.py - Export latent embeddings to CSV
import numpy as np
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from dataset_ivf import IVFSequenceDataset
from raffael_model import ConvLSTMAutoencoder
from huggingface_hub import login, HfApi
from datetime import datetime, timedelta

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def load_model(model_name=None, days_back=30):
    """
    Load ConvLSTM model from HuggingFace or local weights.

    Args:
        model_name: Specific model name to load (e.g., "JensLundsgaard/IVF-ConvLSTM-Model-2025-12-15")
                   If None, will search for most recent models
        days_back: Number of days to search back for models

    Returns:
        Loaded model
    """
    api = HfApi()
    model = None
    model_loaded = False

    # Try specific model name first
    if model_name:
        try:
            print(f"Attempting to load model: {model_name}")
            model = ConvLSTMAutoencoder.from_pretrained(model_name)
            print(f"Successfully loaded model: {model_name}")
            model_loaded = True
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")

    # If specific model failed or not provided, search for recent models
    if not model_loaded:
        print(f"Searching for models from the last {days_back} days...")
        for days in range(days_back):
            date_label = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            model_name = f"JensLundsgaard/IVF-ConvLSTM-Model-{date_label}"
            try:
                print(f"  Trying {model_name}...")
                model = ConvLSTMAutoencoder.from_pretrained(model_name)
                print(f"Successfully loaded model from {date_label}")
                model_loaded = True
                break
            except Exception as e:
                if days < days_back - 1:
                    continue
                else:
                    print(f"Failed to find model within {days_back} days. Last error: {e}")

    # Fall back to local weights if available
    if not model_loaded or model is None:
        print("Could not find any model on HuggingFace, trying local weights...")
        model = ConvLSTMAutoencoder(
            seq_len=50,
            input_channels=1,
            encoder_hidden_dim=256,
            encoder_layers=2,
            decoder_hidden_dim=128,
            decoder_layers=2,
            latent_size=4096,
            use_classifier=False,
            num_classes=2
        )

        if os.path.exists("convlstm_model_weights.pth"):
            try:
                model.load_state_dict(torch.load("convlstm_model_weights.pth", map_location=DEVICE, weights_only=True))
                print("Loaded local convlstm_model_weights.pth")
                model_loaded = True
            except Exception as e:
                print(f"Failed to load local weights: {e}")

        if not model_loaded:
            raise Exception("Could not load model from HuggingFace or local weights")

    return model


def export_latents_to_csv(model_name="JensLundsgaard/IVF-ConvLSTM-Model-2025-12-15", output_csv="latents.csv"):
    """
    Export latent embeddings from the ConvLSTM model to a CSV file.

    Each row contains: cell_id, time_step, and 4096 latent dimensions

    Args:
        model_name: HuggingFace model name or None to search for recent models
        output_csv: Output CSV filename
    """
    # Login to HuggingFace
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_KEY")
    if hf_token:
        login(hf_token)

    # Load dataset
    if not os.path.exists("index.csv"):
        raise FileNotFoundError("index.csv not found. Make sure dataset index is available.")

    ds = IVFSequenceDataset("index.csv", resize=128, norm="minmax01")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Load model
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    model = load_model(model_name=model_name, days_back=30)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded successfully on {DEVICE}!")
    print(f"{'='*60}\n")

    # Collect all latent vectors
    all_latents = []
    cell_ids = []
    time_steps = []

    print(f"Extracting latent embeddings from {len(ds)} sequences...")
    print(f"{'='*60}")
    num_latents = 0
    for idx, (embryo_vol, _, _) in enumerate(loader):
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(ds)} sequences")

        # Get metadata from dataframe
        row = ds.df.iloc[idx]
        cell_id = row.get("cell_id", f"cell_{idx}")
        start_idx = row.get("start_idx", 0)

        # embryo_vol is already (B=1, T, 1, 128, 128) from dataloader
        embryo_vol = embryo_vol.to(DEVICE)

        with torch.no_grad():
            _, z_seq = model(embryo_vol)  # z_seq: (B=1, T, 4096)
        if(num_latents != z_seq.shape[2]):
            num_latents = z_seq.shape[2] 

        # Extract the batch dimension
        z = z_seq[0].cpu().numpy()  # Shape: (T, 4096)

        # Add one row per time step
        for t in range(z.shape[0]):
            all_latents.append(z[t])
            cell_ids.append(cell_id)
            time_steps.append(int(t + start_idx))

    # Create DataFrame
    print(f"\n{'='*60}")
    print(f"Creating CSV with {len(all_latents)} samples...")
    latent_columns = [f"z_{i}" for i in range(num_latents)]

    latents_array = np.array(all_latents)  # Shape: (num_samples, 4096)
    df = pd.DataFrame(latents_array, columns=latent_columns)
    df.insert(0, "cell_id", cell_ids)
    df.insert(1, "time_step", time_steps)

    # De-duplicate: for each (cell_id, time_step) key, average if there are duplicates
    # (due to sequence overlap in the dataset)
    print("De-duplicating overlapping sequences...")
    normed_df = df.groupby(['cell_id', 'time_step']).agg({
        **{f'z_{i}': 'mean' for i in range(num_latents)}
    }).reset_index().sort_values(by=['cell_id', 'time_step'], ascending=[True, True])
    latent_data = normed_df[lat_columns].values  # Shape: (num_rows, 4096)

    # Save the latent data as npy
    np.save(model_name + '.npy', latent_data)

    # Save cell_id and timestep as CSV
    metadata = df[['cell_id', 'time_step']]
    metadata.to_csv(model_name + '.csv', index=False)

    # Save to CSV
    # normed_df.to_csv(output_csv, index=False)
    print(f"{'='*60}")
    print(f"✓ Latent embeddings saved to: {output_csv}")
    print(f"  Total samples (after deduplication): {len(normed_df)}")
    print(f"  Original samples (before deduplication): {len(df)}")
    print(f"  Columns: cell_id, time_step, z_0...z_4095 ({len(normed_df.columns)} total)")
    print(f"  File size: {os.path.getsize(output_csv) / 1024 / 1024:.2f} MB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A simple script using argparse.")

    parser.add_argument("--name", type=str, help="Name of the model", default="JensLundsgaard/IVF-ConvLSTM-Model-2025-12-15")

    args = parser.parse_args()
    export_latents_to_csv(
        model_name=args.name,
        output_csv="latents.csv"
    )

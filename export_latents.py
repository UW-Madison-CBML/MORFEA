# export_latents.py - Export latent embeddings to CSV
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os
from torch.utils.data import DataLoader
from dataset_ivf_embryo import IVFEmbryoDataset
from raffael_model import ConvLSTMAutoencoder
from huggingface_hub import login, HfApi
from datetime import datetime, timedelta

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def addAnnotations(group_name, group, annotations_dir):
    annotation_file = os.path.join(annotations_dir, f"{group_name}_phases.csv")
    df = pd.read_csv(annotation_file, names=['stage_id', 'stage_begin', 'stage_end'])
    new_column = []

    new_column += ["pre_phase"] * (df.iloc[0]["stage_begin"] - 1)
    col_len_seq = []
    for index, row in df.iterrows():
        new_column += [row["stage_id"]] * (row["stage_end"] - row["stage_begin"]+1)
        col_len_seq.append(len(new_column))

    


    new_column += ["post_phase"] * (len(group) - len(new_column))
    new_column = new_column[:len(group)]
    
    group["phase"] = new_column

    
    return group

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
            model = ConvLSTMAutoencoder.from_pretrained("JensLundsgaard/"+model_name)
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


def export_latents_to_csv(model_name="JensLundsgaard/IVF-ConvLSTM-Model-2025-12-15",
                          index_csv="index_embryo.csv",
                          output_csv="latents.csv",
                          limit=None):
    """
    Export latent embeddings from the ConvLSTM model to a CSV file.

    Each row contains: embryo_id, time_step, and latent dimensions

    Args:
        model_name: HuggingFace model name or None to search for recent models
        index_csv: Input index CSV file (default: index_embryo.csv)
        output_csv: Output CSV filename
        limit: Maximum number of embryos to process (None = process all)
    """
    # Login to HuggingFace
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_KEY")
    if hf_token:
        login(hf_token)

    # Load dataset
    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"{index_csv} not found. Run build_index_embryo.py first.")

    ds = IVFEmbryoDataset(index_csv, resize=128, norm="minmax01")

    # Limit number of embryos if specified
    if limit is not None and limit > 0:
        original_len = len(ds.df)
        ds.df = ds.df.iloc[:limit]
        print(f"Limited dataset from {original_len} to {len(ds.df)} embryos")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

    # Load model
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    model = load_model(model_name=model_name, days_back=30)
    
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded successfully on {DEVICE}!")
    print(f"{'='*60}\n")
    model = model.to(DEVICE)

    # Get outputs from both models
    print("testing cosine similariy on half precision model") 
    for idx, embryo_vol in enumerate(loader):
        if idx % 10 != 0:
            continue
        rand_index = np.random.randint(embryo_vol.shape[1]-10)
        embryo_vol = embryo_vol.to(DEVICE)
        embryo_vol_half = embryo_vol.half()
        with torch.no_grad():
            _, output_full = model(embryo_vol[:,rand_index:rand_index+10])
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):    
                _, output_half = model(embryo_vol[:,rand_index:rand_index+10])

        # Ensure outputs are in the same precision for similarity calculation
        output_half_fp32 = output_half.to(torch.float32)
        similarity_outputs = F.cosine_similarity(output_full, output_half_fp32, dim=1)

        print(f"Average Cosine Similarity of model outputs: {similarity_outputs.mean().item()}")
        if(similarity_outputs.mean().item() < 0.98):
            raise ValueError("bad cosine similarity")
    # Collect all latent vectors
    all_latents = []
    embryo_ids = []
    time_steps = []

    print(f"Extracting latent embeddings from {len(ds)} embryos...")
    print(f"{'='*60}")
    num_latents = 0
    for idx, embryo_vol in enumerate(loader):
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(ds)} embryos")

        # Get metadata from dataframe
        row = ds.df.iloc[idx]
        embryo_id = row.get("embryo_id", f"embryo_{idx}")

        # embryo_vol is already (B=1, T, 1, 128, 128) from dataloader
        embryo_vol = embryo_vol.to(DEVICE).half()

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):    
                _, z_seq = model(embryo_vol)  # z_seq: (B=1, T, 4096)
        if(num_latents != z_seq.shape[2]):
            num_latents = z_seq.shape[2] 
        if torch.isinf(z_seq).any():
            raise ValueError("WARNING: Infinite values detected in FP16 latents! (Overflow)")

        # Extract the batch dimension
        z = z_seq[0].cpu().half().numpy()
        # After you are done with the large tensors in the loop:
        del embryo_vol
        del z_seq
        # Add one row per time step
        for t in range(z.shape[0]):
            all_latents.append(z[t])
            embryo_ids.append(embryo_id)
            time_steps.append(int(t))

    # Create DataFrame
    print(f"\n{'='*60}")
    print(f"Creating CSV with {len(all_latents)} samples...")
    latent_columns = [f"z_{i}" for i in range(num_latents)]

    latents_data = np.array(all_latents, dtype=np.float16)  # Shape: (num_samples, latent_size)
    df = pd.DataFrame()
    df.insert(0, "embryo_id", embryo_ids)
    df.insert(1, "time_step", time_steps)

    # Since each embryo is loaded once, there should be no duplicates
    print(f"Total samples: {len(df)}")
    print(f"Unique embryos: {df['embryo_id'].nunique()}")
    normed_df = df

    # Save the latent data as npy
    np.save(model_name + '.npy', latents_data)

    # Save embryo_id and timestep as CSV, optionally joined with grades
    metadata = normed_df[['embryo_id', 'time_step']]

    # Try to load and join with embryo grades
    grades_file = "embryo_dataset_grades.csv"
    if os.path.exists(grades_file):
        print(f"Loading grades from {grades_file}...")
        grades_df = pd.read_csv(grades_file, keep_default_na=False)

        # Join on embryo_id = video_name
        metadata = metadata.merge(
            grades_df,
            left_on='embryo_id',
            right_on='video_name',
            how='left'
        )

        # Drop the duplicate video_name column if it exists
        if 'video_name' in metadata.columns:
            metadata = metadata.drop(columns=['video_name'])

        print(f"Joined with grades. Columns: {list(metadata.columns)}")
    else:
        print(f"Grades file {grades_file} not found, skipping grade join")
    annotations_dir = "embryo_dataset_annotations"
    metadata = metadata.groupby("embryo_id", group_keys = False).apply(lambda group:addAnnotations(group.name,group,annotations_dir)).reset_index()
    metadata.to_csv(model_name + '.csv', index=False)
    torch.cuda.empty_cache()
    print(f"\n{'='*60}")
    print("EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  Latent embeddings saved to: {model_name}.npy")
    print(f"  Metadata saved to: {model_name}.csv")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A simple script using argparse.")

    parser.add_argument("--name", type=str, help="Name of the model", default="JensLundsgaard/IVF-ConvLSTM-Model-2025-12-15")
    parser.add_argument("--index", type=str, help="Input index CSV file", default="index_embryo.csv")
    parser.add_argument("--limit", type=int, default=0,
                       help="Maximum number of embryos to process (default: 25, use 0 for all)")

    args = parser.parse_args()

    # Convert 0 to None for processing all embryos
    limit = None if args.limit == 0 else args.limit

    export_latents_to_csv(
        model_name=args.name,
        index_csv=args.index,
        output_csv="latents.csv",
        limit=limit
    )

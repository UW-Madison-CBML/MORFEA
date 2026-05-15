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
from transformers import VideoMAEModel, AutoImageProcessor
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

    
def export_femi_latents_to_csv(index_csv="index_embryo.csv",
                          output_csv="latents.csv",
                          limit=None):
    # Login to HuggingFace
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_KEY")
    if hf_token:
        login(hf_token)


    processor = AutoImageProcessor.from_pretrained("ihlab/FEMI")
    model = VideoMAEModel.from_pretrained("ihlab/FEMI")

        
    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"{index_csv} not found. Run build_index_embryo.py first.")

    ds = IVFEmbryoDataset(pd.read_csv(index_csv), resize=128, norm="minmax01")

    # Limit number of embryos if specified
    if limit is not None and limit > 0:
        original_len = len(ds.df)
        ds.df = ds.df.iloc[:limit]
        print(f"Limited dataset from {original_len} to {len(ds.df)} embryos")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)

    model = model.to(DEVICE)
    model.eval()

    all_latents = []
    embryo_ids = []
    time_steps = []

    print(f"Extracting latent embeddings from {len(ds)} embryos...")
    print(f"{'='*60}")
    num_latents = 0
    for idx, embryo_vol in enumerate(loader):
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(ds)} embryos")
        row = ds.df.iloc[idx] 
        embryo_imgs = [Image.open(img) for img in row["embryo_paths"].split("|")]
        inputs = processor(images=embryo_imgs, return_tensors="pt")
        with torch.no_grad():
            encoder_outputs = model.vit(**inputs)
        latents = encoder_outputs.last_hidden_state
        print(latents.shape)

        z = z_seq[0].cpu().float().numpy()
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

    latents_data = np.array(all_latents, dtype=np.float32)  # Shape: (num_samples, latent_size)
    df = pd.DataFrame()
    df.insert(0, "embryo_id", embryo_ids)
    df.insert(1, "time_step", time_steps)

    # Since each embryo is loaded once, there should be no duplicates
    print(f"Total samples: {len(df)}")
    print(f"Unique embryos: {df['embryo_id'].nunique()}")
    normed_df = df

    # Save the latent data as npy
    np.save('femi.npy', latents_data)

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
    metadata.to_csv('femi.csv', index=False)
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

    export_femi_latents_to_csv(
        index_csv=args.index,
        output_csv="latents.csv",
        limit=limit
    )

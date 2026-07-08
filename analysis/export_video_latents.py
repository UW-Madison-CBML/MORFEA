# export_video_latents.py - Export latent videos from Gomez et al. to latents via a model on HF
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os
from torch.utils.data import DataLoader
from dataset_ivf_embryo import IVFEmbryoDataset
from ae_model import ConvLSTMAutoencoder
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


def export_video_latents(model, ds, latent_size = None):
    if(latent_size is None):
        latent_size = model.latent_size
        

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=16, pin_memory=True) 

    model.eval()
    print(f"Model loaded successfully on {DEVICE}!")
    print(f"{'='*60}\n")

    # Get outputs from both models
    """print("testing cosine similariy on half precision model") 
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
            raise ValueError("bad cosine similarity")"""
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

        row = ds.df.iloc[idx]
        embryo_id = row.get("embryo_id", f"embryo_{idx}")

        embryo_vol = embryo_vol.to(DEVICE) #.half()

        with torch.no_grad():
            #with torch.amp.autocast(device_type='cuda', dtype=torch.float16):    
            _, z_seq = model(embryo_vol)  
            z_seq = z_seq[:,:,:latent_size]
        if(num_latents != z_seq.shape[2]):
            num_latents = z_seq.shape[2] 
        if torch.isinf(z_seq).any():
            raise ValueError("WARNING: Infinite values detected in FP16 latents! (Overflow)")

        z = z_seq[0].cpu().float().numpy()
        del embryo_vol
        del z_seq
        for t in range(z.shape[0]):
            all_latents.append(z[t])
            embryo_ids.append(embryo_id)
            time_steps.append(int(t))

    print(f"\n{'='*60}")
    print(f"Creating CSV with {len(all_latents)} samples...")
    latent_columns = [f"z_{i}" for i in range(num_latents)]

    latents_data = np.array(all_latents, dtype=np.float16)  # Shape: (num_samples, latent_size)
    df = pd.DataFrame()
    df.insert(0, "embryo_id", embryo_ids)
    df.insert(1, "time_step", time_steps)

    print(f"Total samples: {len(df)}")
    print(f"Unique embryos: {df['embryo_id'].nunique()}")
    normed_df = df


    metadata = normed_df[['embryo_id', 'time_step']]

    grades_file = "embryo_dataset_grades.csv"
    if os.path.exists(grades_file):
        print(f"Loading grades from {grades_file}...")
        grades_df = pd.read_csv(grades_file, keep_default_na=False)

        metadata = metadata.merge(
            grades_df,
            left_on='embryo_id',
            right_on='video_name',
            how='left'
        )

        if 'video_name' in metadata.columns:
            metadata = metadata.drop(columns=['video_name'])

        print(f"Joined with grades. Columns: {list(metadata.columns)}")
    else:
        print(f"Grades file {grades_file} not found, skipping grade join")
    annotations_dir = "embryo_dataset_annotations"
    metadata = metadata.groupby("embryo_id", group_keys = False).apply(lambda group:addAnnotations(group.name,group,annotations_dir)).reset_index()
    model.train()

    return metadata, latents_data

def main(model_name,index_csv):
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_KEY")
    if hf_token:
        login(hf_token)

    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"{index_csv} not found. Run build_index_embryo.py first.")

    ds = IVFEmbryoDataset(pd.read_csv(index_csv), resize=128, norm="minmax01")


    # Load model
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    
    model = ConvLSTMAutoencoder.from_pretrained("JensLundsgaard/"+model_name)

    model = model.to(DEVICE)
    metadata_df, latents_data = export_video_latents(model, ds)
     
    np.save(model_name + '.npy', latents_data)
    
    metadata_df.to_csv(model_name + '.csv', index=False)
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

    args = parser.parse_args()

    
    main(args.name,args.index)


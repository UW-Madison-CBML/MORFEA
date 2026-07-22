import torch
from torch.utils.data import DataLoader
from dataset_ivf_embryo import IVFEmbryoDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from ae_model import ConvGRUAutoencoder
from huggingface_hub import HfApi
from huggingface_hub import login
from datetime import datetime, timedelta

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(model_name):
    login(os.getenv("HF_TOKEN"))
    os.makedirs(os.path.join(f"{model_name}_imgs","A"))
    os.makedirs(os.path.join(f"{model_name}_imgs","B"))
    os.makedirs(os.path.join(f"{model_name}_imgs","C"))
    grades_df = pd.read_csv(os.path.abspath("embryo_dataset_grades.csv")).rename(columns={"video_name":"embryo_id"})
    annotations_dir = "embryo_dataset_annotations"
    df = pd.read_csv(os.path.abspath("index_embryo.csv")).rename(columns={"cell_id":"embryo_id"})
    df = df.merge(grades_df, left_on="embryo_id", right_on="embryo_id", how='left').dropna(subset=["TE"])
    ds = IVFEmbryoDataset(df, resize=128, norm="minmax01", return_augment=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
    model = ConvGRUAutoencoder.from_pretrained("JensLundsgaard/" + model_name)
    model = model.to(DEVICE)
    for idx, (embryo_vol, augment_vol) in enumerate(loader):
        model.eval()

        row = ds.df.iloc[idx]
        cell_id = str(row["embryo_id"])
        annotations_df = pd.read_csv(os.path.join(annotations_dir, f"{cell_id}_phases.csv"), header=None, names=['stage_id', 'stage_begin', 'stage_end'])
        grade = str(row["TE"])

        # pass full sequence to model
        embryo_vol = embryo_vol.to(DEVICE)
        with torch.no_grad():
            recon, _ = model(embryo_vol)
        target_dir = os.path.join(f"{model_name}_imgs", grade)

        os.makedirs(target_dir, exist_ok=True)
        vol_img = embryo_vol[0, :, 0].cpu().detach().numpy()
        recon_img = recon[0, :, 0].cpu().detach().numpy()
         
        for _,phase_row in annotations_df.iterrows():
            idx = (phase_row["stage_begin"] + phase_row["stage_end"]) // 2
            if(idx >= vol_img.shape[0]):
                print(f"idx is {idx}, doesn't fit in {vol_img.shape[0]}")
                continue

            vol = (vol_img[idx] * 255).astype(np.uint8) # don't check idx, it should be fatal if idx isn't in range
            recon = (recon_img[idx]*255).astype(np.uint8)

            plt.imsave(os.path.join(target_dir,str(cell_id)+str(phase_row["stage_id"])+".png"), vol, cmap='gray')
            plt.imsave(os.path.join(target_dir,str(cell_id)+str(phase_row["stage_id"]) + "_recon"+".png"), recon, cmap='gray')
            plt.close()
        # pass full sequence to model
        augment_vol = augment_vol.to(DEVICE)
        with torch.no_grad():
            recon, _ = model(augment_vol)
        target_dir = os.path.join(f"{model_name}_imgs", grade)

        os.makedirs(target_dir, exist_ok=True)
        vol_img = augment_vol[0, :, 0].cpu().detach().numpy()
        recon_img = recon[0, :, 0].cpu().detach().numpy()
         
        for _,phase_row in annotations_df.iterrows():
            idx = (phase_row["stage_begin"] + phase_row["stage_end"]) // 2
            if(idx >= vol_img.shape[0]):
                print(f"idx is {idx}, doesn't fit in {vol_img.shape[0]}")
                continue
            vol = (vol_img[idx] * 255).astype(np.uint8) # don't check idx, it should be fatal if idx isn't in range
            recon = (recon_img[idx]*255).astype(np.uint8)

            plt.imsave(os.path.join(target_dir,str(cell_id)+str(phase_row["stage_id"]) + "_augment.png"), vol, cmap='gray')
            plt.imsave(os.path.join(target_dir,str(cell_id)+str(phase_row["stage_id"]) + "_recon"+"_augment.png"), recon, cmap='gray')
            plt.close()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A simple script using argparse.")

    parser.add_argument("--name", type=str, help="Name of the model", default="JensLundsgaard/IVF-ConvLSTM-Model-2025-12-15")

    args = parser.parse_args()
 
    main(args.name)


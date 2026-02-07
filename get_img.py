import torch
from model import Model
from torch.utils.data import DataLoader
from dataset_ivf import IVFSequenceDataset
import matplotlib.pyplot as plt
import numpy as np
import os

from raffael_model import ConvLSTMAutoencoder
from huggingface_hub import HfApi
from huggingface_hub import login
from datetime import datetime, timedelta

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
def main(model_name):
    login(os.getenv("HF_TOKEN"))
    ds = IVFSequenceDataset(os.path.abspath("index.csv"), resize=128, norm="minmax01")
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    model = None
    api = HfApi()
    model_loaded = False
    
    print(f"Attempting to load model: {model_name}")
    model = ConvLSTMAutoencoder.from_pretrained(model_name)
    print(f"Successfully loaded model from {date_label}")
    model_loaded = True

    model = model.to(DEVICE)
    for idx, (embryo_vol, _, _) in enumerate(loader):
        model.eval()

        # Get cell_id from dataframe for filename
        row = ds.df.iloc[idx]
        cell_id = row.get("cell_id", f"cell_{idx}")

        # Pass full sequence to model
        embryo_vol = embryo_vol.to(DEVICE)
        with torch.no_grad():
            recon, _ = model(embryo_vol)

        vol_img = embryo_vol[0, -1, 0].cpu().detach().numpy()
        recon_img = recon[0, -1, 0].cpu().detach().numpy()

        # Debug: print stats
        print(f"Cell {cell_id}:")
        print(f"  Original - min: {vol_img.min():.3f}, max: {vol_img.max():.3f}, mean: {vol_img.mean():.3f}, std: {vol_img.std():.3f}")
        print(f"  Recon    - min: {recon_img.min():.3f}, max: {recon_img.max():.3f}, mean: {recon_img.mean():.3f}, std: {recon_img.std():.3f}")

        vol_img = (vol_img * 255).astype(np.uint8)
        recon_img = (recon_img * 255).astype(np.uint8)
        comparison = np.concatenate((vol_img, recon_img), axis=1)
        plt.imsave("./imgs/"+str(cell_id)+".png", comparison, cmap='gray')
        plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A simple script using argparse.")

    parser.add_argument("--name", type=str, help="Name of the model", default="JensLundsgaard/IVF-ConvLSTM-Model-2025-12-15")

    args = parser.parse_args()
 
    main(args.name)


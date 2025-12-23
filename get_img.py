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
def main():
    login(os.getenv("HF_TOKEN"))
    ds = IVFSequenceDataset(os.path.abspath("index.csv"), resize=128, norm="minmax01")
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    model = None
    if(model_name == None):
    # Search for model going back up to 30 days
    api = HfApi()
    model_loaded = False
    
    for days_back in range(31):
        date_label = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        model_name = f"JensLundsgaard/IVF-ConvLSTM-Model-{date_label}"
        try:
            print(f"Attempting to load model: {model_name}")
            model = ConvLSTMAutoencoder.from_pretrained(model_name)
            print(f"Successfully loaded model from {date_label}")
            model_loaded = True
            break
        except Exception as e:
            if days_back < 30:
                continue
            else:
                print(f"Failed to find model within 30 days. Last error: {e}")
                raise

    if not model_loaded or model is None:
        print("Could not find any model within the last 30 days, using local weights...")
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
                model.load_state_dict(torch.load("convlstm_model_weights.pth", weights_only=True))
                print("Loaded local convlstm_model_weights.pth")
            except Exception as e:
                print(f"Failed to load local weights: {e}")
                print("Using untrained model - results will be poor!")


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
    main()


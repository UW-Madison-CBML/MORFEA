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
    ds = IVFSequenceDataset(os.path.abspath("index.csv"), resize=500, norm="minmax01")
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    """model = Model()
    if os.path.exists("model_weights.pth"):
        try:
            model.load_state_dict(torch.load("model_weights.pth",weights_only = True))
        except Exception:
            #torch.save(model.state_dict(), f"model_weights.pth")
            print("model has wrong shape")
            return"""
    model = ConvLSTMAutoencoder(
        seq_len=50,
        input_channels=1,
        encoder_hidden_dim=256,
        encoder_layers=2,
        decoder_hidden_dim=128,
        decoder_layers=2,
        latent_size=4000,
        use_classifier=False,
        num_classes=2
    )

    # Search for model going back up to 30 days
    api = HfApi()

    # List the first 10 most downloaded models
    models = api.list_models(sort="downloads", direction=-1, limit=10)

    print(f"Model ID: {model.id} | Downloads: {model.downloads}")
    model_loaded = False
    for days_back in range(31):
        date_label = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        model_name = f"JensLundsgaard/IVF-ConvLSTM-Model-{date_label}"
        try:
            print(f"Attempting to load model: {model_name}")
            model.from_pretrained(model_name)
            print(f"Successfully loaded model from {date_label}")
            model_loaded = True
            break
        except Exception as e:
            if days_back < 30:
                continue
            else:
                print(f"Failed to find model within 30 days. Last error: {e}")
                raise

    if not model_loaded:
        raise Exception("Could not find any model within the last 30 days")


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
        vol_img = embryo_vol[0, -1, 0].cpu().detach().numpy() * 255
        recon_img = recon[0, -1, 0].cpu().detach().numpy() * 255
        comparison = np.concatenate((vol_img, recon_img), axis=1).astype(np.uint8)
        plt.imsave("./imgs/"+str(cell_id)+".png", comparison, cmap='gray')
        plt.close()

if __name__ == "__main__":
    main()


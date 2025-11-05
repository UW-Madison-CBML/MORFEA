import torch
from model import Model, Enc_Model
from torch.utils.data import DataLoader
from dataset_ivf import IVFSequenceDataset
import matplotlib.pyplot as plt
import numpy as np
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
def main():
    ds = IVFSequenceDataset(os.path.abspath("index.csv"), resize=500, norm="minmax01")
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    model = Model()
    if os.path.exists("model_weights.pth"):
        try:
            model.load_state_dict(torch.load("model_weights.pth",weights_only = True))
        except Exception:
            #torch.save(model.state_dict(), f"model_weights.pth")
            print("model has wrong shape")
            return 
    model = model.to(DEVICE)
    for vol, x in loader:
        model.eval()
        vol = vol.to(DEVICE)
        recon, _ = model(vol)
        vol = vol[-1,-1,-1].detach().numpy() * 255
        recon = recon[-1,-1,-1].detach().numpy() * 255
        comparison = np.concatenate((vol, recon), axis=1).astype(np.uint8)
        plt.imsave("./imgs/"+str(x)+".png", comparison, cmap='gray') 
        plt.close()

if __name__ == "__main__":
    main()


import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import math
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
from model import Model
import sys
from torch.utils.data import DataLoader
from dataset_ivf import IVFSequenceDataset
from tqdm import tqdm
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
batch_size = 50
import math
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from huggingface_hub import HfApi
import wandb

#hf_api = HfApi(token=os.getenv("HF_TOKEN"))
wandb.login(key=os.getenv("WANDB_KEY"))
run = wandb.init(
    entity="jenslundsgaard7-uw-madison",
    project="IVF-Training",
    config={
        "learning_rate": 0.02,
        "architecture": "Conv LSTM Autoencoder",
        "dataset": "https://zenodo.org/records/7912264",
        "epochs": 8,
    },
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

def train():
    model = Model()
    if os.path.exists("model_weights.pth"):
        try:
            model.load_state_dict(torch.load("model_weights.pth",weights_only = True))
        except Exception:
            torch.save(model.state_dict(), f"model_weights.pth")
            print("model has wrong shape")
            #return 
    else:
        torch.save(model.state_dict(), f"model_weights.pth")
    model = model.to(DEVICE)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay = 1e-5 )
    scheduler = CosineAnnealingLR(optimizer, T_max=8)

    ds = IVFSequenceDataset(os.path.abspath("index.csv"), resize=500, norm="minmax01")
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(8):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        total = 0.0
        for embryo_vol, empty_well_vol, sample_vol in pbar:
            embryo_size = embryo_vol.shape[0]

            vol = np.stack([embryo_vol, sample_vol], axis = 0)
            vol = vol.to(DEVICE)

            empty_well_vol = empty_well_vol.to(DEVICE)
            recon, lat = model(vol, empty_well = False)
            empty_well_recon, _ = model(empty_well_vol, empty_well=True)
            rec_loss = loss_fn(recon, vol) + loss_fn(empty_well_recon, empty_well_vol)

            embryo_lat = lat[:embryo_size].to(DEVICE)
            sample_lat = lat[embryo_size:].to(DEVICE)

            embryo_lat1 = torch.stack((embryo_lat[1:], embryo_lat[5:], embryo_lat[10:], embryo_lat[20:]), axis=0).to(DEVICE)
            embryo_lat2 = torch.stack((embryo_lat[:-1], embryo_lat[:-5], embryo_lat[:-10], embryo_lat[:-20]), axis=0).to(DEVICE)

            tcl = -1 * math.log( F.cosine_similarity(embryo_lat1, embryo_lat2)/ F.cosine_similarity(embryo_lat, sample_lat))
            loss = rec_loss + (0.1 * tcl)

            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", rec=f"{rec_loss.item():.4f}", sm=f"{smooth.item():.4f}")
        run.log({"lr": scheduler.get_last_lr()[0], "loss": total/len(loader)})
        scheduler.step()
        print(f"epoch {epoch} avg loss={total/len(loader):.4f}")
        torch.save(model.state_dict(), f"model_weights.pth")
    run.finish()
if __name__ == "__main__":
    train()
"""
    embryo_vids = os.listdir()
    np.random.shuffle(embryo_vids)
    select_amount = 1
    embryo_vids = embryo_vids[:int(len(embryo_vids)*select_amount)]
    for i in embryo_vids:
        PATH = "./../model_weights.pth"
        torch.save(model.state_dict(), PATH)
        os.chdir("./"+i)

        print(os.getcwd())
        images = os.listdir()
        try:
            np.array([Image.open(img) for img in images])
        except OSError:
            with open('./../../bad_imgs.txt', 'w') as f:
                f.write(i +'\n')
            os.chdir("./..")
            continue
        images = natsorted(images)
        images = [img for img in images if not os.path.isdir(img)]
        for k in range(2 * (1+ (len(images)//batch_size)) ):
            x = torch.tensor(np.array([Image.open(img) for img in images[
                k*batch_size - ((k & 1) * (batch_size // 2)) : min(len(images)-1,(k+1)*batch_size - ((k & 1) * (batch_size // 2)))
                ]]), dtype=torch.float32).reshape((-1,1,500,500)).to(device)
            if(len(x)) == 0:
                continue
            y = x.clone().to(device)
            print(x.shape)
            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            print(loss)
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        os.chdir("./..")
        print(os.getcwd())

os.chdir("./..")
"""


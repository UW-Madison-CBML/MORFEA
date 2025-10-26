import numpy as np
import torch
import math
from PIL import Image
import os
from natsort import natsorted
from torchsummary import summary
from model import Model
import sys
from torch.utils.data import DataLoader
from dataset_ivf import IVFSequenceDataset
from tqdm import tqdm

batch_size = 50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def train():
    model = Model()
    if os.path.exists("model_weights.pth"):
        model.load_state_dict(torch.load("model_weights.pth",weights_only = True))
    model = model.to(device)
    #print(summary(model, input_size = (1,500,500), batch_size = -1))
    # encoder: convo, downsample (maxpool), convo, downsample..., flatten 
    # rnn: lstm
    # decoder: reshape to 2d img, upsample, convo, upsample, 
    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay = 1e-5 )
    ds = IVFSequenceDataset(os.path.abspath("index.csv"), resize=500, norm="minmax01")
    loader = DataLoader(ds, batch_size=50, shuffle=True, num_workers=4, pin_memory=True)
    for epoch in range(20):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        total = 0.0
        for vol, _ in pbar:
            vol = vol.to(DEVICE)                         # [B,T,1,128,128]
            recon, lat = model(vol)
            rec_loss = loss_fn(recon, vol)
            smooth = ((lat[:,1:]-lat[:,:-1])**2).mean()  # temporal smooth
            loss = rec_loss + 0.1 * smooth # play with this coefficient
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", rec=f"{rec_loss.item():.4f}", sm=f"{smooth.item():.4f}")
        print(f"epoch {epoch} avg loss={total/len(loader):.4f}")
        torch.save(model.state_dict(), f"ae_epoch{epoch}.pth")

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


import numpy as np
import torch
import math
from PIL import Image
import os
from natsort import natsorted
from torchsummary import summary
from model import Model
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = Model().to(device)
if os.path.exists("model_weights.pth"):
    model.load_state_dict(torch.load("model_weights.pth",weights_only = True))
print(summary(model, input_size = (1,500,500), batch_size = -1))
# encoder: convo, downsample (maxpool), convo, downsample..., flatten 
# rnn: lstm
# decoder: reshape to 2d img, upsample, convo, upsample, 
loss_fn = torch.nn.MSELoss(reduction='mean')
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
#go through a random selection of images and run model on each in order, reset directory and lstm hidden state vectors after
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    os.chdir(sys.argv[1])
os.chdir("embryo_dataset")

embryo_vids = os.listdir()
np.random.shuffle(embryo_vids)
select_amount = 1
batch_size = 50
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


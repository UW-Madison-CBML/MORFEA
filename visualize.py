import torch
import os
from natsort import natsorted
from torchsummary import summary
import tphate
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import Model, Enc_Model

model = Model()
if os.path.exists("model_weights.pth"):
    model.load_state_dict(torch.load("model_weights.pth",weights_only = True))
"""
ignore this just me having fun with vim, but still has a use case for grabbing output like torchsummary of a model

:let @a = system("python3 encoder.py") 
"ap

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 8, 498, 498]              80
         MaxPool2d-2          [-1, 8, 166, 166]               0
            Conv2d-3          [-1, 8, 164, 164]             584
         MaxPool2d-4            [-1, 8, 32, 32]               0
            Conv2d-5            [-1, 8, 28, 28]           1,608
         MaxPool2d-6              [-1, 8, 5, 5]               0
           Flatten-7                  [-1, 200]               0
            Linear-8                  [-1, 200]          40,200
           Sigmoid-9                  [-1, 200]               0
================================================================
Total params: 42,472
Trainable params: 42,472
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.95
Forward/backward pass size (MB): 18.58
Params size (MB): 0.16
Estimated Total Size (MB): 19.69
----------------------------------------------------------------
None
"""
enc_model = Enc_Model(model = model)
os.chdir("embryo_dataset")
embryos = os.listdir()
np.random.shuffle(embryos)
for embryo in embryos[:int(len(embryos)*0.2)]:
    os.chdir(embryo)
    print(os.getcwd())
    print(embryo)
    tphate_op = tphate.TPHATE()
    images = natsorted([i for i in os.listdir() if not os.path.isdir(i)])
    try:
        np.array([Image.open(img) for img in images])
    except Exception:
        with open('./../../bad_imgs.txt', 'w') as f:
            f.write(embryo +'\n')
        os.chdir("./..")
        continue
    data_input = torch.tensor(np.array([Image.open(img) for img in images]), dtype=torch.float32).reshape(-1, 1, 500,500)
    model.eval()
    data = enc_model(data_input).detach().numpy()
    print(data.shape)
    data_tphate = tphate_op.fit_transform(data)
    print(data_tphate)
    print(data_tphate.shape)
    data_tphate = data_tphate.reshape(2, len(images)) 
    plt.scatter(data_tphate[0], data_tphate[1])
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(embryo)
    plt.savefig("./../../results/tphate/" + embryo + ".png", dpi=300)
    plt.close()
    os.chdir("./..")

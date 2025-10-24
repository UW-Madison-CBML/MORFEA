import torch
import os
from natsort import natsorted
from torchsummary import summary
import tphate
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__() # Call the constructor of the parent class
        self.conv1 = torch.nn.Conv2d(1, 8, 3)
        self.pool1 = torch.nn.MaxPool2d(3)
        self.conv2 = torch.nn.Conv2d(8, 8, 3)
        self.pool2 = torch.nn.MaxPool2d(5)
        self.conv3 = torch.nn.Conv2d(8, 8, 5)
        self.pool3 = torch.nn.MaxPool2d(5)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(200, 200)
        self.lstm = torch.nn.LSTM(200,200,1)
        self.unflatten = torch.nn.Unflatten(1,(8,5,5))
        self.upsample1 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv4 = torch.nn.Conv2d(8,16,3)
        self.conv5 = torch.nn.Conv2d(16,16,3,padding = 1)
        self.conv6 = torch.nn.Conv2d(16,16,3, padding = 1)
        self.conv7 = torch.nn.Conv2d(16,8,3, padding = 1)
        self.conv8 = torch.nn.Conv2d(8,8,3, padding = 1)
        self.conv9 = torch.nn.Conv2d(8,4,3, padding = 1)
        self.conv10 = torch.nn.Conv2d(4,1,13)
        self.activation = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.activation(x)
        x,(h,c) = self.lstm(x, (torch.zeros(1,200),torch.zeros(1,200))) 
        x = self.unflatten(x)
        x = self.upsample1(x)
        x = self.conv4(x)
        x = self.upsample1(x)
        x = self.conv5(x)
        x = self.upsample1(x)
        x = self.conv6(x)
        x = self.upsample1(x)
        x = self.conv7(x)
        x = self.upsample1(x)
        x = self.conv8(x)
        x = self.upsample1(x)
        x = self.conv9(x)
        x = self.upsample1(x)
        x = self.conv10(x)
        return x

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

class Enc_Model(torch.nn.Module):
    def __init__(self,model = Model()):
        super().__init__() # Call the constructor of the parent class
        self.conv1 = model.conv1
        self.pool1 = model.pool1
        self.conv2 = model.conv2
        self.pool2 = model.pool2
        self.conv3 = model.conv3
        self.pool3 = model.pool3
        self.flatten = model.flatten
        self.linear = model.linear
        self.activation = model.activation
    def forward(self,x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.activation(x)
        return x
enc_model = Enc_Model(model = model)
os.chdir("embryo_dataset")
embryos = os.listdir()
np.random.shuffle(embryos)
for embryo in embryos[:int(len(embryos)*0.1)]:
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

    os.chdir("./..")

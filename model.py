import torch
import torch.nn.functional as F
from torchinfo import summary
import sys
import os
"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Model                                    [1, 50, 1, 500, 500]      --
├─Conv2d: 1-1                            [50, 32, 498, 498]        320
├─ReLU: 1-2                              [50, 32, 498, 498]        --
├─MaxPool2d: 1-3                         [50, 32, 166, 166]        --
├─Conv2d: 1-4                            [50, 32, 164, 164]        9,248
├─ReLU: 1-5                              [50, 32, 164, 164]        --
├─MaxPool2d: 1-6                         [50, 32, 32, 32]          --
├─Conv2d: 1-7                            [50, 8, 30, 30]           2,312
├─ReLU: 1-8                              [50, 8, 30, 30]           --
├─MaxPool2d: 1-9                         [50, 8, 6, 6]             --
├─Flatten: 1-10                          [50, 288]                 --
├─LSTM: 1-11                             [1, 50, 288]              665,856
├─ReLU: 1-12                             [1, 50, 288]              --
├─Linear: 1-13                           [50, 288]                 83,232
├─ReLU: 1-14                             [50, 288]                 --
├─Linear: 1-15                           [50, 200]                 17,800
├─Linear: 1-16                           [50, 200]                 40,200
├─ReLU: 1-17                             [50, 200]                 --
├─LSTM: 1-18                             [1, 50, 200]              321,600
├─ReLU: 1-19                             [1, 50, 200]              --
├─Unflatten: 1-20                        [50, 8, 5, 5]             --
├─UpsamplingBilinear2d: 1-21             [50, 8, 10, 10]           --
├─Conv2d: 1-22                           [50, 32, 8, 8]            2,336
├─ReLU: 1-23                             [50, 32, 8, 8]            --
├─UpsamplingBilinear2d: 1-24             [50, 32, 16, 16]          --
├─Conv2d: 1-25                           [50, 32, 16, 16]          9,248
├─ReLU: 1-26                             [50, 32, 16, 16]          --
├─UpsamplingBilinear2d: 1-27             [50, 32, 32, 32]          --
├─Conv2d: 1-28                           [50, 32, 32, 32]          9,248
├─ReLU: 1-29                             [50, 32, 32, 32]          --
├─UpsamplingBilinear2d: 1-30             [50, 32, 64, 64]          --
├─Conv2d: 1-31                           [50, 32, 64, 64]          9,248
├─ReLU: 1-32                             [50, 32, 64, 64]          --
├─UpsamplingBilinear2d: 1-33             [50, 32, 128, 128]        --
├─Conv2d: 1-34                           [50, 32, 128, 128]        9,248
├─ReLU: 1-35                             [50, 32, 128, 128]        --
├─UpsamplingBilinear2d: 1-36             [50, 32, 256, 256]        --
├─Conv2d: 1-37                           [50, 16, 256, 256]        4,624
├─ReLU: 1-38                             [50, 16, 256, 256]        --
├─UpsamplingBilinear2d: 1-39             [50, 16, 512, 512]        --
├─Conv2d: 1-40                           [50, 1, 500, 500]         2,705
├─ReLU: 1-41                             [50, 1, 500, 500]         --
==========================================================================================
Total params: 1,187,225
Trainable params: 1,187,225
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 75.60
"""
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__() # Call the constructor of the parent class
        self.conv1 = torch.nn.Conv2d(1, 32, 9)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pool1 = torch.nn.MaxPool2d(2)

        self.conv2 = torch.nn.Conv2d(32, 64, 9)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool2 = torch.nn.MaxPool2d(2)

        self.conv3 = torch.nn.Conv2d(64, 128, 5)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.pool3 = torch.nn.MaxPool2d(2)

        self.conv4 = torch.nn.Conv2d(128, 128, 3, padding = 1)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.pool4 = torch.nn.MaxPool2d(2)

        self.conv5 = torch.nn.Conv2d(128, 128, 3, padding = 1)
        self.bn5 = torch.nn.BatchNorm2d(128)
        self.pool5 = torch.nn.MaxPool2d(2)

        self.conv6 = torch.nn.Conv2d(128, 128, 3, padding = 1)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.pool6 = torch.nn.MaxPool2d(2)

        # flatten to 7 * 7 * 128
        self.linear1 = torch.nn.Linear(6272, 4000)
        # reshape to (b,t...)
        self.lstm1 = torch.nn.LSTM(4000,4000,1, batch_first = True)
        # reshape to (b*t...)
        self.linear2 = torch.nn.Linear(4000, 4000)
        self.empty_well_resize_linear = torch.nn.Linear(500, 3500)
        # end encoder dropout
        self.dropout = torch.nn.Dropout(0.2)
        # start decoder
        self.linear3 = torch.nn.Linear(3500, 4000)
        #reshape to (b,t,...)
        self.lstm2 = torch.nn.LSTM(4000,4000,1, batch_first = True)
        # reshape to (b*t....) 
        self.linear4 = torch.nn.Linear(4000, 6272)
        # unflatten
        self.conv1dec = torch.nn.Conv2d(128, 128, 3, padding =1)
        self.bn1dec = torch.nn.BatchNorm2d(128)

        self.conv2dec = torch.nn.Conv2d(128, 128, 3, padding = 1)
        self.bn2dec = torch.nn.BatchNorm2d(128)

        self.conv3dec = torch.nn.Conv2d(128, 128, 3, padding = 1)
        self.bn3dec = torch.nn.BatchNorm2d(128)

        self.conv4dec = torch.nn.Conv2d(128, 128, 3, padding = 1)
        self.bn4dec = torch.nn.BatchNorm2d(128)

        self.conv5dec = torch.nn.Conv2d(128, 64, 3, padding = 1)
        self.bn5dec = torch.nn.BatchNorm2d(64)

        self.conv6dec = torch.nn.Conv2d(64, 32, 3, padding = 1)
        self.bn6dec = torch.nn.BatchNorm2d(32)

        self.conv7dec = torch.nn.Conv2d(32, 1, 3, padding = 1)
        # sigmoid, maybe small avg pool


    def forward(self, x, empty_well = False):
        b,t,_,_,_ = x.shape
        x = x.view(b*t,1,500,500)
        x = F.relu(self.pool1(self.bn1(self.conv1(x))))
        x = F.relu(self.pool2(self.bn2(self.conv2(x))))
        x = F.relu(self.pool3(self.bn3(self.conv3(x))))
        x = F.relu(self.pool4(self.bn4(self.conv4(x))))
        x = F.relu(self.pool5(self.bn5(self.conv5(x))))
        x = F.relu(self.pool6(self.bn6(self.conv6(x))))
        x = x.view(b*t, 7 * 7 * 128)
        x = F.relu(self.linear1(x))
        x = x.view(b,t,4000)
        x,_ = self.lstm1(x); x = F.relu(x)
        x = x.reshape(b*t,4000)
        x = F.relu(self.linear2(x))
        lat_vec = x.view(b,t,4000)
        lat_vec = self.dropout(lat_vec)
        # end encoder start decoder
        if (empty_well):
            x = F.relu(self.empty_well_resize_linear(lat_vec.view(b*t,4000)[:,3500:]).view(b*t,3500))
        else:
            x = lat_vec[:,:,:3500].view(b*t, 3500)
        lat_vec = lat_vec[:,:,:3500]
        x = F.relu(self.linear3(x))
        x = x.view(b,t,4000)
        x, _ = self.lstm2(x); x = F.relu(x)
        x = x.reshape(b*t,4000)
        x = F.relu(self.linear4(x))
        x = x.view(b*t, 128, 7, 7)

        x = F.relu(F.interpolate(self.bn1dec(self.conv1dec(x)), scale_factor= 2, mode='bilinear', align_corners=True))
                   
        x = F.relu(F.interpolate(self.bn2dec(self.conv2dec(x)), scale_factor= 2,mode='bilinear', align_corners=True))
        x = F.relu(F.interpolate(self.bn3dec(self.conv3dec(x)), scale_factor= 2, mode='bilinear', align_corners=True))
        x = F.relu(F.interpolate(self.bn4dec(self.conv4dec(x)), scale_factor= 2, mode='bilinear', align_corners=True))
        x = F.relu(F.interpolate(self.bn5dec(self.conv5dec(x)), scale_factor= 2, mode='bilinear', align_corners=True))
        x = F.relu(F.interpolate(self.bn6dec(self.conv6dec(x)), scale_factor= 2, mode='bilinear', align_corners=True))
        x = F.sigmoid(F.interpolate(self.conv7dec(x), size=(500,500), mode='bilinear', align_corners=True))
        return x.view(b,t,1,500,500),lat_vec
"""class Enc_Model(torch.nn.Module):
    def __init__(self,model = Model()):
        super().__init__() # Call the constructor of the parent class
        self.conv1 = model.conv1
        self.pool1 = model.pool1
        self.conv2 = model.conv2
        self.pool2 = model.pool2
        self.conv3 = model.conv3
        self.flatten = model.flatten
        self.linear1 = model.linear1
        self.activation = model.activation
        self.lstm1 = model.lstm1
    def forward(self,x):
        b,t,_,_,_ = x.shape
        x = self.conv1(x.view(b*t,1,500,500))
        x = self.activation(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = x.view(b,t,288)
        x,_ = self.lstm1(x) 
        x = self.activation(x)
        x = x.view(b*t,288)
        x = self.linear1(x)
        lat_vec = self.activation(x).view(b,t,288)

        return lat_vec[:,:,:200]

"""


def main():
    model = Model()
    print("convlstmae: ", summary(model, input_size = (1,1,1,500,500), empty_well = False))
    #enc_model = Enc_Model(model = model)
    #print("encoder: ", summary(enc_model, input_size = (1,1,1,500,500)))
    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]):
            try:
                model.load_state_dict(torch.load(sys.argv[1],weights_only = True))
                print("model loaded successfully!!!")
            except Exception:
                print("model has wrong shape")
    else:
        print("model not found")
if __name__ == "__main__":
    main()
   #define model
"""
model1 = torch.nn.Sequential(
    torch.nn.Conv2d(1, 8, 3),
    torch.nn.MaxPool2d(3),
    torch.nn.Conv2d(8, 8, 3),
    torch.nn.MaxPool2d(5),
    torch.nn.Conv2d(8, 8, 5),
    torch.nn.MaxPool2d(5),
    torch.nn.Flatten(),
    torch.nn.Linear(200, 200),
    torch.nn.LSTM(200,300,1),
    torch.nn.Flatten(),
    torch.nn.Unflatten(1,(8,5,5)),
    torch.nn.UpsamplingBilinear2d(scale_factor=2),
    torch.nn.Conv2d(8,16,3),
    torch.nn.UpsamplingBilinear2d(scale_factor=2),
    torch.nn.Conv2d(16,16,3,padding = 1),
    torch.nn.UpsamplingBilinear2d(scale_factor=2),
    torch.nn.Conv2d(16,16,3, padding = 1),
    torch.nn.UpsamplingBilinear2d(scale_factor=2),
    torch.nn.Conv2d(16,8,3, padding = 1),
    torch.nn.UpsamplingBilinear2d(scale_factor=2),
    torch.nn.Conv2d(8,8,3, padding = 1),
    torch.nn.UpsamplingBilinear2d(scale_factor=2),
    torch.nn.Conv2d(8,4,3, padding = 1),
    torch.nn.UpsamplingBilinear2d(scale_factor=2),
    torch.nn.Conv2d(4,1,13)
    #torch.nn.Conv2d(4,1,71),
    #torch.nn.Conv2d(1,1,71),
    )

 """

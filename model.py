import torch
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
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.pool1 = torch.nn.MaxPool2d(3)
        self.conv2 = torch.nn.Conv2d(32, 32, 3)
        self.pool2 = torch.nn.MaxPool2d(5)
        self.conv3 = torch.nn.Conv2d(32, 8, 3)
        self.flatten = torch.nn.Flatten()

        self.lstm1 = torch.nn.LSTM(288,288,1, batch_first = True)
        self.linear1 = torch.nn.Linear(288, 288)
        self.empty_well_resize_linear = torch.nn.Linear(88, 200)
        self.linear2 = torch.nn.Linear(200, 200)
        self.lstm2 = torch.nn.LSTM(200,200,1, batch_first = True)
        self.unflatten = torch.nn.Unflatten(1,(8,5,5))
        self.upsample1 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv4 = torch.nn.Conv2d(8,32,3)
        self.conv5 = torch.nn.Conv2d(32,32,3,padding = 1)
        self.conv6 = torch.nn.Conv2d(32,32,3, padding = 1)
        self.conv7 = torch.nn.Conv2d(32,32,3, padding = 1)
        self.conv8 = torch.nn.Conv2d(32,32,3, padding = 1)
        self.conv9 = torch.nn.Conv2d(32,16,3, padding = 1)
        self.conv10 = torch.nn.Conv2d(16,1,13)
        self.activation = torch.nn.ReLU()
    def forward(self, x, empty_well = False):
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
        # end encoder start decoder
        if (empty_well):
            x = self.empty_well_resize_linear(lat_vec.view(b*t,288)[:,200:]).view(b*t,200)
        else:
            x = lat_vec[:,:,:200].view(b*t, 200)
        lat_vec = lat_vec[:,:,:200]
        x = self.linear2(x)
        x = self.activation(x)
        x = x.view(b,t,200)
        x, _ = self.lstm2(x) 
        x = self.activation(x)
        x = x.view(b*t,200)
        x = self.unflatten(x)
        x = self.upsample1(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.upsample1(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.upsample1(x)
        x = self.conv6(x)
        x = self.activation(x)
        x = self.upsample1(x)
        x = self.conv7(x)
        x = self.activation(x)
        x = self.upsample1(x)
        x = self.conv8(x)
        x = self.activation(x)
        x = self.upsample1(x)
        x = self.conv9(x)
        x = self.activation(x)
        x = self.upsample1(x)
        x = self.conv10(x)
        x = self.activation(x)
        return x.view(b,t,1,500,500),lat_vec
class Enc_Model(torch.nn.Module):
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




def main():
    model = Model()
    print("convlstmae: ", summary(model, input_size = (1,50,1,500,500), empty_well = True))
    enc_model = Enc_Model(model = model)
    print("encoder: ", summary(enc_model, input_size = (1,50,1,500,500)))
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

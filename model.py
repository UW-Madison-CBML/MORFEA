import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__() # Call the constructor of the parent class
        self.conv1 = torch.nn.Conv2d(1, 16, 3)
        self.pool1 = torch.nn.MaxPool2d(3)
        self.conv2 = torch.nn.Conv2d(16, 16, 3)
        self.pool2 = torch.nn.MaxPool2d(5)
        self.conv3 = torch.nn.Conv2d(16, 8, 5)
        self.flatten = torch.nn.Flatten()

        self.lstm1 = torch.nn.LSTM(200,200,1, batch_first = True)
        self.linear1 = torch.nn.Linear(200, 200)
        self.linear2 = torch.nn.Linear(200, 200)
        self.lstm2 = torch.nn.LSTM(200,200,1, batch_first = True)
        self.unflatten = torch.nn.Unflatten(1,(8,5,5))
        self.upsample1 = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv4 = torch.nn.Conv2d(8,16,3)
        self.conv5 = torch.nn.Conv2d(16,16,3,padding = 1)
        self.conv6 = torch.nn.Conv2d(16,16,3, padding = 1)
        self.conv7 = torch.nn.Conv2d(16,8,3, padding = 1)
        self.conv8 = torch.nn.Conv2d(8,8,3, padding = 1)
        self.conv9 = torch.nn.Conv2d(8,4,3, padding = 1)
        self.conv10 = torch.nn.Conv2d(4,1,13)
        self.activation = torch.nn.ReLU()
    def forward(self, x):
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
        x = x.view(b,t,200)
        x,_ = self.lstm1(x) 
        x = self.activation(x)
        x = x.view(b*t,200)
        x = self.linear1(x)
        lat_vec = self.activation(x).view(b,t,200)
        # end encoder start decoder
        x = self.linear2(lat_vec)
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
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x,(h,c) = self.lstm1(x, (torch.zeros(1,200),torch.zeros(1,200))) 
        x = self.activation(x)
        x = self.linear1(x)
        x = self.activation(x)
        return x
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

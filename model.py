import torch
import torch.nn.functional as F
from torchinfo import summary
import sys
import os;


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


import torch
import torch.nn.functional as F
from torchinfo import summary
import sys
import os;
from huggingface_hub import PyTorchModelHubMixin

class Model(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__() # Call the constructor of the parent class
        self.conv1 = torch.nn.Conv2d(1, 32, 5, padding = 2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pool1 = torch.nn.MaxPool2d(2)

        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding = 1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool2 = torch.nn.MaxPool2d(2)

        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding = 1)
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
        self.linear2 = torch.nn.Linear(4000, 4000)
        # end encoder dropout
        self.dropout = torch.nn.Dropout(0.2)
        self.linear3 = torch.nn.Linear(2000, 4000)
        self.linear4 = torch.nn.Linear(4000, 6272)
        # unflatten
        # not empty
        self.conv1dec = torch.nn.Conv2d(128, 128, 3, stride = 1)
        self.bn1dec = torch.nn.BatchNorm2d(128)

        self.conv2dec = torch.nn.Conv2d(128, 128, 3, stride = 1)
        self.bn2dec = torch.nn.BatchNorm2d(128)
        
        self.conv3dec = torch.nn.Conv2d(128, 128, 3, stride = 1)
        self.bn3dec = torch.nn.BatchNorm2d(128)

        self.conv4dec = torch.nn.Conv2d(128, 128, 3, stride = 1)
        self.bn4dec = torch.nn.BatchNorm2d(128)

        #fix the numbers here lol 
        self.conv6dec = torch.nn.Conv2d(128, 64, 3, stride = 1)
        self.bn6dec = torch.nn.BatchNorm2d(64)

        self.conv7dec = torch.nn.Conv2d(64, 32, 3, stride = 1)
        self.bn7dec = torch.nn.BatchNorm2d(32)
        
        # empty
        self.conv1dec_e = torch.nn.Conv2d(128, 128, 3, stride = 1)
        self.bn1dec_e = torch.nn.BatchNorm2d(128)

        self.conv2dec_e = torch.nn.Conv2d(128, 128, 3, stride = 1)
        self.bn2dec_e = torch.nn.BatchNorm2d(128)
        
        self.conv3dec_e = torch.nn.Conv2d(128, 128, 3, stride = 1)
        self.bn3dec_e = torch.nn.BatchNorm2d(128)

        self.conv4dec_e = torch.nn.Conv2d(128, 128, 3, stride = 1)
        self.bn4dec_e = torch.nn.BatchNorm2d(128)
        #fix the numbers here lol 
        self.conv6dec_e = torch.nn.Conv2d(128, 64, 3, stride = 1)
        self.bn6dec_e = torch.nn.BatchNorm2d(64)

        self.conv7dec_e = torch.nn.Conv2d(64, 32, 3, stride = 1)
        self.bn7dec_e = torch.nn.BatchNorm2d(32)

        # end
        self.conv8dec = torch.nn.Conv2d(32, 1, 3, padding = 1)
        # sigmoid, maybe small avg pool

        self.skip_proj_32_64 = torch.nn.Conv2d(32, 64, 1)  
        self.skip_proj_64_128 = torch.nn.Conv2d(64, 128, 1) 

        self.skip_proj_128_64 = torch.nn.Conv2d(128, 64, 1)  
        self.skip_proj_64_32 = torch.nn.Conv2d(64, 32, 1)


    def forward(self, x, empty_well = False):
        x = F.relu(self.pool1(self.bn1(self.conv1(x))))

        x_skip = F.avg_pool2d(x, 2) 
        x_skip = self.skip_proj_32_64(x_skip)
        x = F.relu(self.pool2(self.bn2(self.conv2(x))))
        x = x + x_skip

        x_skip = F.avg_pool2d(x, 2)  
        x_skip = self.skip_proj_64_128(x_skip)  
        x = F.relu(self.pool3(self.bn3(self.conv3(x))))
        x = x + x_skip

        x_skip = F.avg_pool2d(x, 2)  
        x = F.relu(self.pool4(self.bn4(self.conv4(x))))
        x = x + x_skip

        x_skip = F.avg_pool2d(x,2)
        x = F.relu(self.pool5(self.bn5(self.conv5(x))))
        x = x + x_skip

        x_skip = F.avg_pool2d(x, 2)
        x = F.relu(self.pool6(self.bn6(self.conv6(x))))
        x = x + x_skip
        
        x = x.view(-1, 128*7*7) 

        x = F.relu(self.linear1(x))
        lat_vec = F.relu(self.linear2(x))

        lat_vec = self.dropout(lat_vec)
        if (empty_well):
            x = lat_vec[:,:2000]
        else:
            x = lat_vec[:, 2000:]
        lat_vec = lat_vec[:, 2000:]
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))

        x = x.view(-1, 128, 7, 7)
        if(empty_well):
            x_skip = x.clone()
            x = F.relu(F.interpolate(self.bn1dec_e(self.conv1dec_e(x)), scale_factor=2, mode='bilinear', align_corners=True))
            x_skip = F.interpolate(x_skip, size = x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + x_skip

            x_skip = x.clone()
            x = F.relu(F.interpolate(self.bn2dec_e(self.conv2dec_e(x)), scale_factor=2, mode='bilinear', align_corners=True))
            x_skip = F.interpolate(x_skip, size = x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + x_skip

            x_skip = x.clone()
            x = F.relu(F.interpolate(self.bn3dec_e(self.conv3dec_e(x)), scale_factor=2, mode='bilinear', align_corners=True))
            x_skip = F.interpolate(x_skip, size = x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + x_skip

            x_skip = x.clone()
            x = F.relu(F.interpolate(self.bn4dec_e(self.conv4dec_e(x)), scale_factor=2, mode='bilinear', align_corners=True))
            x_skip = F.interpolate(x_skip, size = x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + x_skip
            
            x_skip = x.clone()
            x = F.relu(F.interpolate(self.bn6dec_e(self.conv6dec_e(x)), scale_factor=2, mode='bilinear', align_corners=True))
            x_skip = F.interpolate(x_skip, size = x.shape[-2:], mode='bilinear', align_corners=True)
            x_skip = self.skip_proj_128_64(x_skip)
            x = x + x_skip

            x_skip = x.clone()
            x = F.relu(F.interpolate(self.bn7dec_e(self.conv7dec_e(x)), scale_factor=2, mode='bilinear', align_corners=True))
            x_skip = F.interpolate(x_skip, size = x.shape[-2:], mode='bilinear', align_corners=True)
            x_skip = self.skip_proj_64_32(x_skip)
            x = x + x_skip


        else:
            x_skip = x.clone()
            x = F.relu(F.interpolate(self.bn1dec(self.conv1dec(x)), scale_factor=2, mode='bilinear', align_corners=True))
            x_skip = F.interpolate(x_skip, size = x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + x_skip

            x_skip = x.clone()
            x = F.relu(F.interpolate(self.bn2dec(self.conv2dec(x)), scale_factor=2, mode='bilinear', align_corners=True))
            x_skip = F.interpolate(x_skip, size = x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + x_skip

            x_skip = x.clone()
            x = F.relu(F.interpolate(self.bn3dec(self.conv3dec(x)), scale_factor=2, mode='bilinear', align_corners=True))
            x_skip = F.interpolate(x_skip, size = x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + x_skip

            x_skip = x.clone()
            x = F.relu(F.interpolate(self.bn4dec(self.conv4dec(x)), scale_factor=2, mode='bilinear', align_corners=True))
            x_skip = F.interpolate(x_skip, size = x.shape[-2:], mode='bilinear', align_corners=True)
            x = x + x_skip
            
            x_skip = x.clone() 
            x = F.relu(F.interpolate(self.bn6dec(self.conv6dec(x)), scale_factor=2, mode='bilinear', align_corners=True))
            x_skip = F.interpolate(x_skip, size = x.shape[-2:], mode='bilinear', align_corners=True)
            x_skip = self.skip_proj_128_64(x_skip)
            x = x + x_skip

            x_skip = x.clone()
            x = F.relu(F.interpolate(self.bn7dec(self.conv7dec(x)), scale_factor=2, mode='bilinear', align_corners=True))
            x_skip = F.interpolate(x_skip, size = x.shape[-2:], mode='bilinear', align_corners=True)
            x_skip = self.skip_proj_64_32(x_skip)
            x = x + x_skip

        x = F.sigmoid(F.interpolate(self.conv8dec(x), size=(500,500), mode='bilinear', align_corners=True))
        return x, lat_vec

def main():
    model = Model()
    print("convlstmae: ", summary(model, input_size = (1,1,500,500), empty_well = False))
    
    print("convlstmae empty: ", summary(model, input_size = (1,1,500,500), empty_well = True))
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


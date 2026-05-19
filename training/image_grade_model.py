import torch
from torch.nn import Module
import torch.nn.functional as F
from ae_model import ResidualBlock
from torch.nn.utils.rnn import pack_padded_sequence
class ImageGradeModel(Module):

    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.cnn = torch.nn.Sequential(
            ResidualBlock(1, 32, downsample=True),
            ResidualBlock(32, 32, downsample=True),
            ResidualBlock(32, 32, downsample=True),
            ResidualBlock(32, 32, downsample=True), # 8 * 8 * 32
        )


        self.lin1 = torch.nn.Linear(2048, 256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.lstm = torch.nn.LSTM(128, 128, batch_first = True)
        self.lin3 = torch.nn.Linear(128, num_classes)
        

        
    def forward(self, x, lengths):
        B,T, C, H, W = x.shape 
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = x.view(B,T,-1) # last dim should be 2048
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hs, _) = self.lstm(x)
        hs = hs.squeeze(0) # single layer single direction LSTM
        x = self.lin3(hs)
        return x

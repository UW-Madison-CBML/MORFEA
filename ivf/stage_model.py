import torch
from torch.nn import Module
import torch.nn.functional as F


class StageModel(Module):

    def __init__(self, input_size):
        super().__init__()

        self.lin1 = torch.nn.Linear(input_size, 256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.lstm = torch.nn.LSTM(128, 128, batch_first = True)
        self.lin3 = torch.nn.Linear(128, 18)



    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x,_  = self.lstm(x)
        x = F.relu(self.lin3(x))

        return x

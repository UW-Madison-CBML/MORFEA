import torch
import torch.nn.functional as F
class SignatureClassifier(torch.nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_size, 256)
        self.lin2 = torch.nn.Linear(256, 256)
        self.lin3 = torch.nn.Linear(256, 3)
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return x
        

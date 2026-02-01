import torch
import torch.nn.functional as F
class SignatureClassifier(torch.nn.Module):
    def __init__(self, in_size, keep_na=False):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_size, 256)
        self.lin2 = torch.nn.Linear(256, 256)
        self.lin3 = torch.nn.Linear(256, 4 if keep_na else 3)
        self.dropout = torch.nn.Dropout(0.2)
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.dropout(F.relu(self.lin2(x)))
        x = self.lin3(x)
        return x
        

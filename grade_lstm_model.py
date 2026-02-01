import torch
import torch.nn.functional as F
class GradeLSTMClassifier(torch.nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_size, 256)
        self.lin2 = torch.nn.Linear(256, 256)
        self.lin3 = torch.nn.Linear(256, 256)
        self.lin4 = torch.nn.Linear(256, 3)
        self.lstm = torch.nn.LSTM(256, 256, batch_first=True)
        self.dropout = torch.nn.Dropout(0.2)
    def forward(self, x):
        T,L = x.shape 
        x = F.relu(self.lin1(x))
        x = self.dropout(F.relu(self.lin2(x)))
        x = F.relu(self.lin3(x))
        x = x.view(1,T, L) 
        x, _, _ = self.lstm(x)
        x = x.view(T, L)
        x = self.lin4(x)
        return x
 




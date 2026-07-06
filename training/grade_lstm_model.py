import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
class GradeLSTMClassifier(torch.nn.Module):
    def __init__(self, in_size, keep_na=False):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_size, 256)
        self.lin2 = torch.nn.Linear(256, 256)
        self.lin3 = torch.nn.Linear(256, 256)
        self.lstm = torch.nn.LSTM(256, 256, batch_first=True)
        self.lin4 = torch.nn.Linear(256, 4 if keep_na else 3)
        self.dropout = torch.nn.Dropout(0.2)
    def forward(self, x, lengths):
        B,T,L = x.shape 
        x = x.view(B*T,L)
        x = F.relu(self.lin1(x))
        x = self.dropout(F.relu(self.lin2(x)))
        x = F.relu(self.lin3(x))
        x = x.view(B,T,256)
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, (hn, cn) = self.lstm(x_packed)

        x = self.lin4(torch.squeeze(hn,0)) # single layer, single direction so hidden state is (1, batch, lstm_out),  
        
        return x 




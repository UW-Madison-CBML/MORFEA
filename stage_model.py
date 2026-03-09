import torch
from torch.nn import Module
import torch.nn.functional as F


def viterbi_decode_batch(emissions, transitions):
    B, T, C = emissions.shape
    device = emissions.device
    
    path_metrics = torch.full((B, C), -1e9, device=device)
    path_metrics[:, 0] = emissions[:, 0, 0] 
    
    all_scores = [path_metrics]

    for t in range(1, T):
        scores = path_metrics.unsqueeze(1) + transitions.unsqueeze(0)
        
        max_prev_scores, _ = torch.max(scores, dim=2)
        
        path_metrics = max_prev_scores + emissions[:, t, :]
        all_scores.append(path_metrics)

    return torch.stack(all_scores, dim=1) # [B, T, C]
class StageModel(Module):

    def __init__(self, input_size, num_classes=18):
        super().__init__()

        self.lin1 = torch.nn.Linear(input_size, 256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.lstm = torch.nn.LSTM(128, 128, batch_first = True)
        self.lin3 = torch.nn.Linear(128, num_classes)
        self.transition_params = nn.Parameter(torch.randn(num_classes, num_classes))
        self.register_buffer("mask", torch.triu(torch.ones(num_classes, num_classes)))

        with torch.no_grad():
            self.transition_params.copy_(torch.eye(num_classes) * 2.0)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x,_  = self.lstm(x)
        emissions = self.lin3(x)
        
        transitions = self.transition_params.masked_fill(self.mask == 0, -1e9)

        
        return viterbi_decode_batch(emissions, transitions)

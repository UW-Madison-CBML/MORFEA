import torch
from torch.nn import Module
import torch.nn.functional as F
from torchcrf import CRF

class StageModel(Module):

    def __init__(self, input_size, num_classes=18):
        super().__init__()
        self.num_classes = num_classes

        self.lin1 = torch.nn.Linear(input_size, 256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.lstm = torch.nn.LSTM(128, 128, batch_first = True)
        self.lin3 = torch.nn.Linear(128, num_classes)
        
        self.crf = CRF(num_classes, batch_first=True)
        with torch.no_grad():
            self.crf.start_transitions.fill_(-10000.0)
            self.crf.start_transitions[0] = 0.0 
        self.register_buffer("mask", torch.triu(torch.ones(num_classes, num_classes)))

        
    def forward(self, x, mask, tags=None):
        B,T, L = x.shape 
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x, _ = self.lstm(x)
        emissions = self.lin3(x)
        if torch.isnan(emissions).any():
            print("bad emissions") 
        #start_scores = torch.full((self.num_classes,), float("-inf"), device=x.device) # use this if you want to train on just prefixes
        #start_scores[0] = 0.0
        if mask is None and tags is not None:
            mask = torch.ones(tags.shape, dtype=torch.bool) 

        if self.training and tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='token_mean')
            if(torch.isnan(loss).any()):
                print("bad loss")
            return loss
        else:
            decoded = self.crf.decode(emissions, mask=mask)
            decoded_tensor = torch.tensor(decoded, device = emissions.device)
            return F.one_hot(decoded_tensor, num_classes=self.num_classes) # outputting one-hots seems reasonable

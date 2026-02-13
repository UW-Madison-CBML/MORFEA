import torch
from torch.nn import Module
import torch.nn.functional as F


class StageModel(Module):

	def __init__(self, input_size):
		super.__init__()

		self.lin1 = torch.nn.Linear(input_size, 512)
	   	self.lin2 = torch.nn.Linear(512, 256)
		self.lstm = torch.nn.LSTM(256, 256, batch_first = True)
		self.lin3 = torch.nn.Linear(256, 16)
		
		

	def forward(self, x):
		x = F.relu(self.lin1(x))
		x = F.relu(self.lin2(x))
		x,_  = self.lstm(x)
		x = F.relu(self.lin3(x))
	
		return x

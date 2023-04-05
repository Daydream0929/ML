import torch
from torch import nn
from torch.nn import functional as F

class ANN_Model(nn.Module):
    def __init__(self, input_feature = 4, hidden1 = 10, hidden2 = 10, out_feature = 1):
        super().__init__()
        self.f_connected1 = nn.Linear(input_feature, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, out_feature)
    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x
   

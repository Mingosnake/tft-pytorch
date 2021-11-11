import torch
from torch import nn
import torch.nn.functional as F


#Gated Linear Units
class GateLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc_forward = nn.Linear(input_dim, input_dim)
        self.fc_gate = nn.Linear(input_dim, input_dim)

    def forward(self, input):
        #input = [batch size, input dim]

        return F.sigmoid(self.fc_gate(input)) * self.fc_forward(input)



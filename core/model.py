import torch
from torch import nn


#Gated Linear Units
class GateLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc_forward = nn.Linear(input_dim, input_dim)
        self.fc_gate = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        #input = [batch size, input dim]

        return self.sigmoid(self.fc_gate(input)) * self.fc_forward(input)


#Gated Residual Network
class GRN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc_input = nn.Linear(input_dim, input_dim)
        self.fc_context = nn.Linear(input_dim, input_dim, bias=False)
        self.elu = nn.ELU()

        self.fc_forward = nn.Linear(input_dim, input_dim)
        self.glu = GateLayer(input_dim)


    def forward(self, input):
        #input = [batch size, input dim]

        pass
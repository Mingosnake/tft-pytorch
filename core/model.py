import torch
from torch import nn


#Gated Linear Units
class GatingLayer(nn.Module):
    def __init__(self, x_dim, dropout_rate=None):
        super().__init__()

        self.dropout = (nn.Dropout(dropout_rate)
                        if dropout_rate is not None
                        else None)

        self.fc_forward = nn.Linear(x_dim, x_dim)
        self.fc_gate = nn.Linear(x_dim, x_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x = [batch size, x dim]
        if self.dropout is not None:
            x = self.dropout(x)

        return self.sigmoid(self.fc_gate(x)) * self.fc_forward(x)


#Add and Normalization
class AddAndNorm(nn.Module):
    def __init__(self, x_dim):
        super().__init__()

        self.layernorm = nn.LayerNorm(x_dim)

    def forward(self, x_gated, x_res):
        return self.layernorm(x_gated + x_res)


#Gated Residual Network
class GRN(nn.Module):
    def __init__(self, x_dim, c_dim, hid_dim, dropout_rate=None):
        super().__init__()

        self.fc_x = nn.Linear(x_dim, hid_dim)
        self.fc_context = nn.Linear(c_dim, hid_dim, bias=False)
        self.elu = nn.ELU()

        self.fc_forward = nn.Linear(hid_dim, hid_dim)
        self.gating_layer = GatingLayer(hid_dim, dropout_rate)
        self.add_and_norm = AddAndNorm(hid_dim)

    def forward(self, x, c):
        #x = [batch size, x dim]

        hidden = self.elu(self.fc_x(x) + self.fc_context(c))
        hidden = self.fc_forward(hidden)

        return self.add_and_norm(self.gating_layer(hidden), x)
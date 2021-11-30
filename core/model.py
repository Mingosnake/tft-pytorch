import torch
from torch import nn


class GatingLayer(nn.Module):
    """Gated Linear Unit (GLU)."""

    def __init__(self, x_dim, out_dim, dropout_rate=None):
        """
        Args:
            x_dim: Input dimension
            out_dim: Output dimension
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.dropout = (nn.Dropout(dropout_rate)
                        if dropout_rate is not None
                        else None)
        self.fc_forward = nn.Linear(x_dim, out_dim)
        self.fc_gate = nn.Linear(x_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Input to gating layer = [batch size, x dim]

        Returns:
            Return of gating layer = [batch size, out dim]
        """
        if self.dropout is not None:
            x = self.dropout(x)

        return self.sigmoid(self.fc_gate(x)) * self.fc_forward(x)


class AddAndNorm(nn.Module):
    """Add and Normalization."""

    def __init__(self, x_dim):
        """
        Args:
            x_dim: Input dimension
        """
        super().__init__()

        self.layernorm = nn.LayerNorm(x_dim)

    def forward(self, x_gated, x_res):
        """
        Args:
            x_gated: Gated input = [batch size, x dim]
            x_res: Skipped input = [batch size, x dim]

        Returns:
            Added and normalized output = [batch size, x dim]
        """
        return self.layernorm(x_gated + x_res)


class GatedResNet(nn.Module):
    """Gated Residual Network (GRN)."""

    def __init__(self,
                 x_dim,
                 hid_dim,
                 c_dim=None,
                 out_dim=None,
                 dropout_rate=None):
        """
        Args:
            x_dim: Input dimension
            hid_dim: Dimension of GRN
            c_dim: Dimension of context vector (optional)
            out_dim: Output dimension (optional)
            dropout_rate: Dropout rate (optional)
        """
        super().__init__()

        self.name = 'GatedResNet'

        if out_dim is None:
            self.fc_skip = None
            out_dim = hid_dim
        else:
            self.fc_skip = nn.Linear(x_dim, out_dim)

        self.fc_x = nn.Linear(x_dim, hid_dim)
        self.fc_context = (nn.Linear(c_dim, hid_dim, bias=False)
                           if c_dim is not None
                           else None)
        self.elu = nn.ELU()

        self.fc_forward = nn.Linear(hid_dim, hid_dim)
        self.gating_layer = GatingLayer(hid_dim, out_dim, dropout_rate)
        self.add_and_norm = AddAndNorm(hid_dim)

    def forward(self, x, c=None):
        """
        Args:
            x: Input = [batch size, x dim]
            c: Context = [batch size, c dim]

        Returns:
            Output of GRN = [batch size, out dim]
        """
        if (self.fc_context is None) != (c is None):
            raise ValueError(f'{self.name} module is created wrong for c_dim')

        skip = self.fc_skip(x) if self.fc_skip is not None else x
        if c is not None:
            hidden = self.elu(self.fc_x(x) + self.fc_context(c))
        else:
            hidden = self.elu(self.fc_x(x))
        # hidden = [batch size, hid dim]
        hidden = self.fc_forward(hidden)
        # hidden = [batch size, hid dim]

        return self.add_and_norm(self.gating_layer(hidden), skip)


class VarSelectNet(nn.Module):
    """Variable Selection Network.

    Attributes:
        grn_sel_wt: GRN for selection weight
        grn_var_list: list of GRN for each variable
    """

    def __init__(self,
                 var_dim,
                 hid_dim,
                 c_dim=None,
                 dropout_rate=None):
        """
        Args:
            var_dim: Number of variables
            hid_dim: Dimension of hidden layer
            c_dim: Dimension of context vector
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.name = 'VarSelectNet'

        self.c_dim = c_dim

        self.grn_sel_wt = GatedResNet(
            x_dim=var_dim*hid_dim,
            hid_dim=hid_dim,
            c_dim=c_dim,
            out_dim=var_dim,
            dropout_rate=dropout_rate,
        )
        self.softmax = nn.Softmax(dim=1)

        self.grn_var_list = nn.ModuleList([
            GatedResNet(
                x_dim=hid_dim,
                hid_dim=hid_dim,
                dropout_rate=dropout_rate,
            ) for _ in range(var_dim)
        ])

    def forward(self, x: torch.Tensor, c=None):
        """
        Args:
            x: Input = [batch size, var dim, hid dim]
            c: Context = [batch size, c dim]

        Returns:
            output: Output of Variable Selection Network = [batch size, hid dim]
        """
        if (self.c_dim is None) != (c is None):
            raise ValueError(f'{self.name} module is created wrong for c_dim')

        flat = x.view(x.shape[0], -1)
        if c is not None:
            var_sel_wt = self.softmax(self.grn_sel_wt(flat, c))
        else:
            var_sel_wt = self.softmax(self.grn_sel_wt(flat))
        # var_sel_wt = [batch size, var dim]
        var_sel_wt = var_sel_wt.unsqueeze(1)
        # var_sel_wt = [batch size, 1, var dim]

        var_list = []
        for i, grn_var in enumerate(self.grn_var_list):
            var_list.append(grn_var(x[:, i, :]))
        vars = torch.stack(var_list, dim=1)
        # vars = [batch size, var dim, hid dim]
        output = torch.bmm(var_sel_wt, vars)
        # output = [batch size, 1, hid dim]
        output = output.squeeze(1)
        # output = [batch size, hid dim]

        return output


class MultiHeadAttention(nn.Module):
    """Multi Head Attention.
    """
    
    def __init__(self):
        super().__init__()
        
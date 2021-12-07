import torch
from torch import nn


class GatingLayer(nn.Module):
    """Gated Linear Unit (GLU).

    Attributes:
        dropout: Dropout layer
        fc_forward: Feed forward linear layer
        fc_gate: Linear layer for gate vector
        sigmoid: Sigmoid layer
    """

    def __init__(self,
                 x_dim,
                 out_dim,
                 dropout_rate=1.0):
        """
        Args:
            x_dim: Input dimension
            out_dim: Output dimension
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate)
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
        x = self.dropout(x)

        return self.sigmoid(self.fc_gate(x)) * self.fc_forward(x)


class GatedSkipConn(nn.Module):
    """Gated Skip Connection.

    Gating Layer + Add and Normalization.

    Attributes:
        gating_layer: Gating layer
        layernorm: Layer normalization layer
    """

    def __init__(self,
                 x_dim,
                 out_dim,
                 dropout_rate=1.0):
        """
        Args:
            x_dim: Input dimension
            out_dim: Output dimension
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.gating_layer = GatingLayer(x_dim, out_dim, dropout_rate)
        self.layernorm = nn.LayerNorm(out_dim)

    def forward(self, x_gate, x_skip):
        """
        Args:
            x_gate: Input to gating layer = [batch size, x dim]
            x_skip: Skipped input = [batch size, x dim]

        Returns:
            Added and normalized output = [batch size, out dim]
        """
        return self.layernorm(self.gating_layer(x_gate) + x_skip)


class GatedResNet(nn.Module):
    """Gated Residual Network (GRN).

    Attributes:
        fc_skip: Linear layer for different dimension skip connection
        fc_x: Linear layer for input vector
        fc_context: Linear layer for context vector
        elu: Exponential Linear Unit (ELU) layer
        fc_forward: Feed forward linear layer
        gated_skip_conn: Gated skip connection layer
    """

    def __init__(self,
                 x_dim,
                 hid_dim,
                 c_dim=None,
                 out_dim=None,
                 dropout_rate=1.0):
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

        if out_dim is None:  # for usual
            self.fc_skip = None
            out_dim = hid_dim
        else:  # for flattened input's GRN
            self.fc_skip = nn.Linear(x_dim, out_dim)

        self.fc_x = nn.Linear(x_dim, hid_dim)
        self.fc_context = (nn.Linear(c_dim, hid_dim, bias=False)
                           if c_dim is not None
                           else None)
        self.elu = nn.ELU()

        self.fc_forward = nn.Linear(hid_dim, hid_dim)
        self.gated_skip_conn = GatedSkipConn(hid_dim, out_dim, dropout_rate)

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

        skip = x if self.fc_skip is None else self.fc_skip(x)
        if c is not None:
            hidden = self.elu(self.fc_x(x) + self.fc_context(c))
        else:
            hidden = self.elu(self.fc_x(x))
        # hidden = [batch size, hid dim]
        hidden = self.fc_forward(hidden)
        # hidden = [batch size, hid dim]

        return self.gated_skip_conn(hidden, skip)


class VarSelectNet(nn.Module):
    """Variable Selection Network.

    Attributes:
        sel_wt_grn: GRN for selection weights
        var_grn_list: list of GRN for each variable
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

        self.sel_wt_grn = GatedResNet(
            x_dim=var_dim*hid_dim,
            hid_dim=hid_dim,
            c_dim=c_dim,
            out_dim=var_dim,
            dropout_rate=dropout_rate)

        self.softmax = nn.Softmax(dim=1)

        self.var_grn_list = nn.ModuleList([
            GatedResNet(
                x_dim=hid_dim,
                hid_dim=hid_dim,
                dropout_rate=dropout_rate)
            for _ in range(var_dim)])

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
            var_sel_wt = self.softmax(self.sel_wt_grn(flat, c))
        else:
            var_sel_wt = self.softmax(self.sel_wt_grn(flat))
        # var_sel_wt: Variable selection weights = [batch size, var dim]
        var_sel_wt = var_sel_wt.unsqueeze(1)
        # var_sel_wt = [batch size, 1, var dim]

        var_list = []
        for i, var_grn in enumerate(self.var_grn_list):
            var_list.append(var_grn(x[:, i, :]))

        vars = torch.stack(var_list, dim=1)
        # vars = [batch size, var dim, hid dim]
        output = torch.bmm(var_sel_wt, vars)
        # output = [batch size, 1, hid dim]
        output = output.squeeze(1)
        # output = [batch size, hid dim]

        return output


class MultiHeadAttention(nn.Module):
    """Interpretable Multi Head Attention.

    Attributes:
        attn_dim: Dimension of attention layer
        fc_q: Linear layer for query
        fc_k: Linear layer for key
        fc_v: Linear layer for value
        fc_h: Final linear layer for output (combined heads)
        scale: 
    """

    def __init__(self,
                 hid_dim,
                 n_heads,
                 dropout_rate,
                 device='cpu'):
        super().__init__()

        if (hid_dim % n_heads) != 0:
            raise ValueError('hid_dim should be multiple of n_heads')

        self.n_heads = n_heads
        attn_dim = hid_dim // n_heads
        self.attn_dim = attn_dim

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, attn_dim)

        self.fc_h = nn.Linear(attn_dim, hid_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.scale = torch.sqrt(torch.FloatTensor([self.attn_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, attn dim]

        Q = Q.view(
            batch_size, -1, self.n_heads, self.attn_dim).permute(0, 2, 1, 3)
        K = K.view(
            batch_size, -1, self.n_heads, self.attn_dim).permute(0, 2, 1, 3)
        # Q = [batch size, n heads, query len, attn dim]
        # K = [batch size, n heads, key len, attn dim]
        # V = [batch size, value len, attn dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        energy = torch.softmax(energy, dim=-1)
        attention = torch.mean(energy, dim=1)
        # attention = [batch size, query len, key len]

        output = torch.bmm(self.dropout(attention), V)
        # output = [batch size, query len, attn dim]

        output = self.fc_h(output)
        # output = [batch size, query len, hid dim]

        return output, attention


# mask = torch.tril(
#     torch.ones((trg_len, src_len + trg_len), device=device),
#     diagonal=src_len).bool()

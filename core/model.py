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

    def __init__(self, x_dim, out_dim, dropout_rate=1.0):
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
    """Gated skip connection.

    Gating layer + add and normalization.

    Attributes:
        gating_layer: Gating layer
        layernorm: Layer normalization layer
    """

    def __init__(self, x_dim, out_dim, dropout_rate=1.0):
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
    """Gated residual network (GRN).

    Attributes:
        name: Name of Module
        fc_skip: Linear layer for different dimension skip connection
        fc_x: Linear layer for input vector
        fc_context: Linear layer for context vector
        elu: Exponential linear unit (ELU) layer
        fc_forward: Feed forward linear layer
        gated_skip_conn: Gated skip connection layer
    """

    def __init__(
        self, x_dim, hid_dim, c_dim=None, out_dim=None, dropout_rate=1.0
    ):
        """
        Args:
            x_dim: Input dimension
            hid_dim: Dimension of GRN
            c_dim: Dimension of context vector (optional)
            out_dim: Output dimension (optional)
            dropout_rate: Dropout rate (optional)
        """
        super().__init__()

        self.name = "GatedResNet"

        if out_dim is None:  # for usual
            self.fc_skip = None
            out_dim = hid_dim
        else:  # for flattened input's GRN
            self.fc_skip = nn.Linear(x_dim, out_dim)

        self.fc_x = nn.Linear(x_dim, hid_dim)
        self.fc_context = (
            nn.Linear(c_dim, hid_dim, bias=False)
            if c_dim is not None
            else None
        )
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
            raise ValueError(f"{self.name} module is created wrong for c_dim")

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
    """Variable selection network.

    Attributes:
        name: Name of module
        c_dim: Dimension of context vector
        sel_wt_grn: GRN for selection weights
        softmax: Softmax layer
        var_grn_list: List of GRN for each variable
    """

    def __init__(self, var_dim, hid_dim, c_dim=None, dropout_rate=1.0):
        """
        Args:
            var_dim: Number of variables
            hid_dim: Dimension of hidden layer
            c_dim: Dimension of context vector
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.name = "VarSelectNet"

        self.c_dim = c_dim

        self.sel_wt_grn = GatedResNet(
            x_dim=var_dim * hid_dim,
            hid_dim=hid_dim,
            c_dim=c_dim,
            out_dim=var_dim,
            dropout_rate=dropout_rate,
        )

        self.softmax = nn.Softmax(dim=1)

        self.var_grn_list = nn.ModuleList(
            [
                GatedResNet(
                    x_dim=hid_dim, hid_dim=hid_dim, dropout_rate=dropout_rate
                )
                for _ in range(var_dim)
            ]
        )

    def forward(self, x: torch.Tensor, c=None):
        """
        Args:
            x: Input = [batch size, var dim, hid dim]
            c: Context = [batch size, c dim]

        Returns:
            output: Output of variable selection network
                = [batch size, hid dim]
        """
        if (self.c_dim is None) != (c is None):
            raise ValueError(f"{self.name} module is created wrong for c_dim")

        flat = x.view(x.shape[0], -1)
        if c is not None:
            var_sel_wt = self.softmax(self.sel_wt_grn(flat, c))
        else:
            var_sel_wt = self.softmax(self.sel_wt_grn(flat))
        # var_sel_wt: Variable selection weights = [batch size, var dim]
        var_sel_wt = var_sel_wt.unsqueeze(1)
        # var_sel_wt = [batch size, 1, var dim]

        var_list = [
            var_grn(x[:, i, :]) for i, var_grn in enumerate(self.var_grn_list)
        ]
        variables = torch.stack(var_list, dim=1)
        # variables = [batch size, var dim, hid dim]
        output = torch.bmm(var_sel_wt, variables)
        # output = [batch size, 1, hid dim]
        output = output.squeeze(1)
        # output = [batch size, hid dim]

        return output


class MultiHeadAttention(nn.Module):
    """Interpretable multi-head attention.

    Attributes:
        n_heads: Number of heads
        attn_dim: Dimension of attention layer
        fc_q: Linear layer for query
        fc_k: Linear layer for key
        fc_v: Linear layer for value
        fc_h: Final linear layer for output (combined heads)
        dropout: Dropout layer
        scale: Scale factor of dot-product attention
    """

    def __init__(self, hid_dim, n_heads, dropout_rate=1.0):
        """
        Args:
            hid_dim: Dimension of input and output in multi-head attention
            n_heads: Number of heads
            dropout_rate: Dropout rate
            device: Pytorch device
        """
        super().__init__()

        if (hid_dim % n_heads) != 0:
            raise ValueError("hid_dim should be multiple of n_heads")

        self.n_heads = n_heads
        attn_dim = hid_dim // n_heads
        self.attn_dim = attn_dim

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, attn_dim)

        self.fc_h = nn.Linear(attn_dim, hid_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.scale = torch.sqrt(torch.FloatTensor([self.attn_dim]))

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Query for multi-head attention
                = [batch size, query len, hid dim]
            key: Key for multi-head attention
                = [batch size, key len, hid dim]
            value: Value for multi-head attention
                = [batch size, value len, hid dim]
            mask: Masking tensor = [1, 1, query len, key len]

        Returns:
            output: Output of multi-head attention
                = [batch size, query len, hid dim]
            attention: Attention weights = [batch size, query len, key len]
        """
        self.scale = self.scale.to(query.device)
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, attn dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.attn_dim).permute(
            0, 2, 1, 3
        )
        K = K.view(batch_size, -1, self.n_heads, self.attn_dim).permute(
            0, 2, 1, 3
        )
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


class StaticCovariateEncoders(nn.Module):
    """Static covariate encoders.

    Attributes:
        grn_list: List of GRN for contexts
    """

    def __init__(self, hid_dim, dropout_rate=1.0):
        """
        Args:
            hid_dim: Dimension of static covariate encoders
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.grn_list = nn.ModuleList(
            [
                GatedResNet(hid_dim, hid_dim, dropout_rate=dropout_rate)
                for _ in range(4)
            ]
        )

    def forward(self, x):
        """
        Args:
            x: Input of static covariate encoders = [batch size, hid dim]
        
        Returns:
            c_selection: Context vector for variable selection network
                = [batch size, hid dim]
            c_cell: Context vector for initial cell state of LSTM
                = [batch size, hid dim]
            c_hidden: Context vector for initial cell state of LSTM
                = [batch size, hid dim]
            c_enrichment: Context vector for static enrichment layer
                = [batch size, hid dim]
        """
        c_selection = self.grn_list[0](x)
        c_cell = self.grn_list[1](x)
        c_hidden = self.grn_list[2](x)
        c_enrichment = self.grn_list[3](x)

        return c_selection, c_cell, c_hidden, c_enrichment


class Seq2Seq(nn.Module):
    """Sequence to sequence layer for locality enhancement.
    
    Attributes:
        static_sel: Variable selection network for static metadata
        static_encoders: Static covariate encoders
        history_sel: Variable selection network for historical inputs
        future_sel: Variable selection network for known future inputs
        encoder_lstm: Encoder LSTM
        decoder_lstm: Decoder LSTM
        gated_skip_conn: Gated skip connection layer
    """

    def __init__(
        self, static_dim, history_dim, future_dim, hid_dim, dropout_rate=1.0,
    ):
        """
        Args:
            static_dim: Dimension of static metadata
            history_dim: Dimension of historical inputs
            future_dim: Dimension of known future inputs
            hid_dim: Dimension of model
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.static_sel = VarSelectNet(
            static_dim, hid_dim, dropout_rate=dropout_rate
        )
        self.static_encoders = StaticCovariateEncoders(
            hid_dim, dropout_rate=dropout_rate
        )
        self.history_sel = VarSelectNet(
            history_dim, hid_dim, c_dim=hid_dim, dropout_rate=dropout_rate
        )
        self.future_sel = VarSelectNet(
            future_dim, hid_dim, c_dim=hid_dim, dropout_rate=dropout_rate
        )
        self.encoder_lstm = nn.LSTM(
            hid_dim, hid_dim, batch_first=True, dropout=dropout_rate
        )
        self.decoder_lstm = nn.LSTM(
            hid_dim, hid_dim, batch_first=True, dropout=dropout_rate
        )
        self.gated_skip_conn = GatedSkipConn(
            hid_dim, hid_dim, dropout_rate=dropout_rate
        )

    def forward(self, static, history: torch.Tensor, future: torch.Tensor):
        """
        Args:
            static: Static metadata = [batch size, static dim, hid dim]
            history: Historical inputs
                = [batch size, history len, history dim, hid dim]
            future: Known future inputs
                = [batch size, future len, future dim, hid dim]
                
        Returns:
            feature_history: Historical temporal features
                = [batch size, history len, hid dim]
            feature_future: Future temporal features
                = [batch size, future len, hid dim]
            c_enrichment: Context vector for static enrichment layer
                = [batch size, hid dim]
        """
        selected_static = self.static_sel(static)
        # selected_static = [batch size, hid dim]

        c_selection, c_cell, c_hidden, c_enrichment = self.static_encoders(
            selected_static
        )
        # c_selection, ... = [batch size, hid dim]

        history_len = history.shape[1]
        history_list = [
            self.history_sel(history[:, i, :, :], c=c_selection)
            for i in range(history_len)
        ]
        selected_history = torch.stack(history_list, dim=1)
        # selected_history = [batch size, history len, hid dim]

        future_len = future.shape[1]
        future_list = [
            self.future_sel(future[:, i, :, :], c=c_selection)
            for i in range(future_len)
        ]
        selected_future = torch.stack(future_list, dim=1)
        # selected_future = [batch size, future len, hid dim]

        out_lstm_history, (hidden_state, cell_state) = self.encoder_lstm(
            selected_history, (c_hidden, c_cell)
        )
        # out_lstm_history = [batch size, history len, hid dim]
        out_lstm_future, _ = self.decoder_lstm(
            selected_future, (hidden_state, cell_state)
        )
        # out_lstm_future = [batch size, future len, hid dim]

        feature_history_list = [
            self.gated_skip_conn(
                out_lstm_history[:, i, :], selected_history[:, i, :]
            )
            for i in range(history_len)
        ]
        feature_history = torch.stack(feature_history_list, dim=1)
        # feature_history = [batch size, history len, hid dim]

        feature_future_list = [
            self.gated_skip_conn(
                out_lstm_future[:, i, :], selected_future[:, i, :]
            )
            for i in range(future_len)
        ]
        feature_future = torch.stack(feature_future_list, dim=1)
        # feature_future = [batch size, future len, hid dim]

        return feature_history, feature_future, c_enrichment


# mask = torch.tril(
#     torch.ones((trg_len, src_len + trg_len), device=device),
#     diagonal=src_len).bool()

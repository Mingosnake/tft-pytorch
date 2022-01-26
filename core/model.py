import torch
from torch import nn
from torch import Tensor


class GatingLayer(nn.Module):
    """Gated Linear Unit (GLU).

    Attributes:
        dropout: Dropout layer
        fc_forward: Feed forward linear layer
        fc_gate: Linear layer for gate vector
        sigmoid: Sigmoid layer
    """

    def __init__(self, x_dim, out_dim, dropout_rate=0):
        """
        Args:
            x_dim: Input dimension
            out_dim: Output dimension
            dropout_rate: Dropout rate (optional)
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc_forward = nn.Linear(x_dim, out_dim)
        self.fc_gate = nn.Linear(x_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Input to gating layer = [batch size, *, x dim]

        Returns:
            Return of gating layer = [batch size, *, out dim]
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

    def __init__(self, x_dim, out_dim, dropout_rate=0):
        """
        Args:
            x_dim: Input dimension
            out_dim: Output dimension
            dropout_rate: Dropout rate (optional)
        """
        super().__init__()

        self.gating_layer = GatingLayer(x_dim, out_dim, dropout_rate)
        self.layernorm = nn.LayerNorm(out_dim)

    def forward(self, x_gate, x_skip):
        """
        Args:
            x_gate: Input to gating layer = [batch size, *, x dim]
            x_skip: Skipped input = [batch size, *, out dim]

        Returns:
            Added and normalized output = [batch size, *, out dim]
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
        self, x_dim, hid_dim, c_dim=None, out_dim=None, dropout_rate=0
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
        self.c_dim = c_dim
        out_dim = out_dim if out_dim else x_dim

        self.fc_skip = nn.Linear(x_dim, out_dim) if out_dim else None
        self.fc_x = nn.Linear(x_dim, hid_dim)
        if c_dim is not None:
            self.fc_c = nn.Linear(c_dim, hid_dim, bias=False)
        self.elu = nn.ELU()

        self.fc_forward = nn.Linear(hid_dim, hid_dim)
        self.gated_skip_conn = GatedSkipConn(hid_dim, out_dim, dropout_rate)

    def forward(self, x, c=None):
        """
        Args:
            x: Input = [batch size, *, x dim]
            c: Context (optional) = [batch size, 1, c dim]

        Returns:
            Output of GRN = [batch size, *, out dim]
        """
        if (self.c_dim is None) != (c is None):
            raise ValueError(f"{self.name} module is created wrong for c_dim")

        skip = self.fc_skip(x) if self.fc_skip else x
        hidden = self.fc_x(x)
        if c is not None:
            hidden = hidden + self.fc_c(c)
        hidden = self.elu(hidden)
        # hidden = [batch size, *, hid dim]
        hidden = self.fc_forward(hidden)
        # hidden = [batch size, *, hid dim]

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

    def __init__(self, var_dim, hid_dim, c_dim=None, dropout_rate=0):
        """
        Args:
            var_dim: Number of variables
            hid_dim: Dimension of hidden layer
            c_dim: Dimension of context vector (optional)
            dropout_rate: Dropout rate (optional)
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

        self.softmax = nn.Softmax(dim=-2)

        self.var_grn = GatedResNet(
            x_dim=hid_dim, hid_dim=hid_dim, dropout_rate=dropout_rate
        )

    def forward(self, x: Tensor, c=None):
        """
        Args:
            x: Input = [batch size, *, var dim, hid dim]
            c: Context (optional) = [batch size, 1, c dim]

        Returns:
            output: Output of variable selection network
                = [batch size, *, hid dim]
        """
        if (self.c_dim is None) != (c is None):
            raise ValueError(f"{self.name} module is created wrong for c_dim")

        flat_shape = list(x.shape[:-2])
        flat_shape.append(-1)
        flat = x.view(flat_shape)
        var_sel_wt = self.softmax(self.sel_wt_grn(flat, c))
        # var_sel_wt: Variable selection weights = [batch size, *, var dim]
        var_sel_wt = var_sel_wt.unsqueeze(-2)
        # var_sel_wt = [batch size, *, 1, var dim]

        variables = self.var_grn(x)
        # variables = [batch size, *, var dim, hid dim]
        if var_sel_wt.shape[:-2] == variables[:-2]:
            raise ValueError("Matrix shape does not match")
        output = torch.matmul(var_sel_wt, variables)
        # output = [batch size, *, 1, hid dim]
        output = output.squeeze(-2)
        # output = [batch size, *, hid dim]

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

    def __init__(self, hid_dim, n_heads, dropout_rate=0):
        """
        Args:
            hid_dim: Dimension of input and output in multi-head attention
            n_heads: Number of heads
            dropout_rate: Dropout rate (optional)
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

        self.scale = torch.FloatTensor([self.attn_dim ** -0.5])

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Query for multi-head attention
                = [batch size, query len, hid dim]
            key: Key for multi-head attention
                = [batch size, key len, hid dim]
            value: Value for multi-head attention
                = [batch size, value len, hid dim]
            mask: Masking tensor (optional) = [1, 1, query len, key len]

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

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.scale
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

    def __init__(self, hid_dim, dropout_rate=0):
        """
        Args:
            hid_dim: Dimension of static covariate encoders
            dropout_rate: Dropout rate (optional)
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
                = [batch size, 1, hid dim]
            c_cell: Context vector for initial cell state of LSTM
                = [batch size, 1, hid dim]
            c_hidden: Context vector for initial cell state of LSTM
                = [batch size, 1, hid dim]
            c_enrichment: Context vector for static enrichment layer
                = [batch size, 1, hid dim]
        """
        c_selection, c_cell, c_hidden, c_enrichment = tuple(
            grn(x).unsqueeze(1) for grn in self.grn_list
        )

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
        self, static_dim, history_dim, future_dim, hid_dim, dropout_rate=0
    ):
        """
        Args:
            static_dim: Dimension of static metadata
            history_dim: Dimension of historical inputs
            future_dim: Dimension of known future inputs
            hid_dim: Dimension of model
            dropout_rate: Dropout rate (optional)
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

    def forward(self, static, history: Tensor, future: Tensor):
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
                = [batch size, 1, hid dim]
        """
        selected_static = self.static_sel(static)
        # selected_static = [batch size, hid dim]

        c_selection, c_cell, c_hidden, c_enrichment = self.static_encoders(
            selected_static
        )
        # c_selection, ... = [batch size, 1, hid dim]

        selected_history = self.history_sel(history, c=c_selection)
        # selected_history = [batch size, history len, hid dim]

        selected_future = self.future_sel(future, c=c_selection)
        # selected_future = [batch size, future len, hid dim]

        out_lstm_history, (hidden_state, cell_state) = self.encoder_lstm(
            selected_history, (c_hidden, c_cell)
        )
        # out_lstm_history = [batch size, history len, hid dim]
        out_lstm_future, _ = self.decoder_lstm(
            selected_future, (hidden_state, cell_state)
        )
        # out_lstm_future = [batch size, future len, hid dim]

        feature_history = self.gated_skip_conn(
            out_lstm_history, selected_history
        )
        # feature_history = [batch size, history len, hid dim]

        feature_future = self.gated_skip_conn(out_lstm_future, selected_future)
        # feature_future = [batch size, future len, hid dim]

        return feature_history, feature_future, c_enrichment


class TemporalFusionDecoder(nn.Module):
    """Temporal fusion decoder.
    
    Attributes:
        static_enrichment_grn: Gated Residual Network for static enrichment
        multi_head_attention: Interpretable multi-head attention layer
        gated_skip_conn: Gated skip connection layer
        feed_forward_grn: Gated Residual Network for position-wise feed-forward
    """

    def __init__(self, hid_dim, n_heads, dropout_rate=0):
        """
        Args:
            hid_dim: Dimension of model
            n_heads: Number of heads for multi-head attention
            dropout_rate: Dropout rate (optional)
        """
        super().__init__()

        self.static_enrichment_grn = GatedResNet(
            hid_dim, hid_dim, c_dim=hid_dim, dropout_rate=dropout_rate
        )
        self.multi_head_attention = MultiHeadAttention(
            hid_dim, n_heads, dropout_rate=dropout_rate
        )
        self.gated_skip_conn = GatedSkipConn(
            hid_dim, hid_dim, dropout_rate=dropout_rate
        )
        self.feed_forward_grn = GatedResNet(
            hid_dim, hid_dim, dropout_rate=dropout_rate
        )

    def forward(self, feature_history, feature_future, c_enrichment):
        """
        Args:
            feature_history: Historical temporal features
                = [batch size, history len, hid dim]
            feature_future: Future temporal features
                = [batch size, future len, hid dim]
            c_enrichment: Context vector for static enrichment layer
                = [batch size, 1, hid dim]
                
        Returns:
            out_decoder: Output of temporal fusion decoder
                = [batch size, future len, hid dim]
            attention_score: Attention weights
                = [batch size, future len, history len + future len]
        """
        device = feature_history.device
        history_len = feature_history.shape[1]
        future_len = feature_future.shape[1]

        enriched_history = self.static_enrichment_grn(
            feature_history, c=c_enrichment
        )
        # enriched_history = [batch size, history len, hid dim]

        enriched_future = self.static_enrichment_grn(
            feature_future, c=c_enrichment
        )
        # enriched_future = [batch size, future len, hid dim]

        enriched_entire = torch.cat((enriched_history, enriched_future), dim=1)
        # enriched_entire = [batch size, history len + future len, hid dim]

        mask = (
            torch.tril(
                torch.ones(
                    (future_len, history_len + future_len), device=device
                ),
                diagonal=history_len,
            )
            .bool()
            .unsqueeze(0)
            .unsqueeze(1)
        )
        # mask = [1, 1, future len, history len + future len]

        out_attention, attention_score = self.multi_head_attention(
            enriched_future, enriched_entire, enriched_entire, mask=mask
        )
        # out_attention = [batch size, future len, hid dim]
        # attention_score = [batch size, future len, history len + future len]

        skip_conn_attention = self.gated_skip_conn(
            out_attention, enriched_future
        )
        # skip_conn_attention = [batch size, future len, hid dim]

        out_decoder = self.feed_forward_grn(skip_conn_attention)
        # out_decoder = [batch size, future len, hid dim]

        return out_decoder, attention_score


class TemporalFusionTransformer(nn.Module):
    """Temporal fusion transformer.
    """

    def __init__(
        self,
        static_dim,
        history_dim,
        future_dim,
        hid_dim,
        n_heads,
        out_dim,
        dropout_rate=0,
    ):
        super().__init__()

        self.quantiles = [10, 50, 90]
        self.seq_to_seq = Seq2Seq(
            static_dim,
            history_dim,
            future_dim,
            hid_dim,
            dropout_rate=dropout_rate,
        )
        self.temporal_fusion_decoder = TemporalFusionDecoder(
            hid_dim, n_heads, dropout_rate=dropout_rate
        )
        self.gated_skip_conn = GatedSkipConn(
            hid_dim, hid_dim, dropout_rate=dropout_rate
        )
        self.fc_out = nn.Linear(hid_dim, out_dim * len(self.quantiles))

    def forward(self, static, history, future):
        """
        Args:
        """
        future_len = future.shape[1]

        feature_history, feature_future, c_enrichment = self.seq_to_seq(
            static, history, future
        )
        # feature_history = [batch size, history len, hid dim]
        # feature_future = [batch size, future len, hid dim]
        # c_enrichment = [batch size, 1, hid dim]

        out_decoder, attention_score = self.temporal_fusion_decoder(
            feature_history, feature_future, c_enrichment
        )
        # out_decoder = [batch size, future len, hid dim]
        # attention_score = [batch size, future len, history len + future len]

        final_feature = self.gated_skip_conn(out_decoder, feature_future)
        # final_feature = [batch size, future len, hid dim]

        quantile = self.fc_out(final_feature)
        # quantile = [batch size, future len, 3]

        return quantile, attention_score


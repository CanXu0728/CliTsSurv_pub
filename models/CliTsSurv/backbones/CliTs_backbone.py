import copy

import numpy as np
import torch
from torch import nn
import os


class CliTs_attention(nn.Module):
    """Attention mechanism aligned with DDH"""
    def __init__(self, hidden_rnn, input_dim):
        super(CliTs_attention, self).__init__()
        # DDH: input_dim + hidden_rnn -> 1 (attention weight)
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_rnn + input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.attention_soft = nn.Softmax(dim=1)  # On temporal dimension

    def forward(self, hidden, x, lens):
        """
        Args:
            hidden: (batch, T, hidden_rnn) - RNN all-time outputs
            x: (batch, T, input_dim) - original input
            lens: (batch,) - actual lengths

        Returns:
            hidden_attentive: (batch, hidden_rnn) - weighted sum
            x_last: (batch, input_dim) - last observation
            attention_weights: (batch, T) - normalized temporal attention
        """
        batch_size, max_T, _ = hidden.shape
        device = hidden.device

        # Create mask for valid observations
        inputmask = torch.arange(max_T).unsqueeze(0).to(device) >= lens.unsqueeze(1)

        # Extract last observation (the one used for predictions)
        last_observations = lens - 1  # (batch,)
        last_observations_idx = last_observations.unsqueeze(1).repeat(1, max_T)
        index = torch.arange(max_T).repeat(batch_size, 1).to(device)

        last = index == last_observations_idx
        x_last = x[last]  # (batch, input_dim)

        # Concatenate all previous with x_last to measure attention
        concatenation = torch.cat([hidden, x_last.unsqueeze(1).repeat(1, max_T, 1)], -1)

        # Compute attention scores
        attention = self.attention_net(concatenation).squeeze(-1)  # (batch, T)

        # Mask future observations (beyond lens) and not observed data
        attention[inputmask] = -1e10

        # Normalize with softmax (only for sequences with >1 observation)
        attention[last_observations > 0] = self.attention_soft(attention[last_observations > 0])
        attention[last_observations == 0] = 0  # No context for only one observation

        # Save normalized weights (batch, T) before broadcasting
        attention_weights = attention.clone()

        # Weighted sum
        attention = attention.unsqueeze(2).repeat(1, 1, hidden.size(2))  # (batch, T, hidden_rnn)
        hidden_attentive = torch.sum(attention * hidden, dim=1)  # (batch, hidden_rnn)

        return hidden_attentive, x_last, attention_weights


class CliTs_backbone(nn.Module):
    def __init__(self, feature_in, hidden_size=32, num_layers=2, rnn_type='gru'):
        super(CliTs_backbone, self).__init__()

        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.lower()

        rnn_cls = {'lstm': nn.LSTM, 'gru': nn.GRU}.get(self.rnn_type)
        if rnn_cls is None:
            raise ValueError(f"rnn_type must be 'lstm' or 'gru', got '{rnn_type}'")

        self.rnn = rnn_cls(
            input_size=feature_in,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def max_pool_shared_z(self, z):
        max_agents = z.shape[0]
        shared_z, other_z = torch.split(z, [16, 16], dim=1)
        shared_z = shared_z.reshape(-1, max_agents, 16)
        shared_z = torch.max(shared_z, dim=1)[0].reshape(-1, 1, 16)
        shared_z = torch.tile(shared_z, (1, max_agents, 1)).reshape(-1, 16)
        return_z = torch.cat([shared_z, other_z], dim=1)
        return return_z

    def forward(self, x, lens):
        x_ = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        r_out, hidden = self.rnn(x_, None)
        r_out, lens = nn.utils.rnn.pad_packed_sequence(r_out, batch_first=True, total_length=x.shape[1])
        # Return all time points' outputs instead of just the last one
        return r_out, lens  # (batch, T, hidden_size), (batch,)


class InterpretableTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer that also returns self-attention weights.

    Parameters and state_dict layout are identical to nn.TransformerEncoderLayer,
    so checkpoints trained with the standard layer can be loaded directly.
    """

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Call self-attention with need_weights=True to obtain attention maps
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights  # attn_weights: (B, nhead, T, T)


class InterpretableTransformerEncoder(nn.Module):
    """Drop-in replacement for nn.TransformerEncoder that returns attention maps.

    Keeps the same attribute names (`layers`, `norm`) so that a state_dict
    trained with nn.TransformerEncoder loads without remapping.
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attn_maps = []
        for layer in self.layers:
            output, attn_weights = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
            attn_maps.append(attn_weights)
        if self.norm is not None:
            output = self.norm(output)
        return output, attn_maps


class CliTs_transformer_backbone(nn.Module):
    """Transformer backbone for CliTsSurv.

    A learnable CLS token is prepended to the projected input sequence.
    The CLS output after the (interpretable) TransformerEncoder serves as the
    aggregated sequence representation, replacing the RNN + CliTs_attention
    pipeline.

    `forward(..., return_attention=False)` additionally returns the per-layer
    CLS -> visit attention weights, averaged over heads:
        list of (B, T) tensors, one per encoder layer.
    """
    def __init__(self, feature_in, hidden_size=64, num_layers=2,
                 nhead=4, dim_feedforward=256, dropout=0.1):
        super(CliTs_transformer_backbone, self).__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead

        self.input_proj = nn.Linear(feature_in, hidden_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = InterpretableTransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        # norm=None: the original nn.TransformerEncoder was constructed without
        # a final LayerNorm, so checkpoints contain no `transformer.norm.*` keys.
        self.transformer = InterpretableTransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=None,
        )

    def forward(self, x, lens, return_attention=False):
        """
        Args:
            x:    (B, T, feature_in)
            lens: (B,) LongTensor - actual unpadded lengths
            return_attention: bool - also return per-layer CLS attention maps

        Returns:
            cls_out: (B, hidden_size)
            attention (optional): list of (B, T) per-layer CLS->visit attention
        """
        B, T, _ = x.shape

        x_proj = self.input_proj(x)                          # (B, T, hidden_size)
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, hidden_size)
        x_with_cls = torch.cat([cls, x_proj], dim=1)         # (B, T+1, hidden_size)

        # key_padding_mask: True = ignore that position
        # index 0 is CLS (always valid); indices 1..T valid up to lens[b]
        positions = torch.arange(T + 1, device=x.device).unsqueeze(0)  # (1, T+1)
        key_padding_mask = positions > lens.unsqueeze(1)     # (B, T+1)

        out, attn_maps = self.transformer(
            x_with_cls, src_key_padding_mask=key_padding_mask
        )
        cls_out = out[:, 0, :]                               # CLS output: (B, hidden_size)

        if not return_attention:
            return cls_out

        # Extract CLS -> visit attention from each layer, average over heads.
        # attn: (B, nhead, T+1, T+1); CLS is query index 0, visits are keys 1..T
        per_layer_cls_attn = []
        for attn in attn_maps:
            cls_attn = attn[:, :, 0, 1:]          # (B, nhead, T)
            cls_attn = cls_attn.mean(dim=1)       # (B, T) average over heads
            per_layer_cls_attn.append(cls_attn)

        return cls_out, per_layer_cls_attn


class CliTsSurv_backbone(nn.Module):
    """DeepHit version for survival analysis with Attention mechanism (aligned with DDH).

    Supports rnn_type in {'lstm', 'gru', 'transformer'}.
    - lstm / gru  : CliTs_backbone (RNN) + CliTs_attention
    - transformer : CliTs_transformer_backbone (CLS token aggregation)
    In all cases the output shape is (batch, hidden_size + feature_in).

    `forward(..., return_attention=False)` additionally returns temporal
    attention weights:
      - RNN path:        (B, T) tensor (from CliTs_attention)
      - Transformer path: list of (B, T) tensors (one per encoder layer)
    """
    def __init__(self, feature_in, hidden_size=32, num_layers=2, rnn_type='gru',
                 nhead=4, dim_feedforward=None, dropout=0.1):
        super(CliTsSurv_backbone, self).__init__()

        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.lower()
        self.feature_in = feature_in

        if dim_feedforward is None:
            dim_feedforward = hidden_size * 4

        if self.rnn_type == 'transformer':
            self.backbone = CliTs_transformer_backbone(
                feature_in, hidden_size, num_layers, nhead, dim_feedforward, dropout
            )
        else:
            self.backbone = CliTs_backbone(feature_in, hidden_size, num_layers, rnn_type=self.rnn_type)
            self.attention = CliTs_attention(hidden_size, feature_in)

    def forward(self, x, lens, return_attention=False):
        # x: (batch, T, feature_in)
        if not isinstance(lens, torch.Tensor):
            lens = torch.tensor(lens, dtype=torch.long, device=x.device)

        attention = None

        if self.rnn_type == 'transformer':
            if return_attention:
                hidden_attentive, attention = self.backbone(x, lens, return_attention=True)
            else:
                hidden_attentive = self.backbone(x, lens)
            # Extract last valid observation - same role as x_last in the RNN path
            B = x.size(0)
            last_idx = (lens - 1).clamp(min=0)
            idx = last_idx.view(B, 1, 1).expand(B, 1, x.size(2))
            x_last = x.gather(1, idx).squeeze(1)               # (batch, feature_in)
        else:
            r_out, lens = self.backbone(x, lens)               # (batch, T, hidden_size)
            hidden_attentive, x_last, attention = self.attention(r_out, x, lens)

        out = torch.cat([hidden_attentive, x_last], dim=-1)    # (batch, hidden_size + feature_in)

        if return_attention:
            return out, attention
        return out

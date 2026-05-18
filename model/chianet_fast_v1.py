import copy

import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, size, stride=2, hidden_in=64, hidden=64):
        super().__init__()
        pad_len = int(size / 2)
        self.scale = nn.Sequential(
            nn.Conv1d(hidden_in, hidden, size, stride, pad_len),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv1d(hidden, hidden, size, padding=pad_len),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, size, padding=pad_len),
            nn.BatchNorm1d(hidden),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        res_out = self.res(scaled)
        return self.relu(res_out + scaled)


class Encoder(nn.Module):
    def __init__(self, in_channel, output_size=256, filter_size=5, num_blocks=12):
        super().__init__()
        self.filter_size = filter_size
        self.conv_start = nn.Sequential(
            nn.Conv1d(in_channel, 32, 11, 2, 5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        hiddens = [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        self.res_blocks = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        self.conv_end = nn.Conv1d(256, output_size, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        return self.conv_end(x)

    def get_res_blocks(self, n, hidden_ins, hiddens):
        blocks = []
        for _, hidden, hidden_in in zip(range(n), hiddens, hidden_ins):
            blocks.append(ConvBlock(self.filter_size, hidden_in=hidden_in, hidden=hidden))
        return nn.Sequential(*blocks)


class EncoderSplit(Encoder):
    def __init__(self, num_epi, output_size=256, filter_size=5, num_blocks=12):
        super(Encoder, self).__init__()
        self.filter_size = filter_size
        self.conv_start = nn.Sequential(
            nn.Conv1d(5 + num_epi, 32, 11, 2, 5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        hiddens = [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        self.res_blocks = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        self.conv_end = nn.Conv1d(256, output_size, 1)

    def forward(self, x):
        x = self.res_blocks(self.conv_start(x))
        return self.conv_end(x)


class TransformerLayer(torch.nn.TransformerEncoderLayer):
    """Pre-LN Transformer encoder layer matching the training code."""

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src_norm = self.norm1(src)
        src_side, attn_weights = self.self_attn(
            src_norm,
            src_norm,
            src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout1(src_side)

        src_norm = self.norm2(src)
        src_side = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src_side)
        return src, attn_weights


class TransformerEncoder(torch.nn.TransformerEncoder):
    def __init__(self, encoder_layer, num_layers, norm=None, record_attn=False):
        super().__init__(encoder_layer, num_layers)
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.record_attn = record_attn

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attn_weight_list = []

        for mod in self.layers:
            output, attn_weights = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )
            attn_weight_list.append(attn_weights.unsqueeze(0).detach())

        if self.norm is not None:
            output = self.norm(output)

        if self.record_attn:
            return output, torch.cat(attn_weight_list)
        return output

    def _get_clones(self, module, n):
        return torch.nn.modules.ModuleList([copy.deepcopy(module) for _ in range(n)])


class PositionalEncoding(nn.Module):
    def __init__(self, hidden, dropout=0.1, max_len=256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden, 2) * (-np.log(10000.0) / hidden))
        pe = torch.zeros(max_len, 1, hidden)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class AttnModule(nn.Module):
    def __init__(self, hidden=128, layers=8, record_attn=False):
        super().__init__()
        self.record_attn = record_attn
        self.pos_encoder = PositionalEncoding(hidden, dropout=0.1)
        encoder_layer = TransformerLayer(
            hidden,
            nhead=8,
            dropout=0.1,
            dim_feedforward=512,
            batch_first=True,
        )
        self.module = TransformerEncoder(encoder_layer, layers, record_attn=record_attn)

    def forward(self, x):
        x = self.pos_encoder(x)
        return self.module(x)


class ResBlockDilated(nn.Module):
    def __init__(self, size, hidden=64, dil=2):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(hidden, hidden, size, padding=dil, dilation=dil),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, size, padding=dil, dilation=dil),
            nn.BatchNorm2d(hidden),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        res_out = self.res(x)
        return self.relu(res_out + x)


class Decoder(nn.Module):
    def __init__(self, in_channel, hidden=256, filter_size=3, num_blocks=5):
        super().__init__()
        self.filter_size = filter_size
        self.conv_start = nn.Sequential(
            nn.Conv2d(in_channel, hidden, 3, 1, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        self.res_blocks = self.get_res_blocks(num_blocks, hidden)
        self.conv_end_map = nn.Conv2d(hidden, 1, 1)
        self.conv_end_loop = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        contact_map = self.conv_end_map(x)
        loop = self.conv_end_loop(x)
        return contact_map, loop

    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated(self.filter_size, hidden=hidden, dil=dilation))
        return nn.Sequential(*blocks)


class DNASequenceModel(nn.Module):
    def __init__(self, num_genomic_features=1, mid_hidden=256, record_attn=False):
        super().__init__()
        self.encoder = EncoderSplit(num_genomic_features, output_size=mid_hidden, num_blocks=12)
        self.attn = AttnModule(hidden=mid_hidden, record_attn=record_attn)
        self.decoder = Decoder(mid_hidden * 2)
        self.record_attn = record_attn
        self.upper_tri = UpperTri()
        self.log_sigma_map = nn.Parameter(torch.zeros(1))
        self.log_sigma_loop = nn.Parameter(torch.zeros(1))

    def diagonalize(self, x):
        x_i = x.unsqueeze(2).repeat(1, 1, 256, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, 256)
        return torch.cat([x_i, x_j], dim=1)

    def forward(self, sequence, genomic_features):
        x = torch.cat((sequence, genomic_features), dim=1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        if self.record_attn:
            x, attn_weights = self.attn(x)
        else:
            x = self.attn(x)
        x = x.permute(0, 2, 1)
        x = self.diagonalize(x)
        contact_map, loop = self.decoder(x)
        contact_map = self.upper_tri(contact_map).squeeze(1)
        loop = torch.sigmoid(self.upper_tri(loop).squeeze(1))

        if self.record_attn:
            return contact_map, loop, attn_weights, self.log_sigma_map, self.log_sigma_loop
        return contact_map, loop, self.log_sigma_map, self.log_sigma_loop


class UpperTri(nn.Module):
    """Unroll a square matrix to its upper triangular portion."""

    def __init__(self, diagonal_offset=1):
        super().__init__()
        self.diagonal_offset = diagonal_offset

    def forward(self, inputs):
        _, _, seq_len, _ = inputs.size()
        triu_indices = torch.triu_indices(
            seq_len,
            seq_len,
            offset=self.diagonal_offset,
            device=inputs.device,
        )
        return inputs[:, :, triu_indices[0], triu_indices[1]]

    def extra_repr(self):
        return f"diagonal_offset={self.diagonal_offset}"

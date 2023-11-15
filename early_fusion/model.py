import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from config import ACTIVATION_FUNCTIONS, device
from perceiver_pytorch import PerceiverLM, PerceiverIO
from einops import rearrange, repeat

class RNN(nn.Module):
    def __init__(self, d_in, d_out, n_layers=1, bi=True, dropout=0.2, n_to_1=False):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_out, bidirectional=bi, num_layers=n_layers, dropout=dropout)
        self.n_layers = n_layers
        self.d_out = d_out
        self.n_directions = 2 if bi else 1
        self.n_to_1 = n_to_1

    def forward(self, x, x_len):
        x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        rnn_enc = self.rnn(x_packed)

        if self.n_to_1:
            # hiddenstates, h_n, only last layer
            return last_item_from_packed(rnn_enc[0], x_len)
            #batch_size = x.shape[0]
            #h_n = h_n.view(self.n_layers, self.n_directions, batch_size, self.d_out) # (NL, ND, BS, dim)
            #last_layer = h_n[-1].permute(1,0,2) # (BS, ND, dim)
            #x_out = last_layer.reshape(batch_size, self.n_directions * self.d_out) # (BS, ND*dim)

        else:
            x_out = rnn_enc[0]
            x_out = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]

        return x_out

#https://discuss.pytorch.org/t/get-each-sequences-last-item-from-packed-sequence/41118/7
def last_item_from_packed(packed, lengths):
    sum_batch_sizes = torch.cat((
        torch.zeros(2, dtype=torch.int64),
        torch.cumsum(packed.batch_sizes, 0)
    )).to(device)
    sorted_lengths = lengths[packed.sorted_indices].to(device)
    last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0)).to(device)
    last_seq_items = packed.data[last_seq_idxs]
    last_seq_items = last_seq_items[packed.unsorted_indices]
    return last_seq_items

class OutLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=.0, bias=.0):
        super(OutLayer, self).__init__()
        self.fc_1 = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.fc_2 = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.fc_2.bias.data, bias)

    def forward(self, x):
        y = self.fc_2(self.fc_1(x))
        return y

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        v = torch.arange(0, d_model, 2).type(torch.float)
        v = v * -(math.log(1000.0) / d_model)
        div_term = torch.exp(v)
        pe[:, 0::2] = torch.sin(position.type(torch.float) * div_term)
        pe[:, 1::2] = torch.cos(position.type(torch.float) * div_term)
        pe = pe.unsqueeze(0).to(device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params
        self.inp = nn.Linear(params.d_in, params.model_dim, bias=False)

        self.max_seq_len = 1000
        self.pos_emb = PositionalEncoding(d_model=params.model_dim,dropout=0.)

        self.perceiver_io = PerceiverIO(
            dim = params.model_dim,
            queries_dim = params.model_dim,
            depth=6,  # depth of net
            num_latents=256,
            # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=512,  # latent dimension
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=64,  # number of dimensions per cross attention head
            latent_dim_head=64,  # number of dimensions per latent self attention head
            weight_tie_layers=False  # whether to weight tie layers (optional, as indicated in the diagram)
        )
        self.out = OutLayer(params.model_dim, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def generate_sent_masks(self, batch_size, max_seq_length, source_lengths):
        enc_masks = torch.zeros(batch_size, max_seq_length, dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, :src_len] = 1
        return enc_masks

    def forward(self, x, x_len):
        x = self.inp(x)
        x = self.pos_emb(x)
        padding_mask = self.generate_sent_masks(x.shape[0], x.shape[1], x_len).cuda().bool()
        logits = self.perceiver_io(x, mask=padding_mask, queries=x)
        y = self.out(logits)
        y = self.final_activation(y)
        return y,logits

    def set_n_to_1(self, n_to_1):
        self.encoder.n_to_1 = n_to_1

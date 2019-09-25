import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn.inits import reset
from torch_geometric.nn import NNConv, Set2Set


class DDI_MLP(nn.Module):
    def __init__(self, input_feature_dim, num_types, dim):
        super(DDI_MLP, self).__init__()
        self.classifer = nn.Sequential(
            nn.Linear(2*input_feature_dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, num_types)
        )

    def forward(self, h, edge_index, edge_attr):
        h1 = h.index_select(0, edge_index[0])  # row, N,d
        h2 = h.index_select(0, edge_index[1])  # col, N,d
        h3 = torch.cat([h1, h2], -1)  # N,2d
        output = self.classifer(h3)
        return F.sigmoid(output)


class DDIEncoder(nn.Module):
    def __init__(self, input_feature_dim, num_types, dim):
        super(DDIEncoder, self).__init__()
        nn1 = nn.Sequential(
            nn.Linear(num_types, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, input_feature_dim*dim)
        )
        nn2 = nn.Sequential(
            nn.Linear(num_types, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim*dim)
        )

        self.conv1 = NNConv(input_feature_dim, dim, nn1,
                            aggr='mean')
        self.batch_norm = nn.BatchNorm1d(dim)
        self.conv2 = NNConv(dim, dim, nn2, aggr='mean')

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.batch_norm(self.conv1(x, edge_index, edge_attr)))
        output = self.conv2(h, edge_index, edge_attr)
        return output


class DDIDecoder(nn.Module):
    def __init__(self, num_types, dim):
        super(DDIDecoder, self).__init__()
        self.model = DDI_MLP(dim, num_types, dim)

    def forward(self, h, edge_index, edge_attr=None):
        return self.model(h, edge_index, edge_attr)


class DDINet(nn.Module):
    def __init__(self, encoder, decoder, decoder2=None):
        super(DDINet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder  # used for inference
        self.decoder2 = decoder2  # used for test phase in an energy model

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)
        reset(self.decoder2)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilties."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, h, edge_index, edge_attr):
        out = self.decoder(h, edge_index, edge_attr)
        loss = F.binary_cross_entropy(out, edge_attr)
        return loss

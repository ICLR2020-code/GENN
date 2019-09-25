import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn.inits import reset
from torch_geometric.nn import NNConv
from torch_geometric.nn.conv import localedge_conv

from torch_geometric.nn.models.model_DDINet import DDIEncoder


class DDI_Energy_Pooling(torch.nn.Module):
    def __init__(self, dim):
        super(DDI_Energy_Pooling, self).__init__()
        self.lin1 = torch.nn.Linear(dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, h):
        # out = F.relu(self.lin1(h3))
        out = torch.mean(h, 0)
        out = F.relu(self.lin1(out))
        out = F.relu(self.lin2(out).sum())
        return out


class DDI_Energy_Net(torch.nn.Module):
    def __init__(self, encoder, pooling, decoder1=None, decoder2=None):
        super(DDI_Energy_Net, self).__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.decoder1 = decoder1
        self.decoder2 = decoder2

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder1)
        reset(self.decoder2)
        reset(self.pooling)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def pool(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilties."""
        return self.pooling(*args, **kwargs)

    def forward(self, x, edge_index, edge_attr):
        h = self.encoder(x, edge_index, edge_attr)
        energy = self.pooling(h)
        return energy


class DDI_LocalEnergy_Net(torch.nn.Module):
    def __init__(self, input_feature_dim, num_types, dim):
        super(DDI_LocalEnergy_Net, self).__init__()
        self.lin0 = torch.nn.Linear(input_feature_dim, dim)
        self.bn0 = torch.nn.BatchNorm1d(dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.lin1 = torch.nn.Linear(dim, 1)

        nn1 = Sequential(Linear(num_types, 32), torch.nn.BatchNorm1d(
            32), ReLU(), Linear(32, dim * dim))
        self.conv1 = localedge_conv.LocalEdgeConv(
            dim, dim, nn1, aggr='add', root_weight=False)

    def forward(self, x, edge_index, edge_attr):
        x = x.to(self.lin0.weight.device)
        edge_index = edge_index.to(self.lin0.weight.device)
        edge_attr = edge_attr.to(self.lin0.weight.device)
        h = F.relu(self.bn0(self.lin0(x)))
        h = F.relu(self.bn1(self.conv1(h, edge_index, edge_attr)))
        h = self.lin1(h).mean()
        # h = F.relu(self.conv2(h, edge_index, edge_attr)) #n,d
        return h

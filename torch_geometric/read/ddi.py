import sys
import os.path as osp
from itertools import repeat

import torch
import numpy as np
from sklearn import random_projection
from sklearn.model_selection import train_test_split
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.read import read_txt_array
from torch_geometric.utils import remove_self_loops, one_hot
from collections import OrderedDict

try:
    import cPickle as pickle
except ImportError:
    import pickle


def read_ddi_data(folder, prefix, random_seed, data_ratio):
    # data_ratio = 100  # for ratio test purpose
    # random_seed = 3  # for analytis test purpose
    np.random.seed(random_seed)
    names = ['allx', 'graph']
    items = [read_file(folder, prefix, name) for name in names]
    x, graph = items

    edge_index, edge_attr = edge_index_from_dict(graph, num_nodes=x.size(0))

    # if not osp.exists("perm.idx.%d.npy" % random_seed):
    #     perm = np.random.permutation(edge_index.size(1))
    #     np.save(osp.join(folder, "perm.idx.%d.npy" % random_seed), perm)
    # else:
    #     perm = np.load(osp.join(folder, "perm.idx.%d.npy" % random_seed))
    if not osp.exists(osp.join(folder, "perm.idx.seed_%d_ratio_%d.npy" % (random_seed, data_ratio))):
        perm = np.random.permutation(edge_index.size(1))
        perm = perm[:int(edge_index.size(1)*data_ratio/100)]
        np.save(osp.join(osp.join(folder, "perm.idx.seed_%d_ratio_%d.npy" %
                                  (random_seed, data_ratio))), perm)
    else:
        perm = np.load(osp.join(
            osp.join(folder, "perm.idx.seed_%d_ratio_%d.npy" % (random_seed, data_ratio))))

    perm = torch.LongTensor(perm)
    print('original stat', x.size(0))
    print('node count', x.size(0))
    print('edge count', edge_index.size(1))
    print('max in edge_index', max(np.unique(edge_index[1])))

    # reducing size of dataset with data_ratio
    edge_index = edge_index[:, perm]
    edge_attr = edge_attr[perm]

    print('now stat')
    print('node count', x.size(0))
    print('edge count', edge_index.size(1))
    print('max in edge_index', max(np.unique(edge_index[1])))

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, perm=perm)

    return data


def read_file(folder, prefix, name):
    path = osp.join(folder, 'ind.{}.{}'.format(prefix.lower(), name))

    if name == 'test.index':
        return read_txt_array(path, dtype=torch.long)

    with open(path, 'rb') as f:
        if sys.version_info > (3, 0):
            out = pickle.load(f, encoding='latin1')
        else:
            out = pickle.load(f)

    if name == 'graph':
        return out

    out = out.todense() if hasattr(out, 'todense') else out
    print('If input x has nan or inf', np.isinf(out).any(), np.isnan(out).any())

    # for fast training, we discard one-hot encoding and use 32 dimension vector from gaussian distribution
    if prefix == 'ddi_constraint' or prefix == 'decagon':
        if name == 'allx':
            transformer = random_projection.GaussianRandomProjection(
                n_components=32)
            out = transformer.fit_transform(out)
    out = torch.FloatTensor(out)
    return out


def edge_index_from_dict(graph_dict, num_nodes=None):
    row, col = [], []
    edgetype = []
    new_type = 0
    type = 0
    for fake_type, edges in graph_dict.items():
        # print('fake_type: {} to {}'.format(fake_type, type))
        # # num_pairs= sum([len(value) for key, value in edges.items()])
        # num_pairs = len(edges)
        # if num_pairs<10:
        #     continue
        # else:
        #     new_type +=1
        # for key, value in edges.items():
        #
        #     row += repeat(key, len(value))
        #     col += value
        #     edgetype +=repeat(new_type-1, len(value))
        for key, value in edges.items():
            row += repeat(key, len(value))
            col += value
            edgetype += repeat(type, len(value))
        type += 1
    print('edge type ', type)
    print('edgetype length ', len(edgetype))
    edge_attr = one_hot(torch.tensor(edgetype))
    edge_index, edge_attr = collapse(row, col, edge_attr)
    edge_attr = edge_attr > 0
    return edge_index, edge_attr.float()


def collapse(row, col, edge_attr):
    # index = row*num_nodes + col
    dict1 = {}
    for i in range(len(row)):
        if row[i] < col[i]:
            ind = (row[i], col[i])
        else:
            ind = (col[i], row[i])
        if ind not in dict1:
            dict1[ind] = edge_attr[i]
        else:
            dict1[ind] += edge_attr[i]
    new_row = []
    new_col = []
    new_edge_attr = []
    for key in dict1.keys():
        new_row.append(key[0])
        new_col.append(key[1])
        new_edge_attr.append(dict1.get(key))
    edge_index = torch.stack(
        [torch.LongTensor(new_row), torch.LongTensor(new_col)], dim=0)
    new_edge_attr = torch.stack(new_edge_attr, dim=0)
    return edge_index, new_edge_attr


def sample_mask(index, num_nodes):
    mask = torch.zeros((num_nodes, ), dtype=torch.uint8)
    mask[index] = 1
    return mask


if __name__ == "__main__":
    folder = "../../data/DDISynthetic"
    read_ddi_data(folder, "ddi_constraint")

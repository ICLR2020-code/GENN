import torch
import numpy as np
import os
import os.path as osp
import gensim
from tqdm import tqdm
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split
from collections import OrderedDict

from DDItest.get_commanders import base_commanders
from torch_geometric.read.ddi import read_ddi_data
from DDItest.util import metric_report, collect_report


args = base_commanders()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
METHOD_NAME = 'LP'

MODEL_OUTPUT_PATH = 'saved_models/LP_data:{}_seed_{}_ratio:{}/'.format(
    args.data_prefix,
    args.seed,
    args.data_ratio
)
if not osp.exists(MODEL_OUTPUT_PATH):
    os.makedirs(MODEL_OUTPUT_PATH)


# --- load dataset ---
dataset = read_ddi_data(args.data_dir, args.data_prefix,
                        args.seed, args.data_ratio)
x = dataset.x.detach().cpu().numpy()
edge_attr = dataset.edge_attr.detach().cpu().numpy()
perm = dataset.perm
num_edges, num_types = dataset.edge_attr.size()

# --- classification ----


# --- concat two node embedding for training
X = []
Y = edge_attr
perm_edge_index = dataset.edge_index

for i, j in tqdm(zip(perm_edge_index[0], perm_edge_index[1]), desc="concat for training"):
    i = i.item()
    j = j.item()
    X.append(np.concatenate([x[i], x[j]], axis=0))
new_X = []
new_Y = []
for i in range(Y.shape[0]):
    nonzero_index = Y[i, :].nonzero()
    for j in nonzero_index:
        new_X.append(X[i])
        new_Y.append(j[0])
X = np.array(new_X)
Y = np.array(new_Y)
print(len(np.unique(Y)))
# # remove label (#<2) and reorder y
# remove_idx = []
# for i in np.unique(Y):
#     if sum(Y == i) < 2:
#         remove_idx.append((Y == i).nonzero()[0])
# X = np.delete(X, remove_idx, axis=0)
# Y = np.delete(Y, remove_idx, axis=0)
# # reorder y
# for idx, i in enumerate(np.unique(Y)):
#     Y[Y == i] = idx

assert X.shape[0] == Y.shape[0]

# --- training
# 1. sampling

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-args.data_ratio)/100, stratify=Y, random_state=args.seed)

# complete Y_train if unique number < 113
unique_y_train = list(np.unique(Y_train))
add_X = []
add_Y = []
for i in range(113):
    if i in unique_y_train:
        continue
    idx = (Y_test == i).nonzero()[0][0]
    add_X.append(X_test[idx])
    add_Y.append(Y_test[idx])
if len(add_X) != 0:
    X_train = np.r_[X_train, np.array(add_X)]
    Y_train = np.r_[Y_train, np.array(add_Y)]

print('train unique Y:{} test uniuqe Y:{}'.format(
    len(np.unique(Y_train)), len(np.unique(Y_test))
))

classifier = LabelPropagation(
    kernel='rbf', n_jobs=50, max_iter=200, gamma=0.25)
# Y_train[int(len(Y_train)*0.8):] = -1
print('-'*15)
print(X_train.shape, Y_train.shape, len(np.unique(Y_train)))
classifier.fit(X_train, Y_train)

# --- testing
y_prob = classifier.predict_proba(X_test)
# --- report
one_hot_Y = np.zeros((X_test.shape[0], len(np.unique(Y))))
one_hot_Y[np.arange(X_test.shape[0]), Y_test] = 1
test_metrics = metric_report(one_hot_Y, y_prob)
print(test_metrics)
collect_report(METHOD_NAME, args.data_ratio, test_metrics['pr'])

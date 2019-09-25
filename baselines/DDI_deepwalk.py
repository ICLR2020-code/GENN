import torch
import numpy as np
import os
import os.path as osp
import gensim
from tqdm import tqdm
from itertools import combinations
from tensorboardX import SummaryWriter
from torch_geometric.data import Data, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F


from DDItest.get_commanders import deepwalk_commanders
from torch_geometric.read.ddi import read_ddi_data
from DDItest.util import metric_report, noise_dataset, collect_report, collect_pairwise_relation
from torch_geometric.nn.models.model_DDINet import DDI_MLP
import warnings

warnings.filterwarnings("ignore")
MODEL_NAME = 'pytorch.model'
METHOD_NAME = 'DeepWalk'


"""
link: https://github.com/phanein/deepwalk
"""

args = deepwalk_commanders()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

target = 0
dim = 100
threshold = 0.4
train_size = args.train_size
val_size = args.val_size

model_save_path = 'saved_models/DeepWalk_dataset_{}_seed_{}_ratio_{}/'.format(
    args.data_prefix,
    args.seed,
    args.data_ratio
)
EDGE_LIST_PATH = osp.join(model_save_path, 'edge_list.txt')
OUTPUT_EMB_PATH = osp.join(model_save_path, 'embeddings.bin')

if not osp.exists(model_save_path):
    os.makedirs(model_save_path)

# --- load dataset ---
dataset = read_ddi_data(args.data_dir, args.data_prefix,
                        args.seed, args.data_ratio)
x = dataset.x.detach().cpu().numpy()
edge_attr = dataset.edge_attr.detach().cpu().numpy()
num_edges, num_types = dataset.edge_attr.size()

# --- generate deepwalk embedding ---
if not osp.exists(EDGE_LIST_PATH) or not osp.exists(OUTPUT_EMB_PATH):
    data = dataset
    with open(EDGE_LIST_PATH, 'w') as f:
        unique_node = set()
        for i, j in zip(data.edge_index[0], data.edge_index[1]):
            f.write('%d %d\n' % (i, j))
            unique_node.add(i)
            unique_node.add(i)
        # # add nodes not in graph
        for i in set(range(len(x))) - unique_node:
            j = np.random.choice(list(unique_node), size=1)
            f.write('%d %d\n' % (i, j))
    command_str = "deepwalk --format edgelist --input {} --output {} --seed {} --representation-size {} --walk-length 20".format(
        EDGE_LIST_PATH,
        OUTPUT_EMB_PATH,
        args.seed,
        args.representation_size
    )
    os.system(command_str)

# --- load deepwalk emb ---
wv = gensim.models.KeyedVectors.load_word2vec_format(OUTPUT_EMB_PATH).vectors
wv = torch.Tensor(wv)
dataset.x = torch.cat([wv, dataset.x], dim=-1)

# Split datasets.
# generate edge_combination for noise train dataset
# edge_combination = list(combinations(range(dataset.x.size(0)), 2))
train_dataset = Data(
    x=dataset.x, edge_index=dataset.edge_index[:, :int(num_edges*train_size)], edge_attr=dataset.edge_attr[:int(num_edges*train_size), :])
val_dataset = Data(
    x=dataset.x, edge_index=dataset.edge_index[:, int(num_edges*train_size):int(num_edges*val_size)], edge_attr=dataset.edge_attr[int(num_edges*train_size):int(num_edges*val_size), :])
test_dataset = Data(
    x=dataset.x, edge_index=dataset.edge_index[:, int(num_edges*val_size):], edge_attr=dataset.edge_attr[int(num_edges*val_size):, :])


if args.no_cuda:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = train_dataset.to(device)
val_dataset = val_dataset.to(device)
test_dataset = test_dataset.to(device)

model = DDI_MLP(train_dataset.x.size(
    1), num_types=num_types, dim=dim).to(device)
if args.resume_path is not None:
    if osp.exists(osp.join(args.resume_path, MODEL_NAME)):
        print('load pretraining model from ' + args.resume_path)
        model.load_state_dict(torch.load(
            osp.join(args.resume_path, MODEL_NAME)))
else:
    if osp.exists(osp.join(model_save_path, MODEL_NAME)):
        print('load pretraining model from ' + model_save_path)
        model.load_state_dict(torch.load(
            osp.join(model_save_path, MODEL_NAME)))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    optimizer.zero_grad()
    pred = model(train_dataset.x, train_dataset.edge_index,
                 train_dataset.edge_attr)
    loss = F.binary_cross_entropy(pred, train_dataset.edge_attr)
    loss.backward()
    # loss_all += loss.item()
    optimizer.step()
    return loss.item()


def test2(val_data):  # test roc-auc, pr-auc per label type, same as decagon
    model.eval()

    y_prob = model(val_data.x, val_data.edge_index, val_data.edge_attr)

    y, y_prob = val_data.edge_attr.detach().cpu().numpy(), y_prob.detach().cpu().numpy()

    return metric_report(y, y_prob)


if not args.only_test:
    writer = SummaryWriter(model_save_path)

    best_val_roc = None
    best_val_roc_progress = {'cnt': 0, 'value': None}
    for epoch in range(1, args.epoch):

        loss = train(epoch)
        val_metrics = test2(val_dataset)
        val_roc, val_pr = val_metrics['roc'], val_metrics['pr']

        # early stop
        if best_val_roc_progress['value'] is None or val_roc > best_val_roc_progress['value']:
            best_val_roc_progress['cnt'] = 0
            best_val_roc_progress['value'] = val_roc

        if val_roc < best_val_roc_progress['value']:
            best_val_roc_progress['cnt'] += 1
        if best_val_roc_progress['cnt'] > args.n_iter_no_change:
            writer.add_scalar('text/early_stop_epoch', epoch, 0)
            print('early stop')
            break

        if best_val_roc is None or val_roc >= best_val_roc:
            test_metrics = test2(test_dataset)
            best_val_roc = val_roc
            torch.save(model.state_dict(), osp.join(
                model_save_path, MODEL_NAME))
            writer.add_scalar('test/roc', test_metrics['roc'], epoch-1)
            writer.add_scalar('test/pr', test_metrics['pr'], epoch-1)
            writer.add_scalar('test/p@1', test_metrics['p@1'], epoch-1)
            writer.add_scalar('test/p@3', test_metrics['p@3'], epoch-1)
            writer.add_scalar('test/p@5', test_metrics['p@5'], epoch-1)

        writer.add_scalar('train/loss', loss, epoch-1)
        writer.add_scalar('val/roc', val_roc, epoch-1)
        writer.add_scalar('val/pr', val_pr, epoch-1)

        print('Epoch: {:03d}, Loss: {:.7f}, Validation ROC_AUC: {:.7f}, Validation PR_AUC: {:.7f}'.format(
            epoch, loss, val_roc, val_pr)
        )
    writer.close()

else:
    def pairwise_correlation2(val_data):
        model.eval()

        pred = model(val_data.x, val_data.edge_index, val_data.edge_attr)

        y, pred = val_data.edge_attr.detach().cpu().numpy(), pred.detach().cpu().numpy()

        edge_index = val_data.edge_index.detach().cpu().numpy()
        assert len(edge_index[0]) == len(pred)
        collect_pairwise_relation(
            METHOD_NAME, y, pred, edge_index, val_data.x.size(0), threshold=threshold)
    pairwise_correlation2(test_dataset)
    # test_metrics = test2(test_dataset)
    # for k, v in test_metrics.items():
    #     print('{} {}'.format(k, v))
    # collect_report(METHOD_NAME, args.data_ratio, test_metrics['pr'])
    # correlation(test_dataset)
    # pairwise_correlation(test_dataset)
    # print('Test2 roc: {:.7f}, Test2 pr: {:.7f}'.format(test2_roc, test2_pr))

# model.load_state_dict(torch.load("../saved_models/DDINet.model"), strict=False)
# import numpy as np
# for thr in np.arange(0, 1, 0.1):
#     test_acc, test_ap, test_af, test_macro_precision, test_macro_recall, test_macro_f1, test_micro_precision, test_micro_recall, test_micro_f1 = test(
#         test_dataset, threshold=thr)
#     print(thr, ": ", test_acc, test_macro_f1, test_micro_f1)

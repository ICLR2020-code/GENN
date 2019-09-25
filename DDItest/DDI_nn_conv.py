import os.path as osp
from .get_commanders import base_commanders
import os
from .util import noise_dataset, metric_report, collect_report
from itertools import combinations
import sys
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter
import numpy as np

from sklearn.metrics import average_precision_score, f1_score, accuracy_score, precision_recall_fscore_support, \
    roc_auc_score


from torch_geometric.data import Data, DataLoader
# from torch_geometric.utils import remove_self_loops
from torch_geometric.read.ddi import read_ddi_data

from torch_geometric.nn.models.model_DDINet import DDIDecoder, DDINet, DDIEncoder
import warnings
warnings.filterwarnings("ignore")
MODEL_NAME = 'pytorch.model'
METHOD_NAME = 'GNN'

args = base_commanders()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

print('learning rate ', args.lr)
target = 0
dim = 100
threshold = 0.4
train_size = args.train_size
val_size = args.val_size


model_save_path = osp.join(osp.dirname(osp.realpath(
    __file__)), '../saved_models/', 'CONV_dataset_{}_seed_{}_ratio_{}'.format(args.data_prefix, args.seed, args.data_ratio))
if not osp.exists(model_save_path):
    os.makedirs(model_save_path)
print('model_save_dir', model_save_path)

dataset = read_ddi_data(args.data_dir, args.data_prefix,
                        args.seed, data_ratio=args.data_ratio)

num_edges, num_types = dataset.edge_attr.size()

# Split datasets.
# generate edge_combination for noise train dataset
edge_combination = list(combinations(range(dataset.x.size(0)), 2))
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

encoder = DDIEncoder(input_feature_dim=dataset.x.size()
                     [1], num_types=num_types, dim=dim)
decoder = DDIDecoder(num_types, dim)
model = DDINet(encoder, decoder).to(device)
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
    # noise dataset
    # data = noise_dataset(x=train_dataset.x,
    #                      edge_index=train_dataset.edge_index,
    #                      edge_attr=train_dataset.edge_attr,
    #                      edge_combination=edge_combination,
    #                      random_seed=args.seed)
    # data.to(device)
    data = train_dataset
    optimizer.zero_grad()
    h = model.encode(data.x, data.edge_index, data.edge_attr)
    loss = model.recon_loss(h, data.edge_index, data.edge_attr)
    loss.backward()
    optimizer.step()
    return loss.item()


def test2(train_data, val_data):  # test roc-auc, pr-auc per label type, same as decagon
    model.eval()

    h = model.encode(train_data.x, train_data.edge_index, train_data.edge_attr)
    y_prob = model.decode(h, val_data.edge_index, val_data.edge_attr)

    y, y_prob = val_data.edge_attr.detach().cpu().numpy(), y_prob.detach().cpu().numpy()

    return metric_report(y, y_prob)


def correlation(train_data, val_data):
    model.eval()

    h = model.encode(train_data.x, train_data.edge_index, train_data.edge_attr)
    pred = model.decode(h, val_data.edge_index, val_data.edge_attr)

    y, pred = val_data.edge_attr.detach().cpu().numpy(), pred.detach().cpu().numpy()

    edge_index = val_data.edge_index.detach().cpu().numpy()
    assert len(edge_index[0]) == len(pred)

    print('node: ', len(val_data.x), 'label: ', pred.shape[1])
    dditype_ground_truth = np.zeros((pred.shape[1], val_data.x.size(0)))
    dditype_predicted = np.zeros((pred.shape[1], val_data.x.size(0)))
    # save y in dditype_ground_truth
    for i in range(len(y)):
        for j in y[i].nonzero()[0]:
            dditype_ground_truth[j, edge_index[0][i]] += 1
            dditype_ground_truth[j, edge_index[1][i]] += 1

        for j in (pred[i] > threshold).nonzero()[0]:
            dditype_predicted[j, edge_index[0][i]] += 1
            dditype_predicted[j, edge_index[1][i]] += 1

    from scipy.stats import pearsonr
    score = []
    for i in range(len(dditype_ground_truth)):
        co, _ = pearsonr(dditype_ground_truth[i], dditype_predicted[i])
        score.append(co)

    # save in model_file_path
    maxsort_idx = np.argsort(score)[::-1]
    with open(osp.join(model_save_path, 'correlation.txt'), 'w') as fout:
        for idx in maxsort_idx:
            if not np.isnan(score[idx]):
                fout.write('Label:{}\tScore:{:.4f}\n'.format(idx, score[idx]))
            else:
                # print('nan predicted: ', dditype_predicted[idx].nonzero()[0])
                # print('nan groud truth: ',
                    #   dditype_ground_truth[idx].nonzero()[0])
                pass

    print('save score at ', osp.join(model_save_path, 'correlation.txt'))


def pairwise_correlation(train_data, val_data):
    model.eval()

    h = model.encode(train_data.x, train_data.edge_index, train_data.edge_attr)
    pred = model.decode(h, val_data.edge_index, val_data.edge_attr)

    y, pred = val_data.edge_attr.detach().cpu().numpy(), pred.detach().cpu().numpy()

    edge_index = val_data.edge_index.detach().cpu().numpy()
    assert len(edge_index[0]) == len(pred)

    print('node: ', len(val_data.x), 'label: ', pred.shape[1])
    dditype_ground_truth = np.zeros((pred.shape[1], val_data.x.size(0)))
    dditype_predicted = np.zeros((pred.shape[1], val_data.x.size(0)))
    # save y in dditype_ground_truth
    for i in range(len(y)):
        for j in y[i].nonzero()[0]:
            dditype_ground_truth[j, edge_index[0][i]] += 1
            dditype_ground_truth[j, edge_index[1][i]] += 1

        for j in (pred[i] > threshold).nonzero()[0]:
            dditype_predicted[j, edge_index[0][i]] += 1
            dditype_predicted[j, edge_index[1][i]] += 1

    from scipy.stats import pearsonr
    fout_predicted = open(
        osp.join(model_save_path, 'pairwise_correlation_predicted.txt'), 'w')
    fout_truth = open(
        osp.join(model_save_path, 'pairwise_correlation_truth.txt'), 'w')

    score_predicted = {}
    score_truth = {}

    # cal pairwise correlation
    for i, j in combinations(range(pred.shape[1]), 2):
        co_predicted, _ = pearsonr(dditype_predicted[i], dditype_predicted[j])
        co_ground_truth, _ = pearsonr(
            dditype_ground_truth[i], dditype_ground_truth[j])

        score_predicted[(i, j)] = co_predicted
        score_truth[(i, j)] = co_ground_truth

    for k, v in sorted(score_predicted.items(), key=lambda item: item[1], reverse=True):
        if not np.isnan(v):
            fout_predicted.write(
                'Label:{}_{}\tScore:{:.4f}\n'.format(k[0], k[1], v))
    for k, v in sorted(score_truth.items(), key=lambda item: item[1], reverse=True):
        if not np.isnan(v):
            fout_truth.write(
                'Label:{}_{}\tScore:{:.4f}\n'.format(k[0], k[1], v))

    print('save pair-wise score at ', osp.join(model_save_path,
                                               'pairwise_correlation_predicted.txt'))


def conflict_pair_cnt(train_data, val_data):
    # 3, 20, 75, 80, and we set that 75, 80 can't occur in the same time.
    model.eval()

    h = model.encode(train_data.x, train_data.edge_index, train_data.edge_attr)
    pred = model.decode(h, val_data.edge_index, val_data.edge_attr)

    y, pred = val_data.edge_attr.detach().cpu().numpy(), pred.detach().cpu().numpy()

    edge_index = val_data.edge_index.detach().cpu().numpy()
    assert len(edge_index[0]) == len(pred)
    # label2nodes_dict = defaultdict(list)
    # for i in range(len(pred)):
    #     for j, individual_pred in enumerate(pred[i]):
    #         if individual_pred > threshold:
    #             label2nodes_dict[j].extend([edge_index[0][i], edge_index[1][i]])
    node2labels_dict = defaultdict(set)
    # save label for each node
    for i in range(len(pred)):
        for j, individual_pred in enumerate(pred[i]):
            if individual_pred > threshold:
                node2labels_dict[edge_index[0][i]].add(j)
                node2labels_dict[edge_index[1][i]].add(j)
    # cnt conflict edge
    conflict_edges = []
    for i in range(len(pred)):
        bool_conflict = False
        for j, individual_pred in enumerate(pred[i]):
            if individual_pred > threshold:
                if j == 2 and (3 in node2labels_dict[edge_index[0][i]] or 3 in node2labels_dict[edge_index[1][i]]):
                    bool_conflict = True
                if j == 3 and (2 in node2labels_dict[edge_index[1][i]] or 2 in node2labels_dict[edge_index[1][i]]):
                    bool_conflict = True
        if bool_conflict:
            conflict_edges.append(i)
    print('conflict edges: ', len(conflict_edges))

    conflict_cnt = 0
    individual_cnt = {}
    for sample_pred in pred:
        for idx, individual_pred in enumerate(sample_pred):
            if idx not in individual_cnt.keys():
                individual_cnt[idx] = 0
            individual_cnt[idx] += int(individual_pred > threshold)
        if sample_pred[2] > threshold and sample_pred[3] > threshold:
            conflict_cnt += 1
    print('individual cnt:', individual_cnt)
    print('conflict cnt:%d, total cnt:%d' % (conflict_cnt, len(pred)))


# only test
if args.only_test:
    # conflict_pair_cnt(test_dataset)

    # test_metrics = test2(train_dataset, test_dataset)
    # for k, v in test_metrics.items():
    #     print('{} {}'.format(k, v))
    # collect_report(METHOD_NAME, args.data_ratio, test_metrics['pr'])
    def pairwise_correlation2(train_data, val_data):
        model.eval()

        h = model.encode(train_data.x, train_data.edge_index,
                         train_data.edge_attr)
        pred = model.decode(h, val_data.edge_index, val_data.edge_attr)

        y, pred = val_data.edge_attr.detach().cpu().numpy(), pred.detach().cpu().numpy()

        edge_index = val_data.edge_index.detach().cpu().numpy()
        from .util import collect_pairwise_relation

        assert len(edge_index[0]) == len(pred)
        collect_pairwise_relation(
            METHOD_NAME, y, pred, edge_index, val_data.x.size(0), threshold=threshold)
    pairwise_correlation2(train_dataset, test_dataset)
    # correlation(train_dataset, test_dataset)
    # pairwise_correlation(train_dataset, test_dataset)

    sys.exit()

writer = SummaryWriter(model_save_path)

best_val_roc = None
best_val_roc_progress = {'cnt': 0, 'value': None}

for epoch in range(1, args.epoch):
    loss = train(epoch)
    val_metrics = test2(train_dataset, val_dataset)
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
        test_metrics = test2(train_dataset, test_dataset)
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


######### * test auc * ########
# for epoch in range(1, 201):
#     # lr = scheduler.optimizer.param_groups[0]['lr']
#     loss = train(epoch)
#     # val_acc, val_ap, val_af, val_macro_precision, val_macro_recall, val_macro_f1, val_micro_precision, val_micro_recall, val_micro_f1 = test(train_dataset, val_dataset)
#     val_auc, val_prauc = test2(train_dataset, val_dataset)
#     # scheduler.step(loss)
#
#     if best_val_acc is None or val_auc >= best_val_acc:
#         # test_acc, test_ap, test_af, test_macro_precision, test_macro_recall, test_macro_f1, test_micro_precision, test_micro_recall, test_micro_f1 = test(train_dataset, test_dataset)
#         test_auc, test_prauc = test2(train_dataset, test_dataset)
#         best_val_acc = val_auc
#         torch.save(model.state_dict(), "../saved_models/DDINet.model")
#
#     print('Epoch: {:03d}, Loss: {:.7f}, Validation AUC: {:.7f}, '
#           'Test AUC: {:.7f}, Test PR-AUC:  {:.7f}'.format(epoch, loss, val_auc, test_auc, test_prauc))

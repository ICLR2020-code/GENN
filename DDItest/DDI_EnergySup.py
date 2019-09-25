from .get_commanders import energy_commanders
import os.path as osp
import os
from tensorboardX import SummaryWriter
import numpy as np
from .util import noise_dataset, metric_report, collect_report
from itertools import combinations
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, accuracy_score, precision_recall_fscore_support, \
    roc_auc_score


from torch_geometric.data import Data
# from torch_geometric.utils import remove_self_loops
from torch_geometric.read.ddi import read_ddi_data

from torch_geometric.nn.models.model_DDINet import DDIDecoder, DDINet, DDIEncoder
from torch_geometric.nn.models.model_DDI_Energy_Net import DDI_Energy_Pooling, DDI_Energy_Net
import warnings
warnings.filterwarnings("ignore")
MODEL_NAME = 'pytorch.model'
METHOD_NAME = 'EGNN-SUP'

args = energy_commanders()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

ce_loss2_weight = 0 if args.no_ce_loss else 1
energy_phi_weight = 0.1
target = 0
dim = 100
threshold = 0.4
train_size = args.train_size
val_size = args.val_size

model_save_path = osp.join(osp.dirname(osp.realpath(
    __file__)), '../saved_models/', 'EGNN-SUP_dataset_{}_seed_{}_ratio_{}'.format(args.data_prefix, args.seed, args.data_ratio) + str('_no_ce_loss' if args.no_ce_loss else ''))
if not osp.exists(model_save_path):
    os.makedirs(model_save_path)
print('model_save_dir', model_save_path)

if args.no_cuda:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = read_ddi_data(args.data_dir, args.data_prefix,
                        random_seed=args.seed, data_ratio=args.data_ratio)

num_edges, num_types = dataset.edge_attr.size()
dataset = dataset.to(device)

train_dataset = Data(
    x=dataset.x, edge_index=dataset.edge_index[:, :int(num_edges*train_size)], edge_attr=dataset.edge_attr[:int(num_edges*train_size), :])
val_dataset = Data(
    x=dataset.x, edge_index=dataset.edge_index[:, int(num_edges*train_size):int(num_edges*val_size)], edge_attr=dataset.edge_attr[int(num_edges*train_size):int(num_edges*val_size), :])
test_dataset = Data(
    x=dataset.x, edge_index=dataset.edge_index[:, int(num_edges*val_size):], edge_attr=dataset.edge_attr[int(num_edges*val_size):, :])

encoder1 = DDIEncoder(input_feature_dim=dataset.x.size()
                      [1], num_types=num_types, dim=dim)
decoder1 = DDIDecoder(num_types, dim)
decoder2 = DDIDecoder(num_types, dim)

encoder2 = DDIEncoder(input_feature_dim=dataset.x.size()
                      [1], num_types=num_types, dim=dim)
pooling = DDI_Energy_Pooling(dim)

ddi_model = DDINet(encoder1, decoder1, decoder2).to(device)
# ddi_model2 = DDINet(encoder1, decoder2).to(device)
ddi_energy_model = DDI_Energy_Net(encoder2, pooling).to(
    device)  # we should use a different encoder in energy model

# load model
if args.only_test:
    # for testing
    if args.resume_path is not None:
        if osp.exists(osp.join(args.resume_path, MODEL_NAME)):
            print('load pretraining model from ' + args.resume_path)
            ddi_model.load_state_dict(torch.load(
                osp.join(args.resume_path, MODEL_NAME)))
    else:
        if osp.exists(osp.join(model_save_path, MODEL_NAME)):
            print('load pretraining model from ' + model_save_path)
            ddi_model.load_state_dict(torch.load(
                osp.join(model_save_path, MODEL_NAME)))
else:
    if osp.exists(osp.join(args.resume_path, MODEL_NAME)):
        # for fine tuing
        print('load model from ', osp.join(args.resume_path, MODEL_NAME))
        ddi_model.load_state_dict(torch.load(
            osp.join(args.resume_path, MODEL_NAME)), strict=False)
        ddi_model.decoder2.load_state_dict(ddi_model.decoder.state_dict())
        ddi_model.to(device)


optimizer1 = torch.optim.Adam(ddi_model.parameters(), lr=args.lr)
optimizer_energy = torch.optim.Adam(
    ddi_energy_model.parameters(), lr=args.lr, weight_decay=1e-5)
# optimizer2 = torch.optim.Adam(ddi_model2.parameters(), lr=0.01)


def train_max(epoch):  # max phi
    ddi_model.train()
    # noise dataset
    # data = noise_dataset(x=train_dataset.x,
    #                      edge_index=train_dataset.edge_index,
    #                      edge_attr=train_dataset.edge_attr,
    #                      edge_combination=edge_combination,
    #                      random_seed=args.seed)
    # data.to(device)
    data = train_dataset

    optimizer1.zero_grad()
    h = ddi_model.encode(
        data.x, data.edge_index, data.edge_attr)
    pred = ddi_model.decoder(h, data.edge_index)
    loss_max = F.l1_loss(pred, data.edge_attr)
    ce_loss = F.binary_cross_entropy(pred, data.edge_attr)
    energy_phi = ddi_energy_model(
        data.x, data.edge_index, pred)
    energy_y = ddi_energy_model(
        data.x, data.edge_index, data.edge_attr)

    # loss_max = ce_loss - F.relu(loss_max - energy_phi + energy_y)
    #

    h2 = ddi_model.encode(
        data.x, data.edge_index, data.edge_attr)
    pred2 = ddi_model.decoder2(h2, data.edge_index)
    energy_psi = ddi_energy_model(data.x, data.edge_index, pred2)

    pred3 = ddi_model.decoder2(h2, data.edge_index)
    ce_loss2 = F.binary_cross_entropy(pred3, data.edge_attr)
    loss_max = ce_loss - \
        F.relu(loss_max - energy_phi_weight*energy_phi +
               energy_y) + ce_loss2_weight * ce_loss2 + energy_psi

    loss_max.backward()
    optimizer1.step()
    return loss_max.item()


def train_min(epoch):  # min theta
    ddi_energy_model.train()

    # noise dataset
    data = train_dataset

    optimizer_energy.zero_grad()
    h = ddi_model.encode(
        data.x, data.edge_index, data.edge_attr)
    pred = ddi_model.decoder(h, data.edge_index)

    # loss_min = F.binary_cross_entropy(pred, data.edge_attr)
    loss_min = F.l1_loss(pred, data.edge_attr)

    energy_phi = ddi_energy_model(
        data.x, data.edge_index, pred)
    energy_y = ddi_energy_model(
        data.x, data.edge_index, data.edge_attr)
    loss_min = F.relu(loss_min - energy_phi_weight*energy_phi + energy_y)

    loss_min.backward()
    optimizer_energy.step()
    return loss_min.item()


def test2(test_dataset):  # test roc-auc, pr-auc per label type, same as decagon
    ddi_model.eval()

    h = ddi_model.encode(
        train_dataset.x, train_dataset.edge_index, train_dataset.edge_attr)
    y_prob = ddi_model.decoder2(h, test_dataset.edge_index)

    y, y_prob = test_dataset.edge_attr.detach(
    ).cpu().numpy(), y_prob.detach().cpu().numpy()

    return metric_report(y, y_prob)


def correlation(test_dataset):
    ddi_model.eval()

    h = ddi_model.encode(
        train_dataset.x, train_dataset.edge_index, train_dataset.edge_attr)
    pred = ddi_model.decoder2(h, test_dataset.edge_index)

    y, pred = test_dataset.edge_attr.detach().cpu().numpy(), pred.detach().cpu().numpy()

    edge_index = test_dataset.edge_index.detach().cpu().numpy()
    assert len(edge_index[0]) == len(pred)

    print('node: ', len(test_dataset.x), 'label: ', pred.shape[1])
    dditype_ground_truth = np.zeros((pred.shape[1], test_dataset.x.size(0)))
    dditype_predicted = np.zeros((pred.shape[1], test_dataset.x.size(0)))
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


def pair_wise_correlation(test_dataset):
    ddi_model.eval()

    h = ddi_model.encode(
        train_dataset.x, train_dataset.edge_index, train_dataset.edge_attr)
    pred = ddi_model.decoder2(h, test_dataset.edge_index)

    y, pred = test_dataset.edge_attr.detach().cpu().numpy(), pred.detach().cpu().numpy()

    edge_index = test_dataset.edge_index.detach().cpu().numpy()
    assert len(edge_index[0]) == len(pred)

    print('node: ', len(test_dataset.x), 'label: ', pred.shape[1])
    dditype_ground_truth = np.zeros((pred.shape[1], test_dataset.x.size(0)))
    dditype_predicted = np.zeros((pred.shape[1], test_dataset.x.size(0)))
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
        if co_ground_truth > 0.5:
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


def conflict_pair_cnt(test_dataset):
    # 3, 20, 75, 80, and we set that 75, 80 can't occur in the same time.
    ddi_model.eval()

    h = ddi_model.encode(
        train_dataset.x, train_dataset.edge_index, train_dataset.edge_attr)
    pred = ddi_model.decoder2(h, test_dataset.edge_index)

    y, pred = test_dataset.edge_attr.detach().cpu().numpy(), pred.detach().cpu().numpy()

    edge_index = test_dataset.edge_index.detach().cpu().numpy()
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
    # test_metrics = test2(test_dataset)
    # for k, v in test_metrics.items():
    #     print('{} {}'.format(k, v))
    # collect_report(METHOD_NAME, args.data_ratio, test_metrics['pr'])
    # correlation(test_dataset)
    def pair_wise_correlation2(test_dataset):
        ddi_model.eval()

        h = ddi_model.encode(
            train_dataset.x, train_dataset.edge_index, train_dataset.edge_attr)
        pred = ddi_model.decoder2(h, test_dataset.edge_index)

        y, pred = test_dataset.edge_attr.detach().cpu().numpy(), pred.detach().cpu().numpy()

        edge_index = test_dataset.edge_index.detach().cpu().numpy()
        from .util import collect_pairwise_relation

        assert len(edge_index[0]) == len(pred)
        collect_pairwise_relation(
            METHOD_NAME, y, pred, edge_index, test_dataset.x.size(0), threshold=threshold)

    pair_wise_correlation2(test_dataset)
    sys.exit()

writer = SummaryWriter(model_save_path)

best_val_roc = None
best_val_roc_progress = {'cnt': 0, 'value': None}
for epoch in range(1, args.epoch):
    loss1 = train_max(epoch)
    loss2 = train_min(epoch)
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
        torch.save(ddi_model.state_dict(), osp.join(
            model_save_path, MODEL_NAME))
        writer.add_scalar('test/roc', test_metrics['roc'], epoch-1)
        writer.add_scalar('test/pr', test_metrics['pr'], epoch-1)
        writer.add_scalar('test/p@1', test_metrics['p@1'], epoch-1)
        writer.add_scalar('test/p@3', test_metrics['p@3'], epoch-1)
        writer.add_scalar('test/p@5', test_metrics['p@5'], epoch-1)

    writer.add_scalar('train/loss1', loss1, epoch-1)
    writer.add_scalar('train/loss2', loss2, epoch-1)

    writer.add_scalar('val/roc', val_roc, epoch-1)
    writer.add_scalar('val/pr', val_pr, epoch-1)

    print('Epoch: {:03d}, Loss1: {:.7f}, Loss2: {:.7f}, Validation ROC_AUC: {:.7f}, Validation PR_AUC: {:.7f}'.format(
        epoch, loss1, loss2, val_roc, val_pr)
    )
writer.close()

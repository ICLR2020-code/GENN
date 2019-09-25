from torch_geometric.data import Data
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import os
import pandas as pd
import pickle


def pk_load(filename):
    return pickle.load(open(filename, 'rb'))


def pk_save(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))


def noise_dataset(x, edge_index, edge_attr, edge_combination, random_seed=1203, delete_ratio=0.2, insert_ratio=0.2):
    np.random.seed(random_seed)

    total_size = edge_index.size(1)
    del_size = int(total_size * delete_ratio)
    insert_size = int(total_size * insert_ratio)
    print('delete edge size: %d, insert edge size: %d' %
          (del_size, insert_size))

    # random delete
    keep_indices = np.random.choice(
        np.arange(0, total_size),
        total_size - del_size
    )
    keep_edge_index = edge_index[:, keep_indices]
    keep_edge_attr = edge_attr[keep_indices, :]

    # random insert
    insert_edge_indices = np.random.choice(
        range(len(edge_combination)),
        insert_size
    )
    row, col = zip(*edge_combination)
    row = np.array(row)
    col = np.array(col)
    insert_edge_index = torch.stack(
        [torch.LongTensor(row[insert_edge_indices]), torch.LongTensor(col[insert_edge_indices])], dim=0)
    insert_edge_attr = torch.zeros(insert_size, edge_attr.size(1))

    # concat
    new_edge_index = torch.cat([keep_edge_index, insert_edge_index], dim=-1)
    new_edge_attr = torch.cat([keep_edge_attr, insert_edge_attr], dim=0)
    return Data(x=x, edge_index=new_edge_index, edge_attr=new_edge_attr)


def metric_report(y, y_prob):
    rocs = []
    prs = []

    ks = [1, 3, 5]
    pr_score_at_ks = []
    for k in ks:
        pr_at_k = []
        for i in range(y_prob.shape[0]):
            # forloop samples
            y_prob_index_topk = np.argsort(y_prob[i])[::-1][:k]
            inter = set(y_prob_index_topk) & set(y[i].nonzero()[0])
            pr_ith = len(inter) / k
            pr_at_k.append(pr_ith)
        pr_score_at_k = np.mean(pr_at_k)
        pr_score_at_ks.append(pr_score_at_k)

    for i in range(y.shape[1]):
        if(sum(y[:, i]) < 1):
            continue
        roc = roc_auc_score(y[:, i], y_prob[:, i])
        rocs.append(roc)

        prauc = average_precision_score(y[:, i], y_prob[:, i])
        prs.append(prauc)

    roc_auc = sum(rocs)/len(rocs)
    pr_auc = sum(prs)/len(prs)

    return {
        'pr': pr_auc,
        'roc': roc_auc,
        'p@1': pr_score_at_ks[0],
        'p@3': pr_score_at_ks[1],
        'p@5': pr_score_at_ks[2]
    }


def collect_pairwise_relation(method, y, y_pred, edge_index, node_size, threshold=0.4, file_name='pairwise_relation.csv'):

    dditype_ground_truth = np.zeros((y_pred.shape[1], node_size))
    dditype_predicted = np.zeros((y_pred.shape[1], node_size))

    truth_dist = []
    pred_dist = []

    for i in range(len(y)):
        for j in y[i].nonzero()[0]:
            dditype_ground_truth[j, edge_index[0][i]] += 1
            dditype_ground_truth[j, edge_index[1][i]] += 1

        for j in (y_pred[i] > threshold).nonzero()[0]:
            dditype_predicted[j, edge_index[0][i]] += 1
            dditype_predicted[j, edge_index[1][i]] += 1
    for i in range(len(dditype_ground_truth)):
        truth_dist.append({
            'method': 'Truth',
            'ddi_type': i,
            'dist': list(dditype_ground_truth[i])
        })
        pred_dist.append({
            'method': method,
            'ddi_type': i,
            'dist': list(dditype_predicted[i])
        })
    """
    truth_dist = [{
        'method':'Truth',
        'ddi_type': 1,
        'dist': [1,4,5...]
    },]
    """
    if not os.path.exists(file_name):
        data = pd.DataFrame(data=truth_dist, columns=[
                            'method', 'ddi_type', 'dist'])
    else:
        data = pd.read_csv(file_name, index_col=0)
    new_dataframe = pd.DataFrame(data=pred_dist, columns=[
        'method', 'ddi_type', 'dist'])
    data = pd.concat([data, new_dataframe], axis=0, ignore_index=True)
    data.to_csv(file_name)
    print('update complete')


def run_sys_collect_pairwise_relation():
    modules = [
        ['DDItest.DDI_MLP', 3],
        ['baselines.DDI_deepwalk', 3],
        ['DDItest.DDI_nn_conv', 3],
        ['DDItest.DDI_Local_Energy', 2],
        ['DDItest.DDI_Energy', 3],
    ]
    for module in modules:
        command_str = 'python -m {} --only_test --train_size {} --val_size {} --data_ratio {} --seed {} --no_cuda'.format(
            module[0],
            0.6,
            0.6,
            60,
            module[1]
        )
        os.system(command_str)


def collect_report(method, ratio, pr, file_name='ddi_ratio_result2.csv'):
    if not os.path.exists(file_name):
        data = pd.DataFrame(columns=['method', 'ratio', 'pr'])
    else:
        data = pd.read_csv(file_name, index_col=0)
    new_line = {
        'method': method,
        'ratio': ratio,
        'pr': pr
    }
    data = data.append(new_line, ignore_index=True)

    # save
    data.to_csv(file_name)
    print('save complete')


def run_sys_collect_report(file_name='ddi_ratio_result2.csv'):
    # if os.path.exists(file_name):
    #     os.remove(file_name)
    modules = [
        'DDItest.DDI_MLP',
        'baselines.DDI_deepwalk',
        'DDItest.DDI_nn_conv',
        'DDItest.DDI_Local_Energy',
        'DDItest.DDI_Energy',
        'baselines.DDI_LP2'
    ]
    ratios = [
        # 2,
        # 5,
        # 10,
        25
    ]
    seeds = [
        1, 2, 3
    ]
    for module in modules:
        for seed in seeds:
            for ratio in ratios:
                command_str = 'python -m {} --only_test --train_size {} --val_size {} --data_ratio {} --seed {} --no_cuda'.format(
                    module,
                    ratio/100,
                    ratio/100,
                    ratio,
                    seed
                )
                os.system(command_str)


if __name__ == "__main__":
    # run_sys_collect_report()
    run_sys_collect_pairwise_relation()

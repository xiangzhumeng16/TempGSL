import numpy as np
import os
import subprocess
import sys
import time


def split_dataset(c1, c2, train_ratio=0.7):
    pos_train_num = int(round(train_ratio * len(c1)))
    pos_test_num = len(c1) - pos_train_num
    neg_train_num = int(round(train_ratio * len(c2)))
    neg_test_num = len(c2) - neg_train_num
    pos_train_ids = list(np.random.choice(c1, pos_train_num, replace=False))
    pos_test_ids = list(set(c1) - set(pos_train_ids))
    neg_train_ids = list(np.random.choice(c2, neg_train_num, replace=False))
    neg_test_ids = list(set(c2) - set(neg_train_ids))

    print('Numbers: pos_train {}, neg_train {}, pos_test {}, neg_test {}' \
          .format(pos_train_num, neg_train_num, pos_test_num, neg_test_num))

    return pos_train_ids, pos_test_ids, neg_train_ids, neg_test_ids


def getnodes(diff_net, alpha):
    with open("icdm16-egoscan/net.txt", 'w') as f:
        for row in range(1, diff_net.shape[1]):
            for col in range(row):
                f.write("{} {} {}".format(row, col, diff_net[row][col]) + "\n")
        f.close()

    os.chdir("icdm16-egoscan")
    subprocess.call("python densdp.py net.txt {}".format(alpha), shell=True)
    cs = eval(open("tmp/net.txt", "r").readline())
    os.chdir(os.pardir)
    return cs


def st(mat):
    return mat
    # M = np.max(mat)
    # m = np.min(mat)
    # mat = 2/(M-m)*(mat-m)-1
    # return mat/10


def get_selected_summary(paths, nodes):
    g = np.zeros((len(nodes), len(nodes)))
    for f in paths:
        mat = np.loadtxt(f)[nodes][:, nodes]
        g += mat
    return g / len(paths)


def wait(sec=30):
    print("waiting......")
    for i in range(sec + 1):
        if i != sec:
            sys.stdout.write("==")
        else:
            sys.stdout.write("== 100%/100%")
        sys.stdout.flush()
        time.sleep(1)
    print("\n")


def getdataset(c1, c2, diff_net_A, diff_net_B, nodesA, nodesB, summary_c1, summary_c2, args, deep=0):
    pos_train_ids, pos_test_ids, neg_train_ids, neg_test_ids = split_dataset(c1, c2)
    pos_train_x = np.array(
        [getfeature(i, diff_net_A, diff_net_B, nodesA, nodesB, summary_c1, summary_c2, args.mode, args.threshold) for i in
         pos_train_ids])
    pos_test_x = np.array(
        [getfeature(i, diff_net_A, diff_net_B, nodesA, nodesB, summary_c1, summary_c2, args.mode, args.threshold) for i in
         pos_test_ids])
    neg_train_x = np.array(
        [getfeature(i, diff_net_A, diff_net_B, nodesA, nodesB, summary_c1, summary_c2, args.mode, args.threshold) for i in
         neg_train_ids])
    neg_test_x = np.array(
        [getfeature(i, diff_net_A, diff_net_B, nodesA, nodesB, summary_c1, summary_c2, args.mode, args.threshold) for i in
         neg_test_ids])
    train_x = np.concatenate([pos_train_x, neg_train_x], 0)
    train_y = np.array([1] * pos_train_x.shape[0] + [0] * neg_train_x.shape[0]).reshape(-1, 1)
    train_data = np.concatenate([train_x, train_y], 1)
    test_x = np.concatenate([pos_test_x, neg_test_x], 0)
    test_y = np.array([1] * pos_test_x.shape[0] + [0] * neg_test_x.shape[0]).reshape(-1, 1)
    test_data = np.concatenate([test_x, test_y], 1)
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    if deep:
        import torch
        train_x = torch.FloatTensor(train_data[:, :2])
        train_y = torch.LongTensor(train_data[:, -1])
        test_x = torch.FloatTensor(test_data[:, :2])
        test_y = torch.LongTensor(test_data[:, -1])
    else:
        train_x = train_data[:, :2]
        train_y = train_data[:, -1]
        test_x = test_data[:, :2]
        test_y = test_data[:, -1]
    return train_x, test_x, train_y, test_y, pos_train_x, neg_train_x, pos_test_x, neg_test_x


def getfeature(path, diff_A=None, diff_B=None, nodes_A=None, nodes_B=None, sum_c1=None, sum_c2=None, mode=1, threshold=0.1):
    '''
    mode=1: Gong Haisong's implementation
    mode=2: count the subgraph edge number when it is greater than 0.1 on the diff_net
    mode=3: count the whole graph edge number when it is greater than 0.1 on the diff_net
    mode=4: L1-norm
    '''
    if mode == 1:
        mat = st(np.loadtxt(path))
        f = lambda x, y: (x[y][:, y]).sum() / (len(y) ** 2)
        return [f(mat, nodes_A), f(mat, nodes_B)]
    elif mode == 2:
        feat_1, feat_2 = 0, 0
        mat = np.loadtxt(path)
        nodes = nodes_A + nodes_B
        for row in range(diff_A.shape[0]):
            for col in range(row):
                if row in nodes and col in nodes and mat[row][col] != 0:
                    if diff_A[row][col] >= threshold:
                        feat_1 += 1
                    if diff_B[row][col] >= threshold:
                        feat_2 += 1
        return [feat_1, feat_2]
    elif mode == 3:
        feat_1, feat_2 = 0, 0
        mat = np.loadtxt(path)
        for row in range(diff_A.shape[0]):
            for col in range(row):
                if mat[row][col] != 0:
                    if diff_A[row][col] >= threshold:
                        feat_1 += 1
                    if diff_B[row][col] >= threshold:
                        feat_2 += 1
        return [feat_1, feat_2]
    elif mode == 4:
        mat = np.loadtxt(path)
        nodes = nodes_A + nodes_B
        mat_sub = mat[nodes][:, nodes]
        feat1 = np.linalg.norm(mat_sub - sum_c1[nodes][:, nodes], ord=1)
        feat2 = np.linalg.norm(mat_sub - sum_c2[nodes][:, nodes], ord=1)
        return [feat1, feat2]

def get_data_by_txt(txt_path=None):
    pass
    #return adajency_matrix

def train_template_graph(positive_datasets_dir=None, negative_datasets_dir=None, iterations=10)
    pass
    # positive_summary_graph = None
    # negative_summray_graph = None
    # return positive_summary_graph, negative_summray_graph

def get_connectivity_feature(template_graph=None, connectivity_matrix, mode):
    assert mode in (0, 1), 'set mode=0 or 1 '
    
    # if mode==0:
    #     return feature
    # elif mode==1:
    #     return feature

    pass


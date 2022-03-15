from tools import get_summary_graph, calculate_weights, update_summary_graph, get_summary_features, get_specific_graph_list
from dataset import get_dataset
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from scipy import io
import numpy as np
import os
from config import datasets

params_p = np.arange(0, 2, 0.1)
params_alpha = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
epochs = 10

    
for dataset in datasets:
    results = np.zeros([len(params_p), len(params_alpha)])
    data_dir, positive_group, negative_group = dataset
    # data_dir, positive_group, negative_group = 'ADNI', 'LMCI', 'AD'
    save_mat_name = os.path.join('./experiments-results', '-'.join([data_dir, positive_group, negative_group])+'_acc.mat')

    for i, p in enumerate(params_p):
        for j, alpha in enumerate(params_alpha):
            acc_list = []
            for epoch in range(epochs):
                pos_train_features, neg_train_features, pos_test_features, neg_test_features = get_dataset(data_dir, positive_group, negative_group, gamma=p, threshold=alpha, split_ratio=0.7)
                train_x = np.concatenate([pos_train_features, neg_train_features])
                train_y = np.array([0]*pos_train_features.shape[0]+[1]*neg_train_features.shape[0])
                test_x = np.concatenate([pos_test_features, neg_test_features])
                test_y = np.array([0]*pos_test_features.shape[0]+[1]*neg_test_features.shape[0])
                clf = LinearSVC(max_iter=300000)
                clf.fit(train_x, train_y)
                test_result = (clf.predict(test_x) == test_y)
                acc = float(test_result.sum()) / len(test_result)
                acc_list.append(acc)
            results[i, j] = sum(acc_list)/epochs

    io.savemat(save_mat_name, {'results':results})
from tools import get_summary_graph, calculate_weights, update_summary_graph, get_summary_features, get_specific_graph_list
from dataset import get_dataset
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from scipy import io
import numpy as np
import os
from config import datasets

# data_dir, positive_group, negative_group = 'ADNI', 'EMCI', 'AD'
params_alpha = [0.01, 0.05, 0.1, 0.5, 1, 1.5]
epochs = 10
for dataset in datasets:
    data_dir, positive_group, negative_group = dataset
    save_mat_dir = os.path.join('./experiments-results', '-'.join([data_dir, positive_group, negative_group]))
    if not os.path.exists(save_mat_dir):
        os.mkdir(save_mat_dir)

    for j, alpha in enumerate(params_alpha):
        data = {}
        pos_train_features, neg_train_features, pos_test_features, neg_test_features = get_dataset(data_dir, positive_group, negative_group, gamma=0.5, threshold=alpha, split_ratio=0.7)
        train_x = np.concatenate([pos_train_features, neg_train_features])
        pos_features = np.concatenate([pos_train_features, pos_test_features])
        neg_features = np.concatenate([neg_train_features, neg_test_features])
        data['pos_features'] = pos_features
        data['neg_features'] = neg_features
        save_mat_name = os.path.join(save_mat_dir, 'alpha'+'_'+str(j)+'.mat')
        io.savemat(save_mat_name, data)


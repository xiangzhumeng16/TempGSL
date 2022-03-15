#!/usr/bin/env python
# coding: utf-8
from tools import get_summary_graph, calculate_weights, update_summary_graph, get_summary_features, get_specific_graph_list
from dataset import get_dataset
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data_dir, positive_group, negative_group = 'children', 'asd', 'td'

pos_train_features, neg_train_features, pos_test_features, neg_test_features = get_dataset(data_dir, positive_group, negative_group, gamma=1, threshold=0.1, split_ratio=0.7)

plt.scatter(pos_train_features[:, 0], pos_train_features[:, 1], c='r')
plt.scatter(neg_train_features[:, 0], neg_train_features[:, 1], c='b')
plt.title('The distribution on training dataset')
plt.show()

plt.scatter(pos_test_features[:, 0], pos_test_features[:, 1], c='orange')
plt.scatter(neg_test_features[:, 0], neg_test_features[:, 1], c='lightblue')
plt.title('The distribution on testing dataset')
plt.show()

print(f"shape of pos_train_features: {pos_train_features.shape}, shape of neg_train_features:{neg_train_features.shape}, shape of pos_test_features:{pos_test_features.shape}, shape of neg_test_features:{neg_test_features.shape}")

train_x = np.concatenate([pos_train_features, neg_train_features])
train_y = np.array([0]*pos_train_features.shape[0]+[1]*neg_train_features.shape[0])

scaler = MinMaxScaler()
scaler.fit(train_x)
train_x = scaler.transform(train_x)

clf = LinearSVC(max_iter=200000)
clf.fit(train_x, train_y)

test_x = np.concatenate([pos_test_features, neg_test_features])
test_x = scaler.transform(test_x)
test_y = np.array([0]*pos_test_features.shape[0]+[1]*neg_test_features.shape[0])

train_result = (clf.predict(train_x) == train_y)

test_result = (clf.predict(test_x) == test_y)

train_acc = float(train_result.sum()) / len(train_result)

test_acc = float(test_result.sum()) / len(test_result)

print(f"acc of training set is: {train_acc}, and acc of testing set is: {test_acc}")
# encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import argparse
from functools import reduce
import pandas as pd
from utils import *
from dataset import get_dataset

def main(args):

    data_dir = args.data_set
    positive_group = args.positive_group
    negative_group = args.negative_group

    gamma = args.gamma

    split_ratio = args.split_ratio

    threshold = args.threshold

    # SVM training
    total_train_result = 0
    total_test_result = 0
    for j in range(args.test_num):
        # load the datasets
        pos_train_features, neg_train_features, pos_test_features, neg_test_features = get_dataset(data_dir, positive_group, negative_group, gamma, threshold, split_ratio)
        # spliting the datasets
        train_x = np.concatenate([pos_train_features, neg_train_features])
        train_y = np.array([0]*pos_train_features.shape[0]+[1]*neg_train_features.shape[0])

        test_x = np.concatenate([pos_test_features, neg_test_features])
        test_y = np.array([0]*pos_test_features.shape[0]+[1]*neg_test_features.shape[0])
        
        clf = LinearSVC(max_iter=3000)
        clf.fit(train_x, train_y)
        
        # visualization for positive/negative samples
        if j == 4:
            w = clf.coef_.squeeze(0)
            xmin = train_x[:, 0].min()
            xmax = train_x[:, 0].max()
            xx = np.linspace(xmin, xmax, 10)
            yy = -w[0] * xx / w[1] - clf.intercept_[0] / w[1]
            plt.plot(xx, yy)
            plt.scatter(pos_train_features[:, 0], pos_train_features[:, 1], c='r')
            plt.scatter(neg_train_features[:, 0], neg_train_features[:, 1], c='b')
            plt.title('The performance on training dataset')
            plt.show()
            
            plt.plot(xx, yy)
            plt.scatter(pos_test_features[:, 0], pos_test_features[:, 1], c='orange')
            plt.scatter(neg_test_features[:, 0], neg_test_features[:, 1], c='lightblue')
            plt.title('The performance on testing dataset')
            plt.show()

        train_result = (clf.predict(train_x) == train_y)
        test_result = (clf.predict(test_x) == test_y)
        total_train_result += float(train_result.sum()) / len(train_result)
        total_test_result += float(test_result.sum()) / len(test_result)

    print('SVM: Train Acc:{} Test Acc:{}'.format(total_train_result / 5, total_test_result / 5))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--dataset', type=str, default='children', help='dataset name')
    parser.add_argument('--positive_group', type=str, default='asd', help='group one of contrast')
    parser.add_argument('--negative_group', type=str, default='td', help='group two of contrast')
    parser.add_argument('--gamma', type=float, default=1, help='hyper-parameter for adaptive weights')
    parser.add_argument('--split_ratio', type=float, default=0.7, help='dataset splitting ratio')
    parser.add_argument('--threshold', type=float, default=0.1, help='hyper-parameter for contrast graph')
    parser.add_argument('--test_num', type=int, default=5, help='test number')
    
    args = parser.parse_args()
    
    main(args)







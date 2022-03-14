# encoding:utf-8
import numpy as np
from scipy import io
from tools import get_specific_graph_list, get_summary_graph, calculate_weights, update_summary_graph, get_summary_features

# splitting raw dataset into trainingset and testingset
def split_dataset(positive_group_features, negative_group_features, split_ratio):
    # splitting the positive group
    pos_group_num = positive_group_features.shape[0]
    pos_train_num = int(round(split_ratio * pos_group_num))
    pos_test_num = pos_group_num - pos_train_num
    # splitting the negative group
    neg_group_num = negative_group_features.shape[0]
    neg_train_num = int(round(split_ratio * neg_group_num))
    neg_test_num = neg_group_num - pos_train_num
    
    pos_train_ids = list(np.random.choice(range(pos_group_num), pos_train_num, replace=False))
    pos_test_ids = list(set(range(pos_group_num)) - set(pos_train_ids))
    neg_train_ids = list(np.random.choice(range(neg_group_num), neg_train_num, replace=False))
    neg_test_ids = list(set(range(neg_group_num)) - set(neg_train_ids))

    # print('Numbers: pos_train {}, neg_train {}, pos_test {}, neg_test {}' \
    #       .format(pos_train_num, neg_train_num, pos_test_num, neg_test_num))

    # training dataset
    pos_train_features = positive_group_features[pos_train_ids]
    neg_train_features = negative_group_features[neg_train_ids]
    # testing dataset
    pos_test_features = positive_group_features[pos_test_ids]
    neg_test_features = negative_group_features[neg_test_ids]
    return pos_train_features, neg_train_features, pos_test_features, neg_test_features

# get features set according to .dir and groups
def get_dataset(data_dir, positive_group, negative_group, gamma=1.0, threshold=0.1, split_ratio=0.7):

    # load all brain graphs
    pos_graph_list = get_specific_graph_list(data_dir, positive_group)
    neg_graph_list = get_specific_graph_list(data_dir, negative_group)

    # intalize the summary graph for positive and negative groups
    positive_graph = get_summary_graph(pos_graph_list)
    negative_graph = get_summary_graph(neg_graph_list)

    # main loop to optimize the template graphs
    iterations = 5
    for i in range(iterations):
        
        # print(f"pos_graph edges: {positive_graph.sum()}, neg_graph edges: {negative_graph.sum()}")
        
        # calculate weights for two groups
        pos_weights = calculate_weights(positive_graph, pos_graph_list)
        neg_weights = calculate_weights(negative_graph, neg_graph_list)
        positive_graph = get_summary_graph(pos_graph_list, pos_weights)
        negative_graph = get_summary_graph(neg_graph_list, neg_weights)

        # print(pos_weights[:10])
        # print(neg_weights[:10])

        # update summary graph
        positive_template_graph, negative_template_graph = update_summary_graph(positive_graph, negative_graph, threshold)

        # print(f"pos_tem_graph edges: {positive_template_graph.sum()}, neg_tem_graph edges: {negative_template_graph.sum()}")

    # features
    positive_group_features = get_summary_features(pos_graph_list, positive_template_graph, negative_template_graph)
    negative_group_features = get_summary_features(neg_graph_list, positive_template_graph, negative_template_graph)
    
    # save the templates
    template_graphs = {'positive':positive_template_graph, 'negative':negative_template_graph}
    io.savemat(f'./experiments-results/heatmaps/{data_dir}.mat', template_graphs)

    # spliting the features into training set and testing set
    pos_train_features, neg_train_features, pos_test_features, neg_test_features = split_dataset(positive_group_features, negative_group_features, split_ratio)

    return pos_train_features, neg_train_features, pos_test_features, neg_test_features
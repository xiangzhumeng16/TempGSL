import os, math
import numpy as np

def threshold_split_feature(input_file_dir, output_file_dir, ratio = 0.9):
    #  if not output_file_dir, generate the corresponding dir
    if not os.path.exists(output_file_dir):
        os.mkdir(output_file_dir)
        
    for txt_file in os.listdir(input_file_dir):
        # get the matrix from .txt file
        input_data = np.loadtxt(os.path.join(input_file_dir, txt_file), dtype=np.float)
        # set the threshold to split the feature
        threshold = np.percentile(input_data, ratio)
        # generate 0-1 connectivity matrix
        output_data = np.zeros_like(input_data)
        output_data[input_data>threshold] = 1
        np.savetxt(os.path.join(output_file_dir, txt_file, fmt="d", delimiter=" "))

def get_specific_graph_list(data_dir, group):
    
    specific_graph_list = []
    
    for file in os.listdir(os.path.join('./datasets', data_dir, group)):
        graph = np.loadtxt(os.path.join('./datasets', data_dir, group, file), dtype=np.float)
        np.fill_diagonal(graph, 0)
        specific_graph_list.append(graph)
        del graph
    
    return specific_graph_list

def get_summary_graph(specific_graph_list, weights=None):
    if weights is None:
        weights = [1/len(specific_graph_list)]*len(specific_graph_list)

    template_graph = np.zeros_like(specific_graph_list[0])
    
    for i in range(len(weights)):
        template_graph += weights[i]*specific_graph_list[i]
        
    return template_graph

def update_weight(template_graph, specific_graph, gamma):
    diff = np.linalg.norm(template_graph-specific_graph)
    return math.pow(diff, gamma/2-1)

def calculate_weights(template_graph, specific_graph_list, gamma=1):
    weights = []
    
    for specific_graph in specific_graph_list:
        weight = update_weight(template_graph, specific_graph, gamma)
        weights.append(weight)
        del weight
    # weights = [weight/sum(weights) for weight in weights]
    return weights

def update_summary_graph(positive_summary_graph, negative_summary_graph, threshold=0.01, iterations = 1):
    positive_graph = positive_summary_graph.copy()
    negative_graph = negative_summary_graph.copy()
    
    # main loop
    for i in range(iterations):
        
        # updating the positive graph
        tmp_positive_graph = positive_graph.copy()
        positive_graph[positive_graph<negative_graph+threshold] = 0
        
        # updating the positive graph
        negative_graph[negative_graph<tmp_positive_graph+threshold] = 0

        del tmp_positive_graph
    
    return positive_graph, negative_graph

def get_summary_features(sample_graph_list, positive_graph, negative_graph):
    # features list
    features = []
    # edge nums
    for sample_graph in sample_graph_list:
        positive_edges = sample_graph[positive_graph>0].sum()
        negative_edges = sample_graph[negative_graph>0].sum()
        features.append([positive_edges, negative_edges])
        del positive_edges, negative_edges

    features = np.array(features)
    return features

def get_connnectivity_features(sample_graph_list, positive_graph, negative_graph):
    # features list
    features = []

    # connectivity feature with mask
    graph_index = positive_graph+negative_graph
    
    # upper triangle matrix 
    for sample_graph in sample_graph_list:
        # feature selection
        connectivity_matrix = sample_graph[graph_index>0]
        triu_mask = np.tril(np.ones_like(connectivity_matrix))==0
        # add feature into list 
        features.append(connectivity_matrix[triu_mask])

        del connectivity_matrix, triu_mask

    features = np.array(features)

    return features
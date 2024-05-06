# SubGraph Retrieval 
import networkx as nx 
import pickle 
import dill 
import json 
import os 
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
from statistics import mean
from PIL import Image
import itertools
from visualize import visualize

# Each node representation is a tuple of (pytorch_tensor, font_name, case), where font_name is e.g. "a_ABeeZee" and case is e.g. "lower"
# each sample of benchmark: a tuple(query_subgraph, list of all reference subgraphs). the query is a graph object
# each reference subgraph has two values: (sub)graph_file_name k_n.json (k=graph_index, n=representation number between 0 to 9) and the subgraph object 

def load_data(args): 
    # load template (reference) graphs: 
    list_of_graphs = []
    for filename in os.listdir(args.full_data_path):
        # Check if the current item is a file
        file_path = os.path.join(args.full_data_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as graph_file:
                graph = nx.node_link_graph(json.loads(graph_file.read()))
                list_of_graphs.append((filename, graph)) # a list of tuples: (graph_name, graph object)

    # load benchmarks:
    with open(args.benchmark_path, 'rb') as input_file:
        benchmark = dill.load(input_file)

    return list_of_graphs, benchmark

######################################################

def remove_duplicates(lists):
    # Convert each inner list to a tuple after sorting
    tuple_lists = [tuple(sorted(lst)) for lst in lists]
    # Convert list of tuples to a set to remove duplicates
    unique_tuples = set(tuple_lists)
    # Convert set back to list of lists
    unique_lists = [list(t) for t in unique_tuples]
    
    return unique_lists

# given a reference graph and a query graph, find a set of k most similar nodes in reference graph, for each node in the query
def find_most_similar_node_set(ref_graph, query_subgraph, query_node, k):
    similar_nodes = []
    query_vector = np.array(query_subgraph.nodes[query_node]['representation'][0])
    for node in list(ref_graph.nodes):
        node_vector = np.array(ref_graph.nodes[node]['representations'][0])
        similarity = cosine_similarity(query_vector, node_vector)[0][0]
        similar_nodes.append((node, similarity))
    sorted_similar_nodes = sorted(similar_nodes, key=lambda x: x[1], reverse=True)
    top_k_tuples = sorted_similar_nodes[:k]

    return top_k_tuples # tuples (node, similarity)

def graphs_retrieval(args, query_subgraph, reference_graphs):
    # print(list(query_subgraph.nodes))
    retrieved_subgraphs = []
    for graph_name, graph in reference_graphs:
        selected_nodes = []
        for query_node in list(query_subgraph.nodes): 
            # find the closest node to each query node, with weights (dist)
            top_k_tuples = find_most_similar_node_set(graph, query_subgraph, query_node, args.k)
            top_k_nodes = [t[0] for t in top_k_tuples]
            selected_nodes.append(top_k_nodes)
        all_combinations = list(itertools.product(*selected_nodes))
        unique_combinations = remove_duplicates(all_combinations)
        # return a subgraph of the found nodes (it can be a disconnected graph)
        for node_combination in unique_combinations:
            _subgraph_ = graph.subgraph(node_combination)
            retrieved_subgraphs.append((graph_name, _subgraph_))

    return retrieved_subgraphs

########################################################################

def evaluate(args, retrieved_subgraphs, benchmark_subgraphs, reference_graphs):
    all_min_dist = []
    count = 0
    for ret_sg_name, ret_sg in retrieved_subgraphs:
        min_dist = 1000
        for bench_sg_name, bench_sg in benchmark_subgraphs: 
            if str(ret_sg_name) == str(bench_sg_name):  
                count += 1
                # dist = nx.graph_edit_distance(ret_sg, bench_sg, node_match=lambda a,b: a['representations'] == b['representation']) # binary
                dist = nx.graph_edit_distance(ret_sg, bench_sg, node_subst_cost=lambda a,b: 1-cosine_similarity(a['representations'][0], b['representation'][0])[0][0]) # fuzzy
                if dist < min_dist:
                    min_dist = dist 
                    best_bench_sg_name = bench_sg_name
                    best_bench_sg = bench_sg
        if min_dist < 1000:
            all_min_dist.append(min_dist)
            #print('visualizing the benchmark:', best_bench_sg_name)
            #visualize(best_bench_sg, key='representation')
            #print('visualizing the retrieved graph:', ret_sg_name)
            #visualize(ret_sg)
            
    all_min_dist.sort()
    print('number of matches: ', count)
    return all_min_dist




parser = argparse.ArgumentParser(description="graph retrieval from toy dataset")
parser.add_argument('--full_data_path', type=str, help='dataset path (templates)', default='/home/nh/CAD/data/toygraph/full_templates')
parser.add_argument('--benchmark_path', type=str, help='benchmark path', default='/home/nh/CAD/data/toygraph/benchmark.pkl')
parser.add_argument('--k', type=int, help='number of similar nodes to retrieve', default=5)

args = parser.parse_args()

list_of_graphs, benchmark = load_data(args)

# num_queries = len(benchmark)  # 20000
num_queries = len(benchmark) 
query_distances = []
TP_list = []
Recall_list = []


for i in range (num_queries): 
    print('query number', i)
    query_subgraph = benchmark[i][0]  # query subgraph of i'th benchmark sample 
    # print('visualizing the query subgraph')
    # visualize(query_subgraph, key='representation')
    query_benchmark_list = benchmark[i][1] # list of benchmark graphs for the i'th query '''

    print('number of benchmark subgraphs for this query is ', len(query_benchmark_list))
    # print("Retrieving similar subgraphs ...")
    retrieved_subgraphs = graphs_retrieval(args, query_subgraph, list_of_graphs)  
    
    # print('evaluating the retrieval ...')
    query_dist = evaluate(args, retrieved_subgraphs, query_benchmark_list, list_of_graphs)

    # TP = sum(i < 0.5 for i in query_dist)
    TP = query_dist.count(0) # matches with edit_dist = 0 
    recall = TP/len(query_benchmark_list)
    print('TP', TP)
    print('recall', recall)

    TP_list.append(TP)
    Recall_list.append(recall)
    if i%100 == 0: 
        np.save('TP.npy', TP_list)
        np.save('recall.npy', Recall_list)

np.save('TP.npy', TP_list)
np.save('recall.npy', Recall_list)


# note: the feature vectors in query subgraphs and benchmarks is "representation" and in reference graphs is "representations". correct it later (in the collection function)


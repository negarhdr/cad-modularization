import networkx as nx 
import pickle 
import dill 
import json 
import os 
import argparse


def make_graph_collection(args): 
    ### load templates
    ### 200 graphs, each having 10 random representations (of different styles, fonts, etc) --> 2000 types in total
    for filename in os.listdir(args.data_path):
        graph_name = filename
        graph_index = filename.split(".")[0]
        file_path = os.path.join(args.data_path, filename)
        if os.path.isfile(file_path):
            for i in range(10):
                with open(file_path, 'r') as graph_file:
                    graph = nx.node_link_graph(json.loads(graph_file.read()))
                new_graph = graph
                for node_name in list(graph.nodes): 
                    new_graph.nodes[node_name]['representations'] = graph.nodes[node_name]['representations'][i]
                new_graph_data = nx.node_link_data(new_graph)
                ### save new graph 
                new_file_name = graph_index + '_' + str(i) + '.json'
                new_file_path = os.path.join(args.full_data_path, new_file_name)
                with open(new_file_path, "w") as f:
                    json.dump(new_graph_data, f)


parser = argparse.ArgumentParser(description="make a collection of graphs with different representations")
parser.add_argument('--data_path', type=str, help='dataset path (templates)', default='/home/nh/CAD/data/toygraph/templates')
parser.add_argument('--full_data_path', type=str, help='dataset path (templates)', default='/home/nh/CAD/data/toygraph/full_templates')

args = parser.parse_args()
make_graph_collection(args)

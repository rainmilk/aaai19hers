import networkx as nx
import math
from sklearn.utils import shuffle


def read_graph(path, weighted=False, directed=False, remove_isolate=True):
    if weighted:
        G = nx.read_edgelist(path, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
        #G = nx.read_weighted_edgelist(path, nodetype=int, create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph())
        # for edge in G.edges():
        #     G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    if remove_isolate:
        G.remove_nodes_from(list(nx.isolates(G)))

    return G

def split_graph(G,test_ratio):
    edge_list_all = list(G.edges())
    train_G = G.copy()
    test_G = nx.Graph()
    all_edge_num = G.number_of_edges()
    test_edge_num = math.floor(all_edge_num * test_ratio)
    # construct test graph
    min_degree = 3
    remove_edge = []
    edge_list_shuffle = shuffle(edge_list_all)
    count = 0
    for i, edge in enumerate(edge_list_shuffle):
        from_node = edge[0]
        to_node = edge[1]
        if count > test_edge_num:
            break
        if train_G.degree(from_node) >= min_degree and train_G.degree(to_node) >= min_degree:
            remove_edge.append((from_node, to_node))
            train_G.remove_edge(from_node, to_node)
            count = count + 1

    test_G.add_edges_from(remove_edge)
    return test_G,train_G

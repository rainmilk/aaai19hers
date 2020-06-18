import numpy as np
import math
from model.data_utilities import split_negative_test
from sklearn.utils import shuffle
from model.graph_utilities import read_graph
import random
import csv
import networkx as nx


data_path = "datasets/book/"


def network_statistic(data_path):

    G=read_graph(data_path)
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    max_deg= max(list(degree_sequence))
    ave_deg= sum(list(degree_sequence))/len(list(degree_sequence))
    print(max_deg,ave_deg)

def construct_train(data_name):
    test_ratio=0.2
    ui_data_path = data_path + "%s_rating.txt"%(data_name)
    ui_data=np.loadtxt(ui_data_path,dtype=np.int32)

    ui_data=shuffle(ui_data)
    edge_num=len(ui_data)
    test_ui= ui_data[:math.ceil(test_ratio*edge_num),]
    train_ui = ui_data[math.ceil(test_ratio*edge_num):,]
    train_path = data_path + "%s_rating_train.txt"%(data_name)
    test_path = data_path + "%s_rating_test.txt"%(data_name)
    neg_path = "networkRS/%s_rating_test_neg.csv"%(data_name)

    np.savetxt(train_path,train_ui)
    np.savetxt(test_path,test_ui)

    test_data = np.loadtxt(test_path, dtype=np.int32)
    data = np.loadtxt(ui_data_path, dtype=np.int32)
    neg_pro=10
    neg_all=split_negative_test(test_data,data,neg_pro)
    with open(neg_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(neg_all)
    print("split data successfully")

def construct_cold_user():
    name='lastfm'
    user_path = data_path + "%s_userNet.txt"%name
    rating_path = data_path + "%s_rating.txt"%name
    neg_path= "networkRS/%s_rating_test_cold_user_neg.txt"%name
    train_path = "networkRS/%s_rating_train_cold_user.txt"%name
    test_path = "networkRS/%s_rating_test_cold_user.txt"%name
    rating=np.loadtxt(rating_path,dtype=np.int32)
    # G_user=read_graph(user_path)
    # user_list=list(G_user.nodes())

    user_list=list(set(rating[:,0]))
    user_num=len(user_list)
    test_ratio=0.2
    neg_pro=10
    #test_users=random.sample(user_list,math.ceil(test_ratio*user_num))
    test_users=set()
    while len(test_users)<math.ceil(test_ratio*user_num):
        sample_user=random.choice(rating[:,0])
        test_users.add(sample_user)

    test_mask = np.array([u in test_users for u in rating[:,0]])
    train_mask = ~test_mask
    train_data = rating[train_mask,:]
    test_data = rating[test_mask,:]
    # train_data=np.asarray([rating[i,:] for i,u in enumerate(rating[:,0]) if u not in test_users])
    # test_data =np.asarray([rating[i,:] for i, u in enumerate(rating[:,0]) if u in test_users])

    train_data_len=len(train_data)
    train_data= shuffle(train_data)
    #train_data=train_data[:math.ceil(train_data_len*0.5)]
    neg_all=split_negative_test(test_data,rating,neg_pro)
    with open(neg_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(neg_all)

    np.savetxt(train_path,train_data)
    np.savetxt(test_path, test_data)

    print("construct cold user data successfully")

    return True

def construct_cold_item(name):
    rating_path = data_path + "%s_rating.txt"%name
    neg_path = data_path + "%s_rating_test_cold_item_neg.txt"%name
    train_path = data_path + "%s_rating_train_cold_item.txt"%name
    test_path = data_path + "%s_rating_test_cold_item.txt"%name
    rating=np.loadtxt(rating_path,dtype=np.int32)
    # G_item=read_graph(item_path)
    # item_list=list(G_item.nodes())

    item_list=list(set(rating[:,1]))
    item_num=len(item_list)
    test_ratio=0.2
    neg_pro=10
   # test_items=random.sample(item_list,math.ceil(test_ratio*item_num))

    test_items=set()
    while len(test_items)<math.ceil(test_ratio*item_num):
        sample_user=random.choice(rating[:,1])
        test_items.add(sample_user)

    rating=rating[:,[1,0]]

    test_mask = np.array([u in test_items for u in rating[:,0]])
    train_mask = ~test_mask
    train_data = rating[train_mask,:]
    test_data = rating[test_mask,:]

    # train_data=np.asarray([rating[i,:] for i,u in enumerate(rating[:,1]) if u not in test_items])
    # test_data =np.asarray([rating[i,:] for i,u in enumerate(rating[:, 1]) if u in test_items])

    neg_all=split_negative_test(test_data,rating,neg_pro)
    with open(neg_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(neg_all)


    np.savetxt(train_path, train_data)
    np.savetxt(test_path, test_data)

    print("construct cold item data successfully")

    return True

def get_attention_graph_RS(model, G_u, G_i, edge, topK, att_graph_path, order=2):
    first_batch_data = np.zeros([1, topK], dtype=np.int32)
    second_batch_data = np.zeros([1, topK, topK], dtype=np.int32)
    embeddings = model.user_embed.get_weights()[0]
    item_embeddings= model.item_embed.get_weights()[0]
    source_node=edge[0]
    to_node=edge[1]
    edge_score=1

    attention_layer = model.user_model.get_layer("attention_first")
    attention_mid_wt = attention_layer.get_weights()[0]
    attention_mid_b = attention_layer.get_weights()[1]
    attention_out_wt = attention_layer.get_weights()[2]
    attention_out_b = attention_layer.get_weights()[3]

    attention_layer_second = model.second_model.get_layer("attention_second")
    attention_mid_wt_second = attention_layer_second.get_weights()[0]
    attention_mid_b_second = attention_layer_second.get_weights()[1]
    attention_out_wt_second = attention_layer_second.get_weights()[2]
    attention_out_b_second = attention_layer_second.get_weights()[3]

    attention_layer_item = model.item_model.get_layer("target_context")
    attention_mid_wt_item = attention_layer.get_weights()[0]
    attention_mid_b_item = attention_layer.get_weights()[1]
    attention_out_wt_item = attention_layer.get_weights()[2]
    attention_out_b_item = attention_layer.get_weights()[3]

    edgelist = np.zeros([1, 3])
    edgelist[0,:] = [source_node, to_node, edge_score]
    for i in [source_node]:
        target = i

        node_dict = {}
        first_neighbors = set(G_u.neighbors(target))

        first_neighbors = list(first_neighbors)
        nb_first_node = len(first_neighbors)

        first_memory = embeddings[first_neighbors]
        hid_units = np.tanh(np.dot(first_memory, attention_mid_wt) + attention_mid_b)
        scores = np.dot(hid_units, attention_out_wt) + attention_out_b
        scores = np.exp(scores) / np.sum(np.exp(scores))

        target_dup = np.repeat(target, nb_first_node)
        target_first_edge = np.stack([target_dup, np.asarray(first_neighbors), scores]).T
        if nb_first_node > topK:
            top_score = scores[np.argpartition(-scores, topK)][:topK]
            first_neighbors = [first_neighbors[m] for m in np.argpartition(-scores, topK)[:topK]]
            target_dup = np.repeat(target, topK)
            target_first_edge = np.stack([target_dup, np.asarray(first_neighbors), top_score]).T
        edgelist = np.append(edgelist, target_first_edge, axis=0)

        if order > 1:
            # second_inputs = np.zeros([nb_first_node, topK])
            for j, first_node in enumerate(first_neighbors):
                child_nodes = [n for n in G_u.neighbors(first_node) if n != target]
                top_k_nodes = child_nodes
                nb_child = len(child_nodes)
                first_node_embedding = embeddings[first_node]
                second_neighbors_embedding = embeddings[child_nodes]
                hid_units = np.tanh(
                    np.dot(second_neighbors_embedding, attention_mid_wt_second) + attention_mid_b_second)
                attention_values = np.dot(hid_units, attention_out_wt_second) + attention_out_b_second
                # attention_values = np.dot(second_neighbors_embedding, attention_mid_wt_second) + attention_mid_b_second
                first_dup = np.repeat(first_node, nb_child)
                attention_values = np.exp(attention_values) / np.sum(np.exp(attention_values))
                first_second_edge = np.stack([first_dup, child_nodes, attention_values]).T

                if nb_child > topK:
                    top_k_nodes_index = np.argpartition(-attention_values, topK)[:topK]
                    top_k_nodes = [child_nodes[m] for m in top_k_nodes_index]
                    top_k_attention = attention_values[np.argpartition(-attention_values, topK)][:topK]
                    first_dup = np.repeat(first_node, topK)
                    first_second_edge = np.stack([first_dup, top_k_nodes, top_k_attention]).T

                edgelist = np.append(edgelist, first_second_edge, axis=0)
                node_dict[first_node] = top_k_nodes

    for target in [to_node]:

        node_dict = {}
        first_neighbors = set(G_i.neighbors(target))

        first_neighbors = list(first_neighbors)
        nb_first_node = len(first_neighbors)

        first_memory = item_embeddings[first_neighbors]
        hid_units = np.tanh(np.dot(first_memory, attention_mid_wt_item) + attention_mid_b_item)
        scores = np.dot(hid_units, attention_out_wt_item) + attention_out_b_item
        scores = np.exp(scores) / np.sum(np.exp(scores))

        target_dup = np.repeat(target, nb_first_node)
        target_first_edge = np.stack([target_dup, np.asarray(first_neighbors), scores]).T
        if nb_first_node > topK:
            top_score = scores[np.argpartition(-scores, topK)][:topK]
            first_neighbors = [first_neighbors[m] for m in np.argpartition(-scores, topK)[:topK]]
            target_dup = np.repeat(target, topK)
            target_first_edge = np.stack([target_dup, np.asarray(first_neighbors), top_score]).T
        edgelist = np.append(edgelist, target_first_edge, axis=0)


    with open(att_graph_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target', 'weight'])
        writer.writerows(edgelist)
        # np.savetxt(att_graph_path, edgelist)
    print("write edgelist successfully")
    return True

#
# data_name='lastfm'
# construct_cold_item(data_name)
# data_name='book'
# construct_cold_item(data_name)
# user_net_path='networkRS/%s_userNet.txt'%data_name
# data_path = 'networkRS/%s_rating.txt' % data_name
#item_path = 'networkRS/%s_itemNet.txt'%data_name
#network_statistic(item_path )
#
# construct_train(data_name)
#construct_cold_user()

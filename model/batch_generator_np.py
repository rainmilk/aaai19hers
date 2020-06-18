import numpy as np
import random

class BaseGenerator(object):
    def __init__(self, G, model):
        self.G = G
        self.node_list = list(G.nodes())
        self.model = model
        self.node_cache = {}

    def clear_node_cache(self):
        self.node_cache.clear()

    def neighbors(self, n):
        return self.G.predecessors(n) if self.G.is_directed() else self.G.neighbors(n)

    def get_target_batch(self, edge_batch):
        target_batch = []
        positive_batch = []
        negative_batch = []
        for t in edge_batch:
            from_node, to_node = t
            if self.G.is_directed() and random.random() > 0.5: # Swap from node and to node
                to_node, from_node = from_node, to_node

            target_batch.append(from_node)
            positive_batch.append(to_node)
            neighbors = set(self.neighbors(from_node))
            neighbors.add(from_node)
            neighbors.add(to_node)
            while True:
                rnd_node = random.choice(self.node_list)
                if rnd_node not in neighbors:
                    negative_batch.append(rnd_node)
                    break
        return target_batch, positive_batch, negative_batch

    def get_batch_data_sample_k(self, batch_node, topK=50, excluded_node_batch=None, order=2):
        if not isinstance(batch_node, list): batch_node = [batch_node]
        batch_size = len(batch_node)
        first_batch_data = np.zeros([batch_size, topK], dtype=np.int32)
        second_batch_data = None
        for i in range(batch_size):
            target = batch_node[i]
            first_neighbors = set(self.neighbors(target))
            first_neighbors.discard(target)
            if excluded_node_batch is not None: first_neighbors.discard(excluded_node_batch[i])
            first_neighbors = list(first_neighbors)
            if len(first_neighbors) > topK:
                first_neighbors = random.sample(first_neighbors, topK)
            first_batch_data[i, :len(first_neighbors)] = first_neighbors

            if order > 1:
                second_batch_data = np.zeros([batch_size, topK, topK], dtype=np.int32)
                for j, first_node in enumerate(first_neighbors):
                    child_nodes = list(self.neighbors(first_node))
                    if len(child_nodes) > topK:
                        child_nodes = random.sample(child_nodes, topK)
                    second_batch_data[i,j,:len(child_nodes)] = child_nodes

        return first_batch_data, second_batch_data

    def get_node_embed(self):
        return self.model.second_model.get_layer('node_embedding')

    def get_first_layer_weight(self):
        attention_layer = self.model.subgraph_model.get_layer("attention_first")
        attention_mid_wt = attention_layer.get_weights()[0]
        attention_out_wt = attention_layer.get_weights()[1]
        attention_mid_b = attention_layer.get_weights()[2]
        attention_out_b = attention_layer.get_weights()[3]
        return attention_mid_wt, attention_mid_b, attention_out_wt, attention_out_b

    def get_second_layer_weight(self):
        attention_layer_second = self.model.second_model.get_layer("attention_second")
        attention_mid_wt = attention_layer_second.get_weights()[0]
        attention_out_wt = attention_layer_second.get_weights()[1]
        attention_mid_b = attention_layer_second.get_weights()[2]
        attention_out_b = attention_layer_second.get_weights()[3]
        return attention_mid_wt, attention_mid_b, attention_out_wt, attention_out_b

    def get_batch_data_topk(self, batch_node, topK=50, excluded_node_batch=None, predict_batch_size=100, order=2):
        if not isinstance(batch_node, list): batch_node = [batch_node]
        batch_size = len(batch_node)

        embeddings = self.get_node_embed()
        attention_mid_wt, attention_out_wt,  attention_mid_b, attention_out_b = self.get_first_layer_weight()
        attention_mid_wt_second, attention_out_wt_second, attention_mid_b_second, attention_out_b_second = self.get_second_layer_weight()

        first_batch_data = np.zeros([batch_size, topK], dtype=np.int32)
        second_batch_data = None
        for i in range(batch_size):
            target = batch_node[i]
            node_dict =  self.node_cache.get(target)
            if node_dict is None:
                target_embedding = np.expand_dims(embeddings[target], axis=0)
                prune_list = {}
                first_neighbors = set(self.neighbors(target))
                first_neighbors.discard(target)
                if excluded_node_batch is not None: first_neighbors.discard(excluded_node_batch[i])
                first_neighbors = list(first_neighbors)
                nb_first_node = len(first_neighbors)

                if nb_first_node > topK:
                    first_memory = embeddings[first_neighbors]
                    hid_units = np.tanh(np.dot(first_memory, attention_mid_wt) + attention_mid_b)
                    scores = np.dot(hid_units, attention_out_wt) + attention_out_b
                    first_neighbors = [first_neighbors[m] for m in np.argpartition(-scores, topK)[:topK]]

                first_batch_data[i, :nb_first_node] = first_neighbors

                if order > 1:
                    second_batch_data = np.zeros([batch_size, topK, topK], dtype=np.int32)
                    for j, first_node in enumerate(first_neighbors):
                        child_nodes = list(self.neighbors(first_node))
                        top_k_nodes = child_nodes
                        if len(child_nodes) > topK:
                            second_neighbors_embedding = embeddings[child_nodes]
                            # first_repeat = np.repeat(np.expand_dims(embeddings[first_node], 0), len(child_nodes), axis=0)
                            # attention_vectors = np.concatenate([second_neighbors_embedding, first_repeat], axis=-1)
                            hid_units = np.tanh(np.dot(second_neighbors_embedding, attention_mid_wt_second) + attention_mid_b_second)
                            attention_values = np.dot(hid_units, attention_out_wt_second) + attention_out_b_second

                            top_k_nodes_index = np.argpartition(-attention_values, topK)[:topK]
                            top_k_nodes = [child_nodes[m] for m in top_k_nodes_index]
                            second_batch_data[i, j, :len(top_k_nodes)] = top_k_nodes

        return first_batch_data, second_batch_data

    def generate_triplet_batch(self, edge_batch, topK=50, attention_sampling = False):
        batch_node, positive_batch, negative_batch = self.get_target_batch(edge_batch)
        if (attention_sampling):
            first_batch_data, second_batch_data = self.get_batch_data_topk(batch_node,
                                                                           excluded_node_batch=positive_batch, topK=topK)
            positive_first_batch, positive_second_data = self.get_batch_data_topk(positive_batch,
                                                                                  excluded_node_batch=batch_node, topK=topK)
            negative_first_batch, negative_second_data = self.get_batch_data_topk(negative_batch,
                                                                             excluded_node_batch=batch_node, topK=topK)
        else:
            first_batch_data, second_batch_data = self.get_batch_data_sample_k(batch_node,
                                                                           excluded_node_batch=positive_batch,
                                                                           topK=topK)
            positive_first_batch, positive_second_data = self.get_batch_data_sample_k(positive_batch,
                                                                                  excluded_node_batch=batch_node,
                                                                                  topK=topK)
            negative_first_batch, negative_second_data = self.get_batch_data_sample_k(negative_batch,
                                                                                  excluded_node_batch=batch_node,
                                                                                  topK=topK)

        return batch_node, positive_batch, negative_batch,\
               first_batch_data, second_batch_data,\
               positive_first_batch, positive_second_data, \
               negative_first_batch, negative_second_data

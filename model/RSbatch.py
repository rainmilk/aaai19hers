import numpy as np
import random
from model.batch_generator_np import BaseGenerator

class ItemGenerator(BaseGenerator):
    def __init__(self, G, model):
        self.G = G
        self.model = model
        self.node_cache = {}
        self.node_list = list(G.nodes())

    def get_node_embed(self):
        return self.model.item_embed.get_weights()[0]

    def get_first_layer_weight(self):
        attention_layer = self.model.item_model.get_layer("first_context")
        attention_mid_wt = attention_layer.get_weights()[0]
        attention_mid_b = attention_layer.get_weights()[1]
        attention_out_wt = attention_layer.get_weights()[2]
        attention_out_b = attention_layer.get_weights()[3]
        return attention_mid_wt, attention_mid_b, attention_out_wt, attention_out_b


import numpy as np
import random

class TripletGenerator(BaseGenerator):
    def __init__(self, G_U, model, G_UI, G_I):
        super(TripletGenerator, self).__init__(G_U, model)
        self.G_UI = G_UI
        self.itemGenerate = ItemGenerator(G_I, model)
        self.item_list = list(G_I.nodes())
        self.user_items = {u: list(G_UI[G_UI[:, 0] == u, 1]) for u in np.unique(G_UI[:, 0])}

    def clear_node_cache(self):
        self.itemGenerate.clear_node_cache()
        super().clear_node_cache()

    def get_node_embed(self):
        return self.model.user_embed.get_weights()[0]

    def get_first_layer_weight(self):
        attention_layer = self.model.user_model.get_layer("attention_first")
        attention_mid_wt = attention_layer.get_weights()[0]
        attention_mid_b = attention_layer.get_weights()[1]
        attention_out_wt = attention_layer.get_weights()[2]
        attention_out_b = attention_layer.get_weights()[3]
        return attention_mid_wt, attention_mid_b, attention_out_wt, attention_out_b

    def get_second_layer_weight(self):
        attention_layer_second = self.model.second_model.get_layer("attention_second")
        attention_mid_wt = attention_layer_second.get_weights()[0]
        attention_mid_b = attention_layer_second.get_weights()[1]
        attention_out_wt = attention_layer_second.get_weights()[2]
        attention_out_b = attention_layer_second.get_weights()[3]
        return attention_mid_wt, attention_mid_b, attention_out_wt, attention_out_b


    def get_target_batch(self, edge_batch):
        target_batch = []
        positive_batch = []
        negative_batch = []
        for t in edge_batch:
            from_node, to_node = t
            target_batch.append(from_node)
            positive_batch.append(to_node)
            neighbors = set(self.user_items[from_node])
            neighbors.add(from_node)
            neighbors.add(to_node)
            while True:
                rnd_node = random.choice(self.item_list)
                if rnd_node not in neighbors:
                    negative_batch.append(rnd_node)
                    break
        return target_batch, positive_batch, negative_batch


    def generate_triplet_batch(self, edge_batch, topK=50, attention_sampling = False):
        batch_node, positive_batch, negative_batch = self.get_target_batch(edge_batch)
        if (attention_sampling):
            first_batch_data, second_batch_data = self.get_batch_data_topk(batch_node, excluded_node_batch=positive_batch, topK=topK)
            positive_first_batch, _  = self.itemGenerate.get_batch_data_topk(positive_batch, excluded_node_batch=batch_node, topK=topK,order=1)
            negative_first_batch, _  = self.itemGenerate.get_batch_data_topk(negative_batch, excluded_node_batch=batch_node, topK=topK,order=1)
        else:
            first_batch_data, second_batch_data = self.get_batch_data_sample_k(batch_node, excluded_node_batch=positive_batch, topK=topK)
            positive_first_batch, _ = self.itemGenerate.get_batch_data_sample_k(positive_batch, excluded_node_batch=batch_node,  topK=topK,order=1)
            negative_first_batch, _ = self.itemGenerate.get_batch_data_sample_k(negative_batch, excluded_node_batch=batch_node, topK=topK,order=1)

        return batch_node, positive_batch, negative_batch,\
               first_batch_data, second_batch_data,\
               positive_first_batch, \
               negative_first_batch

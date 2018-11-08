from keras.layers import Input, Activation, concatenate, RepeatVector, dot, GRU, multiply, Lambda, Embedding, Dense, Dropout
from keras.models import Model, Sequential
from model.timedistributed import TimeDistributedMultiInput
from model.attentionlayer import PairAttention, AggGatedAttention, TwoLayerAttention, GatedAttention, AverageAttention
from keras import backend as K
from model.masklayers import DropMask, MultiMask


class NetworkRS(object):
    def __init__(self,
                 user_size, item_size, embed_len, score_model,
                 first_dims=1, second_dims=1, embed_regularizer=None, directed=False,
                 mem_filt_alpha=1, mem_agg_alpha=0.001, user_mask=None, item_mask=None, dropout=0):
        self.user_size = user_size
        self.item_size = item_size
        self.user_mask = user_mask
        self.item_mask = item_mask
        self.embed_len = embed_len
        self.user_embed = Embedding(self.user_size + 1, self.embed_len, name='node_embedding',
                                    embeddings_regularizer=embed_regularizer, mask_zero=True)
        self.item_embed = Embedding(self.item_size + 1, self.embed_len, name='node_embedding',
                                    embeddings_regularizer=embed_regularizer, mask_zero=True)

        self.second_dims=second_dims
        self.first_dims = first_dims
        self.score_model = score_model
        self.directed = directed
        self.mem_filt_alpha = mem_filt_alpha
        self.mem_agg_alpha = mem_agg_alpha
        self.dropout = dropout
        self.second_model, self.first_model, self.user_model = self._build_user_model()
        self.item_mem_model, self.item_model = self._build_item_model()
        self.score_model = self._build_score_model()
        self.triplet_model = self._build_triplet_model()

    @property
    def score_input_shape(self):
        return (self.embed_len,)

    def _build_undirected_score_model(self):
        return self.score_model

    def _build_directed_score_model(self):
        return self._build_undirected_score_model()

    def _build_score_model(self):
        if self.directed:
            return self._build_directed_score_model()
        return self._build_undirected_score_model()

    def _build_user_model(self):
        target_input = Input(shape=(1,), dtype='int32', name='target_input')
        target_embedding = self.user_embed(target_input)

        if self.user_mask is not None:
            target_embedding = MultiMask(self.user_size+1, self.user_mask)([target_input, target_embedding])

        first_node_input = Input(shape=(1,), dtype='int32', name='first_input_target')
        second_node_input = Input(shape=(self.second_dims,), dtype='int32', name='second_input_target')
        first_embedded_node = self.user_embed(first_node_input)
        second_embedded_nodes = self.user_embed(second_node_input)
        attention_second = TwoLayerAttention(name='attention_second', mid_units=64,
                                             alpha=self.mem_filt_alpha, keepdims=True)(second_embedded_nodes)
        # attention_second = BatchAttention(name='attention_second', keepdims=True)([second_embedded_nodes, first_embedded_node])

        # concat_embed = concatenate([first_embedded_node, attention_second], axis=1)
        # second_memory = TwoLayerAttention(name='memory_second', mid_units=miduinits, alpha=self.mem_agg_alpha)(concat_embed)
        second_memory = GatedAttention(name='memory_second', mid_units=64, alpha=self.mem_agg_alpha)(
            [first_embedded_node, attention_second])
        second_memory = Dropout(self.dropout)(second_memory)

        second_memory_model = Model(inputs=[target_input, first_node_input, second_node_input], outputs=second_memory)

        second_input = Input(shape=(self.first_dims, self.second_dims), dtype='int32', name='second_input')
        first_input = Input(shape=(self.first_dims,), dtype='int32', name='first_input')

        first_input_1 = Lambda(lambda x : K.expand_dims(x))(first_input)
        target_input_1 = RepeatVector(self.first_dims)(target_input)
        second_memory_dist = TimeDistributedMultiInput(second_memory_model)([target_input_1, first_input_1, second_input])
        attention_first = TwoLayerAttention(name='attention_first', mid_units=64,
                                            alpha=self.mem_filt_alpha, keepdims=True)(second_memory_dist)
        # attention_first = BatchAttention(name='attention_first', keepdims=True)([second_memory_dist, target_embedding])

        # concat_embed = concatenate([target_embedding, attention_first], axis=1)
        # target_memory = TwoLayerAttention(name='memory_final', mid_units=miduinits, alpha=self.mem_agg_alpha)(concat_embed)
        first_memory_model = Model(inputs=[target_input, first_input, second_input], outputs=attention_first, name='first_mem_model')

        target_memory = GatedAttention(name='memory_final', mid_units=64, alpha=self.mem_agg_alpha)(
            [target_embedding, attention_first])

        target_memory = Dropout(self.dropout)(target_memory)
        target_memory = DropMask()(target_memory)

        user_model = Model(inputs=[target_input, first_input, second_input], outputs=target_memory, name='user_model')

        return second_memory_model, first_memory_model, user_model

    def _build_item_model(self):
        target_input = Input(shape=(1,), dtype='int32', name='target_input')
        target_embedding = self.item_embed(target_input)

        if self.item_mask is not None:
            target_embedding = MultiMask(self.item_size+1, self.user_mask)([target_input, target_embedding])

        first_node_input = Input(shape=(self.second_dims,), dtype='int32', name='first_input_target')
        first_embedded_node = self.item_embed(first_node_input)

        first_context = TwoLayerAttention(name='first_context', mid_units=64,
                                             alpha=self.mem_filt_alpha, keepdims=True)(first_embedded_node)
        # attention_second = BatchAttention(name='attention_second', keepdims=True)([second_embedded_nodes, first_embedded_node])

        # concat_embed = concatenate([first_embedded_node, attention_second], axis=1)
        # second_memory = TwoLayerAttention(name='memory_second', mid_units=miduinits, alpha=self.mem_agg_alpha)(concat_embed)
        item_mem_model = Model(inputs=[target_input, first_node_input], outputs=first_context, name='item_mem_model')

        target_context = GatedAttention(name='target_context', mid_units=64, alpha=self.mem_agg_alpha)(
            [target_embedding, first_context])
        target_context = Dropout(self.dropout)(target_context)

        target_context = DropMask()(target_context)

        item_model = Model(inputs=[target_input, first_node_input], outputs=target_context, name='item_model')
        return item_mem_model, item_model

    def _build_triplet_model(self):
        target_input = Input(shape=(1,), dtype='int32', name='target_input')
        positive_input = Input(shape=(1,), dtype='int32', name='positive_input')
        negative_input = Input(shape=(1,), dtype='int32', name='negative_input')
        target_second_input = Input(shape=(self.first_dims, self.second_dims), dtype='int32', name='target_second_input')

        target_first_input = Input(shape=(self.first_dims,), dtype='int32', name='target_first_input')
        positive_first_input = Input(shape=(self.first_dims,), dtype='int32', name='positive_first_input')
        negative_first_input = Input(shape=(self.first_dims,), dtype='int32', name='negative_first_input')

        target_memory = self.user_model([target_input, target_first_input, target_second_input])
        positive_memory = self.item_model([positive_input, positive_first_input])
        negative_memory = self.item_model([negative_input, negative_first_input])

        target_positive_score = self.score_model(inputs=[target_memory, positive_memory])
        target_negative_score = self.score_model(inputs=[target_memory, negative_memory])

        contrastive_scores = concatenate([target_positive_score, target_negative_score], axis=-1, name='contrastive_score')

        triplet_model = Model(inputs=[target_input, target_first_input, target_second_input,
                                      positive_input,positive_first_input,
                                      negative_input,negative_first_input],
                              outputs=contrastive_scores, name='triplet_model')

        return triplet_model


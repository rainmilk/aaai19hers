from keras.layers import Input, add, dot, concatenate, Embedding, Lambda, BatchNormalization
from keras.models import Model
import keras.backend as K


SqueezeEmbed = Lambda(lambda x : K.squeeze(x, 1))

class mlmf(object):
    def __init__(self, nb_user, nb_item, embed_dim, score_model, reg=None):
        self.embed_dim = embed_dim
        self.user_emb = Embedding(nb_user, embed_dim, input_length=1, name='user_embedding', embeddings_regularizer=reg)
        self.item_emb = Embedding(nb_item, embed_dim, input_length=1, name='item_embedding', embeddings_regularizer=reg)
        self.score_model = score_model
        self.contrast_model = self._build()

    def _build(self):
        user_input = Input(shape=(1,), dtype='int32', name="user_input")
        pos_item_input = Input(shape=(1,), dtype='int32', name="pos_item_input")
        neg_item_input = Input(shape=(1,), dtype='int32', name="neg_item_input")

        user_emb = SqueezeEmbed(self.user_emb(user_input))

        pos_item_emb = SqueezeEmbed(self.item_emb(pos_item_input))
        neg_item_emb = SqueezeEmbed(self.item_emb(neg_item_input))

        pos_score = self.score_model([user_emb, pos_item_emb])
        neg_score = self.score_model([user_emb, neg_item_emb])

        contrastive_scores = concatenate([pos_score, neg_score], axis=-1,
                                         name='contrastive_score')

        contrastive_model = Model([user_input, pos_item_input, neg_item_input],
                                  contrastive_scores, name='contrastive_model')

        return contrastive_model
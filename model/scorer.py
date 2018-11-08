from keras.layers import Input, Activation, concatenate, multiply, Lambda, Dense, Dropout, add, subtract
from keras.models import Model
import keras.backend as K


L2Norm = Lambda(lambda x : x / K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)))

def nn_scoremodel(input_shape, edge_rep_len, dropout=0,
                  score_rep_norm=False, score_hid_act='tanh', score_act=None):
    from_node = Input(shape=input_shape)
    to_node = Input(shape=input_shape)

    if score_rep_norm:
        from_node_norm = L2Norm(from_node)
        to_node_norm = L2Norm(to_node)
    else:
        from_node_norm = from_node
        to_node_norm = to_node

    edge_embed = concatenate([from_node_norm, to_node_norm])
    if dropout > 0:
        edge_embed = Dropout(dropout)(edge_embed)

    edge_embed = Dense(edge_rep_len, activation=score_hid_act)(edge_embed)
    if dropout > 0:
        edge_embed = Dropout(dropout)(edge_embed)

    score = Dense(1)(edge_embed)

    if score_act is not None:
        score = Activation(activation=score_act)(score)

    return Model(inputs=[from_node, to_node], outputs=score, name='score_model')


def inner_prod_scoremodel(input_shape, score_rep_norm=False, score_act=None):
    from_node = Input(shape=input_shape)
    to_node = Input(shape=input_shape)

    if score_rep_norm:
        from_node_norm = L2Norm(from_node)
        to_node_norm = L2Norm(to_node)
    else:
        from_node_norm = from_node
        to_node_norm = to_node

    edge_embed = multiply([from_node_norm, to_node_norm], name='edge_embed')
    score = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True))(edge_embed)

    if score_act is not None:
        score = Activation(activation=score_act)(score)

    return Model(inputs=[from_node, to_node], outputs=score, name='score_model')


def fm_scoremodel(input_shape, nb_factor=32, score_rep_norm=False, score_act=None):
    from_node = Input(shape=input_shape)
    to_node = Input(shape=input_shape)

    if score_rep_norm:
        from_node_norm = L2Norm(from_node)
        to_node_norm = L2Norm(to_node)
    else:
        from_node_norm = from_node
        to_node_norm = to_node

    edge_embed = concatenate([from_node_norm, to_node_norm])
    edge_embed = Dense(nb_factor)(edge_embed)
    edge_embed = multiply([edge_embed, edge_embed], name='edge_embedding')

    score = Dense(1)(concatenate([edge_embed, from_node_norm, to_node]))

    if score_act is not None:
        score = Activation(activation=score_act)(score)

    return Model(inputs=[from_node, to_node], outputs=score, name='score_model')
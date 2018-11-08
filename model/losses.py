import keras.backend as K

def infinite_margin_loss(y_true, y_pred):
    delta = -K.log(K.sigmoid(y_pred[:, 0] - y_pred[:, 1]))
    return K.mean(delta)


def max_margin_loss(y_true, y_pred):
    loss = y_true - (y_pred[:, 0] - y_pred[:, 1])
    return K.mean(K.maximum(loss, 0))


def nce_loss(y_true, y_pred):
    loss = K.categorical_crossentropy(y_true, y_pred, True)
    return K.mean(loss)


def hinge_accuracy(y_true, y_pred):
    return K.mean(K.cast(K.equal(y_true, K.sign(y_pred)), K.floatx()), axis=-1)


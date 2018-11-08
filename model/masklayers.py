from keras.layers.core import Lambda, Layer
import keras.backend as K
import numpy as np


class DropMask(Lambda):
    def __init__(self):
        super(DropMask, self).__init__((lambda x : x))
        self.supports_masking = True


class MultiMask(Layer):
    def __init__(self, nb_val, mask_vals, **kwargs):
        mask = np.ones(nb_val, dtype=np.bool)
        mask[mask_vals] = False
        self.mask_vals = K.constant(mask, dtype='bool')
        self.supports_masking = True
        super(MultiMask, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        mask_0 = K.gather(self.mask_vals, inputs[0])
        if mask[1] is not None:
            mask = mask[1] & mask_0
        else:
            mask = mask_0
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def call(self, inputs, mask=None):
        return inputs[1]

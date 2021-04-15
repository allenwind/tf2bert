import tensorflow as tf

# 对多个输入进行merge操作的层

class ReversedConcatenate1D(tf.keras.layers.Layer):
    """对输入进行反向拼接"""

    def __init__(self, axis=-1, **kwargs):
        super(ReversedConcatenate1D, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, mask=None):
        if mask is None:
            mask = tf.ones_like(inputs[..., 0], dtype=tf.bool)
        x_forward = inputs
        x_backward = tf.reverse_sequence(inputs, mask)
        x = tf.concat([x_forward, x_backward], axis=-1)
        x = x * mask
        return x

class LayersConcatenate(tf.keras.layers.Layer):
    """多层输出结果的拼接"""

    def __init__(self, layers, axis=-1, **kwargs):
        super(LayersConcatenate, self).__init__(**kwargs)
        self.layers = layers
        self.axis = axis

    def call(self, inputs):
        x = []
        for layer in self.layers:
            x.append(layer(inputs))
        x = tf.concat(x, self.axis)
        return x

class MaskedConcatenate1D(tf.keras.layers.Layer):
    """支持对齐mask的Concatenate1D"""

    def __init__(self, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return tf.concat(inputs, axis=1)

    def compute_mask(self, inputs, mask=None):
        """对齐mask"""
        if mask is not None:
            masks = []
            for i, m in enumerate(mask):
                if m is None:
                    m = tf.ones_like(inputs[i][..., 0], dtype=tf.bool)
                masks.append(m)
            return tf.concat(masks, axis=1)

    def compute_output_shape(self, input_shape):
        if all([shape[1] for shape in input_shape]):
            seq_len = sum([shape[1] for shape in input_shape])
        else:
            seq_len = None
        return (input_shape[0][0], seq_len, input_shape[0][2])

class MaskedFlatten(tf.keras.layers.Flatten):
    """支持mask的Flatten"""

    def __init__(self, **kwargs):
        super(MaskedFlatten, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

import tensorflow as tf

# 意义匹配层

class MatchingLayer(tf.keras.layers.Layer):
    """匹配层"""

    def __init__(self, func="dot", normalize=False, **kwargs):
        super(MatchingLayer, self).__init__(**kwargs)
        assert func in ("dot", "mul", "plus", "minus", "abs", "concat")
        self.func = func
        self.normalize = normalize

    def build(self, input_shape):
        self._shape1 = input_shape[0]
        self._shape2 = input_shape[1]

    def call(self, inputs, mask=None, **kwargs):
        x1, x2 = inputs
        if self.func == "dot":
            if self.normalize:
                x1 = tf.math.l2_normalize(x1, axis=2)
                x2 = tf.math.l2_normalize(x2, axis=2)
            return tf.expand_dims(tf.einsum("bid,bjd->bij", x1, x2), axis=3)
        if self.func == "mul":
            def func(x, y):
                return x * y
        elif self.func == "plus":
            def func(x, y):
                return x + y
        elif self.func == "minus":
            def func(x, y):
                return x - y
        elif self.func == "abs":
            def func(x, y):
                return tf.abs(x - y)
        else:
            # self.func == "concat":
            def func(x, y):
                return tf.concat([x, y], axis=3)

        x1_exp = tf.stack([x1] * self._shape2[1], 2)
        x2_exp = tf.stack([x2] * self._shape1[1], 1)
        return func(x1_exp, x2_exp)

    def compute_output_shape(self, input_shape):
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if self.func in ("mul", "plus", "minus", "abs"):
            return (shape1[0], shape1[1], shape2[1], shape1[2])
        elif self.func == "dot":
            return (shape1[0], shape1[1], shape2[1], 1)
        else:
            # self.func == "concat"
            return (shape1[0], shape1[1], shape2[1], shape1[2] + shape2[2])

    def get_config(self):
        base_configs = super(MatchingLayer, self).get_config()
        configs = {
            "normalize": self.normalize,
            "func": self.func
        }
        return dict(list(base_configs.items()) + list(configs.items()))

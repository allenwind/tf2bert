import tensorflow as tf
from tensorflow.keras.layers import *
from tf2bert.layers import MultiHeadAttention

# MultiHeadAttention测试主要是mask是否正确

sequence_length = None

inputs = Input(shape=(sequence_length,))
x = Embedding(1000, 128, mask_zero=True)(inputs)
x = [x, x, x]
x = MultiHeadAttention(8, 16)(x)
print(x._keras_mask)

inputs = Input(shape=(sequence_length,))
x = Embedding(1000, 128, mask_zero=True)(inputs)
x = [x, x, x]
x, s = MultiHeadAttention(8, 16, return_attention_scores=True)(x)
print(x._keras_mask)
print(s._keras_mask)

class TestMultiHeadAttention(MultiHeadAttention):

    def call(self, inputs, mask=None, **kwargs):
        q_mask, k_mask, v_mask = mask
        print(q_mask)
        print(k_mask)
        print(v_mask)
        return super().call(inputs, mask, **kwargs)

inputs = Input(shape=(sequence_length,))
x = Embedding(1000, 128, mask_zero=True)(inputs)
x = [x, x, x]
x = TestMultiHeadAttention(8, 16)(x)
print(x._keras_mask)

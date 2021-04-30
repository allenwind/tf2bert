import tensorflow as tf
from tensorflow.keras.layers import *

from tf2bert.initializers import SinusoidalInitializer
from tf2bert.layers import RelativePositionEmbedding
from tf2bert.layers import MultiHeadAttention
from tf2bert.layers import LayerNormalization
from tf2bert.layers import FeedForward, Embedding
from .bert import BERT

class NEZHA(BERT):
    """NEZHA是在BERT的基础上使用经典的相对位置编码，可以处理更长的文本，
    因此这里实现上继承BERT，并在Embedding和位置编码上做相应的修改。
    不同之处：
    - 相对位置编码
    - Whole Word Masking
    - LAMB Optimizer加速训练

    可参考论文：https://arxiv.org/abs/1909.00204
    """

    def build_embeddings(self, inputs):
        """NEZHA的Embedding = Embedding-Token + Embedding-Segment"""
        if self.segment_size > 0:
            x, s = inputs
        else:
            x = inputs[0]

        x = self.build_layer(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name="Embedding-Token"
        )
        if self.segment_size > 0:
            s = self.build_layer(
                inputs=s,
                layer=Embedding,
                input_dim=2, # segment_size=2
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name="Embedding-Segment"
            )
            x = self.build_layer(
                inputs=[x, s],
                layer=Add,
                name="Embedding-Token-Segment"
            )
        x = self.build_layer(
            inputs=x,
            layer=LayerNormalization,
            hidden_initializer=self.initializer,
            name="Embedding-Norm"
        )
        x = self.build_layer(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name="Embedding-Dropout"
        )
        if self.embedding_size != self.hidden_size:
            x = self.build_layer(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name="Embedding-Mapping"
            )
        return x

    def build_hidden_layer(self, inputs, index):
        """NEZHA的隐层架构和BERT差不多：
        Attention->Dropout->Add->LN->FNN->Dropout->Add->LN
        不过这里需要对Attention传入相对位置编码。"""
        attention_name = "Transformer-{}-MultiHeadSelfAttention".format(index)
        callkwargs = {"with_attention_mask": False, "with_position_bias": False}
        x = inputs
        xi = x
        # MultiHeadSelfAttention q=k=v
        x = [x, x, x]

        attention_mask = self.compute_attention_mask(index)
        if attention_mask is not None:
            x.append(attention_mask)
            callkwargs["with_attention_mask"] = True

        position_bias = self.compute_position_bias(xi)
        if position_bias is not None:
            x.append(position_bias)
            callkwargs["with_position_bias"] = "relative_position"

        x = self.build_layer(
            inputs=x,
            layer=MultiHeadAttention,
            callkwargs=callkwargs,
            num_heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.build_layer(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name="{}-Dropout".format(attention_name)
        )
        x = self.build_layer(
            inputs=[xi, x],
            layer=Add,
            name="{}-Add".format(attention_name)
        )
        x = self.build_layer(
            inputs=x,
            layer=LayerNormalization,
            hidden_initializer=self.initializer,
            name="{}-Norm".format(attention_name)
        )

        # position-wise feed-forward networks
        feed_forward_name = "Transformer-{}-FeedForward".format(index)
        xi = x
        x = self.build_layer(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.build_layer(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name="{}-Dropout".format(feed_forward_name)
        )
        x = self.build_layer(
            inputs=[xi, x],
            layer=Add,
            name="{}-Add".format(feed_forward_name)
        )
        x = self.build_layer(
            inputs=x,
            layer=LayerNormalization,
            hidden_initializer=self.initializer,
            name="{}-Norm".format(feed_forward_name)
        )
        return x

    def compute_position_bias(self, inputs=None):
        """处理位置编码，不同Transformer模型有，这里处理相对位置编码。"""
        if self.position_bias is None:
            x = inputs
            x = [x, x]
            embedding_size = 2 * 64 + 1
            self.position_bias = self.build_layer(
                inputs=x,
                layer=RelativePositionEmbedding,
                input_dim=embedding_size,
                output_dim=self.attention_head_size,
                embeddings_initializer=SinusoidalInitializer,
                name="Embedding-Relative-Position",
                trainable=False
            )
        return self.position_bias

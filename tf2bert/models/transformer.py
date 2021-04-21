import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras import initializers
from tensorflow.keras.models import Model

from .base import ModelBuilder
from .base import CheckpointLoader

class Transformer(ModelBuilder, CheckpointLoader):
    """Transformer基类，包括Transformer常用的参数和构建流程"""

    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        hidden_act,
        embedding_size=None,
        dropout_rate=0,
        initializer_range=0.02,
        attention_key_size=None,
        attention_head_size=None,
        sequence_length=None,
        reserved_tokens=None,
        model_name=None,
        **kwargs):
        self.vocab_size = len(reserved_tokens) if reserved_tokens else vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if attention_head_size is None:
            attention_head_size = hidden_size // num_attention_heads
        self.attention_head_size = attention_head_size
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size # FFN的隐层维度
        self.hidden_act = hidden_act # FFN隐层的激活函数
        self.dropout_rate = dropout_rate
        self.initializer_range = initializer_range
        self.embedding_size = embedding_size or hidden_size
        self.sequence_length = sequence_length
        self.reserved_tokens = reserved_tokens
        self.model_name = model_name
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.reset()

    def build(self, checkpoint_path=None):
        if self.builted:
            return self.model
        self.reset()

        self.build_model()
        self.builted = True

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        return self.model

    def reset(self):
        """重置一些参数"""
        self.attention_scores = None
        self.attention_mask = None
        self.position_bias = None
        self.builted = False
        self.layers = {}
        self.model = None
        # 全局初始化器，构建层时的参数初始化器
        self.initializer = initializers.TruncatedNormal(stddev=0.02)

    def load_checkpoint(self, checkpoint):
        """doc:
        https://tensorflow.google.cn/guide/checkpoint?hl=en
        """
        mapping = self.variables_mapping()
        mapping = {k: v for k, v in mapping.items() if k in self.layers}
        checkpoint = tf.train.load_checkpoint(checkpoint)
        for layer, variables in mapping.items():
            layer = self.layers[layer]
            values = [checkpoint.get_tensor(v) for v in variables]
            layer.set_weights(values)

    def build_layer(self, inputs=None, layer=None, callkwargs=None, **kwargs):
        """创建一个隐层并连接输入返回输出以及保存元数据,
        outputs=layer(**kwargs)(inputs, **callkwargs)"""
        if layer is Dropout and self.dropout_rate == 0.0:
            return inputs

        if callkwargs is None:
            callkwargs = {}

        name = kwargs.get("name")
        if name not in self.layers:
            layer = layer(**kwargs)
            name = layer.name
            self.layers[name] = layer

        if inputs is None:
            outputs = self.layers[name]
        else:
            outputs = self.layers[name](inputs, **callkwargs)
        return outputs

    def compute_attention_mask(self, inputs=None):
        """获取attention矩阵的mask，如lm mask、unilm mask，用于语言模型。
        如果是非语言模型，则该接口返回None。"""
        return self.attention_mask

    def compute_position_bias(self, inputs=None):
        """处理位置编码，不同Transformer模型有不同位置编码，如果位置编码是直接与
        TokenEmbedding相加，则该接口返回None。"""
        return self.position_bias

    def show_inputs_outputs(self):
        if self.builted:
            print(self.model.inputs)
            print(self.model.outputs)

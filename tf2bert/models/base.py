from tensorflow.keras.models import Model

class ModelBuilder:
    """创建模型的流程"""

    num_hidden_layers = 0

    def build_inputs(self):
        """返回list形式"""
        raise NotImplementedError

    def build_embeddings(self, inputs):
        raise NotImplementedError

    def build_layer(self, inputs=None, layer=None, callkwargs=None, **kwargs):
        """创建一个层并连接输入返回输出,
        outputs=layer(**kwargs)(inputs, **callkwargs)"""
        raise NotImplementedError

    def build_hidden_layer(self, inputs, index):
        raise NotImplementedError

    def build_all_hidden_layers(self, inputs, num_hidden_layers):
        x = inputs
        for index in range(num_hidden_layers):
            x = self.build_hidden_layer(x, index)
        outputs = x
        return outputs

    def build_outputs(self, inputs):
        """返回list形式"""
        raise NotImplementedError

    def build_model(self):
        inputs = self.build_inputs()
        x = self.build_embeddings(inputs)
        x = self.build_all_hidden_layers(x, self.num_hidden_layers)
        outputs = self.build_outputs(x)
        self.model = Model(inputs, outputs)
        return self.model

class CheckpointLoader:

    def variables_mapping(self):
        """如果需要权重映射，则在这里返回"""
        raise NotImplementedError

    def load_checkpoint(self, checkpoint):
        raise NotImplementedError

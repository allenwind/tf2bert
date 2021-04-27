from tensorflow.keras.models import Model

# 模型构建和权重加载的API定义

class ModelBuilder:
    """创建模型的流程，把常用组件的构建添加到该类上"""

    num_hidden_layers = 0
    model_name = ""

    def build_inputs(self):
        """返回list形式"""
        raise NotImplementedError

    def build_embeddings(self, inputs):
        """构建Embedding层"""
        raise NotImplementedError

    def build_layer(self, inputs=None, layer=None, callkwargs=None, **kwargs):
        """创建一个隐层并连接输入返回输出以及保存元数据,
        outputs=layer(**kwargs)(inputs, **callkwargs)"""
        raise NotImplementedError

    def build_hidden_layer(self, inputs, index):
        """构建隐层"""
        raise NotImplementedError

    def build_all_hidden_layers(self, inputs, num_hidden_layers):
        """构建所有隐层"""
        x = inputs
        for index in range(num_hidden_layers):
            x = self.build_hidden_layer(x, index)
        outputs = x
        return outputs

    def build_outputs(self, inputs):
        """返回list形式"""
        raise NotImplementedError

    def build_model(self):
        """构建完整的模型"""
        self.inputs = self.build_inputs()
        x = self.build_embeddings(self.inputs)
        x = self.build_all_hidden_layers(x, self.num_hidden_layers)
        self.outputs = self.build_outputs(x)
        self.model = Model(self.inputs, self.outputs, name=self.model_name)
        return self.model

class CheckpointLoader:
    """加载checkpoint的抽象"""

    def variables_mapping(self):
        """如果需要权重映射，则在这里返回，否则返回空字典"""
        raise NotImplementedError

    def load_checkpoint(self, checkpoint):
        """从checkpoint文件为layer加载weight"""
        raise NotImplementedError

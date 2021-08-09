import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tf2bert.layers import CRF, CRFModel
from tf2bert.text.tokenizers import Tokenizer
from tf2bert.text.labels import TaggingTokenizer
from tf2bert.models import build_transformer
import dataset

# BERT+CRF解决NER问题
# No matter how old, long live happy
# With friends like you, who needs luck

def batch_pad(X, maxlen=None, dtype="int32"):
    if maxlen is None:
        maxlen = max([len(i) for i in X])
    X = sequence.pad_sequences(
        X,
        maxlen=maxlen,
        dtype=dtype,
        padding="post",
        truncating="post",
        value=0
    )
    return X

class DataGenerator(tf.keras.utils.Sequence):
    """NER训练数据生成器"""

    def __init__(self, X, y, tokenizer, tagger, num_classes, batch_size, maxlen):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.tagger = tagger
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.maxlen = maxlen

    def __len__(self):
        return len(self.X) // self.batch_size

    def __getitem__(self, index):
        # 需要处理对齐的问题
        i = index * self.batch_size
        j = i + self.batch_size
        batch_token_ids, batch_segment_ids = self.tokenizer.batch_encode(self.X[i:j], maxlen=self.maxlen)
        batch_token_ids = batch_pad(batch_token_ids, self.maxlen)
        batch_segment_ids = batch_pad(batch_segment_ids, self.maxlen)
        batch_labels = batch_pad(self.tagger.batch_encode(self.y[i:j]), maxlen=self.maxlen)
        return (batch_token_ids, batch_segment_ids), batch_labels

    def on_epoch_end(self):
        np.random.RandomState(773).shuffle(self.X)
        np.random.RandomState(773).shuffle(self.y)

class Evaluator:
    pass

class NamedEntityFinder:
    pass

config_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_config.json"
token_dict_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/vocab.txt"
checkpoint_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"

load_dataset = dataset.load_msra
X_train, y_train, classes = load_dataset("train", with_labels=True)
num_classes = len(classes)
maxlen = 128 # 注意内存

# 类别映射
tagger = TaggingTokenizer()
tagger.fit(y_train)

# 加载Tokenizer
tokenizer = Tokenizer(token_dict_path, use_lower_case=True)
# 可以根据需要替换模型
bert = build_transformer(
    model="bert+encoder", 
    config_path=config_path, 
    checkpoint_path=checkpoint_path,
    verbose=False
)

inputs = bert.inputs
x = Dense(num_classes)(bert.output)
crf = CRF(
    lr_multiplier=1000,
    trans_initializer="glorot_normal",
    trainable=True
)
outputs = crf(x)
base = Model(inputs=inputs, outputs=outputs)
model = CRFModel(base)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5))

if __name__ == "__main__":
    print(__file__)
    batch_size = 32
    epochs = 10
    X_val, y_val = load_dataset("dev", with_labels=False)
    X_test, y_test = load_dataset("test", with_labels=False)
    dataset_train = DataGenerator(X_train, y_train, tokenizer, tagger, num_classes, batch_size, maxlen)
    dataset_val = DataGenerator(X_val, y_val, tokenizer, tagger, num_classes, batch_size, maxlen)
    dataset_test = DataGenerator(X_test, y_test, tokenizer, tagger, num_classes, batch_size, maxlen)
    
    # for (a, b), c in dataset_train:
    #     print(a.shape)
    #     print(b.shape)
    #     print(c.shape)
    # raise
    model.fit(
        dataset_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=dataset_val,
        validation_batch_size=batch_size
    )

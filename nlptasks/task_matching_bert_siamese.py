import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tf2bert.layers import MaskedGlobalMaxPooling1D
from tf2bert.text.tokenizers import Tokenizer
from tf2bert.models import build_transformer
import dataset

# BERT在文本匹配问题中的应用，siamese架构
# https://arxiv.org/pdf/1908.10084.pdf

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

    def __init__(self, X1, X2, y, tokenizer, num_classes, batch_size, maxlen):
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.tokenizer = tokenizer
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.maxlen = maxlen

    def __len__(self):
        return len(self.y) // self.batch_size

    def __getitem__(self, index):
        i = index * self.batch_size
        j = i + self.batch_size
        # X1
        batch_token_ids1, batch_segment_ids1 = self.tokenizer.batch_encode(self.X1[i:j], maxlen=self.maxlen)
        batch_token_ids1 = batch_pad(batch_token_ids1, self.maxlen)
        batch_segment_ids1 = batch_pad(batch_segment_ids1, self.maxlen)
        # X2
        batch_token_ids2, batch_segment_ids2 = self.tokenizer.batch_encode(self.X2[i:j], maxlen=self.maxlen)
        batch_token_ids2 = batch_pad(batch_token_ids2, self.maxlen)
        batch_segment_ids2 = batch_pad(batch_segment_ids2, self.maxlen)
        # y
        batch_labels = tf.keras.utils.to_categorical(self.y[i:j], num_classes)
        return [(batch_token_ids1, batch_segment_ids1), 
                (batch_token_ids2, batch_segment_ids2)], batch_labels

    def on_epoch_end(self):
        np.random.RandomState(773).shuffle(self.X1)
        np.random.RandomState(773).shuffle(self.X2)
        np.random.RandomState(773).shuffle(self.y)

def split_kfolds(X1, X2, y, n_splits=8):
    X1_train = [j for i, j in enumerate(X1) if i % n_splits != 1]
    X2_train = [j for i, j in enumerate(X2) if i % n_splits != 1]
    y_train = [j for i, j in enumerate(y) if i % n_splits != 1]
    X1_test = [j for i, j in enumerate(X1) if i % n_splits == 1]
    X2_test = [j for i, j in enumerate(X2) if i % n_splits == 1]
    y_test = [j for i, j in enumerate(y) if i % n_splits == 1]
    return (X1_train, X2_train, y_train), (X1_test, X2_test, y_test)

config_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_config.json"
token_dict_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/vocab.txt"
checkpoint_path = "/home/zhiwen/workspace/dataset/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"

X1, X2, y, classes = dataset.load_lcqmc()
num_classes = len(classes)
maxlen = 32 # 注意内存

# 加载Tokenizer
tokenizer = Tokenizer(token_dict_path, use_lower_case=True)
# 可以根据需要替换模型
bert = build_transformer(
    model="bert", 
    config_path=config_path, 
    checkpoint_path=checkpoint_path,
    verbose=False
)

pool = MaskedGlobalMaxPooling1D(return_scores=False)
dropout = Dropout(rate=0.2)
layernorm = LayerNormalization()

def bert_encode(x):
    x = bert(x)
    x = pool(x)
    # x = dropout(x)
    return x

def matching(x1, x2):
    # x*y
    x3 = Multiply()([x1, x2])
    # |x-y|
    x4 = Lambda(lambda x: tf.abs(x[0] - x[1]))([x1, x2])
    x = Concatenate()([x1, x2, x3, x4])
    x = layernorm(x)
    return x

x1_input = Input(shape=(maxlen,), dtype=tf.int32)
s1_input = Input(shape=(maxlen,), dtype=tf.int32)
x2_input = Input(shape=(maxlen,), dtype=tf.int32)
s2_input = Input(shape=(maxlen,), dtype=tf.int32)

input1 = [x1_input, s1_input]
input2 = [x2_input, s2_input]
x1 = bert_encode(input1)
x2 = bert_encode(input2)
x = matching(x1, x2)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=[input1, input2], outputs=outputs)
model.summary()
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(1e-5),
    metrics=["accuracy"]
)

if __name__ == "__main__":
    print(__file__)
    batch_size = 32
    epochs = 10
    (X1_train, X2_train, y_train), \
    (X1_test, X2_test, y_test) = split_kfolds(X1, X2, y, 5)
    dataset_train = DataGenerator(X1_train, X2_train, y_train, tokenizer, num_classes, batch_size, maxlen)
    dataset_val = DataGenerator(X2_test, X2_test, y_test, tokenizer, num_classes, batch_size, maxlen)
    model.fit(
        dataset_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=dataset_val,
        validation_batch_size=batch_size
    )

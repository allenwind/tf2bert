import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tf2bert.layers import MaskedGlobalMaxPooling1D
from tf2bert.text.tokenizers import Tokenizer
from tf2bert.models import build_transformer
import dataset

# BERT在文本匹配问题中的应用，使用[CLS]的输出，S1与S2之间使用[SEP]隔开

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
        self.maxlen = maxlen * 2

    def __len__(self):
        return len(self.y) // self.batch_size

    def __getitem__(self, index):
        i = index * self.batch_size
        j = i + self.batch_size
        # [CLS]X1[SEP]X2[SEP]
        batch_token_ids, batch_segment_ids = self.tokenizer.batch_encode(
            self.X1[i:j],
            self.X2[i:j],
            self.maxlen,
            mode="SEE"
        )
        batch_token_ids = batch_pad(batch_token_ids, self.maxlen)
        batch_segment_ids = batch_pad(batch_segment_ids, self.maxlen)
        # y
        batch_labels = tf.keras.utils.to_categorical(self.y[i:j], num_classes)
        return (batch_token_ids, batch_segment_ids), batch_labels

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
    with_pool=True,
    pool_activation="tanh", # or None
    verbose=False
)

inputs = bert.inputs
x = Dropout(rate=0.2)(bert.output)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)
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

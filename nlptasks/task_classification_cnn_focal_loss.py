import collections
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tf2bert.layers import MaskedGlobalMaxPooling1D
from tf2bert.text.tokenizers import CharTokenizer
import dataset

# 正负样本不平衡的处理方法：
# focal loss

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

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

class FocalLoss(tf.keras.losses.Loss):

    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super(FocalLoss, self).__init__(**kwargs)
        self.alpha = alpha # 权重调节因子
        self.gamma = gamma # 再平衡因子

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0-1e-7)
        loss = - self.alpha * y_true * tf.math.log(y_pred) * tf.pow((1 - y_pred), self.gamma) \
               - (1 - self.alpha) * (1 - y_true) * tf.math.log(1 - y_pred) * tf.pow(y_pred, self.gamma)
        return loss

    # def call(self, y_true, y_pred):
    #     return - y_true * (1 - y_pred) ** self.gamma * tf.math.log(y_pred)

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, X, y, num_classes, batch_size):
        self.X = X
        self.y = y
        self.num_classes = num_classes
        self.batch_size = batch_size

    def __len__(self):
        return len(self.X) // self.batch_size

    def __getitem__(self, index):
        i = index * self.batch_size
        j = i + self.batch_size
        X = self.X[i:j]
        y = self.y[i:j]
        y = tf.keras.utils.to_categorical(y, num_classes)
        return batch_pad(X, maxlen=None), np.array(y)

    def on_epoch_end(self):
        np.random.RandomState(773).shuffle(self.X)
        np.random.RandomState(773).shuffle(self.y)

def split_kfolds(X, y, n_splits=8):
    X_train = [j for i, j in enumerate(X) if i % n_splits != 1]
    y_train = [j for i, j in enumerate(y) if i % n_splits != 1]
    X_test = [j for i, j in enumerate(X) if i % n_splits == 1]
    y_test = [j for i, j in enumerate(y) if i % n_splits == 1]
    return (X_train, y_train), (X_test, y_test)

X, y, classes = dataset.load_hotel_comment()
num_classes = len(classes)

tokenizer = CharTokenizer(mintf=10)
tokenizer.fit(X)
num_words = len(tokenizer)
maxlen = None
embedding_dim = 128
hdim = 128

embedding = Embedding(
    input_dim=num_words,
    output_dim=embedding_dim,
    mask_zero=False,
    name="embedding"
)

conv1 = Conv1D(filters=hdim, kernel_size=2, padding="same", activation=gelu)
conv2 = Conv1D(filters=hdim, kernel_size=2, padding="same", activation=gelu)
conv3 = Conv1D(filters=hdim, kernel_size=3, padding="same", activation=gelu)
pool = MaskedGlobalMaxPooling1D(return_scores=False)

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
x = embedding(inputs)
x = LayerNormalization()(x)
x = conv1(x)
x = conv2(x)
x = conv3(x)
x = pool(x, mask=mask)
x = Dense(128)(x)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
loss = FocalLoss(alpha=0.25, gamma=2.0)
model.compile(
    loss=loss,
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

if __name__ == "__main__":
    print(__file__)
    batch_size = 64
    epochs = 20
    X = tokenizer.transform(X)
    (X_train, y_train), (X_test, y_test) = split_kfolds(X, y, 5)
    dataset_train = DataGenerator(X_train, y_train, num_classes, batch_size)
    dataset_val = DataGenerator(X_test, y_test, num_classes, batch_size)
    model.fit(
        dataset_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=dataset_val,
        validation_batch_size=batch_size,
    )

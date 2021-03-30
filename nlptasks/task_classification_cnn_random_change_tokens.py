import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tf2bert.layers import MaskedGlobalMaxPooling1D
from tf2bert.layers import LayerNormalization
from tf2bert.text.tokenizers import CharTokenizer
import dataset

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# acc: 90.5%+

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

class RandomChange(tf.keras.layers.Layer):
    """随机替换tokens，可以优化的地方：指定tokens集"""
    
    def __init__(self, num_words, rate=0.3, **kwargs):
        super(RandomChange, self).__init__(**kwargs)
        self.num_words = num_words
        self.rate = rate

    def call(self, inputs, training=None):
        # use tf.float32?
        if training:
            batchs = tf.shape(inputs)[0]
            maxlen = tf.shape(inputs)[-1]
            mask = tf.random.uniform((batchs, maxlen), minval=0, maxval=1)
            mask = tf.cast(mask < self.rate, tf.int32)
            tokens = tf.random.uniform((batchs, maxlen), minval=2, maxval=self.num_words, dtype=tf.int32)
            return inputs * (1 - mask) + tokens * mask
        return inputs

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

def compute_class_weight(y):
    """类别权重方法"""
    total = len(y)
    class_weight = {}
    counter = collections.Counter(y)
    print(counter)
    n = len(counter)
    for c, v in counter.items():
        # w = total / (n * v)
        class_weight[c] = total / (n * v)
    return class_weight

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, X, y, num_classes, batch_size):
        self.X = X
        self.y = y
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.on_epoch_end()

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
        np.random.RandomState(87539845).shuffle(self.X)
        np.random.RandomState(87539845).shuffle(self.y)

def split_kfolds(X, y, n_splits=8, shuffle=True):
    if shuffle:
        np.random.RandomState(7832478).shuffle(X)
        np.random.RandomState(7832478).shuffle(y)
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

change = RandomChange(num_words, rate=0.3)
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

inputs = Input(shape=(maxlen,), dtype=tf.int32)
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
x = change(inputs)
x = embedding(x)
x = LayerNormalization()(x)
x = conv1(x)
x = conv2(x)
x = conv3(x)
x = pool(x, mask=mask)
x = Dropout(0.2)(x)
x = Dense(128)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

if __name__ == "__main__":
    print(__file__)
    batch_size = 32
    epochs = 20
    X = tokenizer.transform(X)
    (X_train, y_train), (X_test, y_test) = split_kfolds(X, y, 5)
    class_weight = compute_class_weight(y_train)
    dataset_train = DataGenerator(X_train, y_train, num_classes, batch_size)
    dataset_val = DataGenerator(X_test, y_test, num_classes, batch_size)
    model.fit(
        dataset_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=dataset_val,
        validation_batch_size=batch_size,
        class_weight=class_weight
    )

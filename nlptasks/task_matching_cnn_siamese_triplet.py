import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tf2bert.layers import MaskedGlobalMaxPooling1D
from tf2bert.layers import MaskedGlobalAveragePooling1D
from tf2bert.layers import AttentionPooling1D
from tf2bert.text.tokenizers import CharTokenizer
import dataset

# TODO
# 经典的Siamese架构，使用triplet loss

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

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, X1, X2, y, num_classes, batch_size):
        self.X1 = X1
        self.X2 = X2
        self.y = y
        self.size = len(X1)
        self.num_classes = num_classes
        self.batch_size = batch_size

    def __len__(self):
        return self.size // self.batch_size

    def __getitem__(self, index):
        i = index * self.batch_size
        j = i + self.batch_size
        X1 = self.X1[i:j]
        X2 = self.X2[i:j]
        y = self.y[i:j]
        X1 = batch_pad(X1, maxlen=None)
        X2 = batch_pad(X2, maxlen=None)
        y = tf.keras.utils.to_categorical(y, num_classes)
        y = np.array(y)
        return (X1, X2), y

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

X1, X2, y, classes = dataset.load_lcqmc()
num_classes = len(classes)

tokenizer = CharTokenizer(mintf=10)
tokenizer.fit(X1 + X2)
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
layernom = LayerNormalization()
conv1 = Conv1D(filters=hdim, kernel_size=2, padding="same", activation=gelu)
conv2 = Conv1D(filters=hdim, kernel_size=3, padding="same", activation=gelu)
conv3 = Conv1D(filters=hdim, kernel_size=3, padding="same", activation=gelu)
pool1 = MaskedGlobalMaxPooling1D(return_scores=False)
pool2 = MaskedGlobalAveragePooling1D(return_scores=False)
pool3 = AttentionPooling1D(hdim, return_scores=False)

def encode(x, mask):
    x = embedding(x)
    x = layernom(x)
    x = conv1(x)
    x = conv2(x)
    x = conv3(x)
    p1 = pool1(x, mask=mask)
    p2 = pool2(x, mask=mask)
    p3 = pool3(x, mask=mask)
    x = Concatenate()([p1, p2, p3])
    return x

# build model
input1 = Input(shape=(maxlen,), dtype=tf.int32)
input2 = Input(shape=(maxlen,), dtype=tf.int32)

mask1 = Lambda(lambda x: tf.not_equal(x, 0))(input1)
mask2 = Lambda(lambda x: tf.not_equal(x, 0))(input2)

x1 = encode(input1, mask1)
x2 = encode(input2, mask2)

# x*y
x3 = Multiply()([x1, x2])
# |x-y|
x4 = Lambda(lambda x: tf.abs(x[0] - x[1]))([x1, x2])

x = Concatenate()([x1, x2, x3, x4])
x = Dense(4 * hdim)(x)
outputs = Dense(num_classes, activation="softmax")(x)

inputs = [input1, input2]
model = Model(inputs, outputs)

lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[0, 2, 6],
    values=[2*1e-3, 0.8*1e-3, 0.5*1e-3, 1e-4]
)
adam = tf.keras.optimizers.Adam(lr)
model.compile(
    loss="categorical_crossentropy",
    optimizer=adam,
    metrics=["accuracy"]
)
model.summary()

if __name__ == "__main__":
    print(__file__)
    batch_size = 32
    epochs = 20
    X1 = tokenizer.transform(X1)
    X2 = tokenizer.transform(X2)
    (X1_train, X2_train, y_train), \
    (X1_test, X2_test, y_test) = split_kfolds(X1, X2, y, 5)
    dataset_train = DataGenerator(X1_train, X2_train, y_train, num_classes, batch_size)
    dataset_val = DataGenerator(X1_test, X2_test, y_test, num_classes, batch_size)
    model.fit(
        dataset_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=dataset_val,
        validation_batch_size=batch_size
    )

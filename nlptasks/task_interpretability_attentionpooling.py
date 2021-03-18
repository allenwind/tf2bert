import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tf2bert.layers import AttentionPooling1D
from tf2bert.text.tokenizers import CharTokenizer
from tf2bert.text.rendering import print_color_text
import dataset

# AttentionPooling1D的可视化解释

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

def show_visualization(model, pool_model, text, labelid, id2label, tokenizer):
    X = [text]
    X = tokenizer.transform(X)
    X = batch_pad(X)
    labelid_pred = np.argmax(model.predict(X)[0])
    weights = pool_model.predict(X)[0].flatten()
    print_color_text(text, weights)
    print("=>y_true:", id2label[labelid])
    print("=>y_pred:", id2label[labelid_pred])

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
pool = AttentionPooling1D(hdim, return_scores=True)

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
x = embedding(inputs)
x = LayerNormalization()(x)
x = Dropout(0.2)(x)
x = conv1(x)
x = conv2(x)
x = conv3(x)
x, w = pool(x, mask=mask)
x = Dense(128)(x)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

pool_model = Model(inputs, w)

if __name__ == "__main__":
    print(__file__)
    batch_size = 64
    epochs = 10
    Xv = X[:100].copy()
    yv = y[:100].copy()
    X = tokenizer.transform(X)
    (X_train, y_train), (X_test, y_test) = split_kfolds(X, y, 5)
    dataset_train = DataGenerator(X_train, y_train, num_classes, batch_size)
    dataset_val = DataGenerator(X_test, y_test, num_classes, batch_size)
    model.fit(
        dataset_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=dataset_val,
        validation_batch_size=batch_size
    )

    id2label = {j:i for i,j in classes.items()}
    for text, labelid in zip(Xv, yv):
        show_visualization(model, pool_model, text, labelid, id2label, tokenizer)
        input()

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

from tf2bert.layers import AttentionPooling1D
from tf2bert.text import CharTokenizer


maxlen = 128


inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
x = Embedding(num_words, embedding_dims)(inputs)
x = Dropout(0.2)(x)
x = Conv1D(
    filters=128,
    kernel_size=3,
    padding="same",
    activation="relu",
    strides=1)(x)
x, w = AttentionPooling1D(hdims=128, return_scores=True)(x, mask=mask)
x = Dropout(0.2)(x)
x = Dense(128)(x)
x = gelu(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
model.summary()

model.evaluate()



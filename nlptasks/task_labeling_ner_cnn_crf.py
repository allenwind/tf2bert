import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tf2bert.text.tokenizers import CharTokenizer
from tf2bert.layers import CRF, CRFModel


# CNN+CRF的NER示例

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
x = Dropout(0.2)(x)
x = conv1(x)
x = conv2(x)
x = conv3(x)
x = pool(x, mask=mask)
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

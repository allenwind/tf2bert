import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tf2bert.layers import CRF, CRFModel

vocab_size = 5000
hdims = 128
inputs = Input(shape=(None,), dtype=tf.int32)
x = Embedding(vocab_size, hdims, mask_zero=True)(inputs)
x = Bidirectional(LSTM(hdims, return_sequences=True))(x)
x = Dense(4)(x)
crf = CRF(trans_initializer="orthogonal")
outputs = crf(x)
base = Model(inputs, outputs)
model = CRFModel(base)
model.summary()
model.compile(optimizer="adam")
X = tf.random.uniform((32*100, 64), minval=0, maxval=vocab_size, dtype=tf.int32)
y = tf.random.uniform((32*100, 64), minval=0, maxval=4, dtype=tf.int32)
model.fit(X, y)
y_pred = model.predict(X)
print(y_pred.shape)

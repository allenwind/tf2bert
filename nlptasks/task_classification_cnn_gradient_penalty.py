import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn import metrics

from dataset import load_THUCNews_title_label
from dataset import load_hotel_comment
from dataset import load_weibo_senti_100k
from dataset import load_simplifyweibo_4_moods
from dataset import SimpleTokenizer, find_best_maxlen
from pooling import MaskGlobalMaxPooling1D

# 添加梯度惩罚的loss

class GradientPenalty(tf.keras.Model):
    """在train_step中实现梯度惩罚的逻辑"""

    def compile(self, epsilon, embedding_name, **kwargs):
        super(GradientPenalty, self).compile(**kwargs)
        self.epsilon = epsilon
        self.embedding_name = embedding_name

    def gradient_penalty(self, x, y):
        # 计算gradient penalty
        embedding_layer = self.get_layer(self.embedding_name)
        embeddings = embedding_layer.embeddings
        with tf.GradientTape() as tape:
            tape.watch(embeddings)
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        grads = tape.gradient(loss, embeddings)
        gp = tf.reduce_sum(tf.square(grads))
        return gp

    def train_step(self, data):
        # 计算二阶梯度
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
            gp = self.gradient_penalty(x, y)
            total_loss = loss + 0.5 * self.epsilon * gp

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update(gp=gp)
        return results

# 处理数据
X, y, classes = load_THUCNews_title_label()
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.8,
    random_state=7432
)

num_classes = len(classes)
# 转化成字id
tokenizer = SimpleTokenizer()
tokenizer.fit(X_train)
X_train = tokenizer.transform(X_train)
X_test = tokenizer.transform(X_test)

maxlen = find_best_maxlen(X_train, mode="max")
maxlen = 48

X_train = sequence.pad_sequences(
    X_train,
    maxlen=maxlen,
    dtype="int32",
    padding="post",
    truncating="post",
    value=0.0
)

X_test = sequence.pad_sequences(
    X_test,
    maxlen=maxlen,
    dtype="int32",
    padding="post",
    truncating="post",
    value=0.0
)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# 模型
num_words = len(tokenizer)
embedding_dims = 128

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
embedding = Embedding(
    num_words,
    embedding_dims,
    embeddings_initializer="glorot_normal",
    name="embedding",
    input_length=maxlen)
x = embedding((inputs))
x = Dropout(0.1)(x)
x = Conv1D(filters=128,
           kernel_size=3,
           padding="same",
           activation="relu",
           strides=1)(x)
x, _ = MaskGlobalMaxPooling1D()(x, mask=mask)
x = Dense(128)(x)
x = Dropout(0.1)(x)
x = Activation("relu")(x)
outputs = Dense(num_classes, activation="softmax")(x)

with_gradient_penalty = True
if with_gradient_penalty:
    model = GradientPenalty(inputs=inputs, outputs=outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
        embedding_name="embedding",
        epsilon=1.0
    )
else:
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

model.summary()

# 训练
batch_size = 32
epochs = 10
callbacks = []
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2
)

# 评估
model.evaluate(X_test, y_test)

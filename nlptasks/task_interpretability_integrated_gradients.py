import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tf2bert.layers import MaskedGlobalMaxPooling1D
from tf2bert.text.tokenizers import CharTokenizer
from tf2bert.text.rendering import print_color_text
import dataset

# 积分梯度在文本可视化理解中的应用，并对比纯梯度和积分梯度的效果

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

class GradientLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(GradientLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        output, embedding, label = inputs
        output = tf.gather(output, label, batch_dims=1)
        grads = tf.gradients(output, [embedding])[0] * embedding
        return grads    

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

def show_visualization(model, gradient_model, text, labelid, id2label, tokenizer):
    embeddings = model.layers[1].embeddings # embedding矩阵
    values = tf.Variable(embeddings) # 保存embedding矩阵以便恢复
    X = [text]
    X = tokenizer.transform(X)
    X = batch_pad(X)
    labelid_pred = np.argmax(model.predict(X)[0])
    labelid_pred_in = np.array([labelid_pred])

    # 纯梯度
    based_grads = gradient_model.predict([X, labelid_pred_in])[0]
    based_grads = np.sqrt(np.square(based_grads).sum(axis=1))
    based_grads = (based_grads - based_grads.min()) / (based_grads.max() - based_grads.min())

    grads = []
    n = 30
    for alpha in np.linspace(0, 1, n):
        # 让embedding矩阵渐变
        embeddings.assign(alpha * values)
        pred_grads = gradient_model.predict([X, labelid_pred_in])[0]
        grads.append(pred_grads)

    # 还原embedding矩阵值以便做其他样本的预测
    embeddings.assign(values)

    # 不同embedding值下梯度的均值
    grads = np.mean(grads, axis=0)
    weights = np.sqrt(np.square(grads).sum(axis=1))
    weights = (weights - weights.min()) / (weights.max() - weights.min())

    print("gradients:")
    print_color_text(text, based_grads)
    print("integrated gradients:")
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

# 计算梯度的模型
label_input = Input(shape=(1,), dtype=tf.int32)
inputs = model.inputs + [label_input]

embedding_input = model.layers[1].output
grads = GradientLayer()([model.output, embedding_input, label_input])

gradient_model = Model(inputs, grads)
gradient_model.summary()

if __name__ == "__main__":
    print(__file__)
    batch_size = 64
    epochs = 1
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
        show_visualization(model, gradient_model, text, labelid, id2label, tokenizer)
        input()

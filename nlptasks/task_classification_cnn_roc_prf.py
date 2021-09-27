import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding, BatchNormalization
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn import metrics

import dataset
import evaluation
from dataset import Tokenizer

from tfutils import SaveBestModelOnMemory
# from tfx.layers.embeddings import WordEmbeddingInitializer

# classification 中 multi labels 文件
# 多分类绘制ROC、PRF等曲线的例子

# 用sigmoid进行多标签分类
# [0, 1, 1, 0, 1]

# 处理数据
X, y, categoricals = dataset.load_THUCNews_title_label()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=732)

num_classes = len(categoricals)
# 转化成字id
ctokenizer = Tokenizer()
# 严格的交叉验证，只在训练集上构建全局词表
ctokenizer.fit(X_train)
X_train = ctokenizer.transform(X_train)
X_test = ctokenizer.transform(X_test)

# maxlen = tokenizer.find_best_maxlen(X_train, mode="mean")
maxlen = 48
print("max length is", maxlen)
X_train = sequence.pad_sequences(
    X_train,
    maxlen=maxlen,
    dtype="int32",
    padding="post",
    truncating="post",
    value=0)

X_test = sequence.pad_sequences(
    X_test,
    maxlen=maxlen,
    dtype="int32",
    padding="post",
    truncating="post",
    value=0)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# 模型
input_dim = ctokenizer.vocab_size
# output_dim = tokenizer.find_embedding_dims(input_dim)
output_dim = 128

# wi = WordEmbeddingInitializer(wm.vocab, path="/home/zhiwen/workspace/dataset/word2vec_baike/word2vec_baike")
# input_dim, output_dim = wi.shape

inputs = Input(shape=(maxlen,))  # (batch_size, maxlen)
x = Embedding(input_dim, output_dim,
              embeddings_initializer="glorot_normal",
              input_length=maxlen,
              trainable=True,
              mask_zero=True)(inputs)  # (batch_size, maxlen, output_dim)

x = Dropout(0.2)(x)
x = Conv1D(filters=200,
           kernel_size=2,
           padding="same",
           activation="relu",
           strides=1)(x)

x = Conv1D(filters=200,
           kernel_size=3,
           padding="same",
           activation="relu",
           strides=1)(x)

x = GlobalMaxPooling1D()(x)
x = Dense(100)(x)
x = Dropout(0.2)(x)
x = Activation("relu")(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# 训练
batch_size = 32
epochs = 8

callbacks = [SaveBestModelOnMemory()]
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=callbacks,
          validation_split=0.1)
model.summary()

y_pred = model.predict(X_test)


fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

lw = 1
colors = itertools.cycle(
    ['aqua', 'darkorange', 'cornflowerblue', 'blue', 'red'])
linestyles = itertools.cycle([''])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC to multi-class')
plt.legend(loc="lower right")
plt.show()

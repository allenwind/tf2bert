import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tf2bert.text.tokenizers import Tokenizer
from tf2bert.text.labels import TaggingTokenizer
from tf2bert.text.labels import find_entities_chunking
import dataset

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

def convert(X, y):
    """转换为这种形式[text, (start, end, label), (start, end, label), ...]，
    其中text[start:end]是实体且类型为label。
    """
    data = []
    for text, tags in zip(X, y):
        sample = []
        sample.append(text)
        for label, start, end in find_entities_chunking(tags):
            sample.append((start, end, label))
        data.append(sample)
    return data

def load_data(file="train"):
    X, y = dataset.load_china_people_daily(file)
    return convert(X, y)

train_data = load_data("train")
valid_data = load_data("dev")
test_data = load_data("test")

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, data, batch_size):
        pass

class NamedEntityRecognizer:

    def __init__(self, tagger, batch_size=32):
        self.tagger = tagger
        self.batch_size = batch_size # 批量大小
    
    def predict(self, texts):
        """如果输入大于一个样本，则做batch预测"""
        if isinstance(texts, list):
            return self._predict_batch(texts)
        return self._predict_one(texts)

    def preprocess(self, text):
        tokens = tokenizer.tokenize(text, maxlen=512)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        return mapping, token_ids, segment_ids

    def convert(self, text):
        pass

    def decode_tags(self, mapping):
        pass

    def _predict_one(self, text):
        mapping, token_ids, segment_ids = self.preprocess(text)
        length = len(token_ids)
        token_ids = batch_pad(token_ids)
        segment_ids = batch_pad(segment_ids)
        label_ids = model.predict([token_ids, segment_ids])[0]
        labels = self.tagger.decode(label_ids)
        entities = []
        for label, start, end in find_entities_chunking(labels):
            entities.append((start, end, label))

        # TODO mapping

    def _predict_batch(self, texts):
        pass

class Evaluator(tf.keras.callbacks.Callback):

    def __init__(self, ner, valid_data=None, test_data=None):
        self.ner = ner # 实体识别器
        self.valid_data = valid_data
        self.test_data = test_data
        self.best_valid_f1 = 0.0
        self.best_test_f1 = 0.0

    def evaluate(self, data):
        texts = [sample[0] for sample in data]
        y_true = [set([tuple(i) for i in sample[1:]]) for sample in data]
        y_pred = [set(i) for i in self.ner.predict(texts)]
        X = Y = Z = 1e-10
        for R, T in zip(y_pred, y_true):
            X += len(R & T)
            Y += len(R)
            Z += len(T)
        precision = X / Y
        recall = X / Z
        f1 = 2 * X / (Y + Z)
        return precision, recall, f1

    def on_epoch_end(self, epoch, logs=None):
        template = "precision:{:.5f}, recall:{:.5f}, f1:{:.5f}, best f1:{:.5f}"
        if self.valid_data is not None:
            precision, recall, f1 = self.evaluate(self.valid_data)
            if f1 >= self.best_valid_f1:
                self.best_valid_f1 = f1
                self.model.save_weights("best_model.weights")
            print("valid:", template.format(precision, recall, f1, self.best_valid_f1))
        if self.test_data is not None:
            precision, recall, f1 = self.evaluate(self.test_data)
            if f1 >= self.best_test_f1:
                self.best_test_f1 = f1
            print("test:", template.format(precision, recall, f1, self.best_test_f1))


maxlen = 128
vocab_size = 0
hdims = 256

inputs = Input(shape=(maxlen,))
x = Embedding(input_dim=vocab_size, output_dim=hdims, mask_zero=True)(inputs)
x = Dropout(0.1)(x)
x = LayerNormalization()(x)
x = Bidirectional(LSTM(hdims, return_sequences=True), merge_mode="concat")(x)
x = Dense(num_classes)(x)
crf = CRF(
    lr_multiplier=1,
    trans_initializer="glorot_normal",
    trainable=True
)
outputs = crf(x)

base = Model(inputs=inputs, outputs=outputs)
model = CRFModel(base)
model.summary()
model.compile(optimizer="adam")

if __name__ == "__main__":
    X, y = dataset.load_china_people_daily("train")
    data = convert(X, y)
    for i in data:
        print(i)
        input()


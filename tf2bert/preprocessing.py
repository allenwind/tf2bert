import itertools
import numpy as np
from tensorflow.keras.preprocessing import sequence

def pad(x, maxlen=None, dtype="int32"):
    if maxlen is None:
        maxlen = len(x)
    x = sequence.pad_sequences(
        [x],
        maxlen=maxlen,
        dtype=dtype,
        padding="post",
        truncating="post",
        value=0
    )
    return x[0]

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

def batch_shuffle(inputs, seed=27382):
    return [np.random.RandomState(773).shuffle(x) \
            for x in inputs]

def batch_to_array(*inputs):
    xs = [np.array(x) for x in inputs if x]
    if len(inputs) == 1:
        return xs[0]
    return xs

def batch_paded_generator(X, y, label2id, tokenizer, batch_size, epochs):
    X = tokenizer.transform(X)
    y = batch_tags2ids(y, label2id)
    iterations = (len(X) // batch_size + 1) * epochs * batch_size
    X = itertools.cycle(X)
    y = itertools.cycle(y)
    gen = zip(X, y)
    batch_X = []
    batch_y = []
    for _ in range(iterations):
        sample_x, sample_y = next(gen)
        batch_X.append(sample_x)
        batch_y.append(sample_y)
        if len(batch_X) == batch_size:
            yield batch_pad(batch_X), batch_pad(batch_y)
            batch_X = []
            batch_y = []

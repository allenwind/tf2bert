import itertools
import numpy as np
from tensorflow.keras.preprocessing import sequence

def batch_padded(inputs):
    pass

def batch_shuffle():
    pass

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

def batch_to_array():
    pass

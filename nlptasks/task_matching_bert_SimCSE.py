import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import sequence
from tf2bert.layers import MaskedGlobalMaxPooling1D
from tf2bert.text.tokenizers import Tokenizer
from tf2bert.models import build_transformer
import dataset

# TODO

# paper:
# https://arxiv.org/pdf/2104.08821.pdf

def simcse_loss(y_true, y_pred):
    pass

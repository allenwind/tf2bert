import tensorflow as tf
from .embeddings import SimplePositionEmbedding
from .embeddings import SinusoidalPositionEmbedding
from .embeddings import PositionEmbedding
from .embeddings import Embedding
from .embeddings import CharAlignHybridEmbedding
from .embeddings import RelativePositionEmbedding
from .embeddings import EmbeddingProjector
from .pooling import MaskedGlobalMaxPooling1D
from .pooling import MaskedGlobalAveragePooling1D
from .pooling import AttentionPooling1D
from .pooling import MaskedMinVariancePooling
from .pooling import MultiHeadAttentionPooling1D
from .normalization import LayerNormalization
from .normalization import BatchNormalization
from .attention import MultiHeadAttention, Attention
from .crf import CRF, CRFModel, CRFWrapper
from .merge import ReversedConcatenate1D
from .merge import LayersConcatenate
from .merge import MaskedConcatenate1D
from .merge import MaskedFlatten
from .dense import FeedForward
from .dense import NoisyDense
from .dense import BiasAdd
from .dense import DenseEmbedding
from .cnn import MaskedConv1D
from .cnn import ResidualGatedConv1D
from .regularization import RandomChange

tf.keras.utils.get_custom_objects().update({
    "Embedding": Embedding,
    "PositionEmbedding": PositionEmbedding,
    "SinusoidalPositionEmbedding": SinusoidalPositionEmbedding,
    "LayerNormalization": LayerNormalization
})

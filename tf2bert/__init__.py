__version__ = "0.0.3"

__all__ = ["layers", "match", "math", "models", "text", 
           "activations", "callbacks", "initializers", 
           "losses", "metrics", "optimizers", "utils",
           "preprocessing"]

from . import layers
from . import math
from . import models
from . import text
from . import activations
from . import callbacks
from . import initializers
from . import losses
from . import metrics
from . import optimizers
from . import utils
from . import preprocessing

print("""
  _    __ ___  _
 | |  / _|__ \| |             | |
 | |_| |_   ) | |__   ___ _ __| |
 | __|  _| / /| '_ \ / _ \ '__| __|
 | |_| |  / /_| |_) |  __/ |  | |
  \__|_| |____|_.__/ \___|_|   \__|
""")

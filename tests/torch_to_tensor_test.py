import tensorflow as tf
from tf2bert.utils import torch_to_checkpoint
from tf2bert.utils import list_variables

torch_file = "/home/zhiwen/workspace/dataset/bert/ernie_base/pytorch_model.bin"
tf_file = "../temp/erine_base_model.ckpt"

ck = torch_to_checkpoint(torch_file, tf_file)

rck = tf.train.load_checkpoint(tf_file)

print(list_variables(tf_file, string=True))

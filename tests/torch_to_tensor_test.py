import numpy as np
import tensorflow as tf
from tf2bert.utils import torch_to_checkpoint
from tf2bert.utils import list_variables
from tf2bert.utils import pattern_replace
from tf2bert.utils import transpose
import torch

torch_file = "/home/zhiwen/workspace/dataset/bert/ernie_base/pytorch_model.bin"
tf_file = "../temp/erine_base_model.ckpt"

ck = torch_to_checkpoint(torch_file, tf_file)

rck = tf.train.load_checkpoint(tf_file)

print(list_variables(tf_file, string=True))

torch_checkpoint = torch.load(torch_file, map_location="cpu")
tf_checkpoint = tf.train.load_checkpoint(tf_file)

# 检测是否一致
for name, weight1 in torch_checkpoint.items():
    weight1 = weight1.numpy()
    if any(x in name for x in transpose):
        weight1 = weight1.T
    name = pattern_replace(name)
    weight2 = tf_checkpoint.get_tensor(name)
    assert np.array_equal(weight1, weight2)

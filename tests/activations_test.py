import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tf2bert

x = np.linspace(-3, 3, 1000)

linear = 0.2 * x
sigmoid = tf.sigmoid(x)
tanh = tf.tanh(x)
swish = x * tf.sigmoid(x)
softsign = tf.keras.activations.softsign(x)

relu = tf.keras.activations.relu(x)
selu = tf.keras.activations.selu(x)
elu = tf.keras.activations.elu(x)
softplus = tf.keras.activations.softplus(x)

plt.subplot(131)
plt.plot(x, linear, label="linear")
plt.plot(x, sigmoid, label="sigmoid")
plt.plot(x, tanh, label="tanh")
plt.plot(x, swish, label="swish")
plt.plot(x, softsign, label="softsign")
plt.legend(loc="upper left")

plt.subplot(132)
plt.plot(x, relu, label="relu")
plt.plot(x, selu, label="selu")
plt.plot(x, elu, label="elu")
plt.plot(x, softplus, label="softplus")
plt.legend(loc="upper left")

plt.subplot(133)
plt.plot(x, tf2bert.activations.gelu_erf(x), label="gelu_erf")
plt.plot(x, tf2bert.activations.gelu_tanh(x), label="gelu_tanh")
plt.plot(x, tf2bert.activations.leaky_relu(x), label="leaky_relu")
plt.plot(x, tf2bert.activations.py_swish(x), label="py_swish")
plt.plot(x, tf2bert.activations.py_relu(x), label="py_relu")
plt.plot(x, tf2bert.activations.py_gelu(x), label="py_gelu")
plt.legend(loc="upper left")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import erf

# gelu/swish/softplus 对 relu 的光滑逼近

def psi(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def swish(x, a=1.1702):
    return x / (1 + np.exp(-a * x))

def relu(x):
    return np.maximum(x, 0.0)

def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def softplus(x, a=3):
    return 1 / a * np.log(1 + np.exp(a * x))

def plot_activations():
    x = np.linspace(-4, 4, 1000)
    plt.plot(x, psi(x), label="psi(x)")
    plt.plot(x, relu(x), label="relu")
    plt.plot(x, swish(x), label="swish")
    plt.plot(x, gelu(x), label="gelu")
    plt.plot(x, softplus(x), label="softplus")
    plt.legend(loc="upper right")
    plt.show()

def swish_psi_err(a):
    x = np.arange(0, 5, 0.001)
    e = np.abs(swish(x, a) - psi(x) * x)
    return np.max(e)

# 计算 swish 的参数 a
options = {"xtol": 1e-10, "ftol": 1e-10, "maxiter": 100000}
result = minimize(swish_psi_err, 0, method="Powell", options=options)
print(result.x)

if __name__ == "__main__":
    plot_activations()

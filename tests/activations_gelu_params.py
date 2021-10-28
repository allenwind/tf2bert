import numpy as np
from scipy.special import erf
from scipy.optimize import minimize

def f(x, b):
    a = np.sqrt(2 / np.pi)
    return np.abs(erf(x / np.sqrt(2)) - np.tanh(a * x + b * x**3))

def g(b):
    return np.max([f(x, b) for x in np.arange(0, 5, 0.001)])

options = {"xtol": 1e-10, "ftol": 1e-10, "maxiter": 100000}
result = minimize(g, 0, method="Powell", options=options)
print(result.x)

import numpy as np

def tanh(x):
    x = np.asarray(x, dtype=float)   # handle scalar, list, array
    return np.tanh(x)
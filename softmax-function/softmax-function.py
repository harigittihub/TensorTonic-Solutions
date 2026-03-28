import numpy as np

def softmax(x):
    x = np.array(x)

    # Case 1: 1D array
    if x.ndim == 1:
        x_stable = x - np.max(x)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x)

    # Case 2: 2D array (row-wise softmax)
    elif x.ndim == 2:
        x_stable = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
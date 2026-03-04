import numpy as np

def minmax_scale(X, axis=0, eps=1e-11):
    X = np.array(X, dtype=float)
    
    if X.ndim == 1:
        x_min = X.min()
        x_max = X.max()
        return (X - x_min) / (x_max - x_min + eps)
    
    x_min = X.min(axis=axis, keepdims=True)
    x_max = X.max(axis=axis, keepdims=True)
    
    return (X - x_min) / (x_max - x_min + eps)
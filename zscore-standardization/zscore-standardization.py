import numpy as np

def zscore_standardize(X, axis=0, eps=1e-8):
    X = np.array(X, dtype=float)
    
    if X.ndim == 1:
        mean = np.mean(X)
        std = np.std(X)
        return (X - mean) / (std + eps)
    
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    
    return (X - mean) / (std + eps)
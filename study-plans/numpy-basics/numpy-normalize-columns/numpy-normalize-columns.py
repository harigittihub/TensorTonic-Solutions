import numpy as np

def normalize(data):
    """Returns: np.ndarray of shape (m, n), z-score normalized per column"""
    a= np.array(data, dtype=np.float64)
    col_mean= np.mean(a,axis=0)
    col_std=np.std(a,axis=0)
    return  (a-col_mean)/col_std
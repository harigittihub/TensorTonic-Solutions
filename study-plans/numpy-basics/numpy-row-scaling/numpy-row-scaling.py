import numpy as np

def scale_rows(data, weights):
    """Returns: np.ndarray of shape (m, n), each row scaled by corresponding weight"""
    a= np.array(data, dtype=np.float64)
    b=np.array(weights, dtype=np.float64)
    c= a*b[:,None]
    return c
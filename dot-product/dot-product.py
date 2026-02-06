import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    x= np.array(x, dtype=float)
    y= np.array(y, dtype=float)

    if x.ndim !=1 or y.ndim !=1:
        raise ValueError("Tnput must be 1D array")

    if  x.shape[0] != y.shape[0]:
        raise ValueError("values must be of same length")

    return float(np.sum(x*y))    
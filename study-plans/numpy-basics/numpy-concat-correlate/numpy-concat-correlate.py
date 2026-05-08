import numpy as np

def compare_correlations(a, b):
    """Returns: np.ndarray of shape (3, n, n), stacked correlation matrices"""
    aa= np.array(a, dtype=np.float64)
    bb= np.array(b, dtype=np.float64)
    concat= np.concatenate([aa,bb], axis=0)
    x= np.corrcoef(aa.T)
    y= np.corrcoef(bb.T)
    concate2= np.corrcoef(concat.T)
    return np.stack([x,y, concate2])
import numpy as np

def filter_and_extract(data, row_start, row_stop, threshold):
    """
    Returns: 1D ndarray of float64
    """
    a = np.array(data, dtype=np.float64)
    mask = a[row_start:row_stop]
    return mask[mask>threshold]
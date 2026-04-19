import numpy as np

def select_by_index(arr, indices, axis):
    """
    Returns: 2D ndarray of float64
    """
    a =np.array(arr, dtype=np.float64)
    i= np.array(indices)
    if axis==0:
        return a[indices]
    elif axis==1:
        return a[:,indices]
    else:
        return none
import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    try:
        matrix=np.asarray(matrix, dtype=float)
    except:
        return None

    if matrix.ndim!=2:
        return None

     # Validate axis
    if axis is not None:
        if not isinstance(axis, int):
            return None
        if axis < 0 or axis >= matrix.ndim:
            return None
    

    if norm_type=="l2":
        norm = np.sqrt(np.sum(matrix**2, axis=axis, keepdims=True))            
    elif norm_type  =="l1":
        norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)  
    elif norm_type=="max":
        norm = np.max(np.abs(matrix), axis=axis, keepdims=True)    
    else :
        return None

    norm [norm==0] =1

    return matrix/ norm        
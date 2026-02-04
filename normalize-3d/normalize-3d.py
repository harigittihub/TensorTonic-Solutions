import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v = np.asarray(v,dtype=float)
#compute L2 norm along last axis
    norms = np.linalg.norm(v, axis =-1, keepdims=True)
#Replace zeros with 1 to avoid division by zero
    norms[norms ==0] =1.0

    return v/norms
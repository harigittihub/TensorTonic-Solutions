import numpy as np

def angle_features(angles):
    """Returns: np.ndarray of shape (3, n), rows are sin, cos, tan"""
    a=np.array(angles, dtype=np.float64)
    a1=np.sin(a)
    a2=np.cos(a)
    a3=np.tan(a)
    return a1,a2,a3
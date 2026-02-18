import numpy as np

def angle_between_3d(v, w):
    v = np.array(v)
    w = np.array(w)
    
    # Compute norms
    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)
    
    # Handle zero vector case
    if norm_v == 0 or norm_w == 0:
        return np.nan
    
    # Compute cosine of angle
    cos_theta = np.dot(v, w) / (norm_v * norm_w)
    
    # Numerical stability
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Angle in radians
    theta = np.arccos(cos_theta)
    
    return theta

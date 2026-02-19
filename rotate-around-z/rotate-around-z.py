import numpy as np

def rotate_around_z(points, theta):
    points = np.array(points)
    
    # Handle single point (3,) → reshape to (1,3)
    single_point = False
    if points.ndim == 1:
        points = points.reshape(1, 3)
        single_point = True
    
    # Rotation components
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Apply rotation
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    x_new = x * cos_t - y * sin_t
    y_new = x * sin_t + y * cos_t
    z_new = z
    
    rotated = np.column_stack((x_new, y_new, z_new))
    
    # Return original shape
    if single_point:
        return rotated.flatten()
    
    return rotated

import numpy as np

def covariance_matrix(X):
    X = np.array(X)
    
    # Validate input: must be 2D and at least 2 samples
    if X.ndim != 2 or X.shape[0] <= 1:
        return None
    
    # Step 1: Mean
    mu = np.mean(X, axis=0)
    
    # Step 2: Center data
    X_centered = X - mu
    
    # Step 3: Sample covariance
    N = X.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (N - 1)
    
    return cov_matrix

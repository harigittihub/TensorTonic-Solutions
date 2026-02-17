import numpy as np

def pearson_correlation(X):
    X = np.array(X)
    
    # Validate only basic structure
    if X.ndim != 2 or X.shape[0] <= 1:
        return None
    
    N = X.shape[0]
    
    # Center data
    mu = np.mean(X, axis=0)
    X_centered = X - mu
    
    # Sample covariance
    cov_matrix = (X_centered.T @ X_centered) / (N - 1)
    
    # Standard deviations
    std_dev = np.sqrt(np.diag(cov_matrix))
    
    # Correlation matrix (allow division by zero → nan)
    corr_matrix = cov_matrix / np.outer(std_dev, std_dev)
    
    return corr_matrix

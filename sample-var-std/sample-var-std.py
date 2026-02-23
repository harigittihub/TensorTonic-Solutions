import numpy as np

def sample_var_std(x):
    x = np.array(x)
    
    # Compute mean
    mean = np.mean(x)
    
    # Sample variance (divide by n-1)
    var = np.sum((x - mean) ** 2) / (len(x) - 1)
    
    # Standard deviation
    std = np.sqrt(var)
    
    return float(var), float(std)
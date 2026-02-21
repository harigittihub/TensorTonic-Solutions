import numpy as np

def percentiles(x, q):
    x = np.array(x)
    q = np.array(q)
    
    # Compute percentiles using linear interpolation
    result = np.percentile(x, q, method="linear")
    
    return np.array(result)
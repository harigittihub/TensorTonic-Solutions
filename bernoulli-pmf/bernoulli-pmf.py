import numpy as np

def bernoulli_pmf_and_moments(x, p):
    # Convert to numpy array
    x = np.array(x)
    
    # PMF using vectorized condition
    pmf = np.where(x == 1, p, 1 - p)
    
    # Mean and variance
    mean = float(p)
    var = float(p * (1 - p))
    
    return pmf, mean, var
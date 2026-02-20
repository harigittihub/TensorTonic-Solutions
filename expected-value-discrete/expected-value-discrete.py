import numpy as np

def expected_value_discrete(x, p):
    x = np.array(x)
    p = np.array(p)
    
    # Check shapes match
    if x.shape != p.shape:
        raise ValueError("x and p must have same shape")
    
    # Check probabilities sum to 1 (within tolerance)
    if not np.isclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1")
    
    # Compute expected value
    expected = np.sum(x * p)
    
    return float(expected)
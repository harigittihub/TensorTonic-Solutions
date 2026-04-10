import numpy as np

def swish(x):
    x = np.array(x)
    
    # Stable sigmoid
    sigmoid = np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )
    
    return x * sigmoid
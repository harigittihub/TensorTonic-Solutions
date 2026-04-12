import numpy as np

def selu(x, lam=1.0507009873554805, alpha=1.6732632423543772):
    x = np.array(x)
    
    result = np.where(
        x > 0,
        lam * x,
        lam * alpha * (np.exp(x) - 1)
    )
    
    return [round(val, 4) for val in result]
import numpy as np

def bootstrap_mean(x, n_bootstrap, ci=0.95, rng=None):
    x = np.array(x)
    n = len(x)
    
    # Use provided rng or create one
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate bootstrap indices
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    
    # Compute bootstrap means
    boot_means = x[indices].mean(axis=1)
    
    # Confidence interval
    alpha = 1 - ci
    lower = np.quantile(boot_means, alpha / 2)
    upper = np.quantile(boot_means, 1 - alpha / 2)
    
    return boot_means, float(lower), float(upper)
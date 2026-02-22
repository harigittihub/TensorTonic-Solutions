import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    # Edge cases
    if k < 0:
        return (0.0, 0.0)
    if k > n:
        k = n
    
    # PMF
    pmf = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    
    # CDF
    cdf = 0.0
    for i in range(0, k + 1):
        cdf += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    
    return (float(pmf), float(cdf))
import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Compute log of PMF using log-space for numerical stability
    # log P(X=k) = -lam + k*log(lam) - log(k!)
    # log(k!) = sum(log(1..k))
    
    log_factorial_k = np.sum(np.log(np.arange(1, k + 1))) if k > 0 else 0.0
    log_pmf = -lam + k * np.log(lam) - log_factorial_k
    pmf = np.exp(log_pmf)
    
    # Compute CDF by summing PMF from i=0 to k
    cdf = 0.0
    for i in range(k + 1):
        log_fact_i = np.sum(np.log(np.arange(1, i + 1))) if i > 0 else 0.0
        log_p = -lam + i * np.log(lam) - log_fact_i
        cdf += np.exp(log_p)
    
    return float(pmf), float(cdf)
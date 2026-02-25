import numpy as np

def t_test_one_sample(x, mu0):
    x = np.array(x)
    n = len(x)
    
    # Sample mean
    mean = np.mean(x)
    
    # Sample standard deviation (Bessel correction)
    s = np.std(x, ddof=1)
    
    # Standard error
    se = s / np.sqrt(n)
    
    # t-statistic
    t_stat = (mean - mu0) / se
    
    return float(t_stat)
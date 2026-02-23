import numpy as np

def chi2_independence(C):
    # Convert to numpy array
    C = np.array(C)
    
    # Row sums and column sums
    row_sums = C.sum(axis=1)
    col_sums = C.sum(axis=0)
    
    # Total sum
    total = C.sum()
    
    # Expected frequencies using outer product
    expected = np.outer(row_sums, col_sums) / total
    
    # Chi-square statistic
    chi2 = np.sum((C - expected) ** 2 / expected)
    
    return float(chi2), expected
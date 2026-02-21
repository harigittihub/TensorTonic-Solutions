import numpy as np
from collections import Counter

def mean_median_mode(x):
    x = np.array(x)
    
    # Mean and Median
    mean = float(np.mean(x))
    median = float(np.median(x))
    
    # Mode (smallest value with highest frequency)
    counts = Counter(x)
    max_freq = max(counts.values())
    
    # Filter values with max frequency and take smallest
    mode = float(min([k for k, v in counts.items() if v == max_freq]))
    
    return (mean, median, mode)
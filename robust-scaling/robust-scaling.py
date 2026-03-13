import numpy as np

def robust_scaling(columns):
    if len(columns) == 1:
        return [0]

    x = np.array(columns, dtype=float)
    s = np.sort(x)
    n = len(s)

    median = np.median(s)

    lower = s[:n//2]
    upper = s[(n+1)//2:]

    q1 = np.median(lower)
    q3 = np.median(upper)

    iqr = q3 - q1

    if iqr == 0:
        return [0]*len(x)

    scaled = (x - median) / iqr
    return scaled.tolist()
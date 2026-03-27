import math

def elu(x, alpha=1.0):
    return [xi if xi > 0 else alpha * (math.exp(xi) - 1) for xi in x]
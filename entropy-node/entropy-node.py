import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)

    # avoid log2(0)
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)
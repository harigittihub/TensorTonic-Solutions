import numpy as np

def matrix_inverse(A):
    """
    Compute the inverse of a square matrix A.
    Return None if the matrix is singular.
    """
    A = np.asarray(A, dtype=float)

    # Check 2D
    if A.ndim != 2:
        raise ValueError("Input must be a 2D matrix")

    # Check square
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix must be square")

    # Check singularity
    det = np.linalg.det(A)
    if abs(det) < 1e-10:
        return None   # IMPORTANT: return None, not exception

    return np.linalg.inv(A)

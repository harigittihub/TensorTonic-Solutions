import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        matrix = np.array(matrix, dtype=float)
    except:
        return None

    # Must be 2D
    if matrix.ndim != 2:
        return None

    # Must be square
    n, m = matrix.shape
    if n != m:
        return None

    eigenvalues =np.linalg.eigvals(matrix)    

    eigenvalues = eigenvalues[np.lexsort((eigenvalues.imag, eigenvalues.real))]

    return eigenvalues
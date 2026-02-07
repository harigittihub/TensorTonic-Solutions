import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A)
    rows, cols = A.shape

    T= np.zeros((cols, rows), dtype=A.dtype)

    for i in range (rows):
        for j in range (cols):
            T[j,i] = A[i,j]

    return T        

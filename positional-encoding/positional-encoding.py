import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    """
    
    # Positions: (seq_len, 1)
    pos = np.arange(seq_len)[:, np.newaxis]
    
    # Dimension indices: (d_model,)
    i = np.arange(d_model)
    
    # Compute denominator term
    div_term = np.power(base, (2 * (i // 2)) / d_model)
    
    # Compute angle matrix
    angles = pos / div_term
    
    # Initialize PE
    pe = np.zeros((seq_len, d_model))
    
    # Apply sin to even indices
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    
    # Apply cos to odd indices
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    
    return pe

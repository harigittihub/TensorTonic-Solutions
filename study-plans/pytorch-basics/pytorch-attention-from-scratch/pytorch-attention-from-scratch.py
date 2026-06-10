import torch

def scaled_dot_product_attention(Q, K, V):
    """
    Returns: attention output tensor
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    return weights @ V

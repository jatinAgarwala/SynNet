"""
Computes the sum of a single molecular embedding.
"""
import numpy as np


def func(emb):
    """
    Computes the sum of a single molecular embedding.

    Args:
        emb: Molecular embedding.

    Returns:
        float: The sum of the molecular embedding
    """
    return np.sum(emb)

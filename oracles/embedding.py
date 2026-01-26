"""Embedding oracles."""

import numpy as np


def embedding_lookup(
    weight: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Embedding lookup - gather rows by indices.

    Args:
        weight: Embedding weight matrix [vocab_size, n_embd]
        indices: Token indices [batch, seq_len]

    Returns:
        Embedded tokens [batch, seq_len, n_embd]
    """
    return weight[indices]

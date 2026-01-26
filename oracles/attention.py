"""Attention oracles."""

import numpy as np


def attention_gqa(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Grouped Query Attention.

    Supports different numbers of Q and KV heads via broadcasting.
    n_head must be divisible by n_head_kv.

    Args:
        query: Query tensor [batch, seq_len, n_head, head_dim]
        key: Key tensor [batch, seq_len, n_head_kv, head_dim]
        value: Value tensor [batch, seq_len, n_head_kv, head_dim]
        scale: Scale factor, typically 1.0 / sqrt(head_dim)

    Returns:
        Output tensor [batch, seq_len, n_head, head_dim]
    """
    batch, seq_q, n_head, head_dim = query.shape
    _, seq_kv, n_head_kv, _ = key.shape

    # GQA: Repeat KV heads to match Q heads
    repeat_factor = n_head // n_head_kv
    key = np.repeat(key, repeat_factor, axis=2)
    value = np.repeat(value, repeat_factor, axis=2)

    # Transpose for matmul: [batch, n_head, seq, head_dim]
    q = query.transpose(0, 2, 1, 3)
    k = key.transpose(0, 2, 1, 3)
    v = value.transpose(0, 2, 1, 3)

    # QK^T: [batch, n_head, seq_q, seq_kv]
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale

    # Softmax (numerically stable)
    scores_max = scores.max(axis=-1, keepdims=True)
    scores_exp = np.exp(scores - scores_max)
    attn_weights = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

    # Attention @ V: [batch, n_head, seq_q, head_dim]
    output = np.matmul(attn_weights, v)

    # Transpose back: [batch, seq_q, n_head, head_dim]
    return output.transpose(0, 2, 1, 3)

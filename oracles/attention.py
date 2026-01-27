"""Attention oracles."""

import numpy as np

from oracles.position import rope


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


def attention_block(
    input: np.ndarray,
    positions: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wo: np.ndarray,
    bq: np.ndarray,
    bk: np.ndarray,
    bv: np.ndarray,
    bo: np.ndarray,
    use_bias: bool,
    n_head: int,
    n_head_kv: int,
    n_embd: int,
    rope_freq_base: float = 10000.0,
    rope_freq_scale: float = 1.0,
) -> np.ndarray:
    """Full attention block with QKV projection, RoPE, attention, output projection.

    Args:
        input: Input tensor [batch, seq_len, n_embd]
        positions: Position indices [batch, seq_len]
        wq: Query weight [n_embd, n_embd]
        wk: Key weight [n_embd, n_embd_kv]
        wv: Value weight [n_embd, n_embd_kv]
        wo: Output weight [n_embd, n_embd]
        bq: Query bias [n_embd] (used if use_bias=True)
        bk: Key bias [n_embd_kv]
        bv: Value bias [n_embd_kv]
        bo: Output bias [n_embd]
        use_bias: Whether to add biases
        n_head: Number of query heads
        n_head_kv: Number of key/value heads
        n_embd: Embedding dimension
        rope_freq_base: RoPE frequency base
        rope_freq_scale: RoPE frequency scale

    Returns:
        Output tensor [batch, seq_len, n_embd]
    """
    batch, seq_len, _ = input.shape
    n_embd_kv = wk.shape[1]
    head_dim = n_embd // n_head
    head_dim_kv = n_embd_kv // n_head_kv

    # QKV projections: [batch, seq_len, n_embd] @ [n_embd, n_out] -> [batch, seq_len, n_out]
    q_proj = np.matmul(input, wq)
    k_proj = np.matmul(input, wk)
    v_proj = np.matmul(input, wv)

    # Add biases if enabled
    if use_bias:
        q_proj = q_proj + bq
        k_proj = k_proj + bk
        v_proj = v_proj + bv

    # Reshape for multi-head: [batch, seq_len, n_embd] -> [batch, seq_len, n_head, head_dim]
    q_reshaped = q_proj.reshape(batch, seq_len, n_head, head_dim)
    k_reshaped = k_proj.reshape(batch, seq_len, n_head_kv, head_dim_kv)
    v_reshaped = v_proj.reshape(batch, seq_len, n_head_kv, head_dim_kv)

    # Apply RoPE to query and key
    q_rope = rope(q_reshaped, positions, rope_freq_base, rope_freq_scale)
    k_rope = rope(k_reshaped, positions, rope_freq_base, rope_freq_scale)

    # Compute attention scale
    scale = 1.0 / np.sqrt(head_dim)

    # Run attention
    attn_out = attention_gqa(q_rope, k_rope, v_reshaped, scale)

    # Reshape back: [batch, seq_len, n_head, head_dim] -> [batch, seq_len, n_embd]
    attn_flat = attn_out.reshape(batch, seq_len, n_embd)

    # Output projection
    output_proj = np.matmul(attn_flat, wo)

    # Add output bias if enabled
    if use_bias:
        output_proj = output_proj + bo

    return output_proj

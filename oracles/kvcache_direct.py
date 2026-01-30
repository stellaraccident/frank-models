"""Direct KV Cache oracles for transformer models.

Reference implementations matching the MLIR semantics in
components/kvcache/kvcache_direct.mlir.

Dimension layout (all arrays):
  dim 0: batch_size   - Number of sequences in the batch
  dim 1: seq_len      - Sequence position (max_seq_len for cache storage)
  dim 2: n_head_kv    - Number of KV heads (may differ from Q heads in GQA)
  dim 3: head_dim     - Dimension per attention head
"""

import numpy as np


def allocate(
    batch_size: int,
    max_seq_len: int,
    n_head_kv: int,
    head_dim: int,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate a new batched KV cache initialized to zeros.

    Args:
        batch_size: Number of sequences in batch
        max_seq_len: Maximum sequence length (cache capacity)
        n_head_kv: Number of KV attention heads
        head_dim: Dimension per attention head
        dtype: Data type for cache arrays

    Returns:
        Tuple of (k_cache, v_cache), each [batch_size, max_seq_len, n_head_kv, head_dim]
    """
    shape = (batch_size, max_seq_len, n_head_kv, head_dim)
    k_cache = np.zeros(shape, dtype=dtype)
    v_cache = np.zeros(shape, dtype=dtype)
    return k_cache, v_cache


def gather(
    cache: np.ndarray,  # [batch_size, max_seq_len, n_head_kv, head_dim]
    seq_len: int,
) -> np.ndarray:
    """Gather values from cache for positions [0, seq_len) for all batch elements.

    Args:
        cache: Cache tensor [batch_size, max_seq_len, n_head_kv, head_dim]
        seq_len: Number of positions to gather (same for all batch elements)

    Returns:
        Values [batch_size, seq_len, n_head_kv, head_dim]
    """
    return cache[:, :seq_len, :, :].copy()


def scatter(
    cache: np.ndarray,     # [batch_size, max_seq_len, n_head_kv, head_dim]
    new_vals: np.ndarray,  # [batch_size, new_tokens, n_head_kv, head_dim]
    write_pos: int,
) -> np.ndarray:
    """Write new values to cache at write_pos for all batch elements.

    Args:
        cache: Cache tensor [batch_size, max_seq_len, n_head_kv, head_dim]
        new_vals: New values to write [batch_size, new_tokens, n_head_kv, head_dim]
        write_pos: Starting position to write (same for all batch elements)

    Returns:
        Updated cache tensor (copy, not in-place)
    """
    result = cache.copy()
    new_tokens = new_vals.shape[1]  # dim 1 is seq_len/new_tokens
    result[:, write_pos:write_pos + new_tokens, :, :] = new_vals
    return result

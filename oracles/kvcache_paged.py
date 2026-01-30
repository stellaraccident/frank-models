"""Paged KV Cache oracles for transformer models.

Reference implementations matching the MLIR semantics in
components/kvcache/kvcache_paged.mlir.

Physical cache layout (flat pool, shared across all sequences):
  dim 0: block_idx    - Physical block index in the pool
  dim 1: pos_in_block - Position within block [0, block_size)
  dim 2: n_head_kv    - Number of KV heads
  dim 3: head_dim     - Dimension per attention head

Interface asymmetry (intentional):
  - gather: reads full context, flat [n_ctx_tokens] block indices
  - scatter_single_token: writes ONE token per sequence (decode hot path)
"""

import numpy as np


def allocate(
    n_blocks: int,
    block_size: int,
    n_head_kv: int,
    head_dim: int,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate a new paged KV cache (block pool).

    Args:
        n_blocks: Number of physical blocks in the pool
        block_size: Tokens per block
        n_head_kv: Number of KV attention heads
        head_dim: Dimension per attention head
        dtype: Data type for cache arrays

    Returns:
        Tuple of (k_cache, v_cache), each [n_blocks, block_size, n_head_kv, head_dim]
    """
    shape = (n_blocks, block_size, n_head_kv, head_dim)
    k_cache = np.zeros(shape, dtype=dtype)
    v_cache = np.zeros(shape, dtype=dtype)
    return k_cache, v_cache


def gather(
    cache: np.ndarray,          # [n_blocks, block_size, n_head_kv, head_dim]
    block_indices: np.ndarray,  # [n_ctx_tokens] int32 - flat
    block_size: int,
) -> np.ndarray:
    """Gather values from paged cache using flat block indices.

    Args:
        cache: Block pool [n_blocks, block_size, n_head_kv, head_dim]
        block_indices: [n_ctx_tokens] physical block index per context token (flat)
        block_size: Tokens per block (for computing position within block)

    Returns:
        Gathered values [n_ctx_tokens, n_head_kv, head_dim]
    """
    n_ctx_tokens = len(block_indices)
    n_head_kv = cache.shape[2]
    head_dim = cache.shape[3]

    result = np.zeros((n_ctx_tokens, n_head_kv, head_dim), dtype=cache.dtype)

    for ctx_idx in range(n_ctx_tokens):
        block_idx = block_indices[ctx_idx]
        pos_in_block = ctx_idx % block_size
        result[ctx_idx, :, :] = cache[block_idx, pos_in_block, :, :]

    return result


def scatter_single_token(
    cache: np.ndarray,          # [n_blocks, block_size, n_head_kv, head_dim]
    new_vals: np.ndarray,       # [batch, n_head_kv, head_dim] - ONE token per sequence
    block_indices: np.ndarray,  # [batch] int32 - which block for each sequence
    pos_in_blocks: np.ndarray,  # [batch] int32 - position within block for each sequence
) -> np.ndarray:
    """Scatter one new token per sequence to paged cache (decode phase).

    This is the decode-phase operation where each sequence in the batch
    adds exactly one token.

    Args:
        cache: Block pool [n_blocks, block_size, n_head_kv, head_dim]
        new_vals: New values [batch, n_head_kv, head_dim] - ONE token per sequence
        block_indices: [batch] which block for each sequence
        pos_in_blocks: [batch] position within block for each sequence

    Returns:
        Updated cache (copy, not in-place)
    """
    result = cache.copy()
    batch_size = len(block_indices)

    for i in range(batch_size):
        block_idx = block_indices[i]
        pos = pos_in_blocks[i]
        result[block_idx, pos, :, :] = new_vals[i, :, :]

    return result

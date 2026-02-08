"""Canonical Paged KV Cache oracles.

Reference implementations matching the MLIR semantics in
components/kvcache/kvcache.mlir.

This is the canonical implementation following Ben's device-first philosophy:
all control metadata (block_tables, context_lens, block_indices, pos_in_blocks)
is treated as device data (numpy arrays, not Python lists).

Physical cache layout (block pool):
  K/V: [n_blocks, block_size, n_head_kv, head_dim]

Interface:
  allocate(n_blocks, block_size, n_head_kv, head_dim) -> (k_cache, v_cache)
  gather(k/v_cache, block_tables, context_lens, max_context_len) -> gathered
  scatter_decode(k/v_cache, new_vals, block_indices, pos_in_blocks) -> updated
"""

import numpy as np


def allocate(
    n_blocks: int,
    block_size: int,
    n_head_kv: int,
    head_dim: int,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate a paged KV cache block pool.

    Args:
        n_blocks: Total physical blocks in the pool
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
    cache: np.ndarray,  # [n_blocks, block_size, n_head_kv, head_dim]
    block_tables: np.ndarray,  # [batch, max_blocks_per_seq] int32
    context_lens: np.ndarray,  # [batch] int32
    max_context_len: int,
) -> np.ndarray:
    """Gather values from paged cache using block tables.

    All control data (block_tables, context_lens) treated as device arrays
    to match MLIR semantics.

    Args:
        cache: Block pool [n_blocks, block_size, n_head_kv, head_dim]
        block_tables: [batch, max_blocks_per_seq] mapping logical blocks to physical
        context_lens: [batch] context length per sequence
        max_context_len: Maximum context length (for output shape)

    Returns:
        Gathered values [batch, max_context_len, n_head_kv, head_dim]
        Positions beyond context_lens[b] are zeroed.
    """
    batch = block_tables.shape[0]
    block_size = cache.shape[1]
    n_head_kv = cache.shape[2]
    head_dim = cache.shape[3]

    result = np.zeros((batch, max_context_len, n_head_kv, head_dim), dtype=cache.dtype)

    for b in range(batch):
        ctx_len = int(context_lens[b])
        for ctx_idx in range(min(ctx_len, max_context_len)):
            logical_block = ctx_idx // block_size
            pos_in_block = ctx_idx % block_size
            physical_block = int(block_tables[b, logical_block])
            result[b, ctx_idx, :, :] = cache[physical_block, pos_in_block, :, :]

    return result


def scatter_decode(
    cache: np.ndarray,  # [n_blocks, block_size, n_head_kv, head_dim]
    new_vals: np.ndarray,  # [batch, n_head_kv, head_dim] - ONE token per sequence
    block_indices: np.ndarray,  # [batch] int32 - which block for each sequence
    pos_in_blocks: np.ndarray,  # [batch] int32 - position within block
) -> np.ndarray:
    """Scatter one new token per sequence to paged cache (decode phase).

    All control data (block_indices, pos_in_blocks) treated as device arrays
    to match MLIR semantics.

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
        block_idx = int(block_indices[i])
        pos = int(pos_in_blocks[i])
        result[block_idx, pos, :, :] = new_vals[i, :, :]

    return result

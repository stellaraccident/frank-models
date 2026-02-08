"""Canonical Paged KV Cache oracles.

Reference implementations matching the MLIR semantics in
components/kvcache/kvcache.mlir.

This is the canonical implementation following Ben's device-first philosophy:
all control metadata (block_tables, context_lens, block_indices, pos_in_blocks)
is treated as device data (numpy arrays, not Python lists).

Physical cache layout (unified block pool, shared across all layers):
  K/V: [n_blocks, block_size, n_head_kv, head_dim]

Layer-aware metadata:
  block_tables: [n_layers, batch, max_blocks_per_seq] - physical block mapping
  context_lens: [n_layers, batch] - context length per layer/sequence

Interface (MLIR + Oracle):
  allocate(n_blocks, block_size, n_head_kv, head_dim) -> (k_cache, v_cache)
  gather(cache, layer, block_tables, context_lens, max_context_len) -> gathered
  scatter_decode(cache, layer, new_vals, block_indices, pos_in_blocks) -> updated

Oracle-only (not in MLIR due to shape tracking limitations with nested loops):
  scatter_prefill(cache, layer, new_vals, block_tables, start_positions, seq_lengths, block_size) -> updated

For prefill operations, use repeated scatter_decode calls from the host side.
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
    layer: int,  # which transformer layer (0 to n_layers-1)
    block_tables: np.ndarray,  # [n_layers, batch, max_blocks_per_seq] int32
    context_lens: np.ndarray,  # [n_layers, batch] int32
    max_context_len: int,
) -> np.ndarray:
    """Gather values from paged cache using block tables for a specific layer.

    All control data (block_tables, context_lens) treated as device arrays
    to match MLIR semantics. Layer dimension allows unified cache across all
    transformer layers.

    Args:
        cache: Block pool [n_blocks, block_size, n_head_kv, head_dim]
        layer: Which transformer layer (0 to n_layers-1)
        block_tables: [n_layers, batch, max_blocks_per_seq] mapping logical blocks to physical
        context_lens: [n_layers, batch] context length per layer/sequence
        max_context_len: Maximum context length (for output shape)

    Returns:
        Gathered values [batch, max_context_len, n_head_kv, head_dim]
        Positions beyond context_lens[layer, b] are zeroed.
    """
    # Slice metadata for this layer
    block_tables_layer = block_tables[layer]  # [batch, max_blocks_per_seq]
    context_lens_layer = context_lens[layer]  # [batch]

    batch = block_tables_layer.shape[0]
    block_size = cache.shape[1]
    n_head_kv = cache.shape[2]
    head_dim = cache.shape[3]

    result = np.zeros((batch, max_context_len, n_head_kv, head_dim), dtype=cache.dtype)

    for b in range(batch):
        ctx_len = int(context_lens_layer[b])
        for ctx_idx in range(min(ctx_len, max_context_len)):
            logical_block = ctx_idx // block_size
            pos_in_block = ctx_idx % block_size
            physical_block = int(block_tables_layer[b, logical_block])
            result[b, ctx_idx, :, :] = cache[physical_block, pos_in_block, :, :]

    return result


def scatter_decode(
    cache: np.ndarray,  # [n_blocks, block_size, n_head_kv, head_dim]
    layer: int,  # which transformer layer (unused, for API consistency)
    new_vals: np.ndarray,  # [batch, n_head_kv, head_dim] - ONE token per sequence
    block_indices: np.ndarray,  # [batch] int32 - which block for each sequence
    pos_in_blocks: np.ndarray,  # [batch] int32 - position within block
) -> np.ndarray:
    """Scatter one new token per sequence to paged cache (decode phase).

    All control data (block_indices, pos_in_blocks) treated as device arrays
    to match MLIR semantics.

    Note: layer parameter is included for API consistency with gather, but
    scatter_decode writes directly to physical blocks (no layer slicing needed
    since all layers share the same physical block pool).

    Args:
        cache: Block pool [n_blocks, block_size, n_head_kv, head_dim]
        layer: Which transformer layer (unused, for API consistency)
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


def scatter_prefill(
    cache: np.ndarray,  # [n_blocks, block_size, n_head_kv, head_dim]
    layer: int,  # which transformer layer (for block_tables slicing)
    new_vals: np.ndarray,  # [batch, seq_len, n_head_kv, head_dim] - full sequence
    block_tables: np.ndarray,  # [n_layers, batch, max_blocks_per_seq] int32
    start_positions: np.ndarray,  # [batch] int32 - starting position (usually 0)
    block_size: int,
) -> np.ndarray:
    """Scatter multiple tokens to paged cache (prefill phase).

    Writes a full sequence of K/V values to the cache. Used during prefill
    when processing the initial prompt tokens.

    NOTE: This assumes uniform sequence lengths across the batch (all sequences
    have length = new_vals.shape[1]). For variable-length sequences, pad to
    uniform length or process batches separately by length. Ragged scatter
    (per-sequence seq_lengths) is a potential future optimization.

    Args:
        cache: Block pool [n_blocks, block_size, n_head_kv, head_dim]
        layer: Which transformer layer (for block_tables indexing)
        new_vals: New values [batch, seq_len, n_head_kv, head_dim] - full sequence
        block_tables: [n_layers, batch, max_blocks_per_seq] mapping logical to physical
        start_positions: [batch] starting position in sequence (usually 0)
        block_size: Tokens per block (for computing block indices)

    Returns:
        Updated cache (copy, not in-place)
    """
    # Slice block_tables for this layer
    block_tables_layer = block_tables[layer]  # [batch, max_blocks_per_seq]

    result = cache.copy()
    batch_size, seq_len = new_vals.shape[0], new_vals.shape[1]

    for b in range(batch_size):
        start_pos = int(start_positions[b])

        for s in range(seq_len):
            abs_pos = start_pos + s
            logical_block = abs_pos // block_size
            pos_in_block = abs_pos % block_size
            physical_block = int(block_tables_layer[b, logical_block])

            result[physical_block, pos_in_block, :, :] = new_vals[b, s, :, :]

    return result

"""Sequence-addressed Paged KV Cache oracles.

Reference implementations matching the MLIR semantics in
components/kvcache/kvcache_seq_paged.mlir.

This cache uses (seq_id, position) addressing rather than exposing block indices.
The page table is managed internally.

Physical cache layout:
  K/V blocks: [n_blocks, block_size, n_head_kv, head_dim]

Internal page table:
  page_table: [n_seq, max_blocks_per_seq] i32 - logical block -> physical block
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SeqPagedCache:
    """Sequence-addressed paged KV cache state."""
    k_blocks: np.ndarray      # [n_blocks, block_size, n_head_kv, head_dim]
    v_blocks: np.ndarray      # [n_blocks, block_size, n_head_kv, head_dim]
    page_table: np.ndarray    # [n_seq, max_blocks_per_seq] i32
    block_size: int


def allocate(
    n_seq: int,
    max_seq_len: int,
    n_head_kv: int,
    head_dim: int,
    block_size: int,
    dtype: np.dtype = np.float32,
) -> SeqPagedCache:
    """Allocate a sequence-addressed paged KV cache.

    Pre-allocates blocks for each sequence (simple strategy).
    Sequence i gets blocks [i * blocks_per_seq, (i+1) * blocks_per_seq).

    Args:
        n_seq: Max concurrent sequences
        max_seq_len: Max tokens per sequence
        n_head_kv: Number of KV attention heads
        head_dim: Dimension per attention head
        block_size: Tokens per block
        dtype: Data type for cache arrays

    Returns:
        SeqPagedCache containing K/V blocks and page table
    """
    # Compute blocks per sequence
    blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    n_blocks = n_seq * blocks_per_seq

    # Allocate K/V blocks
    k_blocks = np.zeros((n_blocks, block_size, n_head_kv, head_dim), dtype=dtype)
    v_blocks = np.zeros((n_blocks, block_size, n_head_kv, head_dim), dtype=dtype)

    # Initialize page table: seq i gets blocks [i * blocks_per_seq, (i+1) * blocks_per_seq)
    page_table = np.zeros((n_seq, blocks_per_seq), dtype=np.int32)
    for seq_id in range(n_seq):
        for block_idx in range(blocks_per_seq):
            page_table[seq_id, block_idx] = seq_id * blocks_per_seq + block_idx

    return SeqPagedCache(
        k_blocks=k_blocks,
        v_blocks=v_blocks,
        page_table=page_table,
        block_size=block_size,
    )


def gather(
    cache: SeqPagedCache,
    seq_ids: np.ndarray,      # [batch] i32
    context_lens: np.ndarray,  # [batch] i32
) -> tuple[np.ndarray, np.ndarray]:
    """Gather K/V for attention using sequence IDs and context lengths.

    Args:
        cache: Cache object from allocate
        seq_ids: [batch] which sequences to gather from
        context_lens: [batch] context length per sequence

    Returns:
        (k, v) each [total_tokens, n_head_kv, head_dim]
        where total_tokens = sum(context_lens)
    """
    total_tokens = int(np.sum(context_lens))
    n_head_kv = cache.k_blocks.shape[2]
    head_dim = cache.k_blocks.shape[3]
    block_size = cache.block_size

    k_result = np.zeros((total_tokens, n_head_kv, head_dim), dtype=cache.k_blocks.dtype)
    v_result = np.zeros((total_tokens, n_head_kv, head_dim), dtype=cache.v_blocks.dtype)

    out_idx = 0
    for batch_idx in range(len(seq_ids)):
        seq_id = seq_ids[batch_idx]
        ctx_len = context_lens[batch_idx]

        for pos in range(ctx_len):
            logical_block = pos // block_size
            pos_in_block = pos % block_size
            physical_block = cache.page_table[seq_id, logical_block]

            k_result[out_idx] = cache.k_blocks[physical_block, pos_in_block]
            v_result[out_idx] = cache.v_blocks[physical_block, pos_in_block]
            out_idx += 1

    return k_result, v_result


def scatter_decode(
    cache: SeqPagedCache,
    seq_ids: np.ndarray,      # [batch] i32
    positions: np.ndarray,    # [batch] i32
    new_k: np.ndarray,        # [batch, n_head_kv, head_dim]
    new_v: np.ndarray,        # [batch, n_head_kv, head_dim]
) -> None:
    """Scatter one token per sequence (decode phase).

    Mutates cache in place.

    Args:
        cache: Cache object from allocate
        seq_ids: [batch] which sequences to write to
        positions: [batch] write position per sequence
        new_k: [batch, n_head_kv, head_dim] - one token per sequence
        new_v: [batch, n_head_kv, head_dim]
    """
    block_size = cache.block_size
    batch_size = len(seq_ids)

    for i in range(batch_size):
        seq_id = seq_ids[i]
        pos = positions[i]

        logical_block = pos // block_size
        pos_in_block = pos % block_size
        physical_block = cache.page_table[seq_id, logical_block]

        cache.k_blocks[physical_block, pos_in_block] = new_k[i]
        cache.v_blocks[physical_block, pos_in_block] = new_v[i]

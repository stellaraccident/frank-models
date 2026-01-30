"""Tests for Sequence-Addressed Paged KV Cache component.

Tests the paged KV cache with (seq_id, position) interface.
Page table is managed internally ON THE HOST - callers don't see block indices.

Physical cache layout:
  K/V blocks: [n_blocks, block_size, n_head_kv, head_dim]

Internal page table (host-side):
  page_table: nested !util.list<?> - page_table[seq_id] = list of physical blocks
"""

import numpy as np
import pytest

from iree.runtime import VmVariantList, HalBufferView
from tests.utils import compile_component, assert_close
from oracles.kvcache_seq_paged import (
    allocate as allocate_oracle,
    gather as gather_oracle,
    scatter_decode as scatter_decode_oracle,
)


def create_i32_list(module, values):
    """Create a !util.list<i32> from Python list of ints."""
    lst = VmVariantList(len(values))
    for v in values:
        lst.push_int(int(v))
    return lst


@pytest.fixture(scope="module")
def kvcache_module(iree_cfg):
    """Compile the seq_paged KV cache component once per test module."""
    return compile_component("kvcache/kvcache_seq_paged.mlir", iree_cfg)


class TestAllocate:
    """Tests for allocate operation."""

    def test_basic(self, kvcache_module):
        """Test that allocate creates a cache with correct structure."""
        n_seq, max_seq_len, n_head_kv, head_dim, block_size = 4, 32, 2, 8, 8

        # blocks_per_seq = ceil(32/8) = 4
        # n_blocks = 4 * 4 = 16

        args = VmVariantList(5)
        args.push_int(n_seq)
        args.push_int(max_seq_len)
        args.push_int(n_head_kv)
        args.push_int(head_dim)
        args.push_int(block_size)

        func = kvcache_module.lookup_function("allocate")
        results = VmVariantList(1)
        kvcache_module._context.invoke(func, args, results)

        # Should return a list with 4 elements
        cache = results.get_as_list(0)
        assert cache is not None
        assert len(cache) == 4

        # Check K blocks dimensions
        k_bv = cache.get_as_object(0, HalBufferView)
        k_arr = kvcache_module._buffer_view_to_numpy(k_bv)
        expected_n_blocks = n_seq * ((max_seq_len + block_size - 1) // block_size)
        assert k_arr.shape == (expected_n_blocks, block_size, n_head_kv, head_dim)

        # Check V blocks dimensions
        v_bv = cache.get_as_object(1, HalBufferView)
        v_arr = kvcache_module._buffer_view_to_numpy(v_bv)
        assert v_arr.shape == (expected_n_blocks, block_size, n_head_kv, head_dim)

        # Check page table is a list (host-side)
        page_table = cache.get_as_list(2)
        assert page_table is not None
        assert len(page_table) == n_seq

        # Verify page table content: seq i gets blocks [i * blocks_per_seq, ...)
        blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        for seq_id in range(n_seq):
            seq_blocks = page_table.get_as_list(seq_id)
            assert len(seq_blocks) == blocks_per_seq
            for block_idx in range(blocks_per_seq):
                expected = seq_id * blocks_per_seq + block_idx
                actual = seq_blocks.get_variant(block_idx)
                assert actual == expected


class TestScatterDecode:
    """Tests for scatter_decode operation."""

    def test_single_sequence(self, kvcache_module):
        """Test scatter_decode with a single sequence."""
        np.random.seed(42)
        n_seq, max_seq_len, n_head_kv, head_dim, block_size = 4, 32, 2, 8, 8

        # Allocate cache
        alloc_args = VmVariantList(5)
        alloc_args.push_int(n_seq)
        alloc_args.push_int(max_seq_len)
        alloc_args.push_int(n_head_kv)
        alloc_args.push_int(head_dim)
        alloc_args.push_int(block_size)
        alloc_func = kvcache_module.lookup_function("allocate")
        alloc_results = VmVariantList(1)
        kvcache_module._context.invoke(alloc_func, alloc_args, alloc_results)
        cache = alloc_results.get_as_list(0)

        # Scatter one token to sequence 0 at position 5
        seq_ids = create_i32_list(kvcache_module, [0])
        positions = create_i32_list(kvcache_module, [5])
        new_k = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)
        new_v = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)

        scatter_args = VmVariantList(5)
        scatter_args.push_list(cache)
        scatter_args.push_list(seq_ids)
        scatter_args.push_list(positions)
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
        scatter_func = kvcache_module.lookup_function("scatter_decode")
        scatter_results = VmVariantList(0)
        kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)

        # Verify by reading back K blocks
        k_bv = cache.get_as_object(0, HalBufferView)
        k_blocks = kvcache_module._buffer_view_to_numpy(k_bv)

        # Position 5 should be in block 0 (5 // 8 = 0), position 5 (5 % 8 = 5)
        # Seq 0's block 0 is physical block 0
        assert_close(k_blocks[0, 5], new_k[0])

    def test_multiple_sequences(self, kvcache_module):
        """Test scatter_decode with multiple sequences in batch."""
        np.random.seed(42)
        n_seq, max_seq_len, n_head_kv, head_dim, block_size = 4, 32, 2, 8, 8

        # Allocate cache
        alloc_args = VmVariantList(5)
        alloc_args.push_int(n_seq)
        alloc_args.push_int(max_seq_len)
        alloc_args.push_int(n_head_kv)
        alloc_args.push_int(head_dim)
        alloc_args.push_int(block_size)
        alloc_func = kvcache_module.lookup_function("allocate")
        alloc_results = VmVariantList(1)
        kvcache_module._context.invoke(alloc_func, alloc_args, alloc_results)
        cache = alloc_results.get_as_list(0)

        # Scatter tokens to sequences 0, 1, 2 at different positions
        batch_size = 3
        seq_ids = create_i32_list(kvcache_module, [0, 1, 2])
        positions = create_i32_list(kvcache_module, [3, 10, 0])  # seq1 pos10 crosses block boundary
        new_k = np.random.randn(batch_size, n_head_kv, head_dim).astype(np.float32)
        new_v = np.random.randn(batch_size, n_head_kv, head_dim).astype(np.float32)

        scatter_args = VmVariantList(5)
        scatter_args.push_list(cache)
        scatter_args.push_list(seq_ids)
        scatter_args.push_list(positions)
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
        scatter_func = kvcache_module.lookup_function("scatter_decode")
        scatter_results = VmVariantList(0)
        kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)

        # Verify against oracle
        oracle_cache = allocate_oracle(n_seq, max_seq_len, n_head_kv, head_dim, block_size)
        scatter_decode_oracle(oracle_cache, np.array([0, 1, 2]), np.array([3, 10, 0]), new_k, new_v)

        k_bv = cache.get_as_object(0, HalBufferView)
        k_blocks = kvcache_module._buffer_view_to_numpy(k_bv)
        assert_close(k_blocks, oracle_cache.k_blocks)


class TestGather:
    """Tests for gather operation."""

    def test_single_sequence(self, kvcache_module):
        """Test gather with a single sequence."""
        np.random.seed(42)
        n_seq, max_seq_len, n_head_kv, head_dim, block_size = 4, 32, 2, 8, 8

        # Allocate cache
        alloc_args = VmVariantList(5)
        alloc_args.push_int(n_seq)
        alloc_args.push_int(max_seq_len)
        alloc_args.push_int(n_head_kv)
        alloc_args.push_int(head_dim)
        alloc_args.push_int(block_size)
        alloc_func = kvcache_module.lookup_function("allocate")
        alloc_results = VmVariantList(1)
        kvcache_module._context.invoke(alloc_func, alloc_args, alloc_results)
        cache = alloc_results.get_as_list(0)

        # Fill sequence 0 with some data (scatter multiple tokens)
        context_len = 6
        for pos in range(context_len):
            seq_ids = create_i32_list(kvcache_module, [0])
            positions = create_i32_list(kvcache_module, [pos])
            new_k = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)
            new_v = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)

            scatter_args = VmVariantList(5)
            scatter_args.push_list(cache)
            scatter_args.push_list(seq_ids)
            scatter_args.push_list(positions)
            scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
            scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
            scatter_func = kvcache_module.lookup_function("scatter_decode")
            scatter_results = VmVariantList(0)
            kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)

        # Gather context for sequence 0
        gather_seq_ids = create_i32_list(kvcache_module, [0])
        gather_context_lens = create_i32_list(kvcache_module, [context_len])
        total_tokens = context_len

        gather_args = VmVariantList(4)
        gather_args.push_list(cache)
        gather_args.push_list(gather_seq_ids)
        gather_args.push_list(gather_context_lens)
        gather_args.push_int(total_tokens)
        gather_func = kvcache_module.lookup_function("gather")
        gather_results = VmVariantList(2)
        kvcache_module._context.invoke(gather_func, gather_args, gather_results)

        k_gathered = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(0, HalBufferView)
        )
        v_gathered = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(1, HalBufferView)
        )

        # Check shapes
        assert k_gathered.shape == (context_len, n_head_kv, head_dim)
        assert v_gathered.shape == (context_len, n_head_kv, head_dim)


class TestRoundTrip:
    """Tests for scatter then gather."""

    def test_scatter_then_gather(self, kvcache_module):
        """Test that scatter followed by gather recovers the data."""
        np.random.seed(42)
        n_seq, max_seq_len, n_head_kv, head_dim, block_size = 4, 32, 2, 8, 8

        # Allocate both IREE and oracle caches
        alloc_args = VmVariantList(5)
        alloc_args.push_int(n_seq)
        alloc_args.push_int(max_seq_len)
        alloc_args.push_int(n_head_kv)
        alloc_args.push_int(head_dim)
        alloc_args.push_int(block_size)
        alloc_func = kvcache_module.lookup_function("allocate")
        alloc_results = VmVariantList(1)
        kvcache_module._context.invoke(alloc_func, alloc_args, alloc_results)
        cache = alloc_results.get_as_list(0)

        oracle_cache = allocate_oracle(n_seq, max_seq_len, n_head_kv, head_dim, block_size)

        # Fill sequence 0 with random data
        context_len = 10
        all_k = []
        all_v = []
        for pos in range(context_len):
            seq_ids_list = create_i32_list(kvcache_module, [0])
            positions_list = create_i32_list(kvcache_module, [pos])
            new_k = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)
            new_v = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)
            all_k.append(new_k[0])
            all_v.append(new_v[0])

            # IREE scatter
            scatter_args = VmVariantList(5)
            scatter_args.push_list(cache)
            scatter_args.push_list(seq_ids_list)
            scatter_args.push_list(positions_list)
            scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
            scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
            scatter_func = kvcache_module.lookup_function("scatter_decode")
            scatter_results = VmVariantList(0)
            kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)

            # Oracle scatter
            scatter_decode_oracle(oracle_cache, np.array([0]), np.array([pos]), new_k, new_v)

        expected_k = np.stack(all_k)
        expected_v = np.stack(all_v)

        # Gather from IREE
        gather_seq_ids = create_i32_list(kvcache_module, [0])
        gather_context_lens = create_i32_list(kvcache_module, [context_len])
        total_tokens = context_len

        gather_args = VmVariantList(4)
        gather_args.push_list(cache)
        gather_args.push_list(gather_seq_ids)
        gather_args.push_list(gather_context_lens)
        gather_args.push_int(total_tokens)
        gather_func = kvcache_module.lookup_function("gather")
        gather_results = VmVariantList(2)
        kvcache_module._context.invoke(gather_func, gather_args, gather_results)

        k_gathered = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(0, HalBufferView)
        )
        v_gathered = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(1, HalBufferView)
        )

        # Gather from oracle
        oracle_k, oracle_v = gather_oracle(oracle_cache, np.array([0]), np.array([context_len]))

        # Verify IREE matches oracle
        assert_close(k_gathered, oracle_k)
        assert_close(v_gathered, oracle_v)

        # Verify both match expected data
        assert_close(k_gathered, expected_k)
        assert_close(v_gathered, expected_v)

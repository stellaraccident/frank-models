"""Tests for Canonical Paged KV Cache component.

Tests the canonical KV cache with device-side control data (block_tables,
context_lens, block_indices, pos_in_blocks as tensors, not host lists).

Physical cache layout (block pool):
  K/V: [n_blocks, block_size, n_head_kv, head_dim]

Interface:
  allocate(n_blocks, block_size, n_head_kv, head_dim) -> cache
  gather(cache, block_tables, context_lens, max_context_len) -> (K, V)
  scatter_decode(cache, new_k, new_v, block_indices, pos_in_blocks) -> cache
"""

import numpy as np
import pytest

from iree.runtime import VmVariantList, HalBufferView
from tests.utils import compile_component, assert_close
from oracles.kvcache import (
    allocate as allocate_oracle,
    gather as gather_oracle,
    scatter_decode as scatter_decode_oracle,
)


@pytest.fixture(scope="module")
def kvcache_module(iree_cfg):
    """Compile the canonical KV cache component once per test module."""
    return compile_component("kvcache/kvcache.mlir", iree_cfg)


class TestAllocate:
    """Tests for allocate operation."""

    def test_basic(self, kvcache_module):
        """Test that allocate creates a cache with correct dimensions."""
        n_blocks, block_size, n_head_kv, head_dim = 8, 16, 4, 32

        args = VmVariantList(4)
        args.push_int(n_blocks)
        args.push_int(block_size)
        args.push_int(n_head_kv)
        args.push_int(head_dim)

        func = kvcache_module.lookup_function("allocate")
        results = VmVariantList(1)
        kvcache_module._context.invoke(func, args, results)

        # Should return a list with 2 elements
        cache = results.get_as_list(0)
        assert cache is not None
        assert len(cache) == 2

        # Check K cache dimensions [n_blocks, block_size, n_head_kv, head_dim]
        k_bv = cache.get_as_object(0, HalBufferView)
        k_arr = kvcache_module._buffer_view_to_numpy(k_bv)
        assert k_arr.shape == (n_blocks, block_size, n_head_kv, head_dim)

        # Check V cache dimensions
        v_bv = cache.get_as_object(1, HalBufferView)
        v_arr = kvcache_module._buffer_view_to_numpy(v_bv)
        assert v_arr.shape == (n_blocks, block_size, n_head_kv, head_dim)


class TestGather:
    """Tests for gather operation with device-side block tables and context lengths."""

    def test_basic(self, kvcache_module):
        """Test basic gather with block tables and context lengths."""
        np.random.seed(42)
        n_blocks, block_size, n_head_kv, head_dim = 8, 4, 2, 8
        batch = 2
        max_blocks_per_seq = 4
        max_context_len = 12  # 3 blocks worth

        # Create block tables: seq 0 uses blocks [0,1,2], seq 1 uses blocks [3,4,5]
        block_tables = np.array(
            [[0, 1, 2, 0], [3, 4, 5, 0]], dtype=np.int32
        )  # [batch, max_blocks]
        context_lens = np.array([10, 8], dtype=np.int32)  # [batch]

        # Allocate and fill cache with known data
        k_oracle, v_oracle = allocate_oracle(n_blocks, block_size, n_head_kv, head_dim)

        # Fill blocks 0-5 with random data
        for blk in range(6):
            k_oracle[blk, :, :, :] = np.random.randn(
                block_size, n_head_kv, head_dim
            ).astype(np.float32)
            v_oracle[blk, :, :, :] = np.random.randn(
                block_size, n_head_kv, head_dim
            ).astype(np.float32)

        # Allocate IREE cache
        alloc_args = VmVariantList(4)
        alloc_args.push_int(n_blocks)
        alloc_args.push_int(block_size)
        alloc_args.push_int(n_head_kv)
        alloc_args.push_int(head_dim)
        alloc_func = kvcache_module.lookup_function("allocate")
        alloc_results = VmVariantList(1)
        kvcache_module._context.invoke(alloc_func, alloc_args, alloc_results)
        cache = alloc_results.get_as_list(0)

        # Fill IREE cache using scatter_decode (fill block by block)
        scatter_func = kvcache_module.lookup_function("scatter_decode")
        for blk in range(6):
            for pos in range(block_size):
                new_k = k_oracle[blk : blk + 1, pos, :, :].reshape(1, n_head_kv, head_dim)
                new_v = v_oracle[blk : blk + 1, pos, :, :].reshape(1, n_head_kv, head_dim)
                blk_indices = np.array([blk], dtype=np.int32)
                pos_indices = np.array([pos], dtype=np.int32)

                scatter_args = VmVariantList(5)
                scatter_args.push_list(cache)
                scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
                scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
                scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(blk_indices))
                scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(pos_indices))
                scatter_results = VmVariantList(1)
                kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)
                cache = scatter_results.get_as_list(0)

        # Gather using block tables and context lengths
        gather_args = VmVariantList(4)
        gather_args.push_list(cache)
        gather_args.push_ref(kvcache_module._numpy_to_buffer_view(block_tables))
        gather_args.push_ref(kvcache_module._numpy_to_buffer_view(context_lens))
        gather_args.push_int(max_context_len)
        gather_func = kvcache_module.lookup_function("gather")
        gather_results = VmVariantList(2)
        kvcache_module._context.invoke(gather_func, gather_args, gather_results)

        # Extract results [batch, max_context_len, n_head_kv, head_dim]
        k_result = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(0, HalBufferView)
        )
        v_result = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(1, HalBufferView)
        )

        # Check shapes
        assert k_result.shape == (batch, max_context_len, n_head_kv, head_dim)
        assert v_result.shape == (batch, max_context_len, n_head_kv, head_dim)

        # Compare against oracle
        k_expected = gather_oracle(k_oracle, block_tables, context_lens, max_context_len)
        v_expected = gather_oracle(v_oracle, block_tables, context_lens, max_context_len)

        assert_close(k_result, k_expected)
        assert_close(v_result, v_expected)

    def test_variable_context_lens(self, kvcache_module):
        """Test gather with different context lengths per sequence."""
        np.random.seed(123)
        n_blocks, block_size, n_head_kv, head_dim = 16, 4, 2, 8
        batch = 3
        max_context_len = 12

        # Different context lengths: 5, 9, 3
        context_lens = np.array([5, 9, 3], dtype=np.int32)

        # Block tables: each sequence uses different blocks
        block_tables = np.array(
            [
                [0, 1, 2, 0],  # seq 0: blocks 0,1,2
                [3, 4, 5, 0],  # seq 1: blocks 3,4,5
                [6, 7, 8, 0],  # seq 2: blocks 6,7,8
            ],
            dtype=np.int32,
        )

        # Allocate and fill cache
        k_oracle, v_oracle = allocate_oracle(n_blocks, block_size, n_head_kv, head_dim)
        for blk in range(9):
            k_oracle[blk, :, :, :] = np.random.randn(
                block_size, n_head_kv, head_dim
            ).astype(np.float32)
            v_oracle[blk, :, :, :] = np.random.randn(
                block_size, n_head_kv, head_dim
            ).astype(np.float32)

        # Allocate IREE cache
        alloc_args = VmVariantList(4)
        alloc_args.push_int(n_blocks)
        alloc_args.push_int(block_size)
        alloc_args.push_int(n_head_kv)
        alloc_args.push_int(head_dim)
        alloc_func = kvcache_module.lookup_function("allocate")
        alloc_results = VmVariantList(1)
        kvcache_module._context.invoke(alloc_func, alloc_args, alloc_results)
        cache = alloc_results.get_as_list(0)

        # Fill IREE cache
        scatter_func = kvcache_module.lookup_function("scatter_decode")
        for blk in range(9):
            for pos in range(block_size):
                new_k = k_oracle[blk : blk + 1, pos, :, :].reshape(1, n_head_kv, head_dim)
                new_v = v_oracle[blk : blk + 1, pos, :, :].reshape(1, n_head_kv, head_dim)
                blk_indices = np.array([blk], dtype=np.int32)
                pos_indices = np.array([pos], dtype=np.int32)

                scatter_args = VmVariantList(5)
                scatter_args.push_list(cache)
                scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
                scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
                scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(blk_indices))
                scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(pos_indices))
                scatter_results = VmVariantList(1)
                kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)
                cache = scatter_results.get_as_list(0)

        # Gather
        gather_args = VmVariantList(4)
        gather_args.push_list(cache)
        gather_args.push_ref(kvcache_module._numpy_to_buffer_view(block_tables))
        gather_args.push_ref(kvcache_module._numpy_to_buffer_view(context_lens))
        gather_args.push_int(max_context_len)
        gather_func = kvcache_module.lookup_function("gather")
        gather_results = VmVariantList(2)
        kvcache_module._context.invoke(gather_func, gather_args, gather_results)

        k_result = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(0, HalBufferView)
        )

        # Oracle
        k_expected = gather_oracle(k_oracle, block_tables, context_lens, max_context_len)

        # Verify positions within context_len match
        for b in range(batch):
            ctx_len = context_lens[b]
            assert_close(k_result[b, :ctx_len, :, :], k_expected[b, :ctx_len, :, :])
            # Positions beyond context_len should be zero
            if ctx_len < max_context_len:
                assert np.allclose(k_result[b, ctx_len:, :, :], 0.0)


class TestScatterDecode:
    """Tests for scatter_decode operation (one token per sequence)."""

    def test_basic(self, kvcache_module):
        """Test basic scatter with one token per sequence."""
        np.random.seed(42)
        n_blocks, block_size, n_head_kv, head_dim = 8, 4, 2, 8
        batch_size = 3

        # Each sequence writes to a different block at a different position
        block_indices = np.array([0, 2, 5], dtype=np.int32)
        pos_in_blocks = np.array([3, 1, 0], dtype=np.int32)

        # Allocate cache
        alloc_args = VmVariantList(4)
        alloc_args.push_int(n_blocks)
        alloc_args.push_int(block_size)
        alloc_args.push_int(n_head_kv)
        alloc_args.push_int(head_dim)
        alloc_func = kvcache_module.lookup_function("allocate")
        alloc_results = VmVariantList(1)
        kvcache_module._context.invoke(alloc_func, alloc_args, alloc_results)
        cache = alloc_results.get_as_list(0)

        # Create new K/V data [batch, n_head_kv, head_dim] - ONE token per sequence
        new_k = np.random.randn(batch_size, n_head_kv, head_dim).astype(np.float32)
        new_v = np.random.randn(batch_size, n_head_kv, head_dim).astype(np.float32)

        # Scatter
        scatter_args = VmVariantList(5)
        scatter_args.push_list(cache)
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(block_indices))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(pos_in_blocks))
        scatter_func = kvcache_module.lookup_function("scatter_decode")
        scatter_results = VmVariantList(1)
        kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)

        # Get updated cache
        updated_cache = scatter_results.get_as_list(0)
        k_updated = kvcache_module._buffer_view_to_numpy(
            updated_cache.get_as_object(0, HalBufferView)
        )
        v_updated = kvcache_module._buffer_view_to_numpy(
            updated_cache.get_as_object(1, HalBufferView)
        )

        # Oracle
        k_cache_init, v_cache_init = allocate_oracle(
            n_blocks, block_size, n_head_kv, head_dim
        )
        k_expected = scatter_decode_oracle(
            k_cache_init, new_k, block_indices, pos_in_blocks
        )
        v_expected = scatter_decode_oracle(
            v_cache_init, new_v, block_indices, pos_in_blocks
        )

        assert_close(k_updated, k_expected)
        assert_close(v_updated, v_expected)


class TestRoundTrip:
    """Tests for scatter then gather (decode simulation)."""

    def test_scatter_then_gather(self, kvcache_module):
        """Test that scatter followed by gather recovers the data correctly."""
        np.random.seed(42)
        n_blocks, block_size, n_head_kv, head_dim = 16, 4, 4, 16
        batch = 2

        # Allocate
        alloc_args = VmVariantList(4)
        alloc_args.push_int(n_blocks)
        alloc_args.push_int(block_size)
        alloc_args.push_int(n_head_kv)
        alloc_args.push_int(head_dim)
        alloc_func = kvcache_module.lookup_function("allocate")
        alloc_results = VmVariantList(1)
        kvcache_module._context.invoke(alloc_func, alloc_args, alloc_results)
        cache = alloc_results.get_as_list(0)

        # Track what we've written for verification
        k_oracle, v_oracle = allocate_oracle(n_blocks, block_size, n_head_kv, head_dim)

        # Simulate filling cache: seq 0 gets blocks [0,1], seq 1 gets blocks [2,3]
        scatter_func = kvcache_module.lookup_function("scatter_decode")

        # Fill sequence 0: 6 tokens in blocks 0, 1
        for tok_idx in range(6):
            blk = tok_idx // block_size
            pos = tok_idx % block_size
            new_k = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)
            new_v = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)

            # Update oracle
            k_oracle[blk, pos, :, :] = new_k[0]
            v_oracle[blk, pos, :, :] = new_v[0]

            scatter_args = VmVariantList(5)
            scatter_args.push_list(cache)
            scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
            scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
            scatter_args.push_ref(
                kvcache_module._numpy_to_buffer_view(np.array([blk], dtype=np.int32))
            )
            scatter_args.push_ref(
                kvcache_module._numpy_to_buffer_view(np.array([pos], dtype=np.int32))
            )
            scatter_results = VmVariantList(1)
            kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)
            cache = scatter_results.get_as_list(0)

        # Fill sequence 1: 5 tokens in blocks 2, 3
        for tok_idx in range(5):
            blk = 2 + tok_idx // block_size
            pos = tok_idx % block_size
            new_k = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)
            new_v = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)

            k_oracle[blk, pos, :, :] = new_k[0]
            v_oracle[blk, pos, :, :] = new_v[0]

            scatter_args = VmVariantList(5)
            scatter_args.push_list(cache)
            scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
            scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
            scatter_args.push_ref(
                kvcache_module._numpy_to_buffer_view(np.array([blk], dtype=np.int32))
            )
            scatter_args.push_ref(
                kvcache_module._numpy_to_buffer_view(np.array([pos], dtype=np.int32))
            )
            scatter_results = VmVariantList(1)
            kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)
            cache = scatter_results.get_as_list(0)

        # Now gather using block tables
        block_tables = np.array(
            [
                [0, 1, 0, 0],  # seq 0 uses blocks 0, 1
                [2, 3, 0, 0],  # seq 1 uses blocks 2, 3
            ],
            dtype=np.int32,
        )
        context_lens = np.array([6, 5], dtype=np.int32)
        max_context_len = 8

        gather_args = VmVariantList(4)
        gather_args.push_list(cache)
        gather_args.push_ref(kvcache_module._numpy_to_buffer_view(block_tables))
        gather_args.push_ref(kvcache_module._numpy_to_buffer_view(context_lens))
        gather_args.push_int(max_context_len)
        gather_func = kvcache_module.lookup_function("gather")
        gather_results = VmVariantList(2)
        kvcache_module._context.invoke(gather_func, gather_args, gather_results)

        k_gathered = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(0, HalBufferView)
        )
        v_gathered = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(1, HalBufferView)
        )

        # Compare against oracle
        k_expected = gather_oracle(k_oracle, block_tables, context_lens, max_context_len)
        v_expected = gather_oracle(v_oracle, block_tables, context_lens, max_context_len)

        assert_close(k_gathered, k_expected)
        assert_close(v_gathered, v_expected)

    def test_decode_step(self, kvcache_module):
        """Test a realistic decode step: existing context + one new token per sequence."""
        np.random.seed(123)
        n_blocks, block_size, n_head_kv, head_dim = 16, 4, 2, 8
        batch = 2

        # Allocate
        alloc_args = VmVariantList(4)
        alloc_args.push_int(n_blocks)
        alloc_args.push_int(block_size)
        alloc_args.push_int(n_head_kv)
        alloc_args.push_int(head_dim)
        alloc_func = kvcache_module.lookup_function("allocate")
        alloc_results = VmVariantList(1)
        kvcache_module._context.invoke(alloc_func, alloc_args, alloc_results)
        cache = alloc_results.get_as_list(0)

        k_oracle, v_oracle = allocate_oracle(n_blocks, block_size, n_head_kv, head_dim)
        scatter_func = kvcache_module.lookup_function("scatter_decode")

        # Fill existing context for seq 0: 6 tokens (blocks 0, 1; positions 0-3, 0-1)
        for tok_idx in range(6):
            blk = tok_idx // block_size
            pos = tok_idx % block_size
            new_k = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)
            new_v = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)
            k_oracle[blk, pos, :, :] = new_k[0]
            v_oracle[blk, pos, :, :] = new_v[0]

            scatter_args = VmVariantList(5)
            scatter_args.push_list(cache)
            scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
            scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
            scatter_args.push_ref(
                kvcache_module._numpy_to_buffer_view(np.array([blk], dtype=np.int32))
            )
            scatter_args.push_ref(
                kvcache_module._numpy_to_buffer_view(np.array([pos], dtype=np.int32))
            )
            scatter_results = VmVariantList(1)
            kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)
            cache = scatter_results.get_as_list(0)

        # Fill existing context for seq 1: 5 tokens (blocks 2, 3; positions 0-3, 0)
        for tok_idx in range(5):
            blk = 2 + tok_idx // block_size
            pos = tok_idx % block_size
            new_k = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)
            new_v = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)
            k_oracle[blk, pos, :, :] = new_k[0]
            v_oracle[blk, pos, :, :] = new_v[0]

            scatter_args = VmVariantList(5)
            scatter_args.push_list(cache)
            scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
            scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
            scatter_args.push_ref(
                kvcache_module._numpy_to_buffer_view(np.array([blk], dtype=np.int32))
            )
            scatter_args.push_ref(
                kvcache_module._numpy_to_buffer_view(np.array([pos], dtype=np.int32))
            )
            scatter_results = VmVariantList(1)
            kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)
            cache = scatter_results.get_as_list(0)

        # Decode step: add one new token to each sequence
        # Seq 0: add at position 6 -> block 1, pos 2
        # Seq 1: add at position 5 -> block 3, pos 1
        new_k_decode = np.random.randn(batch, n_head_kv, head_dim).astype(np.float32)
        new_v_decode = np.random.randn(batch, n_head_kv, head_dim).astype(np.float32)
        decode_block_indices = np.array([1, 3], dtype=np.int32)
        decode_pos_in_blocks = np.array([2, 1], dtype=np.int32)

        # Update oracle
        k_oracle[1, 2, :, :] = new_k_decode[0]
        v_oracle[1, 2, :, :] = new_v_decode[0]
        k_oracle[3, 1, :, :] = new_k_decode[1]
        v_oracle[3, 1, :, :] = new_v_decode[1]

        scatter_args = VmVariantList(5)
        scatter_args.push_list(cache)
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k_decode))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v_decode))
        scatter_args.push_ref(
            kvcache_module._numpy_to_buffer_view(decode_block_indices)
        )
        scatter_args.push_ref(
            kvcache_module._numpy_to_buffer_view(decode_pos_in_blocks)
        )
        scatter_results = VmVariantList(1)
        kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)
        cache = scatter_results.get_as_list(0)

        # Gather with updated context lengths (7, 6)
        block_tables = np.array(
            [
                [0, 1, 0, 0],  # seq 0
                [2, 3, 0, 0],  # seq 1
            ],
            dtype=np.int32,
        )
        context_lens = np.array([7, 6], dtype=np.int32)  # After adding one token each
        max_context_len = 8

        gather_args = VmVariantList(4)
        gather_args.push_list(cache)
        gather_args.push_ref(kvcache_module._numpy_to_buffer_view(block_tables))
        gather_args.push_ref(kvcache_module._numpy_to_buffer_view(context_lens))
        gather_args.push_int(max_context_len)
        gather_func = kvcache_module.lookup_function("gather")
        gather_results = VmVariantList(2)
        kvcache_module._context.invoke(gather_func, gather_args, gather_results)

        k_gathered = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(0, HalBufferView)
        )

        k_expected = gather_oracle(k_oracle, block_tables, context_lens, max_context_len)

        # Verify the newly added tokens are present
        # Seq 0, position 6 should be new_k_decode[0]
        assert_close(k_gathered[0, 6, :, :], new_k_decode[0])
        # Seq 1, position 5 should be new_k_decode[1]
        assert_close(k_gathered[1, 5, :, :], new_k_decode[1])

        # Full comparison
        assert_close(k_gathered, k_expected)


class TestBlockTableIndirection:
    """Tests for non-sequential block assignments."""

    def test_non_sequential_blocks(self, kvcache_module):
        """Test gather with non-sequential block assignments (e.g., seq 0 uses [3, 7, 1])."""
        np.random.seed(456)
        n_blocks, block_size, n_head_kv, head_dim = 16, 4, 2, 8
        batch = 2

        # Non-sequential block assignments
        # Seq 0: uses blocks [3, 7, 1] (in that order)
        # Seq 1: uses blocks [5, 2, 9]
        block_tables = np.array(
            [
                [3, 7, 1, 0],
                [5, 2, 9, 0],
            ],
            dtype=np.int32,
        )
        context_lens = np.array([10, 8], dtype=np.int32)
        max_context_len = 12

        # Allocate
        alloc_args = VmVariantList(4)
        alloc_args.push_int(n_blocks)
        alloc_args.push_int(block_size)
        alloc_args.push_int(n_head_kv)
        alloc_args.push_int(head_dim)
        alloc_func = kvcache_module.lookup_function("allocate")
        alloc_results = VmVariantList(1)
        kvcache_module._context.invoke(alloc_func, alloc_args, alloc_results)
        cache = alloc_results.get_as_list(0)

        k_oracle, v_oracle = allocate_oracle(n_blocks, block_size, n_head_kv, head_dim)

        # Fill the specific blocks used by each sequence
        scatter_func = kvcache_module.lookup_function("scatter_decode")
        for blk in [1, 2, 3, 5, 7, 9]:  # All blocks used
            for pos in range(block_size):
                new_k = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)
                new_v = np.random.randn(1, n_head_kv, head_dim).astype(np.float32)
                k_oracle[blk, pos, :, :] = new_k[0]
                v_oracle[blk, pos, :, :] = new_v[0]

                scatter_args = VmVariantList(5)
                scatter_args.push_list(cache)
                scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
                scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
                scatter_args.push_ref(
                    kvcache_module._numpy_to_buffer_view(np.array([blk], dtype=np.int32))
                )
                scatter_args.push_ref(
                    kvcache_module._numpy_to_buffer_view(np.array([pos], dtype=np.int32))
                )
                scatter_results = VmVariantList(1)
                kvcache_module._context.invoke(
                    scatter_func, scatter_args, scatter_results
                )
                cache = scatter_results.get_as_list(0)

        # Gather
        gather_args = VmVariantList(4)
        gather_args.push_list(cache)
        gather_args.push_ref(kvcache_module._numpy_to_buffer_view(block_tables))
        gather_args.push_ref(kvcache_module._numpy_to_buffer_view(context_lens))
        gather_args.push_int(max_context_len)
        gather_func = kvcache_module.lookup_function("gather")
        gather_results = VmVariantList(2)
        kvcache_module._context.invoke(gather_func, gather_args, gather_results)

        k_gathered = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(0, HalBufferView)
        )

        k_expected = gather_oracle(k_oracle, block_tables, context_lens, max_context_len)

        # Verify indirection is working: seq 0, position 0 should come from block 3, pos 0
        assert_close(k_gathered[0, 0, :, :], k_oracle[3, 0, :, :])
        # seq 0, position 4 should come from block 7, pos 0
        assert_close(k_gathered[0, 4, :, :], k_oracle[7, 0, :, :])
        # seq 0, position 8 should come from block 1, pos 0
        assert_close(k_gathered[0, 8, :, :], k_oracle[1, 0, :, :])

        # Full comparison
        assert_close(k_gathered, k_expected)

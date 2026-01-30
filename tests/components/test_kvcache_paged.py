"""Tests for Paged KV Cache component.

Tests the paged KV cache with flat block pool and indirect addressing.

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
import pytest

from iree.runtime import VmVariantList, HalBufferView
from tests.utils import compile_component, assert_close
from oracles.kvcache_paged import (
    allocate as allocate_oracle,
    gather as gather_oracle,
    scatter_single_token as scatter_single_token_oracle,
)


@pytest.fixture(scope="module")
def kvcache_module(iree_cfg):
    """Compile the paged KV cache component once per test module."""
    return compile_component("kvcache/kvcache_paged.mlir", iree_cfg)


class TestAllocate:
    """Tests for allocate operation."""

    def test_basic(self, kvcache_module):
        """Test that allocate creates a cache with correct dimensions."""
        n_blocks, block_size, n_head_kv, head_dim = 8, 16, 4, 32

        # Call allocate(n_blocks, block_size, n_head_kv, head_dim)
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
    """Tests for gather operation (flat context tokens)."""

    def test_basic(self, kvcache_module):
        """Test basic gather with flat block indices."""
        np.random.seed(42)
        n_blocks, block_size, n_head_kv, head_dim = 8, 4, 2, 8
        n_ctx_tokens = 12  # 3 blocks worth

        # Block indices [n_ctx_tokens] - flat
        # Tokens 0-3 in block 0, 4-7 in block 1, 8-11 in block 2
        block_indices = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)

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

        # Scatter some data first so we have something to gather
        # We'll scatter one token at a time to fill the cache
        batch_size = n_ctx_tokens  # Treat each token as a separate "sequence"
        new_k = np.random.randn(batch_size, n_head_kv, head_dim).astype(np.float32)
        new_v = np.random.randn(batch_size, n_head_kv, head_dim).astype(np.float32)

        # Position within block for each token
        pos_in_blocks = np.array([i % block_size for i in range(n_ctx_tokens)], dtype=np.int32)

        scatter_args = VmVariantList(5)
        scatter_args.push_list(cache)
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(block_indices))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(pos_in_blocks))
        scatter_func = kvcache_module.lookup_function("scatter_single_token")
        scatter_results = VmVariantList(1)
        kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)
        cache = scatter_results.get_as_list(0)

        # Gather
        gather_args = VmVariantList(3)
        gather_args.push_list(cache)
        gather_args.push_ref(kvcache_module._numpy_to_buffer_view(block_indices))
        gather_args.push_int(block_size)
        gather_func = kvcache_module.lookup_function("gather")
        gather_results = VmVariantList(2)
        kvcache_module._context.invoke(gather_func, gather_args, gather_results)

        # Extract results [n_ctx_tokens, n_head_kv, head_dim]
        k_result = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(0, HalBufferView)
        )
        v_result = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(1, HalBufferView)
        )

        # Check shapes
        assert k_result.shape == (n_ctx_tokens, n_head_kv, head_dim)
        assert v_result.shape == (n_ctx_tokens, n_head_kv, head_dim)

        # Should match what we scattered
        assert_close(k_result, new_k)
        assert_close(v_result, new_v)


class TestScatterSingleToken:
    """Tests for scatter_single_token operation (decode phase)."""

    def test_basic(self, kvcache_module):
        """Test basic scatter with one token per sequence."""
        np.random.seed(42)
        n_blocks, block_size, n_head_kv, head_dim = 8, 4, 2, 8
        batch_size = 3  # 3 sequences, each adding one token

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
        scatter_func = kvcache_module.lookup_function("scatter_single_token")
        scatter_results = VmVariantList(1)
        kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)

        # Get updated cache and extract K/V
        updated_cache = scatter_results.get_as_list(0)
        k_updated = kvcache_module._buffer_view_to_numpy(
            updated_cache.get_as_object(0, HalBufferView)
        )
        v_updated = kvcache_module._buffer_view_to_numpy(
            updated_cache.get_as_object(1, HalBufferView)
        )

        # Check against oracle
        k_cache_init, v_cache_init = allocate_oracle(n_blocks, block_size, n_head_kv, head_dim)
        k_expected = scatter_single_token_oracle(k_cache_init, new_k, block_indices, pos_in_blocks)
        v_expected = scatter_single_token_oracle(v_cache_init, new_v, block_indices, pos_in_blocks)

        assert_close(k_updated, k_expected)
        assert_close(v_updated, v_expected)


class TestRoundTrip:
    """Tests for scatter then gather (decode simulation)."""

    def test_scatter_then_gather(self, kvcache_module):
        """Test that scatter followed by gather recovers the data."""
        np.random.seed(42)
        n_blocks, block_size, n_head_kv, head_dim = 16, 8, 4, 16
        n_ctx_tokens = 24  # 3 blocks worth of context

        # Simulate: we have context in blocks 0, 1, 2
        # Block indices for gather [n_ctx_tokens]
        block_indices_gather = np.array(
            [0] * 8 + [1] * 8 + [2] * 8, dtype=np.int32
        )

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

        # Fill cache by scattering all tokens (simulating prefill via repeated decode)
        # Each token treated as separate "sequence" for this test
        new_k_all = np.random.randn(n_ctx_tokens, n_head_kv, head_dim).astype(np.float32)
        new_v_all = np.random.randn(n_ctx_tokens, n_head_kv, head_dim).astype(np.float32)
        pos_in_blocks_all = np.array([i % block_size for i in range(n_ctx_tokens)], dtype=np.int32)

        scatter_args = VmVariantList(5)
        scatter_args.push_list(cache)
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k_all))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v_all))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(block_indices_gather))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(pos_in_blocks_all))
        scatter_func = kvcache_module.lookup_function("scatter_single_token")
        scatter_results = VmVariantList(1)
        kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)
        updated_cache = scatter_results.get_as_list(0)

        # Gather
        gather_args = VmVariantList(3)
        gather_args.push_list(updated_cache)
        gather_args.push_ref(kvcache_module._numpy_to_buffer_view(block_indices_gather))
        gather_args.push_int(block_size)
        gather_func = kvcache_module.lookup_function("gather")
        gather_results = VmVariantList(2)
        kvcache_module._context.invoke(gather_func, gather_args, gather_results)

        k_gathered = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(0, HalBufferView)
        )
        v_gathered = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(1, HalBufferView)
        )

        # Should match original data
        assert_close(k_gathered, new_k_all)
        assert_close(v_gathered, new_v_all)

    def test_decode_step(self, kvcache_module):
        """Test a realistic decode step: existing context + one new token per sequence."""
        np.random.seed(123)
        n_blocks, block_size, n_head_kv, head_dim = 16, 4, 2, 8
        batch_size = 2  # 2 sequences

        # Existing context: seq 0 has 6 tokens (blocks 0,1), seq 1 has 5 tokens (blocks 2,3)
        # We'll set up by scattering existing context first

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

        # Fill existing context for seq 0: 6 tokens in blocks 0, 1
        ctx_k_0 = np.random.randn(6, n_head_kv, head_dim).astype(np.float32)
        ctx_v_0 = np.random.randn(6, n_head_kv, head_dim).astype(np.float32)
        ctx_blocks_0 = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
        ctx_pos_0 = np.array([0, 1, 2, 3, 0, 1], dtype=np.int32)

        scatter_args = VmVariantList(5)
        scatter_args.push_list(cache)
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(ctx_k_0))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(ctx_v_0))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(ctx_blocks_0))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(ctx_pos_0))
        scatter_func = kvcache_module.lookup_function("scatter_single_token")
        scatter_results = VmVariantList(1)
        kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)
        cache = scatter_results.get_as_list(0)

        # Fill existing context for seq 1: 5 tokens in blocks 2, 3
        ctx_k_1 = np.random.randn(5, n_head_kv, head_dim).astype(np.float32)
        ctx_v_1 = np.random.randn(5, n_head_kv, head_dim).astype(np.float32)
        ctx_blocks_1 = np.array([2, 2, 2, 2, 3], dtype=np.int32)
        ctx_pos_1 = np.array([0, 1, 2, 3, 0], dtype=np.int32)

        scatter_args = VmVariantList(5)
        scatter_args.push_list(cache)
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(ctx_k_1))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(ctx_v_1))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(ctx_blocks_1))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(ctx_pos_1))
        scatter_results = VmVariantList(1)
        kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)
        cache = scatter_results.get_as_list(0)

        # Now do a decode step: add one new token to each sequence
        # Seq 0: add at block 1, pos 2 (7th token)
        # Seq 1: add at block 3, pos 1 (6th token)
        new_k = np.random.randn(batch_size, n_head_kv, head_dim).astype(np.float32)
        new_v = np.random.randn(batch_size, n_head_kv, head_dim).astype(np.float32)
        new_block_indices = np.array([1, 3], dtype=np.int32)
        new_pos_in_blocks = np.array([2, 1], dtype=np.int32)

        scatter_args = VmVariantList(5)
        scatter_args.push_list(cache)
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_block_indices))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_pos_in_blocks))
        scatter_results = VmVariantList(1)
        kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)
        updated_cache = scatter_results.get_as_list(0)

        # Verify: gather the new token positions and check they match
        # We'll gather just the two new positions
        verify_blocks = np.array([1, 3], dtype=np.int32)  # The blocks we wrote to

        gather_args = VmVariantList(3)
        gather_args.push_list(updated_cache)
        gather_args.push_ref(kvcache_module._numpy_to_buffer_view(verify_blocks))
        gather_args.push_int(block_size)
        gather_func = kvcache_module.lookup_function("gather")
        gather_results = VmVariantList(2)
        kvcache_module._context.invoke(gather_func, gather_args, gather_results)

        k_gathered = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(0, HalBufferView)
        )

        # Position 0 in gather corresponds to block 1, pos 0 (ctx_k_0[4])
        # Position 1 in gather corresponds to block 3, pos 1 (new_k[1])
        # Because gather uses ctx_idx % block_size for position

        # Check that the new token for seq 1 (index 1) is at gathered position 1
        # gathered[1] should equal new_k[1] since gather[1] reads from block 3, pos 1
        assert_close(k_gathered[1], new_k[1])

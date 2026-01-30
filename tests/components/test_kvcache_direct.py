"""Tests for KV Cache component.

Tests the batched direct KV-cache using !util.list<?> encapsulation.

Dimension layout (all tensors):
  dim 0: batch_size   - Number of sequences in the batch
  dim 1: seq_len      - Sequence position (max_seq_len for cache storage)
  dim 2: n_head_kv    - Number of KV heads
  dim 3: head_dim     - Dimension per attention head
"""

import numpy as np
import pytest

from iree.runtime import VmVariantList, HalBufferView
from tests.utils import compile_component, assert_close
from oracles.kvcache_direct import allocate as allocate_oracle, gather as gather_oracle, scatter as scatter_oracle


@pytest.fixture(scope="module")
def kvcache_module(iree_cfg):
    """Compile the KV cache component once per test module."""
    return compile_component("kvcache/kvcache_direct.mlir", iree_cfg)


class TestAllocate:
    """Tests for allocate operation."""

    def test_basic(self, kvcache_module):
        """Test that allocate creates a cache list with correct dimensions."""
        batch_size, max_seq_len, n_head_kv, head_dim = 2, 16, 4, 32

        # Call allocate(batch_size, max_seq_len, n_head_kv, head_dim)
        args = VmVariantList(4)
        args.push_int(batch_size)
        args.push_int(max_seq_len)
        args.push_int(n_head_kv)
        args.push_int(head_dim)

        func = kvcache_module.lookup_function("allocate")
        results = VmVariantList(1)
        kvcache_module._context.invoke(func, args, results)

        # Should return a list with 2 elements
        cache = results.get_as_list(0)
        assert cache is not None
        assert len(cache) == 2

        # Check K cache dimensions [batch_size, max_seq_len, n_head_kv, head_dim]
        k_bv = cache.get_as_object(0, HalBufferView)
        k_arr = kvcache_module._buffer_view_to_numpy(k_bv)
        assert k_arr.shape == (batch_size, max_seq_len, n_head_kv, head_dim)
        assert np.allclose(k_arr, 0.0)  # Should be zero-initialized

        # Check V cache dimensions
        v_bv = cache.get_as_object(1, HalBufferView)
        v_arr = kvcache_module._buffer_view_to_numpy(v_bv)
        assert v_arr.shape == (batch_size, max_seq_len, n_head_kv, head_dim)
        assert np.allclose(v_arr, 0.0)


class TestGather:
    """Tests for gather operation."""

    def test_basic(self, kvcache_module):
        """Test basic gather from batched cache."""
        np.random.seed(42)
        batch_size, max_seq_len, n_head_kv, head_dim = 2, 16, 4, 32
        seq_len = 8

        # Allocate cache
        alloc_args = VmVariantList(4)
        alloc_args.push_int(batch_size)
        alloc_args.push_int(max_seq_len)
        alloc_args.push_int(n_head_kv)
        alloc_args.push_int(head_dim)
        alloc_func = kvcache_module.lookup_function("allocate")
        alloc_results = VmVariantList(1)
        kvcache_module._context.invoke(alloc_func, alloc_args, alloc_results)
        cache = alloc_results.get_as_list(0)

        # Scatter some data first so we have something to gather
        new_k = np.random.randn(batch_size, seq_len, n_head_kv, head_dim).astype(np.float32)
        new_v = np.random.randn(batch_size, seq_len, n_head_kv, head_dim).astype(np.float32)

        scatter_args = VmVariantList(4)
        scatter_args.push_list(cache)
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
        scatter_args.push_int(0)  # write_pos
        scatter_func = kvcache_module.lookup_function("scatter")
        scatter_results = VmVariantList(1)
        kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)
        cache = scatter_results.get_as_list(0)

        # Gather
        gather_args = VmVariantList(2)
        gather_args.push_list(cache)
        gather_args.push_int(seq_len)
        gather_func = kvcache_module.lookup_function("gather")
        gather_results = VmVariantList(2)
        kvcache_module._context.invoke(gather_func, gather_args, gather_results)

        # Extract results [batch_size, seq_len, n_head_kv, head_dim]
        k_result = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(0, HalBufferView)
        )
        v_result = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(1, HalBufferView)
        )

        # Check shapes
        assert k_result.shape == (batch_size, seq_len, n_head_kv, head_dim)
        assert v_result.shape == (batch_size, seq_len, n_head_kv, head_dim)

        # Should match what we scattered
        assert_close(k_result, new_k)
        assert_close(v_result, new_v)


class TestScatter:
    """Tests for scatter operation."""

    def test_basic(self, kvcache_module):
        """Test basic scatter to batched cache."""
        np.random.seed(42)
        batch_size, max_seq_len, n_head_kv, head_dim = 2, 16, 4, 32
        new_tokens = 4
        write_pos = 0

        # Allocate cache
        alloc_args = VmVariantList(4)
        alloc_args.push_int(batch_size)
        alloc_args.push_int(max_seq_len)
        alloc_args.push_int(n_head_kv)
        alloc_args.push_int(head_dim)
        alloc_func = kvcache_module.lookup_function("allocate")
        alloc_results = VmVariantList(1)
        kvcache_module._context.invoke(alloc_func, alloc_args, alloc_results)
        cache = alloc_results.get_as_list(0)

        # Create new K/V data [batch_size, new_tokens, n_head_kv, head_dim]
        new_k = np.random.randn(batch_size, new_tokens, n_head_kv, head_dim).astype(np.float32)
        new_v = np.random.randn(batch_size, new_tokens, n_head_kv, head_dim).astype(np.float32)

        # Scatter
        scatter_args = VmVariantList(4)
        scatter_args.push_list(cache)
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
        scatter_args.push_int(write_pos)
        scatter_func = kvcache_module.lookup_function("scatter")
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
        k_cache_init, v_cache_init = allocate_oracle(batch_size, max_seq_len, n_head_kv, head_dim)
        k_expected = scatter_oracle(k_cache_init, new_k, write_pos)
        v_expected = scatter_oracle(v_cache_init, new_v, write_pos)

        assert_close(k_updated, k_expected)
        assert_close(v_updated, v_expected)


class TestRoundTrip:
    """Tests for scatter then gather."""

    def test_scatter_then_gather(self, kvcache_module):
        """Test that scatter followed by gather recovers the data."""
        np.random.seed(42)
        batch_size, max_seq_len, n_head_kv, head_dim = 4, 32, 4, 64
        new_tokens = 8

        # Allocate
        alloc_args = VmVariantList(4)
        alloc_args.push_int(batch_size)
        alloc_args.push_int(max_seq_len)
        alloc_args.push_int(n_head_kv)
        alloc_args.push_int(head_dim)
        alloc_func = kvcache_module.lookup_function("allocate")
        alloc_results = VmVariantList(1)
        kvcache_module._context.invoke(alloc_func, alloc_args, alloc_results)
        cache = alloc_results.get_as_list(0)

        # Create data [batch_size, new_tokens, n_head_kv, head_dim]
        new_k = np.random.randn(batch_size, new_tokens, n_head_kv, head_dim).astype(np.float32)
        new_v = np.random.randn(batch_size, new_tokens, n_head_kv, head_dim).astype(np.float32)

        # Scatter at position 0
        scatter_args = VmVariantList(4)
        scatter_args.push_list(cache)
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_k))
        scatter_args.push_ref(kvcache_module._numpy_to_buffer_view(new_v))
        scatter_args.push_int(0)
        scatter_func = kvcache_module.lookup_function("scatter")
        scatter_results = VmVariantList(1)
        kvcache_module._context.invoke(scatter_func, scatter_args, scatter_results)
        updated_cache = scatter_results.get_as_list(0)

        # Gather
        gather_args = VmVariantList(2)
        gather_args.push_list(updated_cache)
        gather_args.push_int(new_tokens)
        gather_func = kvcache_module.lookup_function("gather")
        gather_results = VmVariantList(2)
        kvcache_module._context.invoke(gather_func, gather_args, gather_results)

        k_gathered = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(0, HalBufferView)
        )
        v_gathered = kvcache_module._buffer_view_to_numpy(
            gather_results.get_as_object(1, HalBufferView)
        )

        # Should match original new_k, new_v
        assert_close(k_gathered, new_k)
        assert_close(v_gathered, new_v)

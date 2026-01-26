"""Tests for RoPE (Rotary Position Embeddings) component."""

import numpy as np
import pytest

from tests.utils import compile_component, assert_close
from oracles.position import rope as rope_oracle


@pytest.fixture(scope="module")
def rope_module(rt):
    return compile_component("position/rope.mlir", rt)


def test_basic_correctness(rope_module):
    """Test RoPE against oracle with random data."""
    np.random.seed(42)
    batch, seq_len, n_head, head_dim = 2, 4, 8, 16

    input = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32)
    positions = np.arange(seq_len).reshape(1, seq_len).repeat(batch, axis=0).astype(np.int64)
    freq_base = np.float32(10000.0)
    freq_scale = np.float32(1.0)

    iree_result = rope_module.rope(input, positions, freq_base, freq_scale)
    oracle_result = rope_oracle(input, positions, freq_base, freq_scale)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_position_zero(rope_module):
    """Position 0 should have minimal rotation (angle=0 for all dims)."""
    batch, seq_len, n_head, head_dim = 1, 1, 4, 8

    input = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32)
    positions = np.zeros((batch, seq_len), dtype=np.int64)
    freq_base = np.float32(10000.0)
    freq_scale = np.float32(1.0)

    iree_result = rope_module.rope(input, positions, freq_base, freq_scale)
    oracle_result = rope_oracle(input, positions, freq_base, freq_scale)

    # At position 0, cos(0)=1 and sin(0)=0, so output should equal input
    assert_close(iree_result, oracle_result)
    assert_close(iree_result, input)


def test_various_positions(rope_module):
    """Test with non-sequential positions."""
    np.random.seed(123)
    batch, seq_len, n_head, head_dim = 2, 4, 4, 8

    input = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32)
    positions = np.array([[0, 10, 20, 30], [5, 15, 25, 35]], dtype=np.int64)
    freq_base = np.float32(10000.0)
    freq_scale = np.float32(1.0)

    iree_result = rope_module.rope(input, positions, freq_base, freq_scale)
    oracle_result = rope_oracle(input, positions, freq_base, freq_scale)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_different_freq_base(rope_module):
    """Test with different frequency base (like Mixtral's 1000000)."""
    np.random.seed(456)
    batch, seq_len, n_head, head_dim = 2, 4, 4, 16

    input = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32)
    positions = np.arange(seq_len).reshape(1, seq_len).repeat(batch, axis=0).astype(np.int64)
    freq_base = np.float32(1000000.0)  # Mixtral-style
    freq_scale = np.float32(1.0)

    iree_result = rope_module.rope(input, positions, freq_base, freq_scale)
    oracle_result = rope_oracle(input, positions, freq_base, freq_scale)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_freq_scale(rope_module):
    """Test with non-unit frequency scale."""
    np.random.seed(789)
    batch, seq_len, n_head, head_dim = 2, 4, 4, 8

    input = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32)
    positions = np.arange(seq_len).reshape(1, seq_len).repeat(batch, axis=0).astype(np.int64)
    freq_base = np.float32(10000.0)
    freq_scale = np.float32(0.5)

    iree_result = rope_module.rope(input, positions, freq_base, freq_scale)
    oracle_result = rope_oracle(input, positions, freq_base, freq_scale)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)

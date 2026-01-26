"""Tests for SwiGLU activation component."""

import numpy as np
import pytest

from tests.utils import compile_component, assert_close
from oracles.activation import swiglu as swiglu_oracle


@pytest.fixture(scope="module")
def swiglu_module(rt):
    return compile_component("activation/swiglu.mlir", rt)


def test_basic_correctness(swiglu_module):
    """Test SwiGLU against oracle with random data."""
    np.random.seed(42)
    gate = np.random.randn(2, 4, 8).astype(np.float32)
    up = np.random.randn(2, 4, 8).astype(np.float32)

    iree_result = swiglu_module.swiglu(gate, up)
    oracle_result = swiglu_oracle(gate, up)

    assert_close(iree_result, oracle_result)


def test_zeros(swiglu_module):
    """Test with zero inputs."""
    gate = np.zeros((2, 4, 8), dtype=np.float32)
    up = np.zeros((2, 4, 8), dtype=np.float32)

    iree_result = swiglu_module.swiglu(gate, up)
    oracle_result = swiglu_oracle(gate, up)

    assert_close(iree_result, oracle_result)


def test_positive_values(swiglu_module):
    """Test with positive values only."""
    np.random.seed(123)
    gate = np.abs(np.random.randn(2, 4, 8).astype(np.float32)) + 0.1
    up = np.abs(np.random.randn(2, 4, 8).astype(np.float32)) + 0.1

    iree_result = swiglu_module.swiglu(gate, up)
    oracle_result = swiglu_oracle(gate, up)

    assert_close(iree_result, oracle_result)


def test_large_values(swiglu_module):
    """Test numerical stability with large values."""
    np.random.seed(456)
    gate = np.random.randn(2, 4, 8).astype(np.float32) * 10
    up = np.random.randn(2, 4, 8).astype(np.float32) * 10

    iree_result = swiglu_module.swiglu(gate, up)
    oracle_result = swiglu_oracle(gate, up)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)

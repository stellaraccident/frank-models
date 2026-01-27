"""Tests for RMS Normalization component."""

import numpy as np
import pytest

from tests.utils import compile_component, assert_close
from oracles.normalization import rms_norm as rms_norm_oracle


@pytest.fixture(scope="module")
def rms_norm_module(iree_cfg):
    """Compile the RMS norm component once per test module."""
    return compile_component("normalization/rms_norm.mlir", iree_cfg)


def test_basic_correctness(rms_norm_module):
    np.random.seed(42)
    x = np.random.randn(4, 8).astype(np.float32)
    weight = np.random.randn(8).astype(np.float32)
    eps = 1e-6

    iree_result = rms_norm_module.rms_norm_linalg(x, weight, eps)
    oracle_result = rms_norm_oracle(x, weight, eps)

    assert_close(iree_result, oracle_result)


def test_zeros_input(rms_norm_module):
    x = np.zeros((4, 8), dtype=np.float32)
    weight = np.ones(8, dtype=np.float32)
    eps = 1e-6

    iree_result = rms_norm_module.rms_norm_linalg(x, weight, eps)
    oracle_result = rms_norm_oracle(x, weight, eps)

    assert_close(iree_result, oracle_result)


def test_ones_weight(rms_norm_module):
    np.random.seed(123)
    x = np.random.randn(4, 8).astype(np.float32)
    weight = np.ones(8, dtype=np.float32)
    eps = 1e-6

    iree_result = rms_norm_module.rms_norm_linalg(x, weight, eps)
    oracle_result = rms_norm_oracle(x, weight, eps)

    assert_close(iree_result, oracle_result)


def test_large_values(rms_norm_module):
    x = np.full((4, 8), 1000.0, dtype=np.float32)
    weight = np.ones(8, dtype=np.float32)
    eps = 1e-6

    iree_result = rms_norm_module.rms_norm_linalg(x, weight, eps)
    oracle_result = rms_norm_oracle(x, weight, eps)

    assert_close(iree_result, oracle_result)


def test_small_values(rms_norm_module):
    x = np.full((4, 8), 1e-4, dtype=np.float32)
    weight = np.ones(8, dtype=np.float32)
    eps = 1e-6

    iree_result = rms_norm_module.rms_norm_linalg(x, weight, eps)
    oracle_result = rms_norm_oracle(x, weight, eps)

    assert_close(iree_result, oracle_result)


def test_negative_values(rms_norm_module):
    np.random.seed(456)
    x = np.random.randn(4, 8).astype(np.float32) - 5.0
    weight = np.random.randn(8).astype(np.float32)
    eps = 1e-6

    iree_result = rms_norm_module.rms_norm_linalg(x, weight, eps)
    oracle_result = rms_norm_oracle(x, weight, eps)

    assert_close(iree_result, oracle_result)


def test_different_eps(rms_norm_module):
    np.random.seed(789)
    x = np.random.randn(4, 8).astype(np.float32)
    weight = np.ones(8, dtype=np.float32)

    for eps in [1e-5, 1e-6, 1e-8]:
        iree_result = rms_norm_module.rms_norm_linalg(x, weight, eps)
        oracle_result = rms_norm_oracle(x, weight, eps)
        assert_close(iree_result, oracle_result)


def test_various_shapes(rms_norm_module):
    eps = 1e-6

    for batch, hidden in [(1, 16), (8, 32), (16, 64), (32, 128)]:
        np.random.seed(batch * hidden)
        x = np.random.randn(batch, hidden).astype(np.float32)
        weight = np.random.randn(hidden).astype(np.float32)

        iree_result = rms_norm_module.rms_norm_linalg(x, weight, eps)
        oracle_result = rms_norm_oracle(x, weight, eps)
        assert_close(iree_result, oracle_result)

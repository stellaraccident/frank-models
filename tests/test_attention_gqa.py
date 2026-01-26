"""Tests for Multi-Head Attention component.

Note: Current implementation requires n_head == n_head_kv (MHA).
GQA/MQA support (n_head > n_head_kv) is future work.

XFAIL: Compiler crashes with dynamic shapes + collapse/expand around
iree_linalg_ext.attention. Tracked for fix.
"""

import numpy as np
import pytest

from tests.utils import compile_component, assert_close
from oracles.attention import attention_gqa as attention_oracle


@pytest.fixture(scope="module")
def attention_module(rt):
    pytest.xfail("IREE compiler crash: dynamic shapes + attention op")
    return compile_component("attention/attention_gqa.mlir", rt)


def test_mha_basic(attention_module):
    """Test MHA with standard dimensions."""
    np.random.seed(42)
    batch, seq_len, n_head, head_dim = 2, 4, 8, 16

    query = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    key = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    value = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    scale = np.float32(1.0 / np.sqrt(head_dim))

    iree_result = attention_module.attention_gqa(query, key, value, scale)
    oracle_result = attention_oracle(query, key, value, scale)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_mha_small(attention_module):
    """Test MHA with small dimensions."""
    np.random.seed(123)
    batch, seq_len, n_head, head_dim = 1, 3, 4, 8

    query = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    key = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    value = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    scale = np.float32(1.0 / np.sqrt(head_dim))

    iree_result = attention_module.attention_gqa(query, key, value, scale)
    oracle_result = attention_oracle(query, key, value, scale)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_various_scales(attention_module):
    """Test with different scale values."""
    np.random.seed(789)
    batch, seq_len, n_head, head_dim = 2, 4, 4, 32

    query = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    key = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    value = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1

    for scale_val in [0.1, 0.5, 1.0]:
        scale = np.float32(scale_val)
        iree_result = attention_module.attention_gqa(query, key, value, scale)
        oracle_result = attention_oracle(query, key, value, scale)
        assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)

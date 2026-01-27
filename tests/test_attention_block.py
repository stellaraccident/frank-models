"""Tests for Full Attention Block component.

XFAIL: Depends on attention_gqa which crashes with dynamic shapes.
https://github.com/iree-org/iree/issues/23277
"""

import numpy as np
import pytest

from tests.utils import link_and_compile
from oracles.attention import attention_block as attention_block_oracle


@pytest.fixture(scope="module")
def attention_block_module(rt):
    pytest.xfail("iree-org/iree#23277: dynamic shapes + attention op")
    return link_and_compile(
        "attention/attention_block.mlir",
        [
            "position/rope.mlir",
            "attention/attention_gqa.mlir",
        ],
        rt,
    )


def test_mha_no_bias(attention_block_module):
    """Test Multi-Head Attention without biases."""
    np.random.seed(42)
    batch, seq_len = 2, 4
    n_head, n_head_kv = 4, 4  # MHA: same heads for Q and KV
    n_embd = 64
    head_dim = n_embd // n_head

    # Input
    input = np.random.randn(batch, seq_len, n_embd).astype(np.float32) * 0.1
    positions = (
        np.arange(seq_len).reshape(1, seq_len).repeat(batch, axis=0).astype(np.int64)
    )

    # Weights (orthogonal-ish initialization for stability)
    wq = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.1
    wk = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.1
    wv = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.1
    wo = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.1

    # Dummy biases (not used)
    bq = np.zeros(n_embd, dtype=np.float32)
    bk = np.zeros(n_embd, dtype=np.float32)
    bv = np.zeros(n_embd, dtype=np.float32)
    bo = np.zeros(n_embd, dtype=np.float32)

    use_bias = False
    rope_freq_base = np.float32(10000.0)
    rope_freq_scale = np.float32(1.0)

    iree_result = attention_block_module.attention_block(
        input,
        positions,
        wq,
        wk,
        wv,
        wo,
        bq,
        bk,
        bv,
        bo,
        use_bias,
        n_head,
        n_head_kv,
        n_embd,
        rope_freq_base,
        rope_freq_scale,
    )
    oracle_result = attention_block_oracle(
        input,
        positions,
        wq,
        wk,
        wv,
        wo,
        bq,
        bk,
        bv,
        bo,
        use_bias,
        n_head,
        n_head_kv,
        n_embd,
        rope_freq_base,
        rope_freq_scale,
    )

    np.testing.assert_allclose(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_mha_with_bias(attention_block_module):
    """Test Multi-Head Attention with biases enabled."""
    np.random.seed(123)
    batch, seq_len = 1, 8
    n_head, n_head_kv = 8, 8
    n_embd = 128

    input = np.random.randn(batch, seq_len, n_embd).astype(np.float32) * 0.1
    positions = (
        np.arange(seq_len).reshape(1, seq_len).repeat(batch, axis=0).astype(np.int64)
    )

    wq = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.1
    wk = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.1
    wv = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.1
    wo = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.1

    bq = np.random.randn(n_embd).astype(np.float32) * 0.01
    bk = np.random.randn(n_embd).astype(np.float32) * 0.01
    bv = np.random.randn(n_embd).astype(np.float32) * 0.01
    bo = np.random.randn(n_embd).astype(np.float32) * 0.01

    use_bias = True
    rope_freq_base = np.float32(10000.0)
    rope_freq_scale = np.float32(1.0)

    iree_result = attention_block_module.attention_block(
        input,
        positions,
        wq,
        wk,
        wv,
        wo,
        bq,
        bk,
        bv,
        bo,
        use_bias,
        n_head,
        n_head_kv,
        n_embd,
        rope_freq_base,
        rope_freq_scale,
    )
    oracle_result = attention_block_oracle(
        input,
        positions,
        wq,
        wk,
        wv,
        wo,
        bq,
        bk,
        bv,
        bo,
        use_bias,
        n_head,
        n_head_kv,
        n_embd,
        rope_freq_base,
        rope_freq_scale,
    )

    np.testing.assert_allclose(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_gqa(attention_block_module):
    """Test Grouped Query Attention (n_head > n_head_kv)."""
    np.random.seed(456)
    batch, seq_len = 2, 6
    n_head = 8
    n_head_kv = 2  # GQA: 4x fewer KV heads
    n_embd = 64
    n_embd_kv = n_embd * n_head_kv // n_head

    input = np.random.randn(batch, seq_len, n_embd).astype(np.float32) * 0.1
    positions = (
        np.arange(seq_len).reshape(1, seq_len).repeat(batch, axis=0).astype(np.int64)
    )

    wq = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.1
    wk = np.random.randn(n_embd, n_embd_kv).astype(np.float32) * 0.1
    wv = np.random.randn(n_embd, n_embd_kv).astype(np.float32) * 0.1
    wo = np.random.randn(n_embd, n_embd).astype(np.float32) * 0.1

    bq = np.zeros(n_embd, dtype=np.float32)
    bk = np.zeros(n_embd_kv, dtype=np.float32)
    bv = np.zeros(n_embd_kv, dtype=np.float32)
    bo = np.zeros(n_embd, dtype=np.float32)

    use_bias = False
    rope_freq_base = np.float32(1000000.0)  # Mixtral-style
    rope_freq_scale = np.float32(1.0)

    iree_result = attention_block_module.attention_block(
        input,
        positions,
        wq,
        wk,
        wv,
        wo,
        bq,
        bk,
        bv,
        bo,
        use_bias,
        n_head,
        n_head_kv,
        n_embd,
        rope_freq_base,
        rope_freq_scale,
    )
    oracle_result = attention_block_oracle(
        input,
        positions,
        wq,
        wk,
        wv,
        wo,
        bq,
        bk,
        bv,
        bo,
        use_bias,
        n_head,
        n_head_kv,
        n_embd,
        rope_freq_base,
        rope_freq_scale,
    )

    np.testing.assert_allclose(iree_result, oracle_result, rtol=1e-4, atol=1e-5)

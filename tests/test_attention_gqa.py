"""Tests for Multi-Head Attention component.

Note: Current implementation requires n_head == n_head_kv (MHA).
GQA/MQA support (n_head > n_head_kv) is future work.

Fully dynamic shapes crash the compiler (iree-org/iree#23277).
Static-shaped wrapper tests use iree-link to call the dynamic component
from a static @main, relying on IPO to specialize shapes at compile time.
"""

import numpy as np
import pytest

from tests.utils import compile_component, link_and_compile, assert_close
from oracles.attention import attention_gqa as attention_oracle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_static_wrapper(batch, seq_len, n_head, head_dim):
    """Generate a static-shaped wrapper that calls dynamic attention_gqa."""
    shape = f"{batch}x{seq_len}x{n_head}x{head_dim}"
    return f"""
module @test_wrapper {{
  util.func private @attention_components.attention_gqa(
      tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, f32
  ) -> tensor<?x?x?x?xf32>

  util.func public @main(
      %q: tensor<{shape}xf32>, %k: tensor<{shape}xf32>,
      %v: tensor<{shape}xf32>, %scale: f32
  ) -> tensor<{shape}xf32> {{
    %q_dyn = tensor.cast %q : tensor<{shape}xf32> to tensor<?x?x?x?xf32>
    %k_dyn = tensor.cast %k : tensor<{shape}xf32> to tensor<?x?x?x?xf32>
    %v_dyn = tensor.cast %v : tensor<{shape}xf32> to tensor<?x?x?x?xf32>
    %out_dyn = util.call @attention_components.attention_gqa(
        %q_dyn, %k_dyn, %v_dyn, %scale)
        : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, f32)
        -> tensor<?x?x?x?xf32>
    %out = tensor.cast %out_dyn : tensor<?x?x?x?xf32> to tensor<{shape}xf32>
    util.return %out : tensor<{shape}xf32>
  }}
}}"""


def _compile_static_attention(iree_cfg, batch, seq_len, n_head, head_dim):
    """Compile a static-shaped attention wrapper via iree-link."""
    wrapper = _make_static_wrapper(batch, seq_len, n_head, head_dim)
    return link_and_compile(
        main_source=wrapper,
        library_paths=["attention/attention_gqa.mlir"],
        iree_cfg=iree_cfg,
        debug_name=f"attention_gqa_static_{batch}x{seq_len}x{n_head}x{head_dim}",
    )


# ---------------------------------------------------------------------------
# Static wrapper tests (IPO shape specialization)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def static_attention_2x4x8x16(iree_cfg):
    return _compile_static_attention(iree_cfg, 2, 4, 8, 16)


@pytest.fixture(scope="module")
def static_attention_1x3x4x8(iree_cfg):
    return _compile_static_attention(iree_cfg, 1, 3, 4, 8)


@pytest.fixture(scope="module")
def static_attention_2x4x4x32(iree_cfg):
    return _compile_static_attention(iree_cfg, 2, 4, 4, 32)


def test_mha_static_basic(static_attention_2x4x8x16):
    """Test MHA with standard dimensions via static wrapper."""
    np.random.seed(42)
    batch, seq_len, n_head, head_dim = 2, 4, 8, 16

    query = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    key = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    value = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    scale = np.float32(1.0 / np.sqrt(head_dim))

    iree_result = static_attention_2x4x8x16.main(query, key, value, scale)
    oracle_result = attention_oracle(query, key, value, scale)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_mha_static_small(static_attention_1x3x4x8):
    """Test MHA with small dimensions via static wrapper."""
    np.random.seed(123)
    batch, seq_len, n_head, head_dim = 1, 3, 4, 8

    query = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    key = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    value = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    scale = np.float32(1.0 / np.sqrt(head_dim))

    iree_result = static_attention_1x3x4x8.main(query, key, value, scale)
    oracle_result = attention_oracle(query, key, value, scale)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_various_scales_static(static_attention_2x4x4x32):
    """Test with different scale values via static wrapper."""
    np.random.seed(789)
    batch, seq_len, n_head, head_dim = 2, 4, 4, 32

    query = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    key = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    value = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1

    for scale_val in [0.1, 0.5, 1.0]:
        scale = np.float32(scale_val)
        iree_result = static_attention_2x4x4x32.main(query, key, value, scale)
        oracle_result = attention_oracle(query, key, value, scale)
        assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Fully dynamic tests (xfailed - iree-org/iree#23277)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def attention_module(iree_cfg):
    pytest.xfail("iree-org/iree#23277: dynamic shapes + attention op")
    return compile_component("attention/attention_gqa.mlir", iree_cfg)


def test_mha_dynamic_basic(attention_module):
    """Test MHA with fully dynamic shapes (xfailed)."""
    np.random.seed(42)
    batch, seq_len, n_head, head_dim = 2, 4, 8, 16

    query = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    key = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    value = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    scale = np.float32(1.0 / np.sqrt(head_dim))

    iree_result = attention_module.attention_gqa(query, key, value, scale)
    oracle_result = attention_oracle(query, key, value, scale)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_mha_dynamic_small(attention_module):
    """Test MHA with small dimensions, fully dynamic (xfailed)."""
    np.random.seed(123)
    batch, seq_len, n_head, head_dim = 1, 3, 4, 8

    query = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    key = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    value = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    scale = np.float32(1.0 / np.sqrt(head_dim))

    iree_result = attention_module.attention_gqa(query, key, value, scale)
    oracle_result = attention_oracle(query, key, value, scale)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_various_scales_dynamic(attention_module):
    """Test with different scale values, fully dynamic (xfailed)."""
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

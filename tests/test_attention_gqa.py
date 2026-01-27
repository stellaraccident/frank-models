"""Tests for Multi-Head Attention component.

Note: Current implementation requires n_head == n_head_kv (MHA).
GQA/MQA support (n_head > n_head_kv) is future work.

Fully dynamic shapes crash the compiler (iree-org/iree#23277).
Three test tiers use iree-link wrappers with increasing dynamism:
  1. Fully static - all dims concrete
  2. Dynamic batch/seq - n_head/head_dim static, seq_len divisible by 32
  3. Fully dynamic - all dims dynamic (xfailed, iree#23277)
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


def _make_dynamic_batch_seq_wrapper(n_head, head_dim, seq_div=32):
    """Generate a wrapper with dynamic batch/seq_len, static n_head/head_dim.

    seq_len is constrained to be divisible by seq_div.
    """
    dyn_shape = f"?x?x{n_head}x{head_dim}"
    return f"""
module @test_wrapper {{
  util.func private @attention_components.attention_gqa(
      tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, f32
  ) -> tensor<?x?x?x?xf32>

  util.func public @main(
      %q: tensor<{dyn_shape}xf32>, %k: tensor<{dyn_shape}xf32>,
      %v: tensor<{dyn_shape}xf32>, %scale: f32
  ) -> tensor<{dyn_shape}xf32> {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %batch_raw = tensor.dim %q, %c0 : tensor<{dyn_shape}xf32>
    %seq_raw = tensor.dim %q, %c1 : tensor<{dyn_shape}xf32>
    %batch = util.assume.int %batch_raw<umin=1, umax=64> : index
    %seq = util.assume.int %seq_raw<umin={seq_div}, umax=4096, udiv={seq_div}> : index

    %q_dyn = tensor.cast %q : tensor<{dyn_shape}xf32> to tensor<?x?x?x?xf32>
    %k_dyn = tensor.cast %k : tensor<{dyn_shape}xf32> to tensor<?x?x?x?xf32>
    %v_dyn = tensor.cast %v : tensor<{dyn_shape}xf32> to tensor<?x?x?x?xf32>
    %out_dyn = util.call @attention_components.attention_gqa(
        %q_dyn, %k_dyn, %v_dyn, %scale)
        : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, f32)
        -> tensor<?x?x?x?xf32>
    %out = tensor.cast %out_dyn : tensor<?x?x?x?xf32> to tensor<{dyn_shape}xf32>
    util.return %out : tensor<{dyn_shape}xf32>
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


def _compile_dynamic_batch_seq_attention(iree_cfg, n_head, head_dim, seq_div=32):
    """Compile attention with dynamic batch/seq_len, static n_head/head_dim."""
    wrapper = _make_dynamic_batch_seq_wrapper(n_head, head_dim, seq_div)
    return link_and_compile(
        main_source=wrapper,
        library_paths=["attention/attention_gqa.mlir"],
        iree_cfg=iree_cfg,
        debug_name=f"attention_gqa_dynbs_{n_head}x{head_dim}_div{seq_div}",
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
# Dynamic batch/seq_len tests (static n_head/head_dim, seq_len % 32 == 0)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dynamic_attention_8x16(iree_cfg):
    return _compile_dynamic_batch_seq_attention(iree_cfg, n_head=8, head_dim=16)


@pytest.fixture(scope="module")
def dynamic_attention_4x32(iree_cfg):
    return _compile_dynamic_batch_seq_attention(iree_cfg, n_head=4, head_dim=32)


def test_dynamic_batch_seq_single(dynamic_attention_8x16):
    """Test with single batch, seq_len=32."""
    np.random.seed(42)
    batch, seq_len, n_head, head_dim = 1, 32, 8, 16

    query = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    key = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    value = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    scale = np.float32(1.0 / np.sqrt(head_dim))

    iree_result = dynamic_attention_8x16.main(query, key, value, scale)
    oracle_result = attention_oracle(query, key, value, scale)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_dynamic_batch_seq_larger(dynamic_attention_8x16):
    """Test with batch=4, seq_len=64 using the same compiled module."""
    np.random.seed(101)
    batch, seq_len, n_head, head_dim = 4, 64, 8, 16

    query = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    key = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    value = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    scale = np.float32(1.0 / np.sqrt(head_dim))

    iree_result = dynamic_attention_8x16.main(query, key, value, scale)
    oracle_result = attention_oracle(query, key, value, scale)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_dynamic_batch_seq_128(dynamic_attention_8x16):
    """Test with seq_len=128 to exercise longer sequences."""
    np.random.seed(202)
    batch, seq_len, n_head, head_dim = 2, 128, 8, 16

    query = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    key = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    value = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    scale = np.float32(1.0 / np.sqrt(head_dim))

    iree_result = dynamic_attention_8x16.main(query, key, value, scale)
    oracle_result = attention_oracle(query, key, value, scale)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_dynamic_batch_seq_different_head_config(dynamic_attention_4x32):
    """Test dynamic batch/seq with different head configuration."""
    np.random.seed(303)
    batch, seq_len, n_head, head_dim = 3, 64, 4, 32

    query = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    key = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    value = np.random.randn(batch, seq_len, n_head, head_dim).astype(np.float32) * 0.1
    scale = np.float32(1.0 / np.sqrt(head_dim))

    iree_result = dynamic_attention_4x32.main(query, key, value, scale)
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

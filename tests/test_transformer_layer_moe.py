"""Tests for transformer_layer_moe (full MoE transformer layer with parameters)."""

import numpy as np
import pytest

from layers.transformer_layer_moe import (
    MoELayerConfig,
    generate_specialized_module,
    generate_random_params,
)
from oracles.transformer_layer import transformer_layer_moe as oracle
from tests.utils import link_and_compile_with_params, assert_close


# Small Mixtral-like test config.
TEST_CFG = MoELayerConfig(
    n_embd=64,
    n_head=4,
    n_head_kv=2,
    head_dim=16,
    n_ff=128,
    n_expert=4,
    n_expert_used=2,
    batch=1,
    seq_len=4,
    layer_idx=0,
    rms_eps=1e-5,
    rope_freq_base=10000.0,
    rope_freq_scale=1.0,
    use_bias=False,
    normalize_weights=False,
)

# Compute component libraries (linked into the combined main module).
COMPONENT_LIBRARIES = [
    "normalization/rms_norm.mlir",
    "attention/attention_block.mlir",
    "attention/attention_gqa.mlir",
    "position/rope.mlir",
    "moe/moe_ffn_block.mlir",
    "moe/mul_mat_id.mlir",
    "activation/swiglu.mlir",
]


@pytest.fixture(scope="module")
def layer_module(iree_cfg):
    """Compile MoE transformer layer with random parameters.

    The generated module contains the layer function inlined along with
    parameter accessors and the static @forward wrapper. Only compute
    component imports are resolved by iree-link.
    """
    params = generate_random_params(TEST_CFG, seed=42, scale=0.1)
    specialized_mlir = generate_specialized_module(TEST_CFG, scope="model")

    return link_and_compile_with_params(
        main_source=specialized_mlir,
        library_paths=COMPONENT_LIBRARIES,
        iree_cfg=iree_cfg,
        params=params,
        scope="model",
        debug_name="transformer_layer_moe_linked",
    )


@pytest.fixture(scope="module")
def params():
    """Random parameters matching the test config."""
    return generate_random_params(TEST_CFG, seed=42, scale=0.1)


def test_forward_basic(layer_module, params):
    """Test full layer forward pass against oracle."""
    rng = np.random.RandomState(100)
    cfg = TEST_CFG

    input = rng.randn(cfg.batch, cfg.seq_len, cfg.n_embd).astype(np.float32) * 0.1
    positions = np.arange(cfg.seq_len, dtype=np.int64).reshape(cfg.batch, cfg.seq_len)

    iree_result = layer_module.forward(input, positions)

    oracle_result = oracle(
        input, positions, params,
        layer_idx=cfg.layer_idx,
        n_head=cfg.n_head,
        n_head_kv=cfg.n_head_kv,
        n_embd=cfg.n_embd,
        n_ff=cfg.n_ff,
        n_expert=cfg.n_expert,
        n_expert_used=cfg.n_expert_used,
        rms_eps=cfg.rms_eps,
        rope_freq_base=cfg.rope_freq_base,
        rope_freq_scale=cfg.rope_freq_scale,
        use_bias=cfg.use_bias,
        normalize_weights=cfg.normalize_weights,
    )

    assert iree_result.shape == oracle_result.shape
    assert_close(iree_result, oracle_result, rtol=1e-3, atol=1e-4)


def test_output_shape(layer_module):
    """Verify output shape matches input shape."""
    cfg = TEST_CFG
    rng = np.random.RandomState(200)

    input = rng.randn(cfg.batch, cfg.seq_len, cfg.n_embd).astype(np.float32) * 0.1
    positions = np.arange(cfg.seq_len, dtype=np.int64).reshape(cfg.batch, cfg.seq_len)

    result = layer_module.forward(input, positions)
    assert result.shape == (cfg.batch, cfg.seq_len, cfg.n_embd)


def test_residual_passthrough(layer_module, params):
    """With zero params, output should approximate input (residual connections)."""
    cfg = TEST_CFG
    rng = np.random.RandomState(300)

    # Create zero params — residual connections should pass input through.
    zero_params = {k: np.zeros_like(v) for k, v in params.items()}

    oracle_input = rng.randn(cfg.batch, cfg.seq_len, cfg.n_embd).astype(np.float32)
    positions = np.arange(cfg.seq_len, dtype=np.int64).reshape(cfg.batch, cfg.seq_len)

    # With zero weights, attention and MoE outputs should be ~zero,
    # so output ≈ input + 0 + 0 = input.
    # (We can only test this via oracle since the compiled module has fixed params.)
    oracle_result = oracle(
        oracle_input, positions, zero_params,
        layer_idx=cfg.layer_idx,
        n_head=cfg.n_head,
        n_head_kv=cfg.n_head_kv,
        n_embd=cfg.n_embd,
        n_ff=cfg.n_ff,
        n_expert=cfg.n_expert,
        n_expert_used=cfg.n_expert_used,
        rms_eps=cfg.rms_eps,
        rope_freq_base=cfg.rope_freq_base,
        rope_freq_scale=cfg.rope_freq_scale,
        use_bias=cfg.use_bias,
        normalize_weights=cfg.normalize_weights,
    )

    # With zero projection weights, the output should be close to input.
    # rms_norm still normalizes, but zero projections produce zero attention/FFN.
    assert_close(oracle_result, oracle_input, rtol=1e-3, atol=1e-4)

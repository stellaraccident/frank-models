"""Tests for moe_ffn_block (full MoE FFN layer)."""

import numpy as np
import pytest

from tests.utils import link_and_compile, assert_close
from oracles.moe import moe_ffn_block as moe_ffn_block_oracle


@pytest.fixture(scope="module")
def moe_ffn_module(rt):
    """Link moe_ffn_block with its dependencies (mul_mat_id, swiglu)."""
    return link_and_compile(
        "moe/moe_ffn_block.mlir",
        ["moe/mul_mat_id.mlir", "activation/swiglu.mlir"],
        rt,
    )


def test_basic_moe(moe_ffn_module):
    """Test basic MoE FFN with small dimensions."""
    np.random.seed(42)

    # Small dimensions for quick testing
    n_tokens, n_embd, n_ff = 4, 8, 16
    n_expert, n_expert_used = 4, 2

    # Create random weights
    input = np.random.randn(n_tokens, n_embd).astype(np.float32) * 0.1
    gate_inp_w = np.random.randn(n_expert, n_embd).astype(np.float32) * 0.1
    up_exps_w = np.random.randn(n_ff, n_embd, n_expert).astype(np.float32) * 0.1
    gate_exps_w = np.random.randn(n_ff, n_embd, n_expert).astype(np.float32) * 0.1
    down_exps_w = np.random.randn(n_embd, n_ff, n_expert).astype(np.float32) * 0.1

    normalize_weights = False

    iree_result = moe_ffn_module.moe_ffn_block(
        input,
        gate_inp_w,
        up_exps_w,
        gate_exps_w,
        down_exps_w,
        n_expert,
        n_expert_used,
        n_embd,
        n_ff,
        normalize_weights,
    )

    oracle_result = moe_ffn_block_oracle(
        input,
        gate_inp_w,
        up_exps_w,
        gate_exps_w,
        down_exps_w,
        n_expert,
        n_expert_used,
        n_embd,
        n_ff,
        normalize_weights,
    )

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_moe_with_normalization(moe_ffn_module):
    """Test MoE with weight normalization enabled."""
    np.random.seed(123)

    n_tokens, n_embd, n_ff = 3, 8, 12
    n_expert, n_expert_used = 4, 2

    input = np.random.randn(n_tokens, n_embd).astype(np.float32) * 0.1
    gate_inp_w = np.random.randn(n_expert, n_embd).astype(np.float32) * 0.1
    up_exps_w = np.random.randn(n_ff, n_embd, n_expert).astype(np.float32) * 0.1
    gate_exps_w = np.random.randn(n_ff, n_embd, n_expert).astype(np.float32) * 0.1
    down_exps_w = np.random.randn(n_embd, n_ff, n_expert).astype(np.float32) * 0.1

    normalize_weights = True

    iree_result = moe_ffn_module.moe_ffn_block(
        input,
        gate_inp_w,
        up_exps_w,
        gate_exps_w,
        down_exps_w,
        n_expert,
        n_expert_used,
        n_embd,
        n_ff,
        normalize_weights,
    )

    oracle_result = moe_ffn_block_oracle(
        input,
        gate_inp_w,
        up_exps_w,
        gate_exps_w,
        down_exps_w,
        n_expert,
        n_expert_used,
        n_embd,
        n_ff,
        normalize_weights,
    )

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_single_expert(moe_ffn_module):
    """Test with n_expert_used=1 (single expert per token)."""
    np.random.seed(456)

    n_tokens, n_embd, n_ff = 4, 8, 16
    n_expert, n_expert_used = 8, 1

    input = np.random.randn(n_tokens, n_embd).astype(np.float32) * 0.1
    gate_inp_w = np.random.randn(n_expert, n_embd).astype(np.float32) * 0.1
    up_exps_w = np.random.randn(n_ff, n_embd, n_expert).astype(np.float32) * 0.1
    gate_exps_w = np.random.randn(n_ff, n_embd, n_expert).astype(np.float32) * 0.1
    down_exps_w = np.random.randn(n_embd, n_ff, n_expert).astype(np.float32) * 0.1

    normalize_weights = False

    iree_result = moe_ffn_module.moe_ffn_block(
        input,
        gate_inp_w,
        up_exps_w,
        gate_exps_w,
        down_exps_w,
        n_expert,
        n_expert_used,
        n_embd,
        n_ff,
        normalize_weights,
    )

    oracle_result = moe_ffn_block_oracle(
        input,
        gate_inp_w,
        up_exps_w,
        gate_exps_w,
        down_exps_w,
        n_expert,
        n_expert_used,
        n_embd,
        n_ff,
        normalize_weights,
    )

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_mixtral_like_dims(moe_ffn_module):
    """Test with Mixtral-like dimensions (scaled down)."""
    np.random.seed(789)

    # Scaled-down Mixtral: 8 experts, top-2
    n_tokens, n_embd, n_ff = 8, 32, 64
    n_expert, n_expert_used = 8, 2

    input = np.random.randn(n_tokens, n_embd).astype(np.float32) * 0.1
    gate_inp_w = np.random.randn(n_expert, n_embd).astype(np.float32) * 0.1
    up_exps_w = np.random.randn(n_ff, n_embd, n_expert).astype(np.float32) * 0.1
    gate_exps_w = np.random.randn(n_ff, n_embd, n_expert).astype(np.float32) * 0.1
    down_exps_w = np.random.randn(n_embd, n_ff, n_expert).astype(np.float32) * 0.1

    normalize_weights = False  # Mixtral doesn't normalize

    iree_result = moe_ffn_module.moe_ffn_block(
        input,
        gate_inp_w,
        up_exps_w,
        gate_exps_w,
        down_exps_w,
        n_expert,
        n_expert_used,
        n_embd,
        n_ff,
        normalize_weights,
    )

    oracle_result = moe_ffn_block_oracle(
        input,
        gate_inp_w,
        up_exps_w,
        gate_exps_w,
        down_exps_w,
        n_expert,
        n_expert_used,
        n_embd,
        n_ff,
        normalize_weights,
    )

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)

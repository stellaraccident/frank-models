"""Tests for concat_gemm_id_silu (fused concat + gather + GEMM + SwiGLU)."""

import numpy as np
import pytest

from tests.utils import link_and_compile, assert_close
from oracles.moe import concat_gemm_id_silu as concat_gemm_id_silu_oracle


@pytest.fixture(scope="module")
def concat_gemm_id_silu_module(iree_cfg):
    return link_and_compile(
        main_path="moe/concat_gemm_id_silu.mlir",
        library_paths=["activation/swiglu.mlir"],
        iree_cfg=iree_cfg,
    )


def test_single_expert(concat_gemm_id_silu_module):
    """Test with n_expert_used=1 (single expert per token)."""
    np.random.seed(42)
    n_embd, n_ff, n_expert = 3, 4, 8
    n_expert_used, n_tokens = 1, 2

    input = np.random.randn(n_embd, n_tokens).astype(np.float32) * 0.1
    up_exps_w = np.random.randn(n_expert, n_ff, n_embd).astype(np.float32) * 0.1
    gate_exps_w = np.random.randn(n_expert, n_ff, n_embd).astype(np.float32) * 0.1
    ids = np.array([[0, 3]], dtype=np.int32)

    iree_result = concat_gemm_id_silu_module.concat_gemm_id_silu(
        input, up_exps_w, gate_exps_w, ids
    )
    oracle_result = concat_gemm_id_silu_oracle(input, up_exps_w, gate_exps_w, ids, n_ff)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_two_experts(concat_gemm_id_silu_module):
    """Test with n_expert_used=2 (typical MoE configuration)."""
    np.random.seed(123)
    n_embd, n_ff, n_expert = 4, 8, 8
    n_expert_used, n_tokens = 2, 4

    input = np.random.randn(n_embd, n_tokens).astype(np.float32) * 0.1
    up_exps_w = np.random.randn(n_expert, n_ff, n_embd).astype(np.float32) * 0.1
    gate_exps_w = np.random.randn(n_expert, n_ff, n_embd).astype(np.float32) * 0.1
    ids = np.array(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
        dtype=np.int32,
    )

    iree_result = concat_gemm_id_silu_module.concat_gemm_id_silu(
        input, up_exps_w, gate_exps_w, ids
    )
    oracle_result = concat_gemm_id_silu_oracle(input, up_exps_w, gate_exps_w, ids, n_ff)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_same_expert_all_tokens(concat_gemm_id_silu_module):
    """Test edge case: all tokens use the same expert."""
    np.random.seed(456)
    n_embd, n_ff, n_expert = 4, 6, 4
    n_expert_used, n_tokens = 1, 3

    input = np.random.randn(n_embd, n_tokens).astype(np.float32) * 0.1
    up_exps_w = np.random.randn(n_expert, n_ff, n_embd).astype(np.float32) * 0.1
    gate_exps_w = np.random.randn(n_expert, n_ff, n_embd).astype(np.float32) * 0.1
    ids = np.array([[2, 2, 2]], dtype=np.int32)

    iree_result = concat_gemm_id_silu_module.concat_gemm_id_silu(
        input, up_exps_w, gate_exps_w, ids
    )
    oracle_result = concat_gemm_id_silu_oracle(input, up_exps_w, gate_exps_w, ids, n_ff)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_larger_dimensions(concat_gemm_id_silu_module):
    """Test with larger, more realistic dimensions."""
    np.random.seed(789)
    n_embd, n_ff, n_expert = 32, 64, 8
    n_expert_used, n_tokens = 2, 16

    input = np.random.randn(n_embd, n_tokens).astype(np.float32) * 0.1
    up_exps_w = np.random.randn(n_expert, n_ff, n_embd).astype(np.float32) * 0.1
    gate_exps_w = np.random.randn(n_expert, n_ff, n_embd).astype(np.float32) * 0.1
    ids = np.random.randint(0, n_expert, size=(n_expert_used, n_tokens), dtype=np.int32)

    iree_result = concat_gemm_id_silu_module.concat_gemm_id_silu(
        input, up_exps_w, gate_exps_w, ids
    )
    oracle_result = concat_gemm_id_silu_oracle(input, up_exps_w, gate_exps_w, ids, n_ff)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)

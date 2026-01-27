"""Tests for mul_mat_id (expert-selected matrix multiply)."""

import numpy as np
import pytest

from tests.utils import compile_component, assert_close
from oracles.moe import mul_mat_id as mul_mat_id_oracle


@pytest.fixture(scope="module")
def mul_mat_id_module(iree_cfg):
    return compile_component("moe/mul_mat_id.mlir", iree_cfg)


def test_single_expert(mul_mat_id_module):
    """Test with n_expert_used=1 (single expert per token)."""
    np.random.seed(42)
    n_out, n_in, n_expert = 4, 3, 8
    n_expert_used, n_tokens = 1, 2

    weights = np.random.randn(n_out, n_in, n_expert).astype(np.float32) * 0.1
    input = np.random.randn(n_in, n_expert_used, n_tokens).astype(np.float32) * 0.1
    ids = np.array(
        [[0, 3]], dtype=np.int32
    )  # token 0 uses expert 0, token 1 uses expert 3

    iree_result = mul_mat_id_module.mul_mat_id(weights, input, ids)
    oracle_result = mul_mat_id_oracle(weights, input, ids)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_two_experts(mul_mat_id_module):
    """Test with n_expert_used=2 (typical MoE configuration)."""
    np.random.seed(123)
    n_out, n_in, n_expert = 8, 4, 8
    n_expert_used, n_tokens = 2, 4

    weights = np.random.randn(n_out, n_in, n_expert).astype(np.float32) * 0.1
    input = np.random.randn(n_in, n_expert_used, n_tokens).astype(np.float32) * 0.1
    # Each column is [expert_slot_0, expert_slot_1] for that token
    ids = np.array(
        [
            [0, 1, 2, 3],  # first expert choice per token
            [4, 5, 6, 7],  # second expert choice per token
        ],
        dtype=np.int32,
    )

    iree_result = mul_mat_id_module.mul_mat_id(weights, input, ids)
    oracle_result = mul_mat_id_oracle(weights, input, ids)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_same_expert_all_tokens(mul_mat_id_module):
    """Test edge case: all tokens use the same expert."""
    np.random.seed(456)
    n_out, n_in, n_expert = 6, 4, 4
    n_expert_used, n_tokens = 1, 3

    weights = np.random.randn(n_out, n_in, n_expert).astype(np.float32) * 0.1
    input = np.random.randn(n_in, n_expert_used, n_tokens).astype(np.float32) * 0.1
    ids = np.array([[2, 2, 2]], dtype=np.int32)  # all tokens use expert 2

    iree_result = mul_mat_id_module.mul_mat_id(weights, input, ids)
    oracle_result = mul_mat_id_oracle(weights, input, ids)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)


def test_larger_dimensions(mul_mat_id_module):
    """Test with larger, more realistic dimensions."""
    np.random.seed(789)
    n_out, n_in, n_expert = 64, 32, 8
    n_expert_used, n_tokens = 2, 16

    weights = np.random.randn(n_out, n_in, n_expert).astype(np.float32) * 0.1
    input = np.random.randn(n_in, n_expert_used, n_tokens).astype(np.float32) * 0.1
    ids = np.random.randint(0, n_expert, size=(n_expert_used, n_tokens), dtype=np.int32)

    iree_result = mul_mat_id_module.mul_mat_id(weights, input, ids)
    oracle_result = mul_mat_id_oracle(weights, input, ids)

    assert_close(iree_result, oracle_result, rtol=1e-4, atol=1e-5)

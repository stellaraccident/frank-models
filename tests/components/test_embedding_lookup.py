"""Tests for embedding lookup component."""

import numpy as np
import pytest

from tests.utils import compile_component, assert_close
from oracles.embedding import embedding_lookup as embedding_oracle


@pytest.fixture(scope="module")
def embedding_module(iree_cfg):
    return compile_component("embedding/embedding_lookup.mlir", iree_cfg)


def test_basic_correctness(embedding_module):
    """Test embedding lookup against oracle."""
    np.random.seed(42)
    vocab_size = 100
    n_embd = 32

    weight = np.random.randn(vocab_size, n_embd).astype(np.float32)
    indices = np.random.randint(0, vocab_size, size=(2, 8)).astype(np.int64)

    iree_result = embedding_module.embedding_lookup(weight, indices)
    oracle_result = embedding_oracle(weight, indices)

    assert_close(iree_result, oracle_result)


def test_single_batch(embedding_module):
    """Test with batch size 1."""
    np.random.seed(123)
    vocab_size = 50
    n_embd = 16

    weight = np.random.randn(vocab_size, n_embd).astype(np.float32)
    indices = np.random.randint(0, vocab_size, size=(1, 4)).astype(np.int64)

    iree_result = embedding_module.embedding_lookup(weight, indices)
    oracle_result = embedding_oracle(weight, indices)

    assert_close(iree_result, oracle_result)


def test_sequential_indices(embedding_module):
    """Test with sequential token indices."""
    vocab_size = 100
    n_embd = 32

    weight = np.random.randn(vocab_size, n_embd).astype(np.float32)
    indices = np.arange(10).reshape(2, 5).astype(np.int64)

    iree_result = embedding_module.embedding_lookup(weight, indices)
    oracle_result = embedding_oracle(weight, indices)

    assert_close(iree_result, oracle_result)


def test_repeated_indices(embedding_module):
    """Test with repeated token indices."""
    vocab_size = 100
    n_embd = 32

    weight = np.random.randn(vocab_size, n_embd).astype(np.float32)
    indices = np.array([[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.int64)

    iree_result = embedding_module.embedding_lookup(weight, indices)
    oracle_result = embedding_oracle(weight, indices)

    assert_close(iree_result, oracle_result)

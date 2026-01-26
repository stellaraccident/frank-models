"""Test iree-link module linking infrastructure."""

import numpy as np
import pytest

from tests.utils import link_and_compile, compile_component, assert_close


@pytest.fixture(scope="module")
def linked_module(rt):
    """Link use_math with math_ops."""
    return link_and_compile(
        "test_linking/use_math.mlir",
        ["test_linking/math_ops.mlir"],
        rt,
    )


@pytest.fixture(scope="module")
def standalone_math_module(rt):
    """Compile math_ops standalone (no linking needed)."""
    return compile_component("test_linking/math_ops.mlir", rt)


def test_standalone_add(standalone_math_module):
    """Test add_tensors works standalone."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    result = standalone_math_module.add_tensors(a, b)
    assert_close(result, [5.0, 7.0, 9.0])


def test_standalone_mul(standalone_math_module):
    """Test mul_tensors works standalone."""
    a = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    result = standalone_math_module.mul_tensors(a, b)
    assert_close(result, [10.0, 18.0, 28.0])


def test_linked_add_then_mul(linked_module):
    """Test add_then_mul calls linked functions: (a + b) * c."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    c = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    # (1+4)*2=10, (2+5)*2=14, (3+6)*2=18
    result = linked_module.add_then_mul(a, b, c)
    assert_close(result, [10.0, 14.0, 18.0])


def test_linked_larger_tensors(linked_module):
    """Test with larger tensors."""
    np.random.seed(42)
    a = np.random.randn(100).astype(np.float32)
    b = np.random.randn(100).astype(np.float32)
    c = np.random.randn(100).astype(np.float32)
    result = linked_module.add_then_mul(a, b, c)
    expected = (a + b) * c
    assert_close(result, expected)
